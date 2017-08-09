// TODO:
// - don't make Iterator main API; it's a pain to check for Some(true) all the time.
// - go through comments
// - use log2 instead of ln

//! Efficient Bernoulli sampling.
//!
//! The `FastBernoulliTrial` type is an iterator that produces a series of `bool` values,
//! where you choose the probability of it producing `true`. It's designed to help select
//! events to sample when measuring a program's dynamic behavior.
//!
//! Each `bool` value produced has the same probability of being `true`, regardless of its
//! position in the sequence and the values before and after it. Producing a `false` value
//! is quick, just decrementing a counter and comparing it with zero. `FastBernoulliTrial`
//! only consults the underlying random number generator when it returns `true`, so if you
//! are sampling very common events with low probability, the overhead is quite low,
//! despite its desirable statistical properties.
//!
//! ## What is Bernoulli sampling?
//!
//! When gathering statistics about a program's behavior,
//! we may be observing events that occur very frequently
//! (*e.g.*, function calls or memory allocations)
//! and we may be gathering information that is somewhat expensive to produce
//! (*e.g.*, call stacks).
//! Sampling all the events could have a significant impact on the program's performance.
//!
//! Why not just sample every *n*'th event?
//! That technique, called "systematic sampling",
//! is simple and efficient,
//! and it's fine if we imagine a patternless stream of events.
//! But what if we're sampling allocations,
//! and the program happens to have a loop where each iteration does exactly *n* allocations?
//! You would end up sampling the same allocation every time through the loop,
//! and the entire rest of the loop would be invisible to your measurements!
//! More generally, if each iteration does *m* allocations,
//! and *m* and *n* have any common divisor at all,
//! most allocation sites will never be sampled.
//! If they're both even, say,
//! the odd-numbered allocations disappear from your results.
//!
//! Ideally,
//! we'd like each event to have some probability *P* of being sampled,
//! independent of its neighbors and of its position in the sequence.
//! This is called "Bernoulli sampling",
//! and it doesn't suffer from any of the problems mentioned above.
//!
//! One disadvantage of Bernoulli sampling is that
//! you can't be sure exactly how many samples you'll get.
//! Technically, it's possible that you might sample none of them, or all of them.
//! But if the number of events *n* is large, these aren't likely outcomes;
//! you can generally expect somewhere around
//! <i>P</i>⋅<i>n</i>
//! events to be sampled.
//!
//! Another disadvantage of Bernoulli sampling is that
//! you have to generate a random number for every event,
//! which can be slow.
//!
//! *- significant pause -*
//!
//! **But not with this crate!**
//! `FastBernoulliTrial` lets you do true Bernoulli sampling,
//! while generating a fresh random number only when we do decide to sample an event,
//! not on every trial.
//! When it decides not to sample, a call to `<FastBernoulliTrial as Iterator>::next` is
//! nothing but decrementing a counter and comparing it to zero.
//! So the lower your sampling probability is,
//! the less overhead `FastBernoulliTrial` imposes.
//!
//! Probabilities of 0 and 1 are handled efficiently.
//! (In neither case need we ever generate a random number at all.)
//!
//! ## Examples
//!
//!     # use fast_bernoulli_trial::FastBernoulliTrial;
//!     let trials = FastBernoulliTrial::new(0.01).unwrap();
//!
//!     let count = trials.take(10000).filter(|b| *b).count();
//!     println!("Of 10000 items, {} were true.", count);
//!
//! You can supply a specific random number generator to use:
//!
//!     # extern crate fast_bernoulli_trial;
//!     extern crate rand;
//!     use rand::{SeedableRng, XorShiftRng};
//!
//!     # use fast_bernoulli_trial::FastBernoulliTrial;
//!     # fn main() {
//!     let rng = XorShiftRng::from_seed([0x3b3f4150, 0x53704c15,
//!                                       0xf66c1396, 0x09b0136e]);
//!     let trials = FastBernoulliTrial::new_with_rng(0.05, rng);
//!
//!     // Since we specified the RNG and the seed, we always get the same 
//!     // sequence, so this assertion always passes.
//!     assert_eq!(trials.take(10000).filter(|b| *b).count(), 482);
//!     # }
//!
//! ## Why a whole crate for this?
//!
//! I just think geometric distributions are cool.
//! This is as much about the comments as the code.

extern crate rand;
use rand::{Rng, SeedableRng, StdRng};

mod tests;

/// An iterator producing a stream of random `bool` values whose probability of being
/// `true` is under your control.
///
/// This is suitable for choosing some proportion of events to sample out of a series: for
/// each event, draw another `bool` from this iterator, and sample the event if it's
/// `true`.
///
/// The difference between this and simply calling `rand::Rng::next_f64` for
/// each event is that `FastBernoulliTrial` is usually faster: it arranges to
/// generate a random number, not for every value produced, but only for every
/// `true` value produced. The independence of each value is still ensured. If
/// the selected probability of `true` results is low, then most calls to
/// `Iterator::next` simply decrement a counter.
pub struct FastBernoulliTrial<R: Rng = StdRng> {
    /// The likelihood that any given call to `Iterator::next` should return
    /// true. Between 0 and 1 inclusive.
    probability: f64,

    /// The value of `1.0 / f64::ln(1.0 - probability)`, cached for repeated use.
    ///
    /// If `probability` is exactly 0 or exactly 1, we don't use this value.
    /// Otherwise, we guarantee this value is in the range [-2**53, -1/37),
    /// *i.e.* definitely negative, as required by `choose_skip_count`.
    /// See `set_probability` for the details.
    inv_ln_not_probability: f64,

    /// Our random number generator.
    rng: R,

    /// The number of times `Iterator::next` should return `false` before next
    /// returning `true`.
    skip_count: usize
}

impl FastBernoulliTrial<StdRng> {
    /// Return a new `FastBernoulliTrial` iterator.
    /// Calls to `next` return `true` with the given `probability`.
    ///
    /// This uses a freshly seeded random number generator.
    pub fn new(probability: f64) -> std::io::Result<FastBernoulliTrial<StdRng>> {
        Ok(FastBernoulliTrial::<StdRng>::new_with_rng(probability, StdRng::new()?))
    }
}

impl<R: Rng> FastBernoulliTrial<R> {
    /// Return a new `FastBernoulliTrial` iterator.
    /// Calls to `next` return `true` with the given `probability`.
    /// Use `rng` as the underlying random number generator.
    pub fn new_with_rng(probability: f64, rng: R) -> FastBernoulliTrial<R> {
        // Create an FBT with garbage values, and let set_probability initialize
        // everything properly.
        let mut trial = FastBernoulliTrial {
            rng,
            probability: 0.0,
            inv_ln_not_probability: 0.0,
            skip_count: 0
        };
        trial.set_probability(probability);

        trial
    }

    /// Equivalent to calling `self.next()` *n* times, and returning `true` if any of
    /// those calls do. However, like `next`, this runs in fast constant time.
    ///
    /// What is this good for? In some applications, some events are "bigger" than others.
    /// For example, large allocations are more significant than small allocations.
    /// Perhaps we'd like to imagine that we're drawing allocations from a stream of
    /// bytes, and performing a separate Bernoulli trial on every byte from the stream. We
    /// can accomplish this by calling `self.any_of_next_n(S)` for the number of bytes
    /// *S*, and sampling the event if that returns true.
    ///
    /// Of course, this style of sampling needs to be paired with analysis and
    /// presentation that makes the "size" of the event apparent, lest trials with
    /// large values for *S* appear to be indistinguishable from those with small
    /// values for *S*, despite being potentially much more likely to be sampled.
    pub fn any_of_next_n(&mut self, n: usize) -> bool {
        if self.skip_count > n {
            self.skip_count -= n;
            return false;
        }

        self.choose_skip_count()
    }

    /// Change the probability with which this `FastBernoulliTrial` produces `true` values
    /// to the given `probability`.
    pub fn set_probability(&mut self, probability: f64) {
        assert!(0.0 <= probability && probability <= 1.0);
        self.probability = probability;
        if 0.0 < self.probability && self.probability < 1.0 {
            // Let's look carefully at how this calculation plays out in floating- point
            // arithmetic. We'll assume IEEE, but the final Rust code we arrive at would
            // still be fine if our numbers were mathematically perfect. So, while we've
            // considered IEEE's edge cases, we haven't done anything that should be
            // actively bad when using other representations.
            //
            // (In the below, read comparisons as exact mathematical comparisons: when we
            // say something "equals 1", that means it's exactly equal to 1. We treat
            // approximation using intervals with open boundaries: saying a value is in
            // (0,1) doesn't specify how close to 0 or 1 the value gets. When we use
            // closed boundaries like [2**-53, 1], we're careful to ensure the boundary
            // values are actually representable.)
            //
            // - After the comparison above, we know self.probability is in (0,1).
            //
            // - The gaps below 1 are 2**-53, so that interval is (0, 1-2**-53].
            //
            // - Because the floating-point gaps near 1 are wider than those near zero,
            //   there are many small positive doubles ε such that 1-ε rounds to exactly
            //   1. However, 2**-53 can be represented exactly. So 1-self.probability is
            //   in [2**-53, 1].
            //
            // - ln(1 - self.probability) is thus in (-37, 0].
            //
            //   That range includes zero, but when we use self.inv_ln_not_probability,
            //   it would be helpful if we could trust that it's negative. So when ln(1 -
            //   self.probability) is 0, we'll just set self.probability to 0, so that
            //   self.inv_ln_not_probability is not used in choose_skip_count.
            //
            // - How much of the range of self.probability does this cause us to ignore?
            //   The only value for which ln returns 0 is exactly 1; the slope of ln at 1
            //   is 1, so for small ε such that 1 - ε != 1, ln(1 - ε) is -ε, never 0. The
            //   gaps near one are larger than the gaps near zero, so if 1 - ε wasn't 1,
            //   then -ε is representable. So if ln(1 - self.probability) isn't 0, then 1
            //   - self.probability isn't 1, which means that self.probability is at least
            //   2**-53, as discussed earlier. This is a sampling likelihood of roughly
            //   one in ten trillion, which is unlikely to be distinguishable from zero in
            //   practice.
            //
            //   So by forbidding zero, we've tightened our range to (-37, -2**-53].
            //
            // - Finally, 1 / ln(1 - self.probability) is in [-2**53, -1/37). This all
            //   falls readily within the range of an IEEE double.
            //
            // ALL THAT HAVING BEEN SAID: here are the five lines of actual code:
            let ln_not_probability = f64::ln(1.0 - self.probability);
            if ln_not_probability == 0.0 {
                self.probability = 0.0;
            } else {
                self.inv_ln_not_probability = 1.0 / ln_not_probability;
            }
        }

        self.choose_skip_count();
    }

    /// Set the current state of the random number generator to `state`.
    pub fn reseed<S>(&mut self, state: S)
        where R: SeedableRng<S>
    {
        self.rng.reseed(state)
    }
}

impl<R: Rng> Iterator for FastBernoulliTrial<R> {
    type Item = bool;

    fn next(&mut self) -> Option<bool> {
        if self.skip_count > 0 {
            self.skip_count -= 1;
            return Some(false);
        }

        Some(self.choose_skip_count())
    }
}

impl<R: Rng> FastBernoulliTrial<R> {
    /// Choose the next skip count. This also returns the value that
    /// `Iterator::next` should produce, since we have to check for the extreme
    /// values for `self.probability` anyway, and `next` should never produce
    /// `true` at all when `self.probability` is 0.
    fn choose_skip_count(&mut self) -> bool {
        // This comment should just read, "Generate skip counts with a geometric
        // distribution", and leave everyone to go look that up and see why it's the right
        // thing to do, if they don't know already.
        //
        // BUT IF YOU'RE CURIOUS, COMMENTS ARE FREE...
        //
        // Instead of generating a fresh random number for every trial, we can randomly
        // generate a count of how many times we should return false before the next time
        // we return true. We call this a "skip count". Once we've returned true, we
        // generate a fresh skip count, and begin counting down again.
        //
        // Here's an awesome fact: by exercising a little care in the way we generate skip
        // counts, we can produce results indistinguishable from those we would get
        // "rolling the dice" afresh for every trial.
        //
        // In short, skip counts in Bernoulli trials of probability P obey a geometric
        // distribution. If a random variable X is uniformly distributed from [0..1), then
        // `f64::floor(f64::ln(X) / f64::ln(1-P))` has the appropriate geometric
        // distribution for the skip counts.
        //
        // Why that formula?
        //
        // Suppose we're to return `true` with some probability P, say, 0.3. Spread
        // all possible futures along a line segment of length 1. In portion P of
        // those cases, we'll return true on the next call to `trial`; the skip count
        // is 0. For the remaining portion 1-P of cases, the skip count is 1 or more.
        //
        // skip:                0                         1 or more
        //             |------------------^-----------------------------------------|
        // portion:            0.3                            0.7
        //                      P                             1-P
        //
        // But the "1 or more" section of the line is subdivided the same way: *within
        // that section*, in portion P the second call to `trial()` returns true, and in
        // portion 1-P it returns false a second time; the skip count is two or more.
        // So we return true on the second call in proportion 0.7 * 0.3, and skip at
        // least the first two in proportion 0.7 * 0.7.
        //
        // skip:                0                1              2 or more
        //             |------------------^------------^----------------------------|
        // portion:            0.3           0.7 * 0.3          0.7 * 0.7
        //                      P             (1-P)*P            (1-P)^2
        //
        // We can continue to subdivide:
        //
        // skip >= 0:  |------------------------------------------------- (1-P)^0 --|
        // skip >= 1:  |                  ------------------------------- (1-P)^1 --|
        // skip >= 2:  |                               ------------------ (1-P)^2 --|
        // skip >= 3:  |                                 ^     ---------- (1-P)^3 --|
        // skip >= 4:  |                                 .            --- (1-P)^4 --|
        //                                               .
        //                                               ^X, see below
        //
        // In other words, the likelihood of the next n calls to `trial` returning false
        // is (1-P)^n. The longer a run we require, the more the likelihood drops. Further
        // calls may return false too, but this is the probability we'll skip at least n.
        //
        // This is interesting, because we can pick a point along this line segment
        // and see which skip count's range it falls within; the point X above, for
        // example, is within the ">= 2" range, but not within the ">= 3" range, so it
        // designates a skip count of 2. So if we pick points on the line at random
        // and use the skip counts they fall under, that will be indistinguishable
        // from generating a fresh random number between 0 and 1 for each trial and
        // comparing it to P.
        //
        // So to find the skip count for a point X, we must ask: To what whole power
        // must we raise 1-P such that we include X, but the next power would exclude
        // it? This is exactly the logarithm base 1-P, or
        // `f64::floor(f64::ln(X) / f64::ln(1-P)).
        //
        // Our algorithm is then, simply: When constructed, compute an initial skip count.
        // Return false from `Iterator::next` that many times, and then compute a new skip
        // count.
        //
        // For a call to `trial(n)`, if the skip count is greater than n, return false
        // and subtract n from the skip count. If the skip count is less than n,
        // return true and compute a new skip count. Since each trial is independent,
        // it doesn't matter by how much n overshoots the skip count; we can actually
        // compute a new skip count at *any* time without affecting the distribution.
        // This is really beautiful.

        // If the probability is 1.0, every call to `Iterator::next` returns
        // `true`. Make sure `self.skip_count` is 0.
        if self.probability == 1.0 {
            self.skip_count = 0;
            return true;
        }

        // If the probabilility is zero, `trial` never returns true. Don't bother us
        // for a while.
        if self.probability == 0.0 {
            self.skip_count = std::usize::MAX;
            return false;
        }

        // What sorts of values can this call to std::floor produce?
        //
        // Since `self.rng.next_f64` returns a value in [0, 1-2**-53], `f64::log2`
        // returns a value in the range [-infinity, -2**-53], all negative. Since
        // `self.inv_ln_not_probability` is negative (see its comments), the product is
        // positive and possibly infinite. `f64::floor` returns +infinity unchanged.
        // So the result will always be positive.
        //
        // Converting an f64 to an integer that is out of range for that integer is
        // undefined behavior[1], so we must clamp our result to `std::usize::MAX`, to
        // ensure we get an acceptable value for `self.skip_count`.
        //
        // The clamp is written carefully. Note that if we had said:
        //
        //     if skip_count > std::usize::MAX {
        //        skip_count = std::usize::MAX;
        //     }
        //
        // that leads to undefined behavior 64-bit machines: std::usizeSIZE_MAX coerced to
        // double is 2^64, not 2^64-1, so this doesn't actually set skipCount to a
        // value that can be safely assigned to self.skip_count.
        //
        // Jakub Oleson cleverly suggested flipping the sense of the comparison: if
        // we require that skipCount < SIZE_MAX, then because of the gaps (2048)
        // between doubles at that magnitude, the highest double less than 2^64 is
        // 2^64 - 2048, which is fine to store in a size_t.
        //
        // (On 32-bit machines, all size_t values can be represented exactly in
        // double, so all is well.)
        //
        // [1]: https://github.com/rust-lang/issue/10184
        let skip_count = f64::floor(f64::ln(self.rng.next_f64())
                                    * self.inv_ln_not_probability) as usize;
        if skip_count < std::usize::MAX {
            self.skip_count = skip_count;
        } else {
            self.skip_count = std::usize::MAX;
        }

        true
    }
}
