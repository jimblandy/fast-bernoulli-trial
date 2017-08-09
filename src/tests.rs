#![cfg(test)]

use super::FastBernoulliTrial;

use rand::{SeedableRng, XorShiftRng};
use std;

fn make(probability: f64) -> FastBernoulliTrial<XorShiftRng> {
    let rng = XorShiftRng::from_seed([0x3b3f4150, 0x53704c15,
                                      0x09b0136e, 0xf66c1396]);
    FastBernoulliTrial::new_with_rng(probability, rng)
}

#[test]
fn proportions() {
    let mut bernoulli = make(1.0);

    assert!(bernoulli.by_ref().take(100).all(|b| b));

    bernoulli.set_probability(0.001);
    assert_eq!(bernoulli.by_ref().take(1000).filter(|b| *b).count(), 2);

    bernoulli.set_probability(0.5);
    assert_eq!(bernoulli.by_ref().take(1000).filter(|b| *b).count(), 499);

    bernoulli.set_probability(0.85);
    assert_eq!(bernoulli.by_ref().take(1000).filter(|b| *b).count(), 836);

    bernoulli.set_probability(0.0);
    assert_eq!(bernoulli.by_ref().take(1000).filter(|b| *b).count(), 0);
}

#[test]
fn harmonics() {
    const N: usize = 100000;
    const P: f64 = 0.1;

    let trial = make(P);
    let trials: Vec<bool> = trial.take(N).collect();

    // For each harmonic and phase, check that the proportion sampled is
    // within acceptable bounds.
    for harmonic in 1..20 {
        let expected = N as f64 / harmonic as f64 * P;
        let low_expected  = (expected * 0.85) as usize;
        let high_expected = (expected * 1.15) as usize;

        for phase in 0..harmonic {
            let mut count = 0;
            let mut i = phase;
            while i < N {
                if trials[i] {
                    count += 1;
                }
                i += harmonic;
            }

            assert!(low_expected <= count && count <= high_expected);
        }
    }
}

#[test]
fn any_of_next_n() {
    const N: usize = 10000;
    const P: f64 = 0.01;

    let mut trial = make(P);

    // Expected value: 0.01 * 10000 == 100
    assert_eq!((0..N).filter(|_| trial.any_of_next_n(1)).count(), 103);

    // Expected value: (1 - (1 - 0.01) ** 3) == 0.0297,
    // 0.0297 * 10000 == 297
    assert_eq!((0..N).filter(|_| trial.any_of_next_n(3)).count(), 296);

    // Expected value: (1 - (1 - 0.01) ** 10) == 0.0956,
    // 0.0956 * 10000 == 956
    assert_eq!((0..N).filter(|_| trial.any_of_next_n(10)).count(), 946);

    // Expected value: (1 - (1 - 0.01) ** 100) == 0.6339
    // 0.6339 * 10000 == 6339
    assert_eq!((0..N).filter(|_| trial.any_of_next_n(100)).count(), 6348);

    // Expected value: (1 - (1 - 0.01) ** 1000) == 0.9999
    // 0.9999 * 10000 == 9999
    assert_eq!((0..N).filter(|_| trial.any_of_next_n(1000)).count(), 10000);
}

#[test]
fn set_probability() {
    let mut bernoulli = make(1.0);

    // Establish a very high skip count.
    bernoulli.set_probability(0.0);

    // This should re-establish a zero skip count.
    bernoulli.set_probability(1.0);

    // So this should return true.
    assert_eq!(bernoulli.next(), Some(true));
}

#[test]
fn cusp_probabilities() {
    // FastBernoulliTrial takes care to avoid screwing up on edge cases. The
    // checks here all look pretty dumb, but they exercise paths in the code that
    // could exhibit undefined behavior if coded naïvely.

    // IEEE requires these results. They're just here to help persuade the
    // skeptical that the call to `make` below really does pass the largest
    // representable number less than 1.0.
    assert!(1.0 - std::f64::EPSILON / 2.0 <  1.0);
    assert!(1.0 - std::f64::EPSILON / 4.0 == 1.0);

    // This should not be perceptibly different from 1; for 64-bit doubles, this
    // is a one in ten trillion chance of the trial not succeeding. Overflows
    // converting doubles to usize skip counts may change this, though.
    let mut bernoulli = make(1.0 - std::f64::EPSILON / 2.0);

    assert!(bernoulli.by_ref().take(1000).all(|b| b));

    // This should not be perceptibly different from 0; for 64-bit doubles, the
    // FastBernoulliTrial will actually treat this as exactly zero.
    bernoulli.set_probability(std::f64::MIN_POSITIVE);
    assert!(!bernoulli.by_ref().take(1000).any(|b| b));

    // This should be a vanishingly low probability which FastBernoulliTrial does
    // *not* treat as exactly zero.
    bernoulli.set_probability(std::f64::EPSILON);
    assert!(!bernoulli.by_ref().take(1000).any(|b| b));
}
