use alloc::vec::Vec;

use super::sinsemilla_s::SINSEMILLA_S;
use super::{C, HashDomain, IncompletePoint, K, MessageWords, SINSEMILLA_S_AFFINE};
use group::{CurveAffine as _, Group, ff::Field};
use pasta_curves::{arithmetic::CurveAffine, pallas};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use subtle::CtOption;

fn padded_words(bits: &[bool]) -> Vec<u32> {
    let mut padded = bits.to_vec();
    padded.resize(bits.len().div_ceil(K) * K, false);
    padded
        .chunks_exact(K)
        .map(|chunk| super::lebs2ip_k(chunk.try_into().unwrap()))
        .collect()
}

// The pre-optimization algorithm: pad, decode each generator, and perform
// two independent incomplete additions.
fn original_hash(domain: &HashDomain, bits: &[bool]) -> CtOption<pallas::Point> {
    padded_words(bits)
        .into_iter()
        .fold(IncompletePoint::from(domain.Q), |acc, word| {
            let (x, y) = SINSEMILLA_S[word as usize];
            let generator = pallas::Affine::from_xy(x, y).unwrap();
            (acc + generator) + acc
        })
        .into()
}

fn assert_same_point(expected: CtOption<pallas::Point>, actual: CtOption<pallas::Point>) {
    assert_eq!(bool::from(actual.is_some()), bool::from(expected.is_some()));
    if bool::from(expected.is_some()) {
        assert_eq!(actual.unwrap(), expected.unwrap());
    }
}

#[test]
fn message_words_match_padded_bits() {
    let mut rng = StdRng::seed_from_u64(0x53494e53454d494c);
    let bits: Vec<bool> = (0..K * C).map(|_| rng.random()).collect();
    for len in 0..=K * C {
        assert_eq!(
            MessageWords::new(bits[..len].iter().copied()).collect::<Vec<_>>(),
            padded_words(&bits[..len]),
        );
    }
    assert_eq!(MessageWords::new([true].into_iter()).next(), Some(1));
    assert_eq!(
        MessageWords::new(core::iter::repeat_n(true, K)).next(),
        Some((1 << K) - 1),
    );
}

#[test]
fn message_words_do_not_poll_after_end() {
    for len in [0, 1, K, K + 1] {
        let mut calls = 0;
        let bits = core::iter::from_fn(|| {
            calls += 1;
            assert!(calls <= len + 1);
            (calls <= len).then_some(true)
        });
        let mut words = MessageWords::new(bits);
        assert_eq!(words.by_ref().count(), len.div_ceil(K));
        assert_eq!(words.next(), None);
        assert_eq!(words.next(), None);
    }
}

#[test]
#[should_panic]
fn message_words_reject_overlong_input() {
    MessageWords::new(core::iter::repeat_n(false, K * C + 1)).for_each(drop);
}

#[test]
fn hash_matches_original() {
    let mut rng = StdRng::seed_from_u64(0x4841534854455354);
    let domain = HashDomain::new("sinsemilla-equivalence");
    for len in [
        0,
        1,
        K - 1,
        K,
        K + 1,
        509,
        510,
        511,
        520,
        1086,
        K * C - 1,
        K * C,
    ] {
        for bits in [
            vec![false; len],
            vec![true; len],
            (0..len).map(|_| rng.random()).collect(),
        ] {
            assert_same_point(
                original_hash(&domain, &bits),
                domain.hash_to_point(bits.iter().copied()),
            );
        }
    }
}

#[test]
fn hash_preserves_exceptional_domains() {
    let generator = SINSEMILLA_S_AFFINE[0].to_curve();
    let half = pallas::Scalar::from(2).invert().unwrap();
    for q in [
        pallas::Point::identity(),
        generator,
        -generator,
        -(generator * half),
    ] {
        let domain = HashDomain::from_Q(q);
        for len in [0, K, K + 1, 2 * K] {
            let bits = vec![false; len];
            let actual = domain.hash_to_point(bits.iter().copied());
            assert_same_point(original_hash(&domain, &bits), actual);
            assert_eq!(bool::from(actual.is_some()), len == 0);
        }
    }
}

#[test]
#[should_panic]
fn hash_checks_length_after_exception() {
    let domain = HashDomain::from_Q(pallas::Point::identity());
    let _ = domain.hash_to_point(core::iter::repeat_n(false, K * C + 1));
}

#[test]
fn affine_generators_match_coordinates() {
    for j in 0..(1u32 << K) {
        let actual = SINSEMILLA_S[j as usize];
        let cached = SINSEMILLA_S_AFFINE[j as usize];
        assert!(bool::from(cached.is_on_curve()));
        assert_eq!(cached, pallas::Affine::from_xy(actual.0, actual.1).unwrap());
    }
}
