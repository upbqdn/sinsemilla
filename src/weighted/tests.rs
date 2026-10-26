use alloc::vec::Vec;

use group::{Curve, CurveAffine as _, Group};
use pasta_curves::pallas;
use subtle::CtOption;

use super::{GENERATOR_COUNT, UncheckedFixedLengthHashDomain};
use crate::{HashDomain, K, SINSEMILLA_S_AFFINE};

const MERKLE_WORDS: usize = 52;
const MERKLE_DOMAIN: &str = "z.cash:Orchard-MerkleCRH";

fn assert_matches(expected: CtOption<pallas::Point>, actual: pallas::Point) {
    assert!(bool::from(expected.is_some()));
    assert_eq!(actual, expected.unwrap());
}

fn words_to_bits(words: &[u16]) -> Vec<bool> {
    words
        .iter()
        .flat_map(|word| (0..K).map(move |bit| ((word >> bit) & 1) == 1))
        .collect()
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[test]
fn unchecked_evaluator_deliberately_skips_incomplete_addition_failure() {
    let generator_index = 0;
    let generator = SINSEMILLA_S_AFFINE[generator_index];
    let generator_point = pallas::Point::from(generator);
    let domain = HashDomain::from_Q(generator_point);
    let unchecked = UncheckedFixedLengthHashDomain::<1>::new(&domain);
    let words = [u16::try_from(generator_index).expect("a Sinsemilla word fits into u16")];
    let bits = words_to_bits(&words);

    // The first incomplete addition attempts S + S, so the specified
    // partial function returns bottom. The unchecked evaluator computes
    // the corresponding complete group expression instead.
    assert!(!bool::from(
        domain.hash_to_point(bits.iter().copied()).is_some()
    ));
    assert_eq!(
        unchecked.hash_words_to_point(&words),
        generator_point.double() + generator
    );
}

#[test]
#[should_panic(expected = "the weighted evaluator requires a nonidentity Q")]
fn rejects_identity_q() {
    let domain = HashDomain::from_Q(pallas::Point::identity());
    let _ = UncheckedFixedLengthHashDomain::<1>::new(&domain);
}

#[test]
fn fixed_merkle_length_matches_generic_evaluation() {
    let domain = HashDomain::new(MERKLE_DOMAIN);
    let weighted = UncheckedFixedLengthHashDomain::<MERKLE_WORDS>::new(&domain);

    let mut fixtures = vec![
        [0; MERKLE_WORDS],
        [1; MERKLE_WORDS],
        [u16::try_from(GENERATOR_COUNT - 1).unwrap(); MERKLE_WORDS],
    ];
    fixtures.push(core::array::from_fn(|i| u16::try_from(i).unwrap()));
    fixtures.extend((0..=super::FUSED_FIRST_WORDS).map(|first| {
        let mut words = [u16::try_from(GENERATOR_COUNT - 1).unwrap(); MERKLE_WORDS];
        words[0] = u16::try_from(first).unwrap();
        words
    }));

    let mut state = 0x5369_6e73_656d_696c;
    fixtures.extend(
        (0..128).map(|_| {
            core::array::from_fn(|_| u16::try_from(splitmix64(&mut state) & 0x3ff).unwrap())
        }),
    );

    for words in fixtures {
        let bits = words_to_bits(&words);
        let expected = domain.hash_to_point(bits.iter().copied());
        assert_matches(expected, weighted.hash_words_to_point(&words));
        assert_matches(expected, weighted.hash_to_point(bits.iter().copied()));

        let expected = domain.hash(bits.iter().copied());
        assert!(bool::from(expected.is_some()));
        let expected = expected.unwrap();
        assert_eq!(expected, weighted.hash_words(&words));
        assert_eq!(expected, weighted.hash(bits.iter().copied()));
    }
}

#[test]
#[should_panic(expected = "unexpected Sinsemilla word count")]
fn rejects_too_few_message_words() {
    let domain = HashDomain::new(MERKLE_DOMAIN);
    let weighted = UncheckedFixedLengthHashDomain::<2>::new(&domain);
    weighted.hash_to_point(core::iter::repeat_n(false, K));
}

#[test]
#[should_panic(expected = "unexpected Sinsemilla word count")]
fn rejects_too_many_message_words() {
    let domain = HashDomain::new(MERKLE_DOMAIN);
    let weighted = UncheckedFixedLengthHashDomain::<2>::new(&domain);
    weighted.hash_to_point(core::iter::repeat_n(false, 3 * K));
}

#[test]
fn weighted_table_is_a_doubling_chain() {
    let domain = HashDomain::new(MERKLE_DOMAIN);
    let weighted = UncheckedFixedLengthHashDomain::<MERKLE_WORDS>::new(&domain);
    let initial = (0..MERKLE_WORDS).fold(domain.Q, |point, _| point.double());

    assert_eq!(
        weighted.table_bytes(),
        (MERKLE_WORDS + super::FUSED_FIRST_WORDS)
            * GENERATOR_COUNT
            * core::mem::size_of::<pallas::Affine>()
    );

    for (generator, &s) in SINSEMILLA_S_AFFINE.iter().enumerate() {
        assert_eq!(weighted.weighted_generator(0, generator), s);
        for exponent in 0..MERKLE_WORDS - 1 {
            let entry = weighted.weighted_generator(exponent, generator);
            assert!(!bool::from(entry.is_identity()));
            let doubled = pallas::Point::from(entry).double().to_affine();

            // Adjacent rows chain by doubling.
            if exponent + 1 < MERKLE_WORDS - 1 {
                assert_eq!(
                    weighted.weighted_generator(exponent + 1, generator),
                    doubled
                );
            }
        }

        let first_generator =
            pallas::Point::from(weighted.weighted_generator(MERKLE_WORDS - 2, generator)).double();
        assert_eq!(
            weighted.first_accumulator(generator),
            (initial + first_generator).to_affine()
        );
    }
}

#[test]
fn single_word_and_zero_padding_match_generic_evaluation() {
    let domain = HashDomain::new(MERKLE_DOMAIN);
    let single = UncheckedFixedLengthHashDomain::<1>::new(&domain);
    let double = UncheckedFixedLengthHashDomain::<2>::new(&domain);

    for word in [0, 1, 7, 8, 1023] {
        let bits = words_to_bits(&[word]);
        assert_matches(
            domain.hash_to_point(bits.iter().copied()),
            single.hash_words_to_point(&[word]),
        );
    }
    for length in 1..=2 * K {
        let bits: Vec<_> = (0..length).map(|i| i % 3 == 0).collect();
        let expected = domain.hash_to_point(bits.iter().copied());
        let actual = if length <= K {
            single.hash_to_point(bits.iter().copied())
        } else {
            double.hash_to_point(bits.iter().copied())
        };
        assert_matches(expected, actual);
    }
}

#[test]
#[should_panic(expected = "the weighted evaluator requires at least one word")]
fn rejects_zero_word_count() {
    let _ = UncheckedFixedLengthHashDomain::<0>::new(&HashDomain::new(MERKLE_DOMAIN));
}

#[test]
#[should_panic(expected = "Sinsemilla word count exceeds the protocol limit")]
fn rejects_word_count_above_protocol_limit() {
    let _ =
        UncheckedFixedLengthHashDomain::<{ crate::C + 1 }>::new(&HashDomain::new(MERKLE_DOMAIN));
}

#[test]
#[should_panic(expected = "invalid Sinsemilla word")]
fn rejects_invalid_first_word() {
    let weighted = UncheckedFixedLengthHashDomain::<1>::new(&HashDomain::new(MERKLE_DOMAIN));
    weighted.hash_words(&[1024]);
}

#[test]
#[should_panic(expected = "invalid Sinsemilla word")]
fn rejects_invalid_fused_second_word() {
    let weighted = UncheckedFixedLengthHashDomain::<2>::new(&HashDomain::new(MERKLE_DOMAIN));
    weighted.hash_words(&[0, 1024]);
}

#[test]
#[should_panic(expected = "invalid Sinsemilla word")]
fn rejects_invalid_later_word() {
    let weighted = UncheckedFixedLengthHashDomain::<3>::new(&HashDomain::new(MERKLE_DOMAIN));
    weighted.hash_words(&[0, 0, 1024]);
}
