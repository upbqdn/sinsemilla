//! Fixed-length, position-weighted Sinsemilla evaluation.
//!
//! This moves the powers of two in the Sinsemilla recurrence into a
//! precomputed table:
//!
//! `B_i = [2^(N-i)] A_i`, so `B_(i+1) = B_i + [2^(N-i-1)] S[m_i]`.
//!
//! ## Exceptional-case reduction
//!
//! Let `A_0 = Q` and `A_(i+1) = [2] A_i + S[m_i]`. Before the first
//! exceptional incomplete addition, induction gives
//!
//! `A_i = [2^i] Q + sum_j [X_(i,j)] S[j]`,
//!
//! where `X_(i,j)` is the sum of `2^(i-1-t)` over the positions `t < i`
//! for which `m_t = j`.
//!
//! The first incomplete addition in a step can fail only when
//! `A_i = +/-S[m_i]`. If it succeeds, the second can fail only when
//! `A_i + S[m_i] = -A_i`; equality with `A_i` would imply
//! `S[m_i] = O`. Thus every exceptional step gives
//!
//! `[alpha] A_i + S[m_i] = O`
//!
//! for some `alpha` in `{-1, 1, 2}`. Substitution gives the efficiently
//! computable discrete-logarithm relation
//!
//! `[alpha * 2^i] Q
//!     + sum_j [alpha * X_(i,j) + delta_(j,m_i)] S[j] = O`.
//!
//! This relation is nontrivial because its coefficient of `Q` is nonzero.
//! In particular, `i < N` and `N <= C` (enforced using [`C`]) ensure that
//! its magnitude is at most `2^N <= 2^C <= (r_P - 1) / 2`, where `r_P` is
//! the Pallas group order.
//! Therefore an input on which this evaluator differs from [`HashDomain`]
//! yields a nontrivial discrete-logarithm relation among the independently
//! generated Sinsemilla bases. This is the reduction in the
//! [Sinsemilla exceptional-case theorem].
//!
//! Finding such a relation tightly reduces to the discrete logarithm
//! problem [by Jaeger and Tessaro]. The [Pasta curve parameters] give Pallas
//! a group order greater than `2^254`. Accounting conservatively for its
//! efficient order-3 endomorphism, the [Pollard-rho work estimate] is
//! `sqrt(pi * r_P / 12)`, which is greater than `2^126` classical group
//! operations. Triggering an omitted check is therefore considered infeasible
//! under the standard discrete-logarithm relation assumption.
//!
//! ## Required checks and invariants
//!
//! The reduction requires `Q` and every `S[j]` to be nonidentity, and it
//! requires `N <= C`. Construction checks these conditions. Evaluation also
//! checks the exact word count and that every word indexes the Sinsemilla
//! generator table. Callers must additionally use the table with the
//! [`HashDomain`] from which it was constructed; production domains must use
//! the protocol's independently personalized hash-to-curve generators.
//!
//! [Sinsemilla exceptional-case theorem]: https://zips.z.cash/protocol/protocol.pdf#thmsinsemillaex
//! [by Jaeger and Tessaro]: https://eprint.iacr.org/2020/1213.pdf#page=6
//! [Pasta curve parameters]: https://electriccoin.co/blog/the-pasta-curves-for-halo-2-and-beyond/
//! [Pollard-rho work estimate]: https://eprint.iacr.org/2019/1021.pdf#page=13

use alloc::{boxed::Box, vec::Vec};
use core::mem;

use group::{Curve, CurveAffine as _, Group};
use pasta_curves::{arithmetic::CurveAffine as _, pallas};

use super::{C, HashDomain, K, MessageWords, SINSEMILLA_S_AFFINE};

const GENERATOR_COUNT: usize = 1 << K;

/// Orchard levels for which the first two words have a combined table entry.
///
/// These are the widest levels in a batched 1024-leaf tree construction.
const FUSED_FIRST_WORDS: usize = 8;

fn extract(point: pallas::Point) -> pallas::Base {
    point
        .to_affine()
        .coordinates()
        .map(|coordinates| *coordinates.x())
        .unwrap_or_else(pallas::Base::zero)
}

/// An unchecked fixed-word-count Sinsemilla domain with position-weighted
/// generators.
///
/// Each instance is bound to the [`HashDomain`] it was constructed from;
/// instances built for different domains are interchangeable at the type
/// level, so callers must pair each table with messages for its own domain.
///
/// This evaluator deliberately omits Sinsemilla's incomplete-addition checks.
/// Finding an input that triggers one of those exceptional cases for the
/// protocol's independently generated `Q` and `S` points would exhibit a
/// nontrivial discrete-logarithm relation between those points, as shown in
/// the [module-level security argument](self). Callers that require exact
/// partial-function semantics must use [`HashDomain`] instead.
///
/// Construction is intentionally explicit and potentially expensive. Callers
/// should build this once and keep it outside timed or repeated hash paths.
pub struct UncheckedFixedLengthHashDomain<const N: usize> {
    /// Affine entries `W[e][j] = [2^e] S[j]` for `0 <= e < N - 1`,
    /// followed by the first-step accumulators
    /// `[2^N] Q + [2^(N-1)] S[j]`. Rows are flattened row-major.
    weighted_generators: Box<[pallas::Affine]>,
    /// Entries
    /// `[2^N] Q + [2^(N-1)] S[first] + [2^(N-2)] S[second]`, indexed by
    /// `first * GENERATOR_COUNT + second` for the first few `first` words.
    fused_first_two: Box<[pallas::Affine]>,
}

impl<const N: usize> UncheckedFixedLengthHashDomain<N> {
    /// Precomputes the position-weighted table for `domain`.
    ///
    /// # Panics
    ///
    /// Panics if `N` is zero, exceeds the protocol's maximum Sinsemilla word
    /// count, or the domain's `Q` or any precomputed generator is the identity.
    pub fn new(domain: &HashDomain) -> Self {
        assert!(N > 0, "the weighted evaluator requires at least one word");
        assert!(N <= C, "Sinsemilla word count exceeds the protocol limit");
        assert!(
            !bool::from(domain.Q.is_identity()),
            "the weighted evaluator requires a nonidentity Q"
        );

        let mut weighted_generators = Vec::with_capacity(N * GENERATOR_COUNT);

        let mut projective_row: Vec<_> = SINSEMILLA_S_AFFINE
            .iter()
            .copied()
            .map(pallas::Point::from)
            .collect();
        let mut affine_row: Vec<_> = SINSEMILLA_S_AFFINE.to_vec();

        for exponent in 0..N {
            assert!(
                affine_row
                    .iter()
                    .all(|point| !bool::from(point.is_identity()))
            );
            weighted_generators.extend(affine_row.iter().copied());

            if exponent + 1 < N {
                projective_row
                    .iter_mut()
                    .for_each(|point| *point = point.double());
                pallas::Point::batch_normalize(&projective_row, &mut affine_row);
            }
        }

        assert_eq!(weighted_generators.len(), N * GENERATOR_COUNT);

        let initial = (0..N).fold(domain.Q, |point, _| point.double());
        let first_row_start = (N - 1) * GENERATOR_COUNT;
        let first_accumulator_points: Vec<_> = weighted_generators[first_row_start..]
            .iter()
            .map(|generator| initial + generator)
            .collect();
        pallas::Point::batch_normalize(
            &first_accumulator_points,
            &mut weighted_generators[first_row_start..],
        );
        assert!(
            weighted_generators[first_row_start..]
                .iter()
                .all(|point| !bool::from(point.is_identity()))
        );

        let mut fused_first_two = Vec::new();
        if N > 1 {
            fused_first_two.reserve(FUSED_FIRST_WORDS * GENERATOR_COUNT);
            let second_row_start = (N - 2) * GENERATOR_COUNT;
            let second_generators =
                weighted_generators[second_row_start..second_row_start + GENERATOR_COUNT].to_vec();
            let mut normalized = second_generators.clone();

            for first in 0..FUSED_FIRST_WORDS {
                let first_accumulator =
                    pallas::Point::from(weighted_generators[first_row_start + first]);
                let points: Vec<_> = second_generators
                    .iter()
                    .map(|second| first_accumulator + second)
                    .collect();
                pallas::Point::batch_normalize(&points, &mut normalized);
                fused_first_two.extend(normalized.iter().copied());
            }
        }

        Self {
            weighted_generators: weighted_generators.into_boxed_slice(),
            fused_first_two: fused_first_two.into_boxed_slice(),
        }
    }

    /// Evaluates exactly `N` pre-decoded Sinsemilla words to a point.
    ///
    /// # Panics
    ///
    /// Panics if any word is not a valid [`K`]-bit Sinsemilla word.
    pub fn hash_words_to_point(&self, words: &[u16; N]) -> pallas::Point {
        self.evaluate(words.iter().copied())
    }

    /// Evaluates the Sinsemilla hash of exactly `N` pre-decoded words.
    ///
    /// # Panics
    ///
    /// Panics if any word is not a valid [`K`]-bit Sinsemilla word.
    pub fn hash_words(&self, words: &[u16; N]) -> pallas::Base {
        extract(self.hash_words_to_point(words))
    }

    /// Evaluates a bit iterator whose padded representation is exactly `N`
    /// Sinsemilla words.
    ///
    /// # Panics
    ///
    /// Panics if the zero-padded message is not exactly `N` words long.
    pub fn hash_to_point(&self, msg: impl Iterator<Item = bool>) -> pallas::Point {
        self.evaluate(
            MessageWords::new(msg)
                .map(|word| u16::try_from(word).expect("a Sinsemilla word fits into u16")),
        )
    }

    /// Evaluates the Sinsemilla hash of a bit iterator whose padded
    /// representation is exactly `N` words.
    ///
    /// # Panics
    ///
    /// Panics as [`Self::hash_to_point`] does.
    pub fn hash(&self, msg: impl Iterator<Item = bool>) -> pallas::Base {
        extract(self.hash_to_point(msg))
    }

    /// Returns the heap size occupied by the weighted generator table.
    pub fn table_bytes(&self) -> usize {
        (self.weighted_generators.len() + self.fused_first_two.len())
            * mem::size_of::<pallas::Affine>()
    }

    fn evaluate(&self, words: impl Iterator<Item = u16>) -> pallas::Point {
        let mut words = words;
        let first_word = words.next().expect("unexpected Sinsemilla word count");
        let first_generator = usize::from(first_word);
        assert!(first_generator < GENERATOR_COUNT, "invalid Sinsemilla word");
        let mut start = 1;
        let mut point = if N > 1 && first_generator < FUSED_FIRST_WORDS {
            let second_word = words.next().expect("unexpected Sinsemilla word count");
            let second_generator = usize::from(second_word);
            assert!(
                second_generator < GENERATOR_COUNT,
                "invalid Sinsemilla word"
            );
            start = 2;
            self.fused_first_two(first_generator, second_generator)
                .into()
        } else {
            self.first_accumulator(first_generator).into()
        };

        for i in start..N {
            let word = words.next().expect("unexpected Sinsemilla word count");
            let generator_index = usize::from(word);
            assert!(generator_index < GENERATOR_COUNT, "invalid Sinsemilla word");

            let exponent = N - i - 1;
            point += self.weighted_generator(exponent, generator_index);
        }
        assert!(words.next().is_none(), "unexpected Sinsemilla word count");
        point
    }

    fn weighted_generator(&self, exponent: usize, generator: usize) -> pallas::Affine {
        debug_assert!(exponent + 1 < N);
        self.weighted_generators[exponent * GENERATOR_COUNT + generator]
    }

    fn first_accumulator(&self, generator: usize) -> pallas::Affine {
        self.weighted_generators[(N - 1) * GENERATOR_COUNT + generator]
    }

    fn fused_first_two(&self, first: usize, second: usize) -> pallas::Affine {
        debug_assert!(N > 1);
        debug_assert!(first < FUSED_FIRST_WORDS);
        self.fused_first_two[first * GENERATOR_COUNT + second]
    }
}

#[cfg(test)]
mod tests;
