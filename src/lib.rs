//! Implementation of Sinsemilla outside the circuit.

#![no_std]

// Domain constructors require `alloc` for their personalization strings.
#[macro_use]
extern crate alloc;

use group::{Curve, Wnaf};
use pasta_curves::{
    arithmetic::{CurveAffine, CurveExt},
    pallas,
};
use subtle::CtOption;

mod addition;
use self::addition::IncompletePoint;
mod sinsemilla_s;
pub use sinsemilla_s::SINSEMILLA_S;

// These protocol constants are validated against hash-to-curve in the tests.
static SINSEMILLA_S_AFFINE: [pallas::Affine; 1 << K] = {
    let mut points =
        [pallas::Affine::from_xy_unchecked(pallas::Base::zero(), pallas::Base::zero()); 1 << K];
    let mut i = 0;
    while i < points.len() {
        let (x, y) = SINSEMILLA_S[i];
        points[i] = pallas::Affine::from_xy_unchecked(x, y);
        i += 1;
    }
    points
};

/// Number of bits of each message piece in $\mathsf{SinsemillaHashToPoint}$
pub const K: usize = 10;

/// $\frac{1}{2^K}$
pub const INV_TWO_POW_K: [u8; 32] = [
    1, 0, 192, 196, 160, 229, 70, 82, 221, 165, 74, 202, 85, 7, 62, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 240, 63,
];

/// The largest integer such that $2^c \leq (r_P - 1) / 2$, where $r_P$ is the order
/// of Pallas.
pub const C: usize = 253;

// Sinsemilla Q generators

/// SWU hash-to-curve personalization for Sinsemilla $Q$ generators.
pub const Q_PERSONALIZATION: &str = "z.cash:SinsemillaQ";

// Sinsemilla S generators

/// SWU hash-to-curve personalization for Sinsemilla $S$ generators.
pub const S_PERSONALIZATION: &str = "z.cash:SinsemillaS";

/// Converts a little-endian [`K`]-bit string into an integer.
pub fn lebs2ip_k(bits: [bool; K]) -> u32 {
    bits.iter()
        .enumerate()
        .fold(0u32, |acc, (i, b)| acc + if *b { 1 << i } else { 0 })
}

/// Coordinate extractor for Pallas.
///
/// Defined in [Zcash Protocol Spec § 5.4.9.7: Coordinate Extractor for Pallas][concreteextractorpallas].
///
/// [concreteextractorpallas]: https://zips.z.cash/protocol/nu5.pdf#concreteextractorpallas
fn extract_p_bottom(point: CtOption<pallas::Point>) -> CtOption<pallas::Base> {
    point.map(|p| {
        p.to_affine()
            .coordinates()
            .map(|c| *c.x())
            .unwrap_or_else(pallas::Base::zero)
    })
}

/// Converts a bit iterator into zero-padded [`K`]-bit words.
struct MessageWords<I: Iterator<Item = bool>> {
    inner: I,
    bits_read: usize,
    finished: bool,
}

impl<I: Iterator<Item = bool>> MessageWords<I> {
    fn new(inner: I) -> Self {
        Self {
            inner,
            bits_read: 0,
            finished: false,
        }
    }
}

impl<I: Iterator<Item = bool>> Iterator for MessageWords<I> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let mut word = 0;
        for bit_index in 0..K {
            match self.inner.next() {
                Some(bit) => {
                    self.bits_read += 1;
                    assert!(self.bits_read <= K * C);
                    word |= u32::from(bit) << bit_index;
                }
                None => {
                    self.finished = true;
                    return (bit_index > 0).then_some(word);
                }
            }
        }

        Some(word)
    }
}

/// A domain in which $\mathsf{SinsemillaHashToPoint}$ and $\mathsf{SinsemillaHash}$ can
/// be used.
#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub struct HashDomain {
    Q: pallas::Point,
}

impl HashDomain {
    /// Constructs a new `HashDomain` with a specific prefix string.
    pub fn new(domain: &str) -> Self {
        HashDomain {
            Q: pallas::Point::hash_to_curve(Q_PERSONALIZATION)(domain.as_bytes()),
        }
    }

    /// $\mathsf{SinsemillaHashToPoint}$ from [§ 5.4.1.9][concretesinsemillahash].
    ///
    /// [concretesinsemillahash]: https://zips.z.cash/protocol/nu5.pdf#concretesinsemillahash
    pub fn hash_to_point(&self, msg: impl Iterator<Item = bool>) -> CtOption<pallas::Point> {
        self.hash_to_point_inner(msg).into()
    }

    fn hash_to_point_inner(&self, msg: impl Iterator<Item = bool>) -> IncompletePoint {
        MessageWords::new(msg).fold(IncompletePoint::from(self.Q), |acc, word| {
            // A K-bit word is at most 1023, so it fits into usize on every target.
            acc.double_and_add(SINSEMILLA_S_AFFINE[word as usize])
        })
    }

    /// $\mathsf{SinsemillaHash}$ from [§ 5.4.1.9][concretesinsemillahash].
    ///
    /// [concretesinsemillahash]: https://zips.z.cash/protocol/nu5.pdf#concretesinsemillahash
    ///
    /// # Panics
    ///
    /// This panics if the message length is greater than [`K`] * [`C`]
    pub fn hash(&self, msg: impl Iterator<Item = bool>) -> CtOption<pallas::Base> {
        extract_p_bottom(self.hash_to_point(msg))
    }

    /// Constructs a new `HashDomain` from a given `Q`.
    ///
    /// This is only for testing use.
    #[cfg(any(test, feature = "test-dependencies"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-dependencies")))]
    #[allow(non_snake_case)]
    pub fn from_Q(Q: pallas::Point) -> Self {
        HashDomain { Q }
    }

    /// Returns the Sinsemilla $Q$ constant for this domain.
    #[cfg(any(test, feature = "test-dependencies"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-dependencies")))]
    #[allow(non_snake_case)]
    pub fn Q(&self) -> pallas::Point {
        self.Q
    }
}

/// A domain in which $\mathsf{SinsemillaCommit}$ and $\mathsf{SinsemillaShortCommit}$ can
/// be used.
#[derive(Debug)]
#[allow(non_snake_case)]
pub struct CommitDomain {
    M: HashDomain,
    R: pallas::Point,
}

impl CommitDomain {
    /// Constructs a new `CommitDomain` with a specific prefix string.
    pub fn new(domain: &str) -> Self {
        Self::new_with_separate_domains(domain, domain)
    }

    /// Constructs a new `CommitDomain` from different values for `hash_domain` and `blind_domain`
    /// `new_with_separate_domains` is used in the OrchardZSA note commitment, where we use the
    /// OrchardZSA hash domain `z.cash:ZSA-NoteCommit` and reuse the Orchard blind domain
    /// `z.cash:Orchard-NoteCommit`, as specified in
    /// [ZIP 226](https://zips.z.cash/zip-0226#note-structure-commitment).
    pub fn new_with_separate_domains(hash_domain: &str, blind_domain: &str) -> Self {
        let m_prefix = format!("{hash_domain}-M");
        let r_prefix = format!("{blind_domain}-r");
        let hasher_r = pallas::Point::hash_to_curve(&r_prefix);
        CommitDomain {
            M: HashDomain::new(&m_prefix),
            R: hasher_r(&[]),
        }
    }

    /// $\mathsf{SinsemillaCommit}$ from [§ 5.4.8.4][concretesinsemillacommit].
    ///
    /// [concretesinsemillacommit]: https://zips.z.cash/protocol/nu5.pdf#concretesinsemillacommit
    #[allow(non_snake_case)]
    pub fn commit(
        &self,
        msg: impl Iterator<Item = bool>,
        r: &pallas::Scalar,
    ) -> CtOption<pallas::Point> {
        // We use complete addition for the blinding factor.
        CtOption::<pallas::Point>::from(self.M.hash_to_point_inner(msg))
            .map(|p| p + Wnaf::new().scalar(r).base(self.R))
    }

    /// $\mathsf{SinsemillaShortCommit}$ from [§ 5.4.8.4][concretesinsemillacommit].
    ///
    /// [concretesinsemillacommit]: https://zips.z.cash/protocol/nu5.pdf#concretesinsemillacommit
    pub fn short_commit(
        &self,
        msg: impl Iterator<Item = bool>,
        r: &pallas::Scalar,
    ) -> CtOption<pallas::Base> {
        extract_p_bottom(self.commit(msg, r))
    }

    /// Returns the Sinsemilla $R$ constant for this domain.
    #[cfg(any(test, feature = "test-dependencies"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-dependencies")))]
    #[allow(non_snake_case)]
    pub fn R(&self) -> pallas::Point {
        self.R
    }

    /// Returns the Sinsemilla $Q$ constant for this domain.
    #[cfg(any(test, feature = "test-dependencies"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-dependencies")))]
    #[allow(non_snake_case)]
    pub fn Q(&self) -> pallas::Point {
        self.M.Q
    }
}

#[cfg(test)]
mod hash_to_point_tests;

#[cfg(test)]
mod tests {
    use super::K;
    use super::sinsemilla_s::SINSEMILLA_S;
    use group::Curve;
    use pasta_curves::{
        arithmetic::{CurveAffine, CurveExt},
        pallas,
    };

    #[test]
    fn sinsemilla_s() {
        let hasher = pallas::Point::hash_to_curve(super::S_PERSONALIZATION);

        for j in 0..(1u32 << K) {
            let computed = {
                let point = hasher(&j.to_le_bytes()).to_affine().coordinates().unwrap();
                (*point.x(), *point.y())
            };
            let actual = SINSEMILLA_S[j as usize];
            assert_eq!(computed, actual);
        }
    }
}
