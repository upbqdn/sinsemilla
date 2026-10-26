use group::{Curve, Group, ff::Field};
use pasta_curves::pallas;
use rand::{SeedableRng, rngs::StdRng};
use subtle::CtOption;

use super::IncompletePoint;

fn assert_matches_two_additions(p: IncompletePoint, q: pallas::Affine) {
    let expected: CtOption<pallas::Point> = ((p + q) + p).into();
    let actual: CtOption<pallas::Point> = p.double_and_add(q).into();
    assert_eq!(bool::from(actual.is_some()), bool::from(expected.is_some()));
    if bool::from(expected.is_some()) {
        assert_eq!(actual.unwrap(), expected.unwrap());
    }
}

#[test]
fn double_and_add_matches_two_additions() {
    let mut rng = StdRng::seed_from_u64(0x444f55424c454144);
    let generator = pallas::Point::generator();
    for scalar in [
        pallas::Scalar::zero(),
        pallas::Scalar::one(),
        -pallas::Scalar::one(),
    ]
    .into_iter()
    .chain((0..256).map(|_| pallas::Scalar::random(&mut rng)))
    {
        let p = generator * scalar;
        for q in [
            pallas::Point::identity(),
            p,
            -p,
            p.double(),
            -p.double(),
            generator * pallas::Scalar::from(7),
        ] {
            assert_matches_two_additions(IncompletePoint::from(p), q.to_affine());
            let invalid = IncompletePoint::from(p) + p.to_affine();
            assert_matches_two_additions(invalid, q.to_affine());
        }
    }
}
