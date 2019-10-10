use rand::{FromEntropy, SeedableRng};
use rand_core::{self, RngCore};
use serde::Serialize;

pub static EOS: &str = "</s>";

/// Tolerance for small negative values.
const NEGATIVE_TOLERANCE: f32 = 1e-5;

/// Add a small value, to prevent returning Inf on underflow.
#[inline]
pub fn safe_ln(v: f32) -> f32 {
    (v + NEGATIVE_TOLERANCE).ln()
}

/// RNG that reseeds on clone.
///
/// This is a wrapper struct for RNGs implementing the `RngCore`
/// trait.  It adds the following simple behavior: when a
/// `ReseedOnCloneRng` is cloned, the clone is constructed using fresh
/// entropy. This assures that the state of the clone is not related
/// to the cloned RNG.
///
/// The `rand` crate provides similar behavior in the `ReseedingRng`
/// struct. However, `ReseedingRng` requires that the RNG is
/// `BlockRngCore`.
pub struct ReseedOnCloneRng<R>(pub R)
where
    R: RngCore + SeedableRng;

impl<R> RngCore for ReseedOnCloneRng<R>
where
    R: RngCore + SeedableRng,
{
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.fill_bytes(dest)
    }

    #[inline]
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.0.try_fill_bytes(dest)
    }
}

impl<R> Clone for ReseedOnCloneRng<R>
where
    R: RngCore + FromEntropy + SeedableRng,
{
    fn clone(&self) -> Self {
        ReseedOnCloneRng(R::from_entropy())
    }
}

#[derive(Serialize)]
pub(crate) struct VersionInfo {
    finalfusion_version: &'static str,
    git_desc: Option<&'static str>,
}

impl VersionInfo {
    pub(crate) fn new() -> Self {
        VersionInfo {
            finalfusion_version: env!("CARGO_PKG_VERSION"),
            git_desc: option_env!("MAYBE_FINALFRONTIER_GIT_DESC"),
        }
    }
}

#[cfg(test)]
pub use self::test::*;

#[cfg(test)]
mod test {
    use ndarray::{ArrayView, Dimension};
    use rand::{FromEntropy, SeedableRng};
    use rand_core::{self, impls, le, RngCore};

    use super::ReseedOnCloneRng;

    #[derive(Clone)]
    struct BogusRng(pub u64);

    impl RngCore for BogusRng {
        fn next_u32(&mut self) -> u32 {
            self.next_u64() as u32
        }

        fn next_u64(&mut self) -> u64 {
            self.0 += 1;
            self.0
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) {
            impls::fill_bytes_via_next(self, dest)
        }

        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
            Ok(self.fill_bytes(dest))
        }
    }

    impl SeedableRng for BogusRng {
        type Seed = [u8; 8];

        fn from_seed(seed: Self::Seed) -> Self {
            let mut state = [0u64; 1];
            le::read_u64_into(&seed, &mut state);
            BogusRng(state[0])
        }
    }

    pub fn close(a: f32, b: f32, eps: f32) -> bool {
        let diff = (a - b).abs();
        if diff > eps {
            return false;
        }

        true
    }

    pub fn all_close(a: &[f32], b: &[f32], eps: f32) -> bool {
        for (&av, &bv) in a.iter().zip(b) {
            if !close(av, bv, eps) {
                return false;
            }
        }

        true
    }

    pub fn array_all_close<Ix>(a: ArrayView<f32, Ix>, b: ArrayView<f32, Ix>, eps: f32) -> bool
    where
        Ix: Dimension,
    {
        for (&av, &bv) in a.iter().zip(b) {
            if !close(av, bv, eps) {
                return false;
            }
        }

        true
    }

    #[test]
    fn reseed_on_clone_rng() {
        let bogus_rng = BogusRng::from_entropy();
        let bogus_rng_clone = bogus_rng.clone();
        assert_eq!(bogus_rng.0, bogus_rng_clone.0);

        let reseed = ReseedOnCloneRng(bogus_rng);
        let reseed_clone = reseed.clone();
        // One in 2^64 probability of collision given good entropy source.
        assert_ne!((reseed.0).0, (reseed_clone.0).0);
    }
}
