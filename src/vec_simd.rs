use ndarray::{ArrayView1, ArrayViewMut1};

/// Dot product: u · v
///
/// If the CPU supports SSE or AVX instructions, the dot
/// product is SIMD-vectorized.
pub fn dot(u: ArrayView1<f32>, v: ArrayView1<f32>) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx") {
            if is_x86_feature_detected!("fma") {
                return unsafe { avx_fma::dot(u, v) };
            }

            return unsafe { avx::dot(u, v) };
        } else if is_x86_feature_detected!("sse") {
            return unsafe { sse::dot(u, v) };
        }
    }

    dot_unvectorized(
        u.as_slice().expect("Cannot use vector u as slice"),
        v.as_slice().expect("Cannot use vector v as slice"),
    )
}

/// Scaling: u = au
///
/// If the CPU supports SSE or AVX instructions, scaling is
/// SIMD-vectorized.
pub fn scale(mut u: ArrayViewMut1<f32>, a: f32) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx") {
            return unsafe { avx::scale(u, a) };
        } else if is_x86_feature_detected!("sse") {
            return unsafe { sse::scale(u, a) };
        }
    }

    scale_unvectorized(u.as_slice_mut().expect("Cannot use vector u as slice"), a)
}

/// Scaled addition: *u = u + av*
///
/// If the CPU supports SSE or AVX instructions, scaled addition is
/// SIMD-vectorized.
pub fn scaled_add(mut u: ArrayViewMut1<f32>, v: ArrayView1<f32>, a: f32) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx") {
            return unsafe { avx::scaled_add(u, v, a) };
        } else if is_x86_feature_detected!("sse") {
            return unsafe { sse::scaled_add(u, v, a) };
        }
    }

    scaled_add_unvectorized(
        u.as_slice_mut().expect("Cannot use vector u as slice"),
        v.as_slice().expect("Cannot use vector v as slice"),
        a,
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;

    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use ndarray::{ArrayView1, ArrayViewMut1};

    use super::{dot_unvectorized, scale_unvectorized, scaled_add_unvectorized};

    #[target_feature(enable = "sse")]
    #[allow(dead_code)]
    pub unsafe fn dot(u: ArrayView1<f32>, v: ArrayView1<f32>) -> f32 {
        assert_eq!(u.len(), v.len());

        let mut u = u
            .as_slice()
            .expect("Cannot apply SIMD instructions on non-contiguous data.");
        let mut v = &v
            .as_slice()
            .expect("Cannot apply SIMD instructions on non-contiguous data.")[..u.len()];

        let mut sums = _mm_setzero_ps();

        while u.len() >= 4 {
            let ux4 = _mm_loadu_ps(&u[0] as *const f32);
            let vx4 = _mm_loadu_ps(&v[0] as *const f32);

            sums = _mm_add_ps(_mm_mul_ps(ux4, vx4), sums);

            u = &u[4..];
            v = &v[4..];
        }

        sums = _mm_hadd_ps(sums, sums);
        sums = _mm_hadd_ps(sums, sums);

        _mm_cvtss_f32(sums) + dot_unvectorized(u, v)
    }

    #[target_feature(enable = "sse")]
    #[allow(dead_code)]
    pub unsafe fn scale(mut u: ArrayViewMut1<f32>, a: f32) {
        let mut u = u
            .as_slice_mut()
            .expect("Cannot apply SIMD instructions on non-contiguous data.");

        let ax4 = _mm_set1_ps(a);

        while u.len() >= 4 {
            let mut ux4 = _mm_loadu_ps(&u[0] as *const f32);
            ux4 = _mm_mul_ps(ux4, ax4);
            _mm_storeu_ps(&mut u[0] as *mut f32, ux4);
            u = &mut { u }[4..];
        }

        scale_unvectorized(u, a);
    }

    #[target_feature(enable = "sse")]
    #[allow(dead_code, clippy::float_cmp)]
    pub unsafe fn scaled_add(mut u: ArrayViewMut1<f32>, v: ArrayView1<f32>, a: f32) {
        assert_eq!(u.len(), v.len());

        let mut u = u
            .as_slice_mut()
            .expect("Cannot apply SIMD instructions on non-contiguous data.");
        let mut v = &v
            .as_slice()
            .expect("Cannot apply SIMD instructions on non-contiguous data.")[..u.len()];

        if a == 1f32 {
            while u.len() >= 4 {
                let mut ux4 = _mm_loadu_ps(&u[0] as *const f32);
                let vx4 = _mm_loadu_ps(&v[0] as *const f32);
                ux4 = _mm_add_ps(ux4, vx4);
                _mm_storeu_ps(&mut u[0] as *mut f32, ux4);
                u = &mut { u }[4..];
                v = &v[4..];
            }
        } else {
            let ax4 = _mm_set1_ps(a);

            while u.len() >= 4 {
                let mut ux4 = _mm_loadu_ps(&u[0] as *const f32);
                let vx4 = _mm_loadu_ps(&v[0] as *const f32);
                ux4 = _mm_add_ps(ux4, _mm_mul_ps(vx4, ax4));
                _mm_storeu_ps(&mut u[0] as *mut f32, ux4);
                u = &mut { u }[4..];
                v = &v[4..];
            }
        }

        scaled_add_unvectorized(u, v, a);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;

    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use ndarray::{ArrayView1, ArrayViewMut1};

    use super::{dot_unvectorized, scale_unvectorized, scaled_add_unvectorized};

    #[target_feature(enable = "avx")]
    pub unsafe fn dot(u: ArrayView1<f32>, v: ArrayView1<f32>) -> f32 {
        assert_eq!(u.len(), v.len());

        let mut u = u
            .as_slice()
            .expect("Cannot apply SIMD instructions on non-contiguous data.");
        let mut v = &v
            .as_slice()
            .expect("Cannot apply SIMD instructions on non-contiguous data.")[..u.len()];

        let mut sums = _mm256_setzero_ps();

        while u.len() >= 8 {
            let ux8 = _mm256_loadu_ps(&u[0] as *const f32);
            let vx8 = _mm256_loadu_ps(&v[0] as *const f32);

            sums = _mm256_add_ps(_mm256_mul_ps(ux8, vx8), sums);

            u = &u[8..];
            v = &v[8..];
        }

        sums = _mm256_hadd_ps(sums, sums);
        sums = _mm256_hadd_ps(sums, sums);

        // Sum sums[0..4] and sums[4..8].
        let sums = _mm_add_ps(_mm256_castps256_ps128(sums), _mm256_extractf128_ps(sums, 1));

        _mm_cvtss_f32(sums) + dot_unvectorized(u, v)
    }

    #[target_feature(enable = "avx")]
    pub unsafe fn scale(mut u: ArrayViewMut1<f32>, a: f32) {
        let mut u = u
            .as_slice_mut()
            .expect("Cannot apply SIMD instructions on non-contiguous data.");

        let ax8 = _mm256_set1_ps(a);

        while u.len() >= 8 {
            let mut ux8 = _mm256_loadu_ps(&mut u[0] as *const f32);
            ux8 = _mm256_mul_ps(ux8, ax8);
            _mm256_storeu_ps(&mut u[0] as *mut f32, ux8);
            u = &mut { u }[8..];
        }

        scale_unvectorized(u, a);
    }

    #[target_feature(enable = "avx")]
    #[allow(clippy::float_cmp)]
    pub unsafe fn scaled_add(mut u: ArrayViewMut1<f32>, v: ArrayView1<f32>, a: f32) {
        assert_eq!(u.len(), v.len());

        let mut u = u
            .as_slice_mut()
            .expect("Cannot apply SIMD instructions on non-contiguous data.");
        let mut v = &v
            .as_slice()
            .expect("Cannot apply SIMD instructions on non-contiguous data.")[..u.len()];

        if a == 1f32 {
            while u.len() >= 8 {
                let mut ux8 = _mm256_loadu_ps(&u[0] as *const f32);
                let vx8 = _mm256_loadu_ps(&v[0] as *const f32);

                ux8 = _mm256_add_ps(ux8, vx8);

                _mm256_storeu_ps(&mut u[0] as *mut f32, ux8);
                u = &mut { u }[8..];
                v = &v[8..];
            }
        } else {
            let ax8 = _mm256_set1_ps(a);

            while u.len() >= 8 {
                let mut ux8 = _mm256_loadu_ps(&mut u[0] as *const f32);
                let vx8 = _mm256_loadu_ps(&v[0] as *const f32);

                ux8 = _mm256_add_ps(ux8, _mm256_mul_ps(vx8, ax8));

                _mm256_storeu_ps(&mut u[0] as *mut f32, ux8);
                u = &mut { u }[8..];
                v = &v[8..];
            }
        }

        scaled_add_unvectorized(u, v, a);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx_fma {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;

    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use ndarray::ArrayView1;

    use super::dot_unvectorized;

    #[target_feature(enable = "avx", enable = "fma")]
    pub unsafe fn dot(u: ArrayView1<f32>, v: ArrayView1<f32>) -> f32 {
        assert_eq!(u.len(), v.len());

        let mut u = u
            .as_slice()
            .expect("Cannot apply SIMD instructions on non-contiguous data.");
        let mut v = &v
            .as_slice()
            .expect("Cannot apply SIMD instructions on non-contiguous data.")[..u.len()];

        let mut sums = _mm256_setzero_ps();

        while u.len() >= 8 {
            let ux8 = _mm256_loadu_ps(&u[0] as *const f32);
            let vx8 = _mm256_loadu_ps(&v[0] as *const f32);

            sums = _mm256_fmadd_ps(ux8, vx8, sums);

            u = &u[8..];
            v = &v[8..];
        }

        sums = _mm256_hadd_ps(sums, sums);
        sums = _mm256_hadd_ps(sums, sums);

        // Sum sums[0..4] and sums[4..8].
        let sums = _mm_add_ps(_mm256_castps256_ps128(sums), _mm256_extractf128_ps(sums, 1));

        _mm_cvtss_f32(sums) + dot_unvectorized(u, v)
    }
}

pub fn dot_unvectorized(u: &[f32], v: &[f32]) -> f32 {
    assert_eq!(u.len(), v.len());
    u.iter().zip(v).map(|(&a, &b)| a * b).sum()
}

#[allow(clippy::float_cmp)]
fn scaled_add_unvectorized(u: &mut [f32], v: &[f32], a: f32) {
    assert_eq!(u.len(), v.len());

    if a == 1f32 {
        for i in 0..u.len() {
            u[i] += v[i];
        }
    } else {
        for i in 0..u.len() {
            u[i] += v[i] * a;
        }
    }
}

fn scale_unvectorized(u: &mut [f32], a: f32) {
    for c in u {
        *c *= a;
    }
}

/// Normalize a vector by its l2 norm.
///
/// The l2 norm is returned.
#[inline]
pub fn l2_normalize(v: ArrayViewMut1<f32>) -> f32 {
    let norm = dot(v.view(), v.view()).sqrt();
    scale(v, 1.0 / norm);
    norm
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    use crate::util::{all_close, array_all_close, close};

    use super::{dot_unvectorized, l2_normalize, scale_unvectorized, scaled_add_unvectorized};

    #[cfg(target_feature = "sse")]
    use super::sse;

    #[cfg(target_feature = "avx")]
    use super::avx;

    #[cfg(all(target_feature = "avx", target_feature = "fma"))]
    use super::avx_fma;

    #[test]
    fn add_unvectorized_test() {
        let u = &mut [1., 2., 3., 4., 5.];
        let v = &[5., 3., 3., 2., 1.];
        scaled_add_unvectorized(u, v, 1.0);
        assert!(all_close(u, &[6.0, 5.0, 6.0, 6.0, 6.0], 1e-5));
    }

    #[test]
    #[cfg(target_feature = "sse")]
    fn add_sse_test() {
        let mut u = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let v = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let mut check = u.clone();
        scaled_add_unvectorized(check.as_slice_mut().unwrap(), v.as_slice().unwrap(), 1.0);
        unsafe { sse::scaled_add(u.view_mut(), v.view(), 1.0) };
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }

    #[test]
    #[cfg(target_feature = "avx")]
    fn add_avx_test() {
        let mut u = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let v = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let mut check = u.clone();
        scaled_add_unvectorized(check.as_slice_mut().unwrap(), v.as_slice().unwrap(), 1.0);
        unsafe { avx::scaled_add(u.view_mut(), v.view(), 1.0) };
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }

    #[test]
    #[cfg(target_feature = "sse")]
    fn dot_sse_test() {
        let u = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let v = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        assert!(close(
            unsafe { sse::dot(u.view(), v.view()) },
            dot_unvectorized(u.as_slice().unwrap(), v.as_slice().unwrap()),
            1e-5
        ));
    }

    #[test]
    #[cfg(target_feature = "avx")]
    fn dot_avx_test() {
        let u = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let v = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        assert!(close(
            unsafe { avx::dot(u.view(), v.view()) },
            dot_unvectorized(u.as_slice().unwrap(), v.as_slice().unwrap()),
            1e-5
        ));
    }

    #[test]
    #[cfg(all(target_feature = "avx", target_feature = "fma"))]
    fn dot_avx_fma_test() {
        let u = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let v = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        assert!(close(
            unsafe { avx_fma::dot(u.view(), v.view()) },
            dot_unvectorized(u.as_slice().unwrap(), v.as_slice().unwrap()),
            1e-5
        ));
    }

    #[test]
    fn dot_unvectorized_test() {
        let u = [1f32, -2f32, -3f32];
        let v = [2f32, 4f32, -2f32];
        let w = [-1f32, 3f32, 2.5f32];

        assert!(close(dot_unvectorized(&u, &v), 0f32, 1e-5));
        assert!(close(dot_unvectorized(&u, &w), -14.5f32, 1e-5));
        assert!(close(dot_unvectorized(&v, &w), 5f32, 1e-5));
    }

    #[test]
    fn scaled_add_unvectorized_test() {
        let u = &mut [1., 2., 3., 4., 5.];
        let v = &[5., 3., 3., 2., 1.];
        scaled_add_unvectorized(u, v, 0.5);
        assert!(all_close(u, &[3.5, 3.5, 4.5, 5.0, 5.5], 1e-5));
    }

    #[test]
    #[cfg(target_feature = "sse")]
    fn scaled_add_sse_test() {
        let mut u = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let v = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let mut check = u.clone();
        scaled_add_unvectorized(check.as_slice_mut().unwrap(), v.as_slice().unwrap(), 2.5);
        unsafe { sse::scaled_add(u.view_mut(), v.view(), 2.5) };
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }

    #[test]
    #[cfg(target_feature = "avx")]
    fn scaled_add_avx_test() {
        let mut u = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let v = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let mut check = u.clone();
        scaled_add_unvectorized(check.as_slice_mut().unwrap(), v.as_slice().unwrap(), 2.5);
        unsafe { avx::scaled_add(u.view_mut(), v.view(), 2.5) };
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }

    #[test]
    fn scale_unvectorized_test() {
        let s = &mut [1., 2., 3., 4., 5.];
        scale_unvectorized(s, 0.5);
        assert!(all_close(s, &[0.5, 1.0, 1.5, 2.0, 2.5], 1e-5));
    }

    #[test]
    #[cfg(target_feature = "sse")]
    fn scale_sse_test() {
        let mut u = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let mut check = u.clone();
        scale_unvectorized(check.as_slice_mut().unwrap(), 2.);
        unsafe { sse::scale(u.view_mut(), 2.) };
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }

    #[test]
    #[cfg(target_feature = "avx")]
    fn scale_avx_test() {
        let mut u = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        let mut check = u.clone();
        scale_unvectorized(check.as_slice_mut().unwrap(), 2.);
        unsafe { avx::scale(u.view_mut(), 2.) };
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }

    #[test]
    fn l2_normalize_test() {
        let mut u = Array1::from(vec![1., -2., -1., 3., -3., 1.]);
        assert!(close(l2_normalize(u.view_mut()), 5., 1e-5));
        assert!(all_close(
            &[0.2, -0.4, -0.2, 0.6, -0.6, 0.2],
            u.as_slice().unwrap(),
            1e-5
        ));

        // Normalization should result in a unit vector.
        let mut v = Array1::random((102,), Uniform::new_inclusive(-1.0, 1.0));
        l2_normalize(v.view_mut());
        assert!(close(v.dot(&v).sqrt(), 1.0, 1e-5));
    }
}
