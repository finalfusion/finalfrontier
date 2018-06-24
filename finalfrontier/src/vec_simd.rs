use ndarray::{ArrayView1, ArrayViewMut1};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Dot product: u · v
///
/// This SIMD-vectorized function computes the dot product
/// (BLAS sdot).
cfg_if! {
    if #[cfg(target_feature = "avx")] {
        pub fn dot(u: ArrayView1<f32>, v: ArrayView1<f32>) -> f32 {
            unsafe { dot_f32x8(u, v) }
        }
    } else if #[cfg(target_feature = "sse")] {
        pub fn dot(u: ArrayView1<f32>, v: ArrayView1<f32>) -> f32 {
            unsafe { dot_f32x4(u, v) }
        }
    } else {
        pub fn dot(u: ArrayView1<f32>, v: ArrayView1<f32>) -> f32 {
            dot_unvectorized(u, v)
        }
    }
}

/// Scaling: u = au
///
/// This function performs SIMD-vectorized scaling (BLAS sscal).
cfg_if! {
    if #[cfg(target_feature = "avx")] {
        pub fn scale(u: ArrayViewMut1<f32>, a: f32) {
            unsafe { scale_f32x8(u, a) }
        }
    } else if #[cfg(target_feature = "sse")] {
        pub fn scale(u: ArrayViewMut1<f32>, a: f32) {
            unsafe { scale_f32x4(u, a) }
        }
    } else {
        pub fn scale(u: ArrayViewMut1<f32>, a: f32) {
            scale_unvectorized(u, a)
        }
    }
}

/// Scaled addition: *u = u + av*
///
/// This function performs SIMD-vectorized scaled addition (BLAS saxpy).
cfg_if! {
    if #[cfg(target_feature = "avx")] {
        pub fn scaled_add(u: ArrayViewMut1<f32>, v: ArrayView1<f32>, a: f32) {
            unsafe { scaled_add_f32x8(u, v, a) }
        }
    } else if #[cfg(target_feature = "sse")] {
        pub fn scaled_add(u: ArrayViewMut1<f32>, v: ArrayView1<f32>, a: f32) {
            unsafe { scaled_add_f32x4(u, v, a) }
        }
    } else {
        pub fn scaled_add(u: ArrayViewMut1<f32>, v: ArrayView1<f32>, a: f32) {
            scaled_add_unvectorized(u, v, a)
        }
    }
}

#[allow(dead_code)]
unsafe fn dot_f32x4(u: ArrayView1<f32>, v: ArrayView1<f32>) -> f32 {
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

#[cfg(target_feature = "avx")]
unsafe fn dot_f32x8(u: ArrayView1<f32>, v: ArrayView1<f32>) -> f32 {
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

        // Future: support FMA?
        // sums = _mm256_fmadd_ps(a, b, sums);

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

pub fn dot_unvectorized(u: &[f32], v: &[f32]) -> f32 {
    assert_eq!(u.len(), v.len());
    u.iter().zip(v).map(|(&a, &b)| a * b).sum()
}

#[allow(dead_code)]
unsafe fn scaled_add_f32x4(mut u: ArrayViewMut1<f32>, v: ArrayView1<f32>, a: f32) {
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

#[cfg(target_feature = "avx")]
unsafe fn scaled_add_f32x8(mut u: ArrayViewMut1<f32>, v: ArrayView1<f32>, a: f32) {
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

#[allow(dead_code)]
unsafe fn scale_f32x4(mut u: ArrayViewMut1<f32>, a: f32) {
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

#[cfg(target_feature = "avx")]
unsafe fn scale_f32x8(mut u: ArrayViewMut1<f32>, a: f32) {
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

fn scale_unvectorized(u: &mut [f32], a: f32) {
    for i in 0..u.len() {
        u[i] *= a;
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;
    use ndarray_rand::RandomExt;
    use rand::distributions::Range;

    use util::{all_close, array_all_close, close};

    use super::{
        dot_f32x4, dot_unvectorized, scale_f32x4, scale_unvectorized, scaled_add_f32x4,
        scaled_add_unvectorized,
    };

    #[cfg(target_feature = "avx")]
    use super::{dot_f32x8, scale_f32x8, scaled_add_f32x8};

    #[test]
    fn add_unvectorized_test() {
        let u = &mut [1., 2., 3., 4., 5.];
        let v = &[5., 3., 3., 2., 1.];
        scaled_add_unvectorized(u, v, 1.0);
        assert!(all_close(u, &[6.0, 5.0, 6.0, 6.0, 6.0], 1e-5));
    }

    #[test]
    fn add_f32x4_test() {
        let mut u = Array1::random((102,), Range::new(-1.0, 1.0));
        let v = Array1::random((102,), Range::new(-1.0, 1.0));
        let mut check = u.clone();
        scaled_add_unvectorized(check.as_slice_mut().unwrap(), v.as_slice().unwrap(), 1.0);
        unsafe { scaled_add_f32x4(u.view_mut(), v.view(), 1.0) };
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }

    #[test]
    #[cfg(target_feature = "avx")]
    fn add_f32x8_test() {
        let mut u = Array1::random((102,), Range::new(-1.0, 1.0));
        let v = Array1::random((102,), Range::new(-1.0, 1.0));
        let mut check = u.clone();
        scaled_add_unvectorized(check.as_slice_mut().unwrap(), v.as_slice().unwrap(), 1.0);
        unsafe { scaled_add_f32x8(u.view_mut(), v.view(), 1.0) };
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }

    #[test]
    fn dot_f32x4_test() {
        let u = Array1::random((102,), Range::new(-1.0, 1.0));
        let v = Array1::random((102,), Range::new(-1.0, 1.0));
        assert!(close(
            unsafe { dot_f32x4(u.view(), v.view()) },
            dot_unvectorized(u.as_slice().unwrap(), v.as_slice().unwrap()),
            1e-5
        ));
    }

    #[test]
    #[cfg(target_feature = "avx")]
    fn dot_f32x8_test() {
        let u = Array1::random((102,), Range::new(-1.0, 1.0));
        let v = Array1::random((102,), Range::new(-1.0, 1.0));
        assert!(close(
            unsafe { dot_f32x8(u.view(), v.view()) },
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
    fn scaled_add_f32x4_test() {
        let mut u = Array1::random((102,), Range::new(-1.0, 1.0));
        let v = Array1::random((102,), Range::new(-1.0, 1.0));
        let mut check = u.clone();
        scaled_add_unvectorized(check.as_slice_mut().unwrap(), v.as_slice().unwrap(), 2.5);
        unsafe { scaled_add_f32x4(u.view_mut(), v.view(), 2.5) };
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }

    #[test]
    #[cfg(target_feature = "avx")]
    fn scaled_add_f32x8_test() {
        let mut u = Array1::random((102,), Range::new(-1.0, 1.0));
        let v = Array1::random((102,), Range::new(-1.0, 1.0));
        let mut check = u.clone();
        scaled_add_unvectorized(check.as_slice_mut().unwrap(), v.as_slice().unwrap(), 2.5);
        unsafe { scaled_add_f32x8(u.view_mut(), v.view(), 2.5) };
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }

    #[test]
    fn scale_unvectorized_test() {
        let s = &mut [1., 2., 3., 4., 5.];
        scale_unvectorized(s, 0.5);
        assert!(all_close(s, &[0.5, 1.0, 1.5, 2.0, 2.5], 1e-5));
    }

    #[test]
    fn scale_f32x4_test() {
        let mut u = Array1::random((102,), Range::new(-1.0, 1.0));
        let mut check = u.clone();
        scale_unvectorized(check.as_slice_mut().unwrap(), 2.);
        unsafe { scale_f32x4(u.view_mut(), 2.) };
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }

    #[test]
    #[cfg(target_feature = "avx")]
    fn scale_f32x8_test() {
        let mut u = Array1::random((102,), Range::new(-1.0, 1.0));
        let mut check = u.clone();
        scale_unvectorized(check.as_slice_mut().unwrap(), 2.);
        unsafe { scale_f32x8(u.view_mut(), 2.) };
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }
}
