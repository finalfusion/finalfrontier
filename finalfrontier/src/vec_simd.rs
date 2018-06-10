use ndarray::{ArrayView1, ArrayViewMut1};

use std::simd::f32x4;

#[cfg(feature = "avx-accel")]
use std::simd::f32x8;

cfg_if! {
    if #[cfg(feature = "avx-accel")] {
        /// Scaling: u = au
        ///
        /// This function performs SIMD-vectorized scaling (BLAS sscal).
        pub fn scale(u: ArrayViewMut1<f32>, a: f32) {
            scale_f32x8(u, a)
        }

        /// Scaled addition: *u = u + av*
        ///
        /// This function performs SIMD-vectorized scaled addition (BLAS saxpy).
        pub fn scaled_add(u: ArrayViewMut1<f32>, v: ArrayView1<f32>, a: f32) {
            scaled_add_f32x8(u, v, a)
        }
    } else {
        /// Scaling: u = au
        ///
        /// This function performs SIMD-vectorized scaling (BLAS sscal).
        pub fn scale(u: ArrayViewMut1<f32>, a: f32) {
            scale_f32x4(u, a)
        }

        /// Scaled addition: *u = u + av*
        ///
        /// This function performs SIMD-vectorized scaled addition (BLAS saxpy).
        pub fn scaled_add(u: ArrayViewMut1<f32>, v: ArrayView1<f32>, a: f32) {
            scaled_add_f32x4(u, v, a)
        }
    }
}

#[allow(dead_code)]
fn scaled_add_f32x4(mut u: ArrayViewMut1<f32>, v: ArrayView1<f32>, a: f32) {
    assert_eq!(u.len(), v.len());

    let mut u = u
        .as_slice_mut()
        .expect("Cannot apply SIMD instructions on non-contiguous data.");
    let mut v = &v
        .as_slice()
        .expect("Cannot apply SIMD instructions on non-contiguous data.")[..u.len()];

    if a == 1f32 {
        while u.len() >= 4 {
            let mut ux4 = f32x4::load_unaligned(u);
            let vx4 = f32x4::load_unaligned(v);
            ux4 += vx4;
            ux4.store_unaligned(u);
            u = &mut { u }[4..];
            v = &v[4..];
        }
    } else {
        let ax4 = f32x4::splat(a);

        while u.len() >= 4 {
            let mut ux4 = f32x4::load_unaligned(u);
            let vx4 = f32x4::load_unaligned(v);
            ux4 += vx4 * ax4;
            ux4.store_unaligned(u);
            u = &mut { u }[4..];
            v = &v[4..];
        }
    }

    scaled_add_unvectorized(u, v, a);
}

#[cfg(feature = "avx-accel")]
fn scaled_add_f32x8(mut u: ArrayViewMut1<f32>, v: ArrayView1<f32>, a: f32) {
    assert_eq!(u.len(), v.len());

    let mut u = u
        .as_slice_mut()
        .expect("Cannot apply SIMD instructions on non-contiguous data.");
    let mut v = &v
        .as_slice()
        .expect("Cannot apply SIMD instructions on non-contiguous data.")[..u.len()];

    if a == 1f32 {
        while u.len() >= 8 {
            let mut ux8 = f32x8::load_unaligned(u);
            let vx8 = f32x8::load_unaligned(v);
            ux8 += vx8;
            ux8.store_unaligned(u);
            u = &mut { u }[8..];
            v = &v[8..];
        }
    } else {
        let ax8 = f32x8::splat(a);

        while u.len() >= 8 {
            let mut ux8 = f32x8::load_unaligned(u);
            let vx8 = f32x8::load_unaligned(v);
            ux8 += vx8 * ax8;
            ux8.store_unaligned(u);
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
fn scale_f32x4(mut u: ArrayViewMut1<f32>, a: f32) {
    let mut u = u
        .as_slice_mut()
        .expect("Cannot apply SIMD instructions on non-contiguous data.");

    let ax4 = f32x4::splat(a);

    while u.len() >= 4 {
        let mut ux4 = f32x4::load_unaligned(u);
        ux4 *= ax4;
        ux4.store_unaligned(u);
        u = &mut { u }[4..];
    }

    scale_unvectorized(u, a);
}

#[cfg(feature = "avx-accel")]
fn scale_f32x8(mut u: ArrayViewMut1<f32>, a: f32) {
    let mut u = u.as_slice_mut()
        .expect("Cannot apply SIMD instructions on non-contiguous data.");

    let ax8 = f32x8::splat(a);

    while u.len() >= 8 {
        let mut ux8 = f32x8::load_unaligned(u);
        ux8 *= ax8;
        ux8.store_unaligned(u);
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

    use util::{all_close, array_all_close};

    use super::{scale_f32x4, scale_unvectorized, scaled_add_f32x4, scaled_add_unvectorized};

    #[cfg(feature = "avx-accel")]
    use super::{scale_f32x8, scaled_add_f32x8};

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
        scaled_add_f32x4(u.view_mut(), v.view(), 1.0);
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }

    #[test]
    #[cfg(feature = "avx-accel")]
    fn add_f32x8_test() {
        let mut u = Array1::random((102,), Range::new(-1.0, 1.0));
        let v = Array1::random((102,), Range::new(-1.0, 1.0));
        let mut check = u.clone();
        scaled_add_unvectorized(check.as_slice_mut().unwrap(), v.as_slice().unwrap(), 1.0);
        scaled_add_f32x8(u.view_mut(), v.view(), 1.0);
        assert!(array_all_close(check.view(), u.view(), 1e-5));
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
        let mut u = Array1::random((100,), Range::new(-1.0, 1.0));
        let v = Array1::random((100,), Range::new(-1.0, 1.0));
        let mut check = u.clone();
        scaled_add_unvectorized(check.as_slice_mut().unwrap(), v.as_slice().unwrap(), 2.5);
        scaled_add_f32x4(u.view_mut(), v.view(), 2.5);
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }

    #[test]
    #[cfg(feature = "avx-accel")]
    fn scaled_add_f32x8_test() {
        let mut u = Array1::random((100,), Range::new(-1.0, 1.0));
        let v = Array1::random((100,), Range::new(-1.0, 1.0));
        let mut check = u.clone();
        scaled_add_unvectorized(check.as_slice_mut().unwrap(), v.as_slice().unwrap(), 2.5);
        scaled_add_f32x8(u.view_mut(), v.view(), 2.5);
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
        let mut u = Array1::random((100,), Range::new(-1.0, 1.0));
        let mut check = u.clone();
        scale_unvectorized(check.as_slice_mut().unwrap(), 2.);
        scale_f32x4(u.view_mut(), 2.);
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }

    #[test]
    #[cfg(feature = "avx-accel")]
    fn scale_f32x8_test() {
        let mut u = Array1::random((100,), Range::new(-1.0, 1.0));
        let mut check = u.clone();
        scale_unvectorized(check.as_slice_mut().unwrap(), 2.);
        scale_f32x8(u.view_mut(), 2.);
        assert!(array_all_close(check.view(), u.view(), 1e-5));
    }
}
