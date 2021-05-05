use criterion::{black_box, criterion_group, criterion_main, Criterion};
use finalfrontier::vec_simd;
use ndarray::Array1;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

const ARRAY_SIZE: usize = 512;

fn random_array(n: usize) -> Array1<f32> {
    Array1::random((n,), Normal::new(0.0, 0.5).unwrap())
}

fn dot_avx(c: &mut Criterion) {
    let u = random_array(ARRAY_SIZE);
    let v = random_array(ARRAY_SIZE);
    c.bench_function("dot_avx", move |b| {
        b.iter(|| black_box(unsafe { vec_simd::avx::dot(u.view(), v.view()) }))
    });
}

fn dot_fma(c: &mut Criterion) {
    let u = random_array(ARRAY_SIZE);
    let v = random_array(ARRAY_SIZE);
    c.bench_function("dot_fma", move |b| {
        b.iter(|| black_box(unsafe { vec_simd::avx_fma::dot(u.view(), v.view()) }))
    });
}

fn dot_ndarray(c: &mut Criterion) {
    let u = random_array(ARRAY_SIZE);
    let v = random_array(ARRAY_SIZE);
    c.bench_function("dot_ndarray", move |b| b.iter(|| black_box(u.dot(&v))));
}

fn dot_sse(c: &mut Criterion) {
    let u = random_array(ARRAY_SIZE);
    let v = random_array(ARRAY_SIZE);
    c.bench_function("dot_sse", move |b| {
        b.iter(|| black_box(unsafe { vec_simd::sse::dot(u.view(), v.view()) }))
    });
}

fn dot_unvectorized(c: &mut Criterion) {
    let u = random_array(ARRAY_SIZE);
    let v = random_array(ARRAY_SIZE);
    c.bench_function("dot_unvectorized", move |b| {
        b.iter(|| {
            black_box(vec_simd::dot_unvectorized(
                u.as_slice().unwrap(),
                v.as_slice().unwrap(),
            ))
        });
    });
}

criterion_group!(
    benches,
    dot_avx,
    dot_fma,
    dot_ndarray,
    dot_sse,
    dot_unvectorized
);
criterion_main!(benches);
