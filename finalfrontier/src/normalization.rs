//! Embedding normalization.

use ndarray::{ArrayViewMut1, ArrayViewMut2};

/// Trait for normalization of word embeddings.
pub trait Normalization {
    /// Normalize the given word embedding.
    fn normalize(v: ArrayViewMut1<f32>);

    /// Normalize the embedding matrix.
    fn normalize_matrix(mut v: ArrayViewMut2<f32>) {
        for mut v in v.outer_iter_mut() {
            Self::normalize(v);
        }
    }
}

/// Do not normalize word embeddings.
pub struct NoNormalization;

impl Normalization for NoNormalization {
    fn normalize(_: ArrayViewMut1<f32>) {}
}

/// Normalize word embeddings by their L2 norm.
pub struct L2Normalization;

impl Normalization for L2Normalization {
    fn normalize(mut v: ArrayViewMut1<f32>) {
        let l2norm = v.dot(&v).sqrt();
        if l2norm != 0f32 {
            v /= l2norm;
        }
    }
}
