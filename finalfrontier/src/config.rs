/// Model types.
#[derive(Copy, Clone, Debug)]
pub enum ModelType {
    // The skip-gram model (Mikolov, 2013).
    SkipGram,
}

/// Losses.
#[derive(Copy, Clone, Debug)]
pub enum LossType {
    /// Logistic regression with negative sampling.
    LogisticNegativeSampling,
}

/// Embedding model hyperparameters.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// The model type.
    pub model: ModelType,

    /// The loss function used for the model.
    pub loss: LossType,

    /// The number of preceding and succeeding tokens that will be consider
    /// as context during training.
    ///
    /// For example, a context size of 5 will consider the 5 tokens preceding
    /// and the 5 tokens succeeding the focus token.
    pub context_size: u32,

    /// Discard threshold.
    ///
    /// The discard threshold is used to compute the discard probability of
    /// a token. E.g. with a threshold of 0.00001 tokens with approximately
    /// that probability will never be discarded.
    pub discard_threshold: f32,

    /// Word embedding dimensionality.
    pub dims: u32,

    /// The number of training epochs.
    pub epochs: u32,

    /// Minimum token count.
    ///
    /// No word-specific embeddings will be trained for tokens occurring less
    /// than this count.
    pub min_count: u32,

    /// Minimum n-gram length for subword units (inclusive).
    pub min_n: u32,

    /// Maximum n-gram length for subword units (inclusive).
    pub max_n: u32,

    /// Bucket exponent. The model will use 2^bucket_exp buckets.
    ///
    /// A typical value for this parameter is 21, which gives roughly 2M
    /// buckets.
    pub buckets_exp: u32,

    /// Number of negative samples to use for each context word.
    pub negative_samples: u32,

    /// The initial learning rate.
    pub lr: f32,
}
