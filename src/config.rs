use anyhow::{bail, Result};
use serde::Serialize;

/// Model types.
#[derive(Copy, Clone, Debug, Serialize)]
pub enum ModelType {
    // The skip-gram model (Mikolov, 2013).
    SkipGram,

    // The structured skip-gram model (Ling et al., 2015).
    StructuredSkipGram,

    // The directional skip-gram model (Song et al., 2018).
    DirectionalSkipgram,
}

impl ModelType {
    pub fn try_from(model: u8) -> Result<ModelType> {
        match model {
            0 => Ok(ModelType::SkipGram),
            1 => Ok(ModelType::StructuredSkipGram),
            2 => Ok(ModelType::DirectionalSkipgram),
            _ => bail!("Unknown model type: {}", model),
        }
    }

    pub fn try_from_str(model: &str) -> Result<ModelType> {
        match model {
            "skipgram" => Ok(ModelType::SkipGram),
            "structgram" => Ok(ModelType::StructuredSkipGram),
            "dirgram" => Ok(ModelType::DirectionalSkipgram),
            _ => bail!("Unknown model type: {}", model),
        }
    }
}

/// Losses.
#[derive(Copy, Clone, Debug, Serialize)]
pub enum LossType {
    /// Logistic regression with negative sampling.
    LogisticNegativeSampling,
}

impl LossType {
    pub fn try_from(model: u8) -> Result<LossType> {
        match model {
            0 => Ok(LossType::LogisticNegativeSampling),
            _ => bail!("Unknown model type: {}", model),
        }
    }
}

/// Common embedding model hyperparameters.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct CommonConfig {
    /// The loss function used for the model.
    pub loss: LossType,

    /// Word embedding dimensionality.
    pub dims: u32,

    /// The number of training epochs.
    pub epochs: u32,

    /// Number of negative samples to use for each context word.
    pub negative_samples: u32,

    /// The initial learning rate.
    pub lr: f32,

    /// Exponent in zipfian distribution.
    ///
    /// This is s in *f(k) = 1 / (k^s H_{N, s})*.
    pub zipf_exponent: f64,
}

/// Hyperparameters for Dependency Embeddings.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(tag = "type")]
#[serde(rename = "Depembeds")]
pub struct DepembedsConfig {
    /// Maximum depth to extract dependency contexts from.
    pub depth: u32,

    /// Include the ROOT as dependency context.
    pub use_root: bool,

    /// Lowercase all tokens when used as context.
    pub normalize: bool,

    /// Projectivize dependency graphs before training.
    pub projectivize: bool,

    /// Extract untyped dependency contexts.
    ///
    /// Only takes the attached word-form into account.
    pub untyped: bool,
}

/// Hyperparameters for Subword vocabs.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename = "SubwordVocab")]
#[serde(tag = "type")]
pub struct SubwordVocabConfig<V> {
    /// Minimum token count.
    ///
    /// No word-specific embeddings will be trained for tokens occurring less
    /// than this count.
    pub min_count: u32,

    /// Discard threshold.
    ///
    /// The discard threshold is used to compute the discard probability of
    /// a token. E.g. with a threshold of 0.00001 tokens with approximately
    /// that probability will never be discarded.
    pub discard_threshold: f32,

    /// Minimum n-gram length for subword units (inclusive).
    pub min_n: u32,

    /// Maximum n-gram length for subword units (inclusive).
    pub max_n: u32,

    /// Indexer specific parameters.
    pub indexer: V,
}

/// Hyperparameters for bucket-vocabs.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename = "Buckets")]
#[serde(tag = "type")]
pub struct BucketConfig {
    /// Bucket exponent. The model will use 2^bucket_exp buckets.
    ///
    /// A typical value for this parameter is 21, which gives roughly 2M
    /// buckets.
    pub buckets_exp: u32,
}

/// Hyperparameters for ngram-vocabs.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename = "NGrams")]
#[serde(tag = "type")]
pub struct NGramConfig {
    /// Minimum NGram count.
    ///
    /// Ngrams occurring less than `min_count` times in in-vocabulary tokens
    /// will be ignored.
    pub min_ngram_count: u32,
}

/// Hyperparameters for simple vocabs.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename = "SimpleVocab")]
#[serde(tag = "type")]
pub struct SimpleVocabConfig {
    /// Minimum token count.
    ///
    /// No word-specific embeddings will be trained for tokens occurring less
    /// than this count.
    pub min_count: u32,

    /// Discard threshold.
    ///
    /// The discard threshold is used to compute the discard probability of
    /// a token. E.g. with a threshold of 0.00001 tokens with approximately
    /// that probability will never be discarded.
    pub discard_threshold: f32,
}

/// Hyperparameters for SkipGram-like models.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(tag = "type")]
#[serde(rename = "SkipGramLike")]
pub struct SkipGramConfig {
    /// The model type.
    pub model: ModelType,

    /// The number of preceding and succeeding tokens that will be consider
    /// as context during training.
    ///
    /// For example, a context size of 5 will consider the 5 tokens preceding
    /// and the 5 tokens succeeding the focus token.
    pub context_size: u32,
}
