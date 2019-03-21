use std::borrow::Borrow;
use std::cmp;
use std::hash::Hash;
use std::iter::FusedIterator;
use std::sync::Arc;

use failure::{err_msg, Error};
use rand::{Rng, SeedableRng};
use serde::Serialize;

use crate::sampling::{BandedRangeGenerator, ZipfRangeGenerator};
use crate::train_model::{Indices, NegativeSamples, TrainIterFrom, Trainer};
use crate::util::ReseedOnCloneRng;
use crate::{CommonConfig, ModelType, SkipGramConfig, Vocab};

/// Skipgram Trainer
///
/// The `SkipgramTrainer` holds the information and logic necessary to transform a tokenized
/// sentence into an iterator of focus and context tuples. The struct is cheap to clone because
/// the vocabulary is shared between clones.
#[derive(Clone)]
pub struct SkipgramTrainer<R, V> {
    vocab: Arc<V>,
    rng: R,
    range_gen: BandedRangeGenerator<R, ZipfRangeGenerator<R>>,
    common_config: CommonConfig,
    skipgram_config: SkipGramConfig,
}

impl<R, V> SkipgramTrainer<ReseedOnCloneRng<R>, V>
where
    R: Rng + Clone + SeedableRng,
    V: Vocab,
{
    /// Constructs a new `SkipgramTrainer`.
    pub fn new(
        vocab: V,
        rng: R,
        common_config: CommonConfig,
        skipgram_config: SkipGramConfig,
    ) -> Self {
        let vocab = Arc::new(vocab);
        let rng = ReseedOnCloneRng(rng);
        let band_size = match skipgram_config.model {
            ModelType::SkipGram => 1,
            ModelType::StructuredSkipGram => skipgram_config.context_size * 2,
            ModelType::DirectionalSkipgram => 2,
        };

        let range_gen = BandedRangeGenerator::new(
            rng.clone(),
            ZipfRangeGenerator::new_with_exponent(
                rng.clone(),
                vocab.len(),
                common_config.zipf_exponent,
            ),
            band_size as usize,
        );
        SkipgramTrainer {
            vocab,
            rng,
            range_gen,
            common_config,
            skipgram_config,
        }
    }
}

impl<S, R, V> TrainIterFrom<[S]> for SkipgramTrainer<R, V>
where
    S: Hash + Eq,
    R: Rng + Clone,
    V: Vocab,
    V::VocabType: Borrow<S>,
{
    type Iter = SkipGramIter<R>;
    type Contexts = Vec<usize>;

    fn train_iter_from(&mut self, sequence: &[S]) -> Self::Iter {
        let mut ids = Vec::new();
        for t in sequence {
            if let Some(idx) = self.vocab.idx(t) {
                if self.rng.gen_range(0f32, 1f32) < self.vocab.discard(idx) {
                    ids.push(idx);
                }
            }
        }
        SkipGramIter::new(self.rng.clone(), ids, self.skipgram_config)
    }
}

impl<R, V> NegativeSamples for SkipgramTrainer<R, V>
where
    R: Rng,
    V: Vocab,
{
    fn negative_sample(&mut self, output: usize) -> usize {
        loop {
            let negative = self.range_gen.next().unwrap();
            if negative != output {
                return negative;
            }
        }
    }
}

impl<R, V, VC> Trainer for SkipgramTrainer<R, V>
where
    R: Rng + Clone,
    V: Vocab<Config = VC>,
    VC: Serialize,
{
    type InputVocab = V;
    type Metadata = SkipgramMetadata<VC>;

    fn input_indices(&self, idx: usize) -> Indices {
        if let Some(indices) = self.vocab.idx2indices(idx).unwrap() {
            let mut v = indices.to_vec();
            v.push(idx as u64);
            Indices::Multiple(v)
        } else {
            Indices::Single([idx as u64])
        }
    }

    fn input_vocab(&self) -> &V {
        &self.vocab
    }

    fn try_into_input_vocab(self) -> Result<V, Error> {
        match Arc::try_unwrap(self.vocab) {
            Ok(vocab) => Ok(vocab),
            Err(_) => Err(err_msg("Cannot unwrap input vocab.")),
        }
    }

    fn n_input_types(&self) -> usize {
        self.input_vocab().n_input_types()
    }

    fn n_output_types(&self) -> usize {
        match self.skipgram_config.model {
            ModelType::StructuredSkipGram => {
                self.vocab.len() * 2 * self.skipgram_config.context_size as usize
            }
            ModelType::SkipGram => self.vocab.len(),
            ModelType::DirectionalSkipgram => self.vocab.len() * 2,
        }
    }

    fn config(&self) -> &CommonConfig {
        &self.common_config
    }

    fn to_metadata(&self) -> SkipgramMetadata<VC> {
        SkipgramMetadata {
            common_config: self.common_config,
            skipgram_config: self.skipgram_config,
            vocab_config: self.vocab.config(),
        }
    }
}

/// Iterator over focus identifier and associated context identifiers in a sentence.
pub struct SkipGramIter<R> {
    ids: Vec<usize>,
    rng: R,
    i: usize,
    model_type: ModelType,
    ctx_size: usize,
}

impl<R> SkipGramIter<R>
where
    R: Rng + Clone,
{
    /// Constructs a new `SkipGramIter`.
    ///
    /// The `rng` is used to determine the window size for each focus token.
    pub fn new(rng: R, ids: Vec<usize>, skip_config: SkipGramConfig) -> Self {
        SkipGramIter {
            ids,
            rng,
            i: 0,
            model_type: skip_config.model,
            ctx_size: skip_config.context_size as usize,
        }
    }

    fn output_(&self, token: usize, focus_idx: usize, offset_idx: usize) -> usize {
        match self.model_type {
            ModelType::StructuredSkipGram => {
                let offset = if offset_idx < focus_idx {
                    (offset_idx + self.ctx_size) - focus_idx
                } else {
                    (offset_idx - focus_idx - 1) + self.ctx_size
                };

                (token * self.ctx_size * 2) + offset
            }
            ModelType::SkipGram => token,
            ModelType::DirectionalSkipgram => {
                let offset = if offset_idx < focus_idx { 0 } else { 1 };

                (token * 2) + offset
            }
        }
    }
}

impl<R> Iterator for SkipGramIter<R>
where
    R: Rng + Clone,
{
    type Item = (usize, Vec<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.ids.len() {
            // Bojanowski, et al., 2017 uniformly sample the context size between 1 and c.
            let context_size = self.rng.gen_range(1, self.ctx_size + 1) as usize;
            let left = self.i - cmp::min(self.i, context_size);
            let right = cmp::min(self.i + context_size + 1, self.ids.len());
            let contexts = (left..right)
                .filter(|&idx| idx != self.i)
                .map(|idx| self.output_(self.ids[idx], self.i, idx))
                .fold(Vec::with_capacity(right - left), |mut contexts, idx| {
                    contexts.push(idx);
                    contexts
                });
            self.i += 1;
            return Some((self.ids[self.i - 1], contexts));
        }
        None
    }
}

impl<R> FusedIterator for SkipGramIter<R> where R: Rng + Clone {}

/// Metadata for Skipgramlike training algorithms.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct SkipgramMetadata<V>
where
    V: Serialize,
{
    common_config: CommonConfig,
    #[serde(rename = "model_config")]
    skipgram_config: SkipGramConfig,
    vocab_config: V,
}
