use std::borrow::Borrow;
use std::hash::Hash;
use std::iter::FusedIterator;
use std::sync::Arc;
use std::{cmp, mem};

use failure::{err_msg, Error};
use rand::{Rng, SeedableRng};
use serde::Serialize;

use crate::idx::WordIdx;
use crate::sampling::{BandedRangeGenerator, ZipfRangeGenerator};
use crate::train_model::{NegativeSamples, TrainIterFrom, Trainer};
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

impl<S, R, V, I> TrainIterFrom<[S]> for SkipgramTrainer<R, V>
where
    S: Hash + Eq,
    R: Rng + Clone,
    V: Vocab<IdxType = I>,
    V::VocabType: Borrow<S>,
    I: WordIdx,
{
    type Iter = SkipGramIter<R, I>;
    type Focus = I;
    type Contexts = Vec<usize>;

    fn train_iter_from(&mut self, sequence: &[S]) -> Self::Iter {
        let mut ids = Vec::new();
        for t in sequence {
            if let Some(idx) = self.vocab.idx(t) {
                if self.rng.gen_range(0f32, 1f32) < self.vocab.discard(idx.word_idx() as usize) {
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

impl<R, V> Trainer for SkipgramTrainer<R, V>
where
    R: Rng + Clone,
    V: Vocab,
    V::Config: Serialize,
{
    type InputVocab = V;
    type Metadata = SkipgramMetadata<V::Config>;

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

    fn to_metadata(&self) -> SkipgramMetadata<V::Config> {
        SkipgramMetadata {
            common_config: self.common_config,
            skipgram_config: self.skipgram_config,
            vocab_config: self.vocab.config(),
        }
    }
}

/// Iterator over focus identifier and associated context identifiers in a sentence.
pub struct SkipGramIter<R, I> {
    ids: Vec<I>,
    rng: R,
    i: usize,
    model_type: ModelType,
    ctx_size: usize,
}

impl<R, I> SkipGramIter<R, I>
where
    R: Rng + Clone,
    I: WordIdx,
{
    /// Constructs a new `SkipGramIter`.
    ///
    /// The `rng` is used to determine the window size for each focus token.
    pub fn new(rng: R, ids: Vec<I>, skip_config: SkipGramConfig) -> Self {
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

impl<R, I> Iterator for SkipGramIter<R, I>
where
    R: Rng + Clone,
    I: WordIdx,
{
    type Item = (I, Vec<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.ids.len() {
            // Bojanowski, et al., 2017 uniformly sample the context size between 1 and c.
            let context_size = self.rng.gen_range(1, self.ctx_size + 1) as usize;
            let left = self.i - cmp::min(self.i, context_size);
            let right = cmp::min(self.i + context_size + 1, self.ids.len());
            let contexts = (left..right)
                .filter(|&idx| idx != self.i)
                .map(|idx| self.output_(self.ids[idx].word_idx() as usize, self.i, idx))
                .fold(Vec::with_capacity(right - left), |mut contexts, idx| {
                    contexts.push(idx);
                    contexts
                });

            // swap the representation possibly containing multiple indices with one that only
            // contains the distinct word index since we need the word index for context lookups.
            let mut word_idx = WordIdx::from_word_idx(self.ids[self.i].word_idx());
            mem::swap(&mut self.ids[self.i], &mut word_idx);
            self.i += 1;
            return Some((word_idx, contexts));
        }
        None
    }
}

impl<R, I> FusedIterator for SkipGramIter<R, I>
where
    R: Rng + Clone,
    I: WordIdx,
{
}

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
