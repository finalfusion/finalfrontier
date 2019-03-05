use std::cmp;
use std::iter::FusedIterator;
use std::sync::Arc;

use rand::{Rng, SeedableRng};

use train_model::{NegativeSamples, TrainIterFrom, Trainer};
use util::ReseedOnCloneRng;
use {Config, ModelType, SubwordVocab, Vocab};

/// Skipgram Trainer
///
/// The `SkipgramTrainer` holds the information and logic necessary to transform a tokenized
/// sentence into an iterator of focus and context tuples. The struct is cheap to clone because
/// the vocabulary is shared between clones.
#[derive(Clone)]
pub struct SkipgramTrainer<R> {
    vocab: Arc<SubwordVocab>,
    rng: R,
    config: Config,
}

impl<R> SkipgramTrainer<ReseedOnCloneRng<R>>
where
    R: Rng + Clone + SeedableRng,
{
    /// Constructs a new `SkipgramTrainer`.
    pub fn new(vocab: SubwordVocab, rng: R, config: Config) -> Self {
        let vocab = Arc::new(vocab);
        let rng = ReseedOnCloneRng(rng);
        SkipgramTrainer { vocab, rng, config }
    }
}

impl<S, R> TrainIterFrom<[S]> for SkipgramTrainer<R>
where
    S: AsRef<str>,
    R: Rng + Clone,
{
    type Iter = SkipGramIter<R>;
    type Contexts = Vec<usize>;

    fn train_iter_from(&mut self, sequence: &[S]) -> Self::Iter {
        let mut ids = Vec::new();
        for t in sequence {
            if let Some(idx) = self.vocab.idx(t.as_ref()) {
                if self.rng.gen_range(0f32, 1f32) < self.vocab.discard(idx) {
                    ids.push(idx);
                }
            }
        }
        SkipGramIter::new(self.rng.clone(), ids, self.config)
    }
}

impl<R> NegativeSamples for SkipgramTrainer<R>
where
    R: Rng,
{
    fn negative_sample(&mut self, output: usize) -> usize {
        loop {
            let negative = match self.config.model {
                ModelType::StructuredSkipGram => {
                    let context_size = self.config.context_size as usize;
                    let offset = output % (context_size * 2);
                    let rand_type = self.rng.gen_range(0, self.vocab.len());
                    // in structured skipgram the offset into the output matrix is calculated as:
                    // (vocab_idx * context_size * 2) + offset
                    rand_type * context_size * 2 + offset
                }
                ModelType::SkipGram => self.rng.gen_range(0, self.vocab.len()),
            };
            if negative != output {
                return negative;
            }
        }
    }
}

impl<R> Trainer for SkipgramTrainer<R>
where
    R: Rng + Clone,
{
    type InputVocab = SubwordVocab;

    fn input_indices(&self, idx: usize) -> Vec<u64> {
        let mut v = self.vocab.subword_indices_idx(idx).unwrap().to_vec();
        v.push(idx as u64);
        v
    }

    fn input_vocab(&self) -> &SubwordVocab {
        &self.vocab
    }

    fn n_input_types(&self) -> usize {
        let n_buckets = 2usize.pow(self.config.buckets_exp as u32);
        n_buckets + self.vocab.len()
    }

    fn n_output_types(&self) -> usize {
        match self.config.model {
            ModelType::StructuredSkipGram => {
                self.vocab.len() * 2 * self.config.context_size as usize
            }
            ModelType::SkipGram => self.vocab.len(),
        }
    }

    fn config(&self) -> &Config {
        &self.config
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
    pub fn new(rng: R, ids: Vec<usize>, config: Config) -> Self {
        SkipGramIter {
            ids,
            rng,
            i: 0,
            model_type: config.model,
            ctx_size: config.context_size as usize,
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
