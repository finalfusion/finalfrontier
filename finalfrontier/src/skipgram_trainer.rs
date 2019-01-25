use rand::Rng;
use std::cmp;
use std::iter::FusedIterator;
use Config;
use ModelType;

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
