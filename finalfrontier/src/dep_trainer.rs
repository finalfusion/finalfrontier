use std::sync::Arc;

use conllx::graph::Sentence;
use failure::{err_msg, Error};
use rand::{Rng, SeedableRng};
use serde::Serialize;

use crate::sampling::ZipfRangeGenerator;
use crate::train_model::{NegativeSamples, TrainIterFrom};
use crate::util::ReseedOnCloneRng;
use crate::{
    CommonConfig, DepembedsConfig, Dependency, DependencyIterator, Idx, SimpleVocab,
    SimpleVocabConfig, SubwordVocab, SubwordVocabConfig, Trainer, Vocab,
};

/// Dependency embeddings Trainer.
///
/// The `DepembedsTrainer` holds the information and logic necessary to transform a
/// `conllx::Sentence` into an iterator of focus and context tuples. The struct is cheap to clone
/// because the vocabulary is shared between clones.
#[derive(Clone)]
pub struct DepembedsTrainer<R> {
    dep_config: DepembedsConfig,
    common_config: CommonConfig,
    input_vocab: Arc<SubwordVocab>,
    output_vocab: Arc<SimpleVocab<Dependency>>,
    range_gen: ZipfRangeGenerator<R>,
    rng: R,
}

impl<R> DepembedsTrainer<R> {
    pub fn dep_config(&self) -> DepembedsConfig {
        self.dep_config
    }
}

impl<R> DepembedsTrainer<ReseedOnCloneRng<R>>
where
    R: Rng + Clone + SeedableRng,
{
    /// Constructs a new `DepTrainer`.
    pub fn new(
        input_vocab: SubwordVocab,
        output_vocab: SimpleVocab<Dependency>,
        common_config: CommonConfig,
        dep_config: DepembedsConfig,
        rng: R,
    ) -> Self {
        let rng = ReseedOnCloneRng(rng);
        let range_gen = ZipfRangeGenerator::new_with_exponent(
            rng.clone(),
            output_vocab.len(),
            common_config.zipf_exponent,
        );
        DepembedsTrainer {
            common_config,
            dep_config,
            input_vocab: Arc::new(input_vocab),
            output_vocab: Arc::new(output_vocab),
            range_gen,
            rng,
        }
    }
}

impl<R> NegativeSamples for DepembedsTrainer<R>
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

impl<R> TrainIterFrom<Sentence, Vec<u64>> for DepembedsTrainer<R>
where
    R: Rng,
{
    type Iter = Box<Iterator<Item = (Vec<u64>, Vec<usize>)>>;
    type Contexts = Vec<usize>;

    fn train_iter_from(&mut self, sentence: &Sentence) -> Self::Iter {
        let mut tokens = vec![None; sentence.len() - 1];
        for (idx, token) in sentence.iter().filter_map(|node| node.token()).enumerate() {
            if let Some(vocab_idx) = self.input_vocab.idx(token.form()) {
                if self.rng.gen_range(0f32, 1f32)
                    < self.input_vocab.discard(vocab_idx.single_idx() as usize)
                {
                    tokens[idx] = Some(vocab_idx)
                }
            }
        }

        let mut contexts = vec![Vec::new(); sentence.len() - 1];
        let graph = sentence.dep_graph();
        for (focus, dep) in DependencyIterator::new_from_config(&graph, self.dep_config)
            .filter(|(focus, _dep)| tokens[*focus] != None)
        {
            if let Some(dep_id) = self.output_vocab.idx(&dep) {
                if self.rng.gen_range(0f32, 1f32) < self.output_vocab.discard(dep_id.single_idx()) {
                    contexts[focus].push(dep_id.single_idx())
                }
            }
        }
        Box::new(
            tokens
                .into_iter()
                .zip(contexts.into_iter())
                .filter(|(focus, _)| focus.is_some())
                .map(|(focus, ctx)| (focus.unwrap(), ctx)),
        )
    }
}

impl<R> Trainer for DepembedsTrainer<R>
where
    R: Rng,
{
    type InputVocab = SubwordVocab;
    type Metadata = DepembedsMetadata<SubwordVocabConfig, SimpleVocabConfig>;

    fn input_vocab(&self) -> &SubwordVocab {
        &self.input_vocab
    }

    fn try_into_input_vocab(self) -> Result<Self::InputVocab, Error> {
        match Arc::try_unwrap(self.input_vocab) {
            Ok(vocab) => Ok(vocab),
            Err(_) => Err(err_msg("Cannot unwrap input vocab.")),
        }
    }

    fn n_input_types(&self) -> usize {
        let n_buckets = 2usize.pow(self.input_vocab().config().buckets_exp);
        n_buckets + self.input_vocab().len()
    }

    fn n_output_types(&self) -> usize {
        self.output_vocab.len()
    }

    fn config(&self) -> &CommonConfig {
        &self.common_config
    }

    fn to_metadata(&self) -> Self::Metadata {
        DepembedsMetadata {
            common_config: self.common_config,
            dep_config: self.dep_config,
            input_vocab_config: self.input_vocab.config(),
            output_vocab_config: self.output_vocab.config(),
        }
    }
}

/// Metadata for dependency embeddings.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct DepembedsMetadata<IC, OC>
where
    IC: Serialize,
    OC: Serialize,
{
    common_config: CommonConfig,
    #[serde(rename = "model_config")]
    dep_config: DepembedsConfig,
    input_vocab_config: IC,
    output_vocab_config: OC,
}
