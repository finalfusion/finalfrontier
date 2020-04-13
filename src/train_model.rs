use std::io::{Seek, Write};
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use finalfusion::io::WriteEmbeddings;
use finalfusion::metadata::Metadata;
use finalfusion::norms::NdNorms;
use finalfusion::prelude::{Embeddings, VocabWrap};
use finalfusion::storage::NdArray;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::Serialize;
use toml::Value;

use crate::hogwild::HogwildArray2;
use crate::idx::WordIdx;
use crate::io::TrainInfo;
use crate::util::VersionInfo;
use crate::vec_simd::{l2_normalize, scale, scaled_add};
use crate::{CommonConfig, Vocab, WriteModelBinary};

/// Training model.
///
/// Instances of this type represent training models. Training models have
/// an input matrix, an output matrix, and a trainer. The input matrix
/// represents observed inputs, whereas the output matrix represents
/// predicted outputs. The output matrix is typically discarded after
/// training. The trainer holds lexical information, such as word ->
/// index mappings and word discard probabilities. Additionally the trainer
/// provides the logic to transform some input to an iterator of training
/// examples.
///
/// `TrainModel` stores the matrices as `HogwildArray`s to share parameters
/// between clones of the same model. The trainer is also shared between
/// clones due to memory considerations.
#[derive(Clone)]
pub struct TrainModel<T> {
    trainer: T,
    input: HogwildArray2<f32>,
    output: HogwildArray2<f32>,
}

impl<T> From<T> for TrainModel<T>
where
    T: Trainer,
{
    /// Construct a model from a Trainer.
    ///
    /// This randomly initializes the input and output matrices using a
    /// uniform distribution in the range [-1/dims, 1/dims).
    ///
    /// The number of rows of the input matrix is the vocabulary size
    /// plus the number of buckets for subword units. The number of rows
    /// of the output matrix is the number of possible outputs for the model.
    fn from(trainer: T) -> TrainModel<T> {
        let config = *trainer.config();
        let init_bound = 1.0 / config.dims as f32;
        let distribution = Uniform::new_inclusive(-init_bound, init_bound);

        let input = Array2::random(
            (trainer.input_vocab().n_input_types(), config.dims as usize),
            distribution,
        )
        .into();
        let output = Array2::random(
            (trainer.n_output_types(), config.dims as usize),
            distribution,
        )
        .into();
        TrainModel {
            trainer,
            input,
            output,
        }
    }
}

impl<T> TrainModel<T>
where
    T: Trainer,
{
    /// Get the model configuration.
    pub fn config(&self) -> &CommonConfig {
        &self.trainer.config()
    }
}

impl<V, T> TrainModel<T>
where
    T: Trainer<InputVocab = V>,
    V: Vocab,
{
    /// Get this model's input vocabulary.
    pub fn input_vocab(&self) -> &V {
        self.trainer.input_vocab()
    }
}

impl<T> TrainModel<T> {
    /// Get this model's trainer mutably.
    pub fn trainer(&mut self) -> &mut T {
        &mut self.trainer
    }

    /// Get the mean input embedding of the given indices.
    pub(crate) fn mean_input_embedding<'a, I>(&self, idx: &'a I) -> Array1<f32>
    where
        I: WordIdx,
        &'a I: IntoIterator<Item = u64>,
    {
        if idx.len() == 1 {
            self.input
                .view()
                .row(idx.into_iter().next().unwrap() as usize)
                .to_owned()
        } else {
            Self::mean_embedding(self.input.view(), idx)
        }
    }

    /// Get the mean input embedding of the given indices.
    fn mean_embedding<'a, I>(embeds: ArrayView2<f32>, indices: &'a I) -> Array1<f32>
    where
        I: WordIdx,
        &'a I: IntoIterator<Item = u64>,
    {
        let mut embed = Array1::zeros((embeds.ncols(),));
        let len = indices.len();
        for idx in indices {
            scaled_add(
                embed.view_mut(),
                embeds.index_axis(Axis(0), idx as usize),
                1.0,
            );
        }

        scale(embed.view_mut(), 1.0 / len as f32);

        embed
    }

    /// Get the input embedding with the given index.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn input_embedding(&self, idx: usize) -> ArrayView1<f32> {
        self.input.subview(Axis(0), idx)
    }

    /// Get the input embedding with the given index mutably.
    #[inline]
    pub(crate) fn input_embedding_mut(&mut self, idx: usize) -> ArrayViewMut1<f32> {
        self.input.subview_mut(Axis(0), idx)
    }

    pub(crate) fn into_parts(self) -> Result<(T, Array2<f32>)> {
        let input = match Arc::try_unwrap(self.input.into_inner()) {
            Ok(input) => input.into_inner(),
            Err(_) => bail!("Cannot unwrap input matrix."),
        };

        Ok((self.trainer, input))
    }

    /// Get the output embedding with the given index.
    #[inline]
    pub(crate) fn output_embedding(&self, idx: usize) -> ArrayView1<f32> {
        self.output.subview(Axis(0), idx)
    }

    /// Get the output embedding with the given index mutably.
    #[inline]
    pub(crate) fn output_embedding_mut(&mut self, idx: usize) -> ArrayViewMut1<f32> {
        self.output.subview_mut(Axis(0), idx)
    }
}

impl<W, T, V, M> WriteModelBinary<W> for TrainModel<T>
where
    W: Seek + Write,
    T: Trainer<InputVocab = V, Metadata = M>,
    V: Vocab + Into<VocabWrap>,
    V::VocabType: ToString,
    for<'a> &'a V::IdxType: IntoIterator<Item = u64>,
    M: Serialize,
{
    fn write_model_binary(self, write: &mut W, mut train_info: TrainInfo) -> Result<()> {
        let (trainer, mut input_matrix) = self.into_parts()?;
        let mut metadata = Value::try_from(trainer.to_metadata())?;
        let build_info = Value::try_from(VersionInfo::new())?;
        let metadata_table = metadata
            .as_table_mut()
            .ok_or_else(|| anyhow!("Metadata has to be 'Table'."))?;
        metadata_table.insert("version_info".to_string(), build_info);
        train_info.set_end();
        let train_info = Value::try_from(train_info)?;
        metadata_table.insert("training_info".to_string(), train_info);

        // Compute and write word embeddings.
        let mut norms = vec![0f32; trainer.input_vocab().len()];
        for (i, (norm, word)) in norms
            .iter_mut()
            .zip(trainer.input_vocab().types())
            .take(trainer.input_vocab().len())
            .enumerate()
        {
            let input = trainer.input_vocab().idx(word.label()).unwrap();
            let mut embed = Self::mean_embedding(input_matrix.view(), &input);
            *norm = l2_normalize(embed.view_mut());
            input_matrix.index_axis_mut(Axis(0), i).assign(&embed);
        }

        let vocab: VocabWrap = trainer.try_into_input_vocab()?.into();
        let storage = NdArray::new(input_matrix);
        let norms = NdNorms::new(Array1::from(norms));

        Embeddings::new(Some(Metadata::new(metadata)), vocab, storage, norms)
            .write_embeddings(write)
            .map_err(|err| err.into())
    }
}

/// Trainer Trait.
pub trait Trainer {
    type InputVocab: Vocab;
    type Metadata;

    /// Get the trainer's input vocabulary.
    fn input_vocab(&self) -> &Self::InputVocab;

    /// Destruct the trainer and get the input vocabulary.
    fn try_into_input_vocab(self) -> Result<Self::InputVocab>;

    /// Get the number of possible input types.
    ///
    /// In a model with subword units this value is calculated as:
    /// `2^n_buckets + input_vocab.len()`.
    fn n_input_types(&self) -> usize;

    /// Get the number of possible outputs.
    ///
    /// In a structured skipgram model this value is calculated as:
    /// `output_vocab.len() * context_size * 2`
    fn n_output_types(&self) -> usize;

    /// Get this Trainer's common hyperparameters.
    fn config(&self) -> &CommonConfig;

    /// Get this Trainer's configuration.
    fn to_metadata(&self) -> Self::Metadata;
}

/// TrainIterFrom.
///
/// This trait defines how some input `&S` is transformed into an iterator of training examples.
pub trait TrainIterFrom<'a, S>
where
    S: ?Sized,
{
    type Iter: Iterator<Item = (Self::Focus, Self::Contexts)>;
    type Focus;
    type Contexts: IntoIterator<Item = usize>;

    fn train_iter_from(&mut self, sequence: &S) -> Self::Iter;
}

/// Negative Samples
///
/// This trait defines a method on how to draw a negative sample given some output. The return value
/// should follow the distribution of the underlying output vocabulary.
pub trait NegativeSamples {
    fn negative_sample(&mut self, output: usize) -> usize;
}

#[cfg(test)]
mod tests {
    use finalfusion::subword::FinalfusionHashIndexer;
    use ndarray::Array2;
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    use super::TrainModel;
    use crate::config::SubwordVocabConfig;
    use crate::idx::WordWithSubwordsIdx;
    use crate::skipgram_trainer::SkipgramTrainer;
    use crate::util::all_close;
    use crate::{
        BucketConfig, CommonConfig, LossType, ModelType, SkipGramConfig, SubwordVocab, VocabBuilder,
    };

    const TEST_COMMON_CONFIG: CommonConfig = CommonConfig {
        dims: 3,
        epochs: 5,
        loss: LossType::LogisticNegativeSampling,
        lr: 0.05,
        negative_samples: 5,
        zipf_exponent: 0.5,
    };

    const TEST_SKIP_CONFIG: SkipGramConfig = SkipGramConfig {
        context_size: 5,
        model: ModelType::SkipGram,
    };

    const VOCAB_CONF: SubwordVocabConfig<BucketConfig> = SubwordVocabConfig {
        discard_threshold: 1e-4,
        min_count: 2,
        max_n: 6,
        min_n: 3,
        indexer: BucketConfig { buckets_exp: 21 },
    };

    #[test]
    pub fn model_embed_methods() {
        let mut vocab_config = VOCAB_CONF.clone();
        vocab_config.min_count = 1;

        let common_config = TEST_COMMON_CONFIG.clone();
        let skipgram_config = TEST_SKIP_CONFIG.clone();
        // We just need some bogus vocabulary
        let mut builder: VocabBuilder<_, String> = VocabBuilder::new(vocab_config);
        builder.count("bla".to_string());
        let vocab: SubwordVocab<_, FinalfusionHashIndexer> = builder.into();

        let input = Array2::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.])
            .unwrap()
            .into();
        let output = Array2::from_shape_vec((2, 3), vec![-1., -2., -3., -4., -5., -6.])
            .unwrap()
            .into();

        let mut model = TrainModel {
            trainer: SkipgramTrainer::new(
                vocab,
                XorShiftRng::from_entropy(),
                common_config,
                skipgram_config,
            ),
            input,
            output,
        };

        // Input embeddings
        assert!(all_close(
            model.input_embedding(0).as_slice().unwrap(),
            &[1., 2., 3.],
            1e-5
        ));
        assert!(all_close(
            model.input_embedding(1).as_slice().unwrap(),
            &[4., 5., 6.],
            1e-5
        ));

        // Mutable input embeddings
        assert!(all_close(
            model.input_embedding_mut(0).as_slice().unwrap(),
            &[1., 2., 3.],
            1e-5
        ));
        assert!(all_close(
            model.input_embedding_mut(1).as_slice().unwrap(),
            &[4., 5., 6.],
            1e-5
        ));

        // Output embeddings
        assert!(all_close(
            model.output_embedding(0).as_slice().unwrap(),
            &[-1., -2., -3.],
            1e-5
        ));
        assert!(all_close(
            model.output_embedding(1).as_slice().unwrap(),
            &[-4., -5., -6.],
            1e-5
        ));

        // Mutable output embeddings
        assert!(all_close(
            model.output_embedding_mut(0).as_slice().unwrap(),
            &[-1., -2., -3.],
            1e-5
        ));
        assert!(all_close(
            model.output_embedding_mut(1).as_slice().unwrap(),
            &[-4., -5., -6.],
            1e-5
        ));

        // Mean input embedding.
        assert!(all_close(
            model
                .mean_input_embedding(&WordWithSubwordsIdx::new(0, vec![1]))
                .as_slice()
                .unwrap(),
            &[2.5, 3.5, 4.5],
            1e-5
        ));
    }
}
