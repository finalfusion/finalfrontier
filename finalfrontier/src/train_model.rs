use std::io::{Seek, Write};
use std::iter::FusedIterator;
use std::sync::Arc;

use failure::{err_msg, Error};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rust2vec::{
    embeddings::Embeddings, io::WriteEmbeddings, metadata::Metadata, storage::NdArray,
    vocab::SubwordVocab as R2VSubwordVocab,
};
use toml::Value;

use hogwild::HogwildArray2;
use vec_simd::{l2_normalize, scale, scaled_add};
use {Config, SubwordVocab, Vocab, WriteModelBinary};

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
    config: Config,
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
            (trainer.n_input_types(), config.dims as usize),
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
            config,
        }
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

    /// Get the model configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Get the mean input embedding of the given indices.
    pub(crate) fn mean_input_embedding(&self, indices: &[u64]) -> Array1<f32> {
        Self::mean_embedding(self.input.view(), indices)
    }

    /// Get the mean input embedding of the given indices.
    fn mean_embedding(embeds: ArrayView2<f32>, indices: &[u64]) -> Array1<f32> {
        let mut embed = Array1::zeros((embeds.cols(),));

        for &idx in indices.iter() {
            scaled_add(
                embed.view_mut(),
                embeds.index_axis(Axis(0), idx as usize),
                1.0,
            );
        }

        scale(embed.view_mut(), 1.0 / indices.len() as f32);

        embed
    }

    /// Get the input embedding with the given index.
    #[inline]
    pub(crate) fn input_embedding(&self, idx: usize) -> ArrayView1<f32> {
        self.input.subview(Axis(0), idx)
    }

    /// Get the input embedding with the given index mutably.
    #[inline]
    pub(crate) fn input_embedding_mut(&mut self, idx: usize) -> ArrayViewMut1<f32> {
        self.input.subview_mut(Axis(0), idx)
    }

    pub(crate) fn into_parts(self) -> Result<(Config, T, Array2<f32>), Error> {
        let input = match Arc::try_unwrap(self.input.into_inner()) {
            Ok(input) => input.into_inner(),
            Err(_) => return Err(err_msg("Cannot unwrap input matrix.")),
        };

        Ok((self.config, self.trainer, input))
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

impl<W, T> WriteModelBinary<W> for TrainModel<T>
where
    W: Seek + Write,
    T: Trainer<InputVocab = SubwordVocab>,
{
    fn write_model_binary(self, write: &mut W) -> Result<(), Error> {
        let (config, trainer, mut input_matrix) = self.into_parts()?;

        let words = trainer
            .input_vocab()
            .types()
            .iter()
            .map(|l| l.label().to_owned())
            .collect::<Vec<_>>();
        let vocab = R2VSubwordVocab::new(words, config.min_n, config.max_n, config.buckets_exp);
        let metadata = Metadata(Value::try_from(config)?);

        // Compute and write word embeddings.
        let mut norms = vec![0f32; trainer.input_vocab().len()];
        for i in 0..trainer.input_vocab().len() {
            let input = trainer.input_indices(i);
            let mut embed = Self::mean_embedding(input_matrix.view(), &input);
            norms[i] = l2_normalize(embed.view_mut());
            input_matrix.index_axis_mut(Axis(0), i).assign(&embed);
        }

        let storage = NdArray(input_matrix);

        Embeddings::new(Some(metadata), vocab, storage).write_embeddings(write)
    }
}

/// Trainer Trait.
pub trait Trainer {
    type InputVocab: Vocab;

    /// Given an input index get all associated indices.
    fn input_indices(&self, idx: usize) -> Vec<u64>;

    /// Get the trainer's input vocabulary.
    fn input_vocab(&self) -> &Self::InputVocab;

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

    /// Get this Trainer's `Config`
    fn config(&self) -> &Config;
}

/// TrainIterFrom.
///
/// This trait defines how some input `&S` is transformed into an iterator of training examples.
pub trait TrainIterFrom<S>
where
    S: ?Sized,
{
    type Iter: Iterator<Item = (usize, Self::Contexts)> + FusedIterator;
    type Contexts: Sized + IntoIterator<Item = usize>;

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
    use ndarray::Array2;
    use rand::FromEntropy;
    use rand_xorshift::XorShiftRng;

    use super::TrainModel;
    use skipgram_trainer::SkipgramTrainer;
    use util::all_close;
    use {Config, LossType, ModelType, VocabBuilder};

    const TEST_CONFIG: Config = Config {
        buckets_exp: 21,
        context_size: 5,
        dims: 3,
        discard_threshold: 1e-4,
        epochs: 5,
        loss: LossType::LogisticNegativeSampling,
        lr: 0.05,
        min_count: 2,
        max_n: 6,
        min_n: 3,
        model: ModelType::SkipGram,
        negative_samples: 5,
        zipf_exponent: 0.5,
    };

    #[test]
    pub fn model_embed_methods() {
        let mut config = TEST_CONFIG.clone();
        config.min_count = 1;

        // We just need some bogus vocabulary
        let builder: VocabBuilder<String> = VocabBuilder::new(TEST_CONFIG.clone());
        let vocab = builder.into();

        let input = Array2::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.])
            .unwrap()
            .into();
        let output = Array2::from_shape_vec((2, 3), vec![-1., -2., -3., -4., -5., -6.])
            .unwrap()
            .into();

        let mut model = TrainModel {
            config,
            trainer: SkipgramTrainer::new(vocab, XorShiftRng::from_entropy(), config),
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
            model.mean_input_embedding(&[0, 1]).as_slice().unwrap(),
            &[2.5, 3.5, 4.5],
            1e-5
        ));
    }
}
