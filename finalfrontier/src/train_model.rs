use std::io::{Seek, Write};
use std::sync::Arc;

use failure::{err_msg, Error};
use finalfusion::{
    embeddings::Embeddings,
    io::WriteEmbeddings,
    metadata::Metadata,
    storage::NdArray,
    vocab::{SubwordVocab as FiFuSubwordVocab, Vocab as FiFuVocab},
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use toml::Value;

use hogwild::HogwildArray2;
use vec_simd::{l2_normalize, scale, scaled_add};
use {Config, ModelType, SubwordVocab, Vocab, WriteModelBinary};

/// Training model.
///
/// Instances of this type represent training models. Training models have
/// an input matrix, an output matrix, and a vocabulary. The input matrix
/// represents observed words, whereas the output matrix represents predicted
/// words. The output matrix is typically discarded after training. The
/// vocabulary holds lexical information, such as word -> index mappings
/// and word discard probabilities.
///
/// `TrainModel` stores the matrices as `HogwildArray`s to share parameters
/// between clones of the same model. The vocabulary is also shared between
/// clones due to memory considerations.
#[derive(Clone)]
pub struct TrainModel {
    config: Config,
    vocab: Arc<SubwordVocab>,
    input: HogwildArray2<f32>,
    output: HogwildArray2<f32>,
}

impl TrainModel where {
    /// Construct a model from a vocabulary.
    ///
    /// This randomly initializes the input and output matrices using a
    /// uniform distribution in the range [-1/dims, 1/dims).
    ///
    /// The number of rows of the input matrix is the vocabulary size
    /// plus the number of buckets for subword units. The number of rows
    /// of the output matrix is the vocabulary size.
    pub fn from_vocab(vocab: SubwordVocab, config: Config) -> Self {
        let init_bound = 1.0 / config.dims as f32;
        let distribution = Uniform::new_inclusive(-init_bound, init_bound);

        let n_buckets = 2usize.pow(config.buckets_exp as u32);

        let input = Array2::random(
            (vocab.len() + n_buckets, config.dims as usize),
            distribution,
        )
        .into();

        let output_vocab_size = match config.model {
            ModelType::SkipGram => vocab.len(),
            ModelType::StructuredSkipGram => vocab.len() * config.context_size as usize * 2,
        };
        let output = Array2::random((output_vocab_size, config.dims as usize), distribution).into();

        TrainModel {
            config,
            vocab: Arc::new(vocab),
            input,
            output,
        }
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

    /// Get the input embedding with the given index mutably.
    #[inline]
    pub(crate) fn input_embedding_mut(&mut self, idx: usize) -> ArrayViewMut1<f32> {
        self.input.subview_mut(Axis(0), idx)
    }

    pub(crate) fn into_parts(self) -> Result<(Config, SubwordVocab, Array2<f32>), Error> {
        let vocab = match Arc::try_unwrap(self.vocab) {
            Ok(vocab) => vocab,
            Err(_) => return Err(err_msg("Cannot unwrap vocabulary.")),
        };

        let input = match Arc::try_unwrap(self.input.into_inner()) {
            Ok(input) => input.into_inner(),
            Err(_) => return Err(err_msg("Cannot unwrap input matrix.")),
        };

        Ok((self.config, vocab, input))
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

    /// Get the model's vocabulary.
    pub fn vocab(&self) -> &SubwordVocab {
        &self.vocab
    }
}

impl<W> WriteModelBinary<W> for TrainModel
where
    W: Seek + Write,
{
    fn write_model_binary(self, write: &mut W) -> Result<(), Error> {
        let (config, subword_vocab, mut input_matrix) = self.into_parts()?;

        let words = subword_vocab
            .types()
            .iter()
            .map(|l| l.label().to_owned())
            .collect::<Vec<_>>();
        let vocab = FiFuSubwordVocab::new(words, config.min_n, config.max_n, config.buckets_exp);
        let metadata = Metadata(Value::try_from(config)?);

        // Compute and write word embeddings.
        let mut norms = vec![0f32; vocab.len()];
        for i in 0..subword_vocab.len() {
            let mut input = subword_vocab.subword_indices_idx(i).unwrap().to_owned();
            input.push(i as u64);
            let mut embed = Self::mean_embedding(input_matrix.view(), &input);
            norms[i] = l2_normalize(embed.view_mut());
            input_matrix.index_axis_mut(Axis(0), i).assign(&embed);
        }

        let storage = NdArray(input_matrix);

        Embeddings::new(Some(metadata), vocab, storage).write_embeddings(write)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ndarray::Array2;

    use super::TrainModel;
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
            vocab: Arc::new(vocab),
            input,
            output,
        };

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
