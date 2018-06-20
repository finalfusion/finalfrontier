use std::io::Write;
use std::sync::Arc;

use byteorder::{LittleEndian, WriteBytesExt};
use failure::Error;
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Range;

use vec_simd::{scale, scaled_add};
use {Config, HogwildArray2, Vocab, WriteModelBinary};

/// Training model.
///
/// Instances of this type represent training models. Training models have
/// an input matrix, an output matrix, and a vocabulary. The input matrix
/// represents observed words, whereas the output matrix represents predicted
/// words. The output matrix is typically discarded after training. The
/// vocabulary holds lexical information, such as token -> index mappings
/// and token discard probabilities.
///
/// `TrainModel` stores the matrices as `HogwildArray`s to share parameters
/// between clones of the same model. The vocabulary is also shared between
/// clones due to memory considerations.
#[derive(Clone)]
pub struct TrainModel {
    config: Config,
    vocab: Arc<Vocab>,
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
    pub fn from_vocab(vocab: Vocab, config: Config) -> Self {
        let init_bound = 1.0 / config.dims as f32;
        let range = Range::new(-init_bound, init_bound);

        let n_buckets = 2usize.pow(config.buckets_exp as u32);

        let input = Array2::random((vocab.len() + n_buckets, config.dims as usize), range).into();
        let output = Array2::random((vocab.len(), config.dims as usize), range).into();

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
    pub fn mean_input_embedding(&self, indices: &[u64]) -> Array1<f32> {
        let mut embed = Array1::zeros((self.config.dims as usize,));

        for &idx in indices.iter() {
            scaled_add(embed.view_mut(), self.input_embedding(idx as usize), 1.0);
        }

        scale(embed.view_mut(), 1.0 / indices.len() as f32);

        embed
    }

    /// Get the input embedding with the given index.
    #[inline]
    pub fn input_embedding(&self, idx: usize) -> ArrayView1<f32> {
        self.input.subview(Axis(0), idx)
    }

    /// Get the input embedding with the given index mutably.
    #[inline]
    pub fn input_embedding_mut(&mut self, idx: usize) -> ArrayViewMut1<f32> {
        self.input.subview_mut(Axis(0), idx)
    }

    /// Get the output embedding with the given index.
    #[inline]
    pub fn output_embedding(&self, idx: usize) -> ArrayView1<f32> {
        self.output.subview(Axis(0), idx)
    }

    /// Get the output embedding with the given index mutably.
    #[inline]
    pub fn output_embedding_mut(&mut self, idx: usize) -> ArrayViewMut1<f32> {
        self.output.subview_mut(Axis(0), idx)
    }

    /// Get the model's vocabulary.
    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }
}

impl<W> WriteModelBinary<W> for TrainModel
where
    W: Write,
{
    fn write_model_binary(&self, write: &mut W) -> Result<(), Error> {
        write.write_all(&[b'D', b'F', b'F'])?;
        write.write_u32::<LittleEndian>(1)?;
        write.write_u32::<LittleEndian>(self.config.context_size)?;
        write.write_u32::<LittleEndian>(self.config.dims)?;
        write.write_f32::<LittleEndian>(self.config.discard_threshold)?;
        write.write_u32::<LittleEndian>(self.config.epochs)?;
        write.write_u32::<LittleEndian>(self.config.min_count)?;
        write.write_u32::<LittleEndian>(self.config.min_n)?;
        write.write_u32::<LittleEndian>(self.config.max_n)?;
        write.write_u32::<LittleEndian>(self.config.buckets_exp)?;
        write.write_u32::<LittleEndian>(self.config.negative_samples)?;
        write.write_f32::<LittleEndian>(self.config.lr)?;
        write.write_u64::<LittleEndian>(self.vocab.n_tokens() as u64)?;
        write.write_u64::<LittleEndian>(self.vocab.len() as u64)?;

        for token in self.vocab.types() {
            write.write_u32::<LittleEndian>(token.token().len() as u32)?;
            write.write_all(token.token().as_bytes())?;
            write.write_u64::<LittleEndian>(token.count() as u64)?;
        }

        for &v in self.input.as_slice().unwrap() {
            write.write_f32::<LittleEndian>(v)?;
        }

        Ok(())
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
    };

    #[test]
    pub fn model_embed_methods() {
        let mut config = TEST_CONFIG.clone();
        config.min_count = 1;

        // We just need some bogus vocabulary
        let builder = VocabBuilder::new(TEST_CONFIG.clone());
        let vocab = builder.build();

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
