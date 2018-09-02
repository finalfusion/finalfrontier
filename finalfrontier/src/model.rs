use std::io::{Read, Write};
use std::sync::Arc;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use failure::{err_msg, Error};
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Range;

use hogwild::HogwildArray2;
use vec_simd::{scale, scaled_add};
use {
    Config, LossType, ModelType, ReadModelBinary, Vocab, WordCount, WriteModelBinary,
    WriteModelText,
};

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
    pub(crate) fn mean_input_embedding(&self, indices: &[u64]) -> Array1<f32> {
        let mut embed = Array1::zeros((self.config.dims as usize,));

        for &idx in indices.iter() {
            scaled_add(embed.view_mut(), self.input_embedding(idx as usize), 1.0);
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
        write.write_u8(self.config.model as u8)?;
        write.write_u8(self.config.loss as u8)?;
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

        for word in self.vocab.words() {
            write.write_u32::<LittleEndian>(word.word().len() as u32)?;
            write.write_all(word.word().as_bytes())?;
            write.write_u64::<LittleEndian>(word.count() as u64)?;
        }

        for &v in self.input.as_slice().unwrap() {
            write.write_f32::<LittleEndian>(v)?;
        }

        Ok(())
    }
}

/// Word embedding model.
///
/// This data type is used for models post-training. It stores the vocabulary
/// and embedding matrix. The model can be used to retrieve word embeddings.
pub struct Model {
    config: Config,
    vocab: Vocab,
    embed_matrix: Array2<f32>,
}

impl Model {
    /// Get the model configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Get the embedding for the given word.
    ///
    /// This method will return `None` iff the word is unknown and no n-grams
    /// could be extracted. Otherwise, this method will always return an
    /// embedding.
    pub fn embedding(&self, word: &str) -> Option<Array1<f32>> {
        let mut embed = Array1::zeros((self.config.dims as usize,));

        let indices = self.vocab.indices(word);
        if indices.is_empty() {
            return None;
        }

        for &idx in indices.iter() {
            scaled_add(
                embed.view_mut(),
                self.embed_matrix.subview(Axis(0), idx as usize),
                1.0,
            );
        }

        scale(embed.view_mut(), 1.0 / indices.len() as f32);

        Some(embed)
    }

    /// Get the vocabulary.
    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }
}

impl<R> ReadModelBinary<R> for Model
where
    R: Read,
{
    fn read_model_binary(read: &mut R) -> Result<Self, Error> {
        let mut header = [0u8; 3];
        read.read_exact(&mut header)?;
        if header != [b'D', b'F', b'F'] {
            return Err(err_msg("Incorrect file format"));
        }

        let version = read.read_u32::<LittleEndian>()?;
        if version != 1 {
            return Err(err_msg("Unknown file version"));
        }

        let model = ModelType::try_from(read.read_u8()?)?;
        let loss = LossType::try_from(read.read_u8()?)?;
        let context_size = read.read_u32::<LittleEndian>()?;
        let dims = read.read_u32::<LittleEndian>()?;
        let discard_threshold = read.read_f32::<LittleEndian>()?;
        let epochs = read.read_u32::<LittleEndian>()?;
        let min_count = read.read_u32::<LittleEndian>()?;
        let min_n = read.read_u32::<LittleEndian>()?;
        let max_n = read.read_u32::<LittleEndian>()?;
        let buckets_exp = read.read_u32::<LittleEndian>()?;
        let negative_samples = read.read_u32::<LittleEndian>()?;
        let lr = read.read_f32::<LittleEndian>()?;

        let config = Config {
            context_size,
            dims,
            discard_threshold,
            epochs,
            loss,
            model,
            min_count,
            min_n,
            max_n,
            buckets_exp,
            negative_samples,
            lr,
        };

        let n_tokens = read.read_u64::<LittleEndian>()?;
        let vocab_len = read.read_u64::<LittleEndian>()?;
        let mut words = Vec::with_capacity(vocab_len as usize);
        for _ in 0..vocab_len {
            let word_len = read.read_u32::<LittleEndian>()?;
            let mut bytes = vec![0; word_len as usize];
            read.read_exact(&mut bytes)?;
            let word = String::from_utf8(bytes)?;
            let count = read.read_u64::<LittleEndian>()? as usize;

            words.push(WordCount::new(word, count));
        }

        let vocab = Vocab::new(config.clone(), words, n_tokens as usize);

        let n_embeds = vocab_len as usize + 2usize.pow(config.buckets_exp);
        let mut data = vec![0f32; n_embeds * config.dims as usize];
        read.read_f32_into::<LittleEndian>(&mut data)?;

        Ok(Model {
            config: config.clone(),
            vocab,
            embed_matrix: Array2::from_shape_vec((n_embeds, config.dims as usize), data)?,
        })
    }
}

impl<W> WriteModelText<W> for Model
where
    W: Write,
{
    fn write_model_text(&self, write: &mut W, write_dims: bool) -> Result<(), Error> {
        if write_dims {
            writeln!(
                write,
                "{} {}",
                self.vocab.words().len(),
                self.embed_matrix.shape()[1]
            )?;
        }

        for word in self.vocab.words() {
            let embed = self
                .embedding(word.word())
                .expect("Word without an embedding");
            let embed_str = embed
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<String>>()
                .join(" ");
            writeln!(write, "{} {}", word.word(), embed_str)?;
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
