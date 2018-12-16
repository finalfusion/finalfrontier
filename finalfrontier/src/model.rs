//! Embedding prediction model.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::iter::Enumerate;
use std::mem;
use std::slice;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use failure::{err_msg, Error};
use memmap::{Mmap, MmapOptions};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Dimension, Ix2};

use io::MODEL_VERSION;
use vec_simd::{l2_normalize, scale, scaled_add};
use {
    Config, LossType, MmapModelBinary, ModelType, ReadModelBinary, Vocab, WordCount,
    WriteModelText, WriteModelWord2Vec,
};

/// Embedding matrix
///
/// This enum wraps several embedding matrix representations.
enum EmbeddingMatrix {
    /// In-memory `ndarray` matrix.
    NDArray(Array2<f32>),

    /// Memory-mapped matrix.
    Mmap { map: Mmap, shape: Ix2 },
}

impl EmbeddingMatrix {
    /// Get embedding matrix view.
    fn view(&self) -> ArrayView2<f32> {
        match self {
            EmbeddingMatrix::NDArray(a) => a.view(),
            EmbeddingMatrix::Mmap { map, shape } => unsafe {
                ArrayView2::from_shape_ptr(*shape, map.as_ptr() as *const f32)
            },
        }
    }
}

/// Word embedding model.
///
/// This data type is used for models post-training. It stores the vocabulary
/// and embedding matrix. The model can be used to retrieve word embeddings.
pub struct Model {
    config: Config,
    vocab: Vocab,
    embed_matrix: EmbeddingMatrix,
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
        // For known words, return the precomputed embedding.
        if let Some(index) = self.vocab.word_idx(word) {
            return Some(
                self.embed_matrix
                    .view()
                    .index_axis(Axis(0), index as usize)
                    .to_owned(),
            );
        }

        // For unknown words, gather subword indices and compute the embedding.
        let indices = self.vocab.subword_indices(word);
        if indices.is_empty() {
            return None;
        }

        let mut embed = Array1::zeros((self.config.dims as usize,));
        for &idx in indices.iter() {
            scaled_add(
                embed.view_mut(),
                self.embed_matrix.view().index_axis(Axis(0), idx as usize),
                1.0,
            );
        }

        // Return the average embedding.
        scale(embed.view_mut(), 1.0 / indices.len() as f32);

        // Normalize predicted vector by its l2-norm.
        l2_normalize(embed.view_mut());

        Some(embed)
    }

    pub fn embedding_matrix(&self) -> ArrayView2<f32> {
        self.embed_matrix.view()
    }

    /// Get an iterator over known words and their embeddings.
    pub fn iter(&self) -> Iter {
        Iter {
            view: self.embed_matrix.view(),
            inner: self.vocab.words().iter().enumerate(),
        }
    }

    /// Get the vocabulary.
    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }
}

impl<'a> IntoIterator for &'a Model {
    type Item = (&'a str, ArrayView1<'a, f32>);
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl MmapModelBinary for Model {
    fn mmap_model_binary(f: File) -> Result<Self, Error> {
        let mut read = BufReader::new(&f);

        read_model_binary_header(&mut read)?;
        let config = read_model_binary_config(&mut read)?;
        let vocab = read_model_binary_vocab(&config, &mut read)?;

        let n_embeds = vocab.len() as usize + 2usize.pow(config.buckets_exp);
        let shape = Ix2(n_embeds, config.dims as usize);

        // Set up memory mapping.
        let offset = read.seek(SeekFrom::Current(0))?;
        let mut mmap_opts = MmapOptions::new();
        let map = unsafe {
            mmap_opts
                .offset(offset)
                .len(shape.size() * mem::size_of::<f32>())
                .map(&f)?
        };

        Ok(Model {
            config: config.clone(),
            vocab,
            embed_matrix: EmbeddingMatrix::Mmap { map, shape },
        })
    }
}

impl<R> ReadModelBinary<R> for Model
where
    R: Read,
{
    fn read_model_binary(read: &mut R) -> Result<Self, Error> {
        read_model_binary_header(read)?;
        let config = read_model_binary_config(read)?;
        let vocab = read_model_binary_vocab(&config, read)?;

        let n_embeds = vocab.len() as usize + 2usize.pow(config.buckets_exp);
        let mut data = vec![0f32; n_embeds * config.dims as usize];
        read.read_f32_into::<LittleEndian>(&mut data)?;
        let embed_matrix = Array2::from_shape_vec((n_embeds, config.dims as usize), data)?;

        Ok(Model {
            config: config.clone(),
            vocab,
            embed_matrix: EmbeddingMatrix::NDArray(embed_matrix),
        })
    }
}

fn read_model_binary_header<R>(read: &mut R) -> Result<(), Error>
where
    R: Read,
{
    let mut header = [0u8; 3];
    read.read_exact(&mut header)?;
    if header != [b'D', b'F', b'F'] {
        return Err(err_msg("Incorrect file format"));
    }

    let version = read.read_u32::<LittleEndian>()?;
    if version != MODEL_VERSION {
        return Err(err_msg("Unknown file version"));
    }

    Ok(())
}

fn read_model_binary_config<R>(read: &mut R) -> Result<Config, Error>
where
    R: Read,
{
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

    Ok(Config {
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
    })
}

fn read_model_binary_vocab<R>(config: &Config, read: &mut R) -> Result<Vocab, Error>
where
    R: Read,
{
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

    Ok(Vocab::new(config.clone(), words, n_tokens as usize))
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
                self.embed_matrix.view().shape()[1]
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

impl<W> WriteModelWord2Vec<W> for Model
where
    W: Write,
{
    fn write_model_word2vec(&self, write: &mut W) -> Result<(), Error> {
        write!(
            write,
            "{} {}\n",
            self.vocab.len(),
            self.embed_matrix.view().shape()[1]
        )?;

        for word in self.vocab.words() {
            write!(write, "{} ", word.word())?;

            let embed = self
                .embedding(word.word())
                .expect("Word without an embedding");

            // Write embedding to a vector with little-endian encoding.
            for v in embed.iter() {
                write.write_f32::<LittleEndian>(*v)?;
            }

            write.write(&[0x0a])?;
        }

        Ok(())
    }
}

/// Iterator over known words and their embeddings.
///
/// This iterator is created by the [`Model::iter`] method.
pub struct Iter<'a> {
    view: ArrayView2<'a, f32>,

    // Note, we cannot use AxisIter, because Model uses ephemeral
    // arrays for memory-mapped embedding matrices. So, we use
    // indexing instead.
    inner: Enumerate<slice::Iter<'a, WordCount>>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = (&'a str, ArrayView1<'a, f32>);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(idx, word_count)| (word_count.word(), self.view.index_axis_move(Axis(0), idx)))
    }
}

#[cfg(test)]
mod tests {
    use super::{EmbeddingMatrix, Model};

    use ndarray::{arr2, Axis};

    use {Config, LossType, ModelType, VocabBuilder};

    pub const TEST_CONFIG: Config = Config {
        buckets_exp: 21,
        context_size: 5,
        dims: 3,
        discard_threshold: 1e-4,
        epochs: 5,
        loss: LossType::LogisticNegativeSampling,
        lr: 0.05,
        min_count: 1,
        max_n: 6,
        min_n: 3,
        model: ModelType::SkipGram,
        negative_samples: 5,
    };

    #[test]
    pub fn test_iter() {
        let mut builder = VocabBuilder::new(TEST_CONFIG.clone());
        builder.count("test");
        builder.count("test");
        builder.count("test");
        builder.count("this");
        builder.count("this");
        builder.count("!");
        let vocab = builder.build();

        let test_matrix = arr2(&[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);

        let model = Model {
            config: TEST_CONFIG.clone(),
            vocab,
            embed_matrix: EmbeddingMatrix::NDArray(test_matrix.clone()),
        };

        let mut iter = model.iter();
        assert_eq!(
            iter.next(),
            Some(("test", test_matrix.index_axis(Axis(0), 0)))
        );
        assert_eq!(
            iter.next(),
            Some(("this", test_matrix.index_axis(Axis(0), 1)))
        );
        assert_eq!(iter.next(), Some(("!", test_matrix.index_axis(Axis(0), 2))));
        assert_eq!(iter.next(), None);
    }
}
