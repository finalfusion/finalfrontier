use std::convert::TryFrom;
use std::fs::File;
use std::io::{self, BufRead, Lines, Read, Seek, SeekFrom, Write};

use anyhow::{anyhow, Context, Error, Result};
use chrono::{DateTime, Local};
use indicatif::{ProgressBar, ProgressStyle};
use memmap::{Mmap, MmapOptions};
use serde::Serialize;

pub struct FileProgress {
    inner: File,
    progress: ProgressBar,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingFormat {
    FastText,
    FinalFusion,
    Word2Vec,
    Text,
    TextDims,
}

impl TryFrom<&str> for EmbeddingFormat {
    type Error = Error;

    fn try_from(format: &str) -> Result<Self> {
        use self::EmbeddingFormat::*;

        match format {
            "fasttext" => Ok(FastText),
            "finalfusion" => Ok(FinalFusion),
            "word2vec" => Ok(Word2Vec),
            "text" => Ok(Text),
            "textdims" => Ok(TextDims),
            unknown => Err(anyhow!("Unknown embedding format: {}", unknown)),
        }
    }
}

/// A progress bar that implements the `Read` trait.
///
/// This wrapper of `indicatif`'s `ProgressBar` updates progress based on the
/// current offset within the file.
impl FileProgress {
    pub fn new(file: File) -> io::Result<Self> {
        let metadata = file.metadata()?;
        let progress = ProgressBar::new(metadata.len());
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{bar:30} {bytes}/{total_bytes} ETA: {eta_precise}"),
        );

        Ok(FileProgress {
            inner: file,
            progress,
        })
    }
}

impl Read for FileProgress {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n_read = self.inner.read(buf)?;
        let pos = self.inner.seek(SeekFrom::Current(0))?;
        self.progress.set_position(pos);
        Ok(n_read)
    }
}

impl Drop for FileProgress {
    fn drop(&mut self) {
        self.progress.finish();
    }
}

/// Sentence iterator.
///
/// This iterator consumes a reader with tokenized sentences:
///
/// - One sentence per line.
/// - Tokens separated by a space.
///
/// It produces `Vec`s with the tokens, adding an end-of-sentence marker
/// to the end of the sentence. Lines that are empty or only consist of
/// whitespace are discarded.
pub struct SentenceIterator<R> {
    lines: Lines<R>,
}

impl<R> SentenceIterator<R>
where
    R: BufRead,
{
    pub fn new(read: R) -> Self {
        SentenceIterator {
            lines: read.lines(),
        }
    }
}

impl<R> Iterator for SentenceIterator<R>
where
    R: BufRead,
{
    type Item = Result<Vec<String>>;

    fn next(&mut self) -> Option<Self::Item> {
        for line in &mut self.lines {
            let line = match line {
                Ok(ref line) => line.trim(),
                Err(err) => return Some(Err(err.into())),
            };

            // Skip empty lines.
            if !line.is_empty() {
                return Some(Ok(whitespace_tokenize(line)));
            }
        }

        None
    }
}

/// Get thread-specific data.
///
/// This function will return a memory map of the corpus data. The initial
/// starting position for the given thread is also returned. This starting
/// Position will always be the beginning of a sentence.
pub fn thread_data_text(f: &File, thread: usize, n_threads: usize) -> Result<(Mmap, usize)> {
    assert!(
        thread < n_threads,
        "Thread {} out of index [0, {})",
        thread,
        n_threads
    );

    let size = f.metadata().context("Cannot get file metadata")?.len();
    let chunk_size = size as usize / n_threads;

    let mmap = unsafe { MmapOptions::new().map(&f)? };

    if thread == 0 {
        return Ok((mmap, 0));
    }

    let mut start = thread * chunk_size;
    while start < mmap.len() {
        let next = mmap[start];
        start += 1;
        if next == b'\n' {
            break;
        }
    }

    Ok((mmap, start))
}

/// Get thread-specific data for a CoNLL-U corpus.
///
/// This function will return a memory map of the corpus data. The initial
/// starting position for the given thread is also returned. This starting
/// Position will always be the beginning of a sentence.
pub fn thread_data_conllu(f: &File, thread: usize, n_threads: usize) -> Result<(Mmap, usize)> {
    assert!(
        thread < n_threads,
        "Thread {} out of index [0, {})",
        thread,
        n_threads
    );

    let size = f.metadata().context("Cannot get file metadata")?.len();
    let chunk_size = size as usize / n_threads;

    let mmap = unsafe { MmapOptions::new().map(&f)? };

    if thread == 0 {
        return Ok((mmap, 0));
    }

    let mut start = thread * chunk_size;
    while start < mmap.len() - 1 {
        let next = mmap[start];
        start += 1;
        if next == b'\n' && mmap[start] == b'\n' {
            start += 1;
            break;
        }
    }

    Ok((mmap, start))
}

/// Meta information about training.
#[derive(Clone, Serialize)]
pub struct TrainInfo {
    corpus: String,
    output: String,
    n_threads: usize,
    start_datetime: String,
    end_datetime: Option<String>,
}

impl TrainInfo {
    /// Construct new TrainInfo.
    ///
    /// Constructs TrainInfo with `start_datetime` set to the current datetime. `end_datetime` is
    /// set to `None` and can be set through `TrainInfo::set_end`.
    pub fn new(corpus: String, output: String, n_threads: usize) -> Self {
        let start_datetime: DateTime<Local> = Local::now();
        TrainInfo {
            corpus,
            output,
            n_threads,
            start_datetime: start_datetime.format("%Y-%m-%d %H:%M:%S").to_string(),
            end_datetime: None,
        }
    }

    /// Get the corpus path.
    pub fn corpus(&self) -> &str {
        &self.corpus
    }

    /// Get the output file.
    pub fn output(&self) -> &str {
        &self.output
    }

    /// Get the number of threads.
    pub fn n_threads(&self) -> usize {
        self.n_threads
    }

    /// Get the start datetime.
    pub fn start_datetime(&self) -> &str {
        &self.start_datetime
    }

    /// Get the end datetime.
    pub fn end_datetime(&self) -> Option<&str> {
        self.end_datetime.as_deref()
    }

    /// Set the end datetime to current datetime.
    pub fn set_end(&mut self) {
        let start_datetime: DateTime<Local> = Local::now();
        self.end_datetime = Some(start_datetime.format("%Y-%m-%d %H:%M:%S").to_string());
    }
}

/// Trait for writing models in binary format.
pub trait WriteModelBinary<W>
where
    W: Write,
{
    fn write_model_binary(
        self,
        write: &mut W,
        train_info: TrainInfo,
        format: EmbeddingFormat,
    ) -> Result<()>;
}

fn whitespace_tokenize(line: &str) -> Vec<String> {
    line.split_whitespace()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>()
}

/// Trait for writing models in text format.
pub trait WriteModelText<W>
where
    W: Write,
{
    /// Write the model in text format.
    ///
    /// This function only writes the word embeddings. The subword
    /// embeddings are discarded.
    ///
    /// The `write_dims` parameter indicates whether the first line
    /// should contain the dimensionality of the embedding matrix.
    fn write_model_text(&self, write: &mut W, write_dims: bool) -> Result<()>;
}

/// Trait for writing models in binary format.
pub trait WriteModelWord2Vec<W>
where
    W: Write,
{
    fn write_model_word2vec(&self, write: &mut W) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Cursor;

    use super::SentenceIterator;
    use super::{thread_data_conllu, thread_data_text};

    #[test]
    fn sentence_iterator_test() {
        let v = b"This is a sentence .\nAnd another one .\n".to_vec();
        let c = Cursor::new(v);
        let mut iter = SentenceIterator::new(c);
        assert_eq!(
            iter.next().unwrap().unwrap(),
            vec!["This", "is", "a", "sentence", "."]
        );
        assert_eq!(
            iter.next().unwrap().unwrap(),
            vec!["And", "another", "one", "."]
        );
        assert!(iter.next().is_none());
    }

    #[test]
    fn sentence_iterator_no_newline_test() {
        let v = b"This is a sentence .\nAnd another one .".to_vec();
        let c = Cursor::new(v);
        let mut iter = SentenceIterator::new(c);
        assert_eq!(
            iter.next().unwrap().unwrap(),
            vec!["This", "is", "a", "sentence", "."]
        );
        assert_eq!(
            iter.next().unwrap().unwrap(),
            vec!["And", "another", "one", "."]
        );
        assert!(iter.next().is_none());
    }

    #[test]
    fn sentence_iterator_empty_test() {
        let v = b"".to_vec();
        let c = Cursor::new(v);
        let mut iter = SentenceIterator::new(c);
        assert!(iter.next().is_none());
    }

    #[test]
    fn sentence_iterator_empty_newline_test() {
        let v = b"\n \n   \n".to_vec();
        let c = Cursor::new(v);
        let mut iter = SentenceIterator::new(c);
        assert!(iter.next().is_none());
    }

    static CHUNKING_TEST_DATA: &str =
        "a b c\nd e f\ng h i\nj k l\nm n o\np q r\ns t u\nv w x\ny z\n";

    static CHUNKING_TEST_DATA_DEPS: &str =
        "a b c\nd e f\n\ng h i\nj k l\n\nm n o\np q r\n\ns t u\nv w x\ny z\n";

    #[test]
    fn thread_data_test() {
        let f = File::open("testdata/chunking.txt").unwrap();

        let (mmap, start) = thread_data_text(&f, 0, 3).unwrap();
        assert_eq!(
            &*mmap,
            CHUNKING_TEST_DATA.as_bytes(),
            "Memory mapping is incorrect"
        );
        assert_eq!(start, 0, "Incorrect start index");

        let (mmap, start) = thread_data_text(&f, 1, 3).unwrap();
        assert_eq!(
            &*mmap,
            CHUNKING_TEST_DATA.as_bytes(),
            "Memory mapping is incorrect"
        );
        assert_eq!(start, 18, "Incorrect start index");

        let (mmap, start) = thread_data_text(&f, 2, 3).unwrap();
        assert_eq!(
            &*mmap,
            CHUNKING_TEST_DATA.as_bytes(),
            "Memory mapping is incorrect"
        );
        assert_eq!(start, 36, "Incorrect start index");
    }

    #[test]
    fn deps_thread_data_test() {
        // file size is 55 bytes
        // starts scanning at index 19
        // first double linebreak is at 26
        // second at 39
        let f = File::open("testdata/dep_chunking.txt").unwrap();
        let (mmap, start) = thread_data_conllu(&f, 0, 3).unwrap();
        assert_eq!(
            &*mmap,
            CHUNKING_TEST_DATA_DEPS.as_bytes(),
            "Memory mapping is incorrect"
        );
        assert_eq!(start, 0, "Incorrect start index");

        let (mmap, start) = thread_data_conllu(&f, 1, 3).unwrap();
        assert_eq!(
            &*mmap,
            CHUNKING_TEST_DATA_DEPS.as_bytes(),
            "Memory mapping is incorrect"
        );
        assert_eq!(start, 26, "Incorrect start index");

        let (mmap, start) = thread_data_conllu(&f, 2, 3).unwrap();
        assert_eq!(
            &*mmap,
            CHUNKING_TEST_DATA_DEPS.as_bytes(),
            "Memory mapping is incorrect"
        );
        assert_eq!(start, 39, "Incorrect start index");
    }

    #[should_panic]
    #[test]
    fn thread_data_out_of_bounds_test() {
        let f = File::open("testdata/chunking.txt").unwrap();
        let _ = thread_data_conllu(&f, 3, 3).unwrap();
    }
}
