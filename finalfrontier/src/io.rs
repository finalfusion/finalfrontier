use std::io::{self, BufRead, Lines, Write};

use failure::Error;

use crate::util::EOS;

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
    type Item = Result<Vec<String>, io::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        for line in &mut self.lines {
            let line = match line {
                Ok(ref line) => line.trim(),
                Err(err) => return Some(Err(err)),
            };

            // Skip empty lines.
            if !line.is_empty() {
                return Some(Ok(whitespace_tokenize(line)));
            }
        }

        None
    }
}

/// Trait for writing models in binary format.
pub trait WriteModelBinary<W>
where
    W: Write,
{
    fn write_model_binary(self, write: &mut W) -> Result<(), Error>;
}

fn whitespace_tokenize(line: &str) -> Vec<String> {
    let mut tokens = line
        .split_whitespace()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    tokens.push(EOS.to_string());
    tokens
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
    fn write_model_text(&self, write: &mut W, write_dims: bool) -> Result<(), Error>;
}

/// Trait for writing models in binary format.
pub trait WriteModelWord2Vec<W>
where
    W: Write,
{
    fn write_model_word2vec(&self, write: &mut W) -> Result<(), Error>;
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::SentenceIterator;
    use crate::util::EOS;

    #[test]
    fn sentence_iterator_test() {
        let v = b"This is a sentence .\nAnd another one .\n".to_vec();
        let c = Cursor::new(v);
        let mut iter = SentenceIterator::new(c);
        assert_eq!(
            iter.next().unwrap().unwrap(),
            vec!["This", "is", "a", "sentence", ".", EOS]
        );
        assert_eq!(
            iter.next().unwrap().unwrap(),
            vec!["And", "another", "one", ".", EOS]
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
            vec!["This", "is", "a", "sentence", ".", EOS]
        );
        assert_eq!(
            iter.next().unwrap().unwrap(),
            vec!["And", "another", "one", ".", EOS]
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

}
