use std::io::{self, BufRead, Lines};

use util::EOS;

/// Sentence iterator.
///
/// This this iterator consumes a reader with tokenized sentences:
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
        let mut line;
        loop {
            line = match self.lines.next()? {
                Ok(line) => line,
                Err(err) => return Some(Err(err)),
            };

            // Trim leading and trailing whitespace.
            line = line.trim().to_owned();

            // Skip empty lines.
            if !line.is_empty() {
                break;
            }
        }

        let mut tokens = line.split_whitespace()
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        tokens.push(EOS.to_string());

        Some(Ok(tokens))
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::SentenceIterator;
    use util::EOS;

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
