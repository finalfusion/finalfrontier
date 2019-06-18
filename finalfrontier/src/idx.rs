use std::iter::FusedIterator;
use std::{option, slice};

/// A single lookup index.
#[derive(Copy, Clone)]
pub struct SingleIdx {
    word_idx: u64,
}

/// A lookup index with associated subword indices.
#[derive(Clone)]
pub struct WordWithSubwordsIdx {
    word_idx: u64,
    subwords: Vec<u64>,
}

impl WordWithSubwordsIdx {
    pub(crate) fn new(word_idx: u64, subwords: impl Into<Vec<u64>>) -> Self {
        WordWithSubwordsIdx {
            word_idx,
            subwords: subwords.into(),
        }
    }
}

/// Vocabulary indexing trait.
///
/// This trait defines methods shared by indexing types.
pub trait WordIdx: Clone {
    /// Return the unique word index for the WordIdx.
    fn word_idx(&self) -> u64;

    /// Build a new WordIdx containing only a single index.
    fn from_word_idx(word_idx: u64) -> Self;

    /// Return the number of indices.
    fn len(&self) -> usize;
}

impl WordIdx for SingleIdx {
    fn word_idx(&self) -> u64 {
        self.word_idx
    }

    fn from_word_idx(word_idx: u64) -> Self {
        SingleIdx { word_idx }
    }

    fn len(&self) -> usize {
        1
    }
}

impl<'a> IntoIterator for &'a SingleIdx {
    type Item = u64;
    type IntoIter = option::IntoIter<u64>;

    fn into_iter(self) -> Self::IntoIter {
        Some(self.word_idx).into_iter()
    }
}

impl WordIdx for WordWithSubwordsIdx {
    fn word_idx(&self) -> u64 {
        self.word_idx
    }

    fn from_word_idx(word_idx: u64) -> Self {
        WordWithSubwordsIdx {
            word_idx,
            subwords: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        1 + self.subwords.len()
    }
}

impl<'a> IntoIterator for &'a WordWithSubwordsIdx {
    type Item = u64;
    type IntoIter = IdxIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        IdxIter {
            word_idx: Some(self.word_idx),
            subwords: self.subwords.iter(),
        }
    }
}

/// Iterator over Indices.
pub struct IdxIter<'a> {
    word_idx: Option<u64>,
    subwords: slice::Iter<'a, u64>,
}

impl<'a> Iterator for IdxIter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(idx) = self.subwords.next() {
            Some(*idx)
        } else {
            self.word_idx.take()
        }
    }
}

impl<'a> FusedIterator for IdxIter<'a> {}

#[cfg(test)]
mod test {
    use crate::idx::{SingleIdx, WordIdx, WordWithSubwordsIdx};

    #[test]
    fn test_idx_iter() {
        let with_subwords = WordWithSubwordsIdx::new(0, vec![24, 4, 42]);
        let mut idx_iter = (&with_subwords).into_iter();
        assert_eq!(24, idx_iter.next().unwrap());
        assert_eq!(4, idx_iter.next().unwrap());
        assert_eq!(42, idx_iter.next().unwrap());
        assert_eq!(0, idx_iter.next().unwrap());
        assert_eq!(0, with_subwords.word_idx());

        let single = SingleIdx::from_word_idx(0);
        let mut idx_iter = (&single).into_iter();
        assert_eq!(0, idx_iter.next().unwrap());
        assert_eq!(0, single.word_idx());
    }
}
