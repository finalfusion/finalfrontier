// pub (crate) mod ngram;
pub(crate) mod simple;
pub(crate) mod subword;

use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;

use crate::idx::WordIdx;

const BOW: char = '<';
const EOW: char = '>';

pub type Word = CountedType<String>;
#[derive(Clone, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct CountedType<T> {
    count: usize,
    label: T,
}

impl<T> CountedType<T> {
    /// Construct a new type.
    pub(crate) fn new(label: T, count: usize) -> Self {
        CountedType { label, count }
    }
    pub fn count(&self) -> usize {
        self.count
    }
    pub fn label(&self) -> &T {
        &self.label
    }
}

impl CountedType<String> {
    /// The string representation of the word.
    pub fn word(&self) -> &str {
        &self.label
    }
}

/// Trait for lookup of indices.
pub trait Vocab {
    type VocabType: Hash + Eq;
    type IdxType: WordIdx;
    type Config;

    /// Return this vocabulary's config.
    fn config(&self) -> Self::Config;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the number of entries in the vocabulary.
    fn len(&self) -> usize {
        self.types().len()
    }

    /// Get the index of the entry, will return None if the item is not present.
    fn idx<Q>(&self, key: &Q) -> Option<Self::IdxType>
    where
        Self::VocabType: Borrow<Q>,
        Q: Hash + ?Sized + Eq;

    /// Get the discard probability of the entry with the given index.
    fn discard(&self, idx: usize) -> f32;

    /// Get the number of possible input types.
    fn n_input_types(&self) -> usize;

    /// Get all types in the vocabulary.
    fn types(&self) -> &[CountedType<Self::VocabType>];

    /// Get the number of types in the corpus.
    ///
    /// This returns the number of types in the corpus that the vocabulary
    /// was constructed from, **before** removing types that are below the
    /// minimum count.
    fn n_types(&self) -> usize;
}

/// Generic builder struct to count types.
///
/// Items are added to the vocabulary and counted using the `count` method.
/// There is no explicit build method, conversion is done via implementing
/// `From<VocabBuilder<T>>`.
pub struct VocabBuilder<C, T> {
    config: C,
    items: HashMap<T, usize>,
    n_items: usize,
}

impl<C, T> VocabBuilder<C, T>
where
    T: Hash + Eq,
{
    pub fn new(config: C) -> Self {
        VocabBuilder {
            config,
            items: HashMap::new(),
            n_items: 0,
        }
    }

    pub fn count<S>(&mut self, item: S)
    where
        S: Into<T>,
    {
        self.n_items += 1;
        let cnt = self.items.entry(item.into()).or_insert(0);
        *cnt += 1;
    }
}

/// Create discard probabilities based on threshold, specific counts and total counts.
pub(crate) fn create_discards<S>(
    discard_threshold: f32,
    types: &[CountedType<S>],
    n_tokens: usize,
) -> Vec<f32> {
    let mut discards = Vec::with_capacity(types.len());

    for item in types {
        let p = item.count() as f32 / n_tokens as f32;
        let p_discard = discard_threshold / p + (discard_threshold / p).sqrt();

        // Not a proper probability, upper bound at 1.0.
        discards.push(1f32.min(p_discard));
    }

    discards
}

/// Create lookup.
pub(crate) fn create_indices<S>(types: &[CountedType<S>]) -> HashMap<S, usize>
where
    S: Hash + Eq + Clone,
{
    let mut token_indices = HashMap::new();

    for (idx, item) in types.iter().enumerate() {
        token_indices.insert(item.label.clone(), idx);
    }

    // Invariant: The index size should be the same as the number of
    // types.
    assert_eq!(types.len(), token_indices.len());

    token_indices
}

/// Add begin/end-of-word brackets.
pub(crate) fn bracket(word: &str) -> String {
    let mut bracketed = String::new();
    bracketed.push(BOW);
    bracketed.push_str(word);
    bracketed.push(EOW);

    bracketed
}
