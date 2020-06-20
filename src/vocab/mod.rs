pub(crate) mod simple;
pub(crate) mod subword;

use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;

use serde::Serialize;
use superslice::Ext;

use crate::idx::WordIdx;
use std::cmp::Reverse;

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
    pub fn new(label: T, count: usize) -> Self {
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

/// Cutoff to determine vocabulary size.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "type", content = "value")]
pub enum Cutoff {
    /// Cutoff based on minimum frequency, items appearing less than
    /// `min_count` times are discarded.
    MinCount(usize),
    /// Cutoff based on a target size, up to `target_size` items are kept
    /// in the vocabulary. If the item at `target_size+1` appears `n` times,
    /// all items with frequency `n` and smaller are discarded.
    TargetSize(usize),
}

impl Cutoff {
    pub(crate) fn filter<T, S>(
        &self,
        items: impl IntoIterator<Item = (T, usize)>,
    ) -> Vec<CountedType<S>>
    where
        T: Hash + Eq + Into<S>,
        S: Hash + Eq + Clone + Ord,
    {
        match self {
            Cutoff::MinCount(min_count) => filter_minfreq(items, *min_count),
            Cutoff::TargetSize(target_size) => filter_targetsize(items, *target_size),
        }
    }
}

fn filter_minfreq<T, S>(
    items: impl IntoIterator<Item = (T, usize)>,
    min_count: usize,
) -> Vec<CountedType<S>>
where
    T: Hash + Eq + Into<S>,
    S: Hash + Eq + Clone + Ord,
{
    let mut types: Vec<_> = items
        .into_iter()
        .filter(|(_, count)| *count >= min_count as usize)
        .map(|(item, count)| CountedType::new(item.into(), count))
        .collect();
    types.sort_unstable_by(|w1, w2| w2.cmp(&w1));
    types
}

fn filter_targetsize<T, S>(
    items: impl IntoIterator<Item = (T, usize)>,
    target_size: usize,
) -> Vec<CountedType<S>>
where
    T: Hash + Eq + Into<S>,
    S: Hash + Eq + Clone + Ord,
{
    let mut items = items
        .into_iter()
        .map(|(item, count)| CountedType::new(item.into(), count))
        .collect::<Vec<_>>();
    items.sort_unstable_by(|i1, i2| i2.cmp(&i1));

    if target_size > items.len() {
        return items;
    }

    let cutoff_idx =
        items.lower_bound_by_key(&Reverse(items[target_size].count), |key| Reverse(key.count));
    items.truncate(cutoff_idx);
    items
}

#[cfg(test)]
mod test {
    use crate::{Cutoff, Word};

    #[test]
    pub fn target_size_unique_counts() {
        let cutoff = Cutoff::TargetSize(3);
        let items = vec![("a", 10), ("b", 3), ("c", 12), ("d", 5)];
        let filtered: Vec<Word> = cutoff.filter(items);
        let target_items = vec![
            Word::new("c".to_string(), 12),
            Word::new("a".to_string(), 10),
            Word::new("d".to_string(), 5),
        ];
        assert!(
            filtered == target_items,
            format!("{:#?}\n != \n {:#?}", filtered, target_items)
        );
    }

    #[test]
    pub fn target_size_discard_equal() {
        let cutoff = Cutoff::TargetSize(3);
        let items = vec![("a", 10), ("b", 3), ("c", 12), ("e", 12), ("d", 10)];
        let filtered: Vec<Word> = cutoff.filter(items);
        let target_items = vec![
            Word::new("e".to_string(), 12),
            Word::new("c".to_string(), 12),
        ];
        assert!(
            filtered == target_items,
            format!("{:#?}\n != \n {:#?}", filtered, target_items)
        );
    }

    #[test]
    pub fn target_size_0() {
        let cutoff = Cutoff::TargetSize(0);
        let items = vec![("a", 10), ("b", 3), ("c", 12), ("e", 12), ("d", 10)];
        let filtered: Vec<Word> = cutoff.filter(items);
        let target_items = vec![];
        assert!(
            filtered == target_items,
            format!("{:#?}\n != \n {:#?}", filtered, target_items)
        );
    }

    #[test]
    pub fn target_size_large() {
        let cutoff = Cutoff::TargetSize(10);
        let items = vec![("a", 10), ("b", 3), ("c", 12), ("e", 12), ("d", 10)];
        let filtered: Vec<Word> = cutoff.filter(items);
        let target_items = vec![
            Word::new("e".to_string(), 12),
            Word::new("c".to_string(), 12),
            Word::new("d".to_string(), 10),
            Word::new("a".to_string(), 10),
            Word::new("b".to_string(), 3),
        ];
        assert!(
            filtered == target_items,
            format!("{:#?}\n != \n {:#?}", filtered, target_items)
        );
    }

    #[test]
    pub fn target_size_all_equal_too_many() {
        let cutoff = Cutoff::TargetSize(3);
        let items = vec![("a", 10), ("b", 10), ("c", 10), ("e", 10), ("d", 10)];
        let filtered: Vec<Word> = cutoff.filter(items);
        let target_items = vec![];
        assert!(
            filtered == target_items,
            format!("{:#?}\n != \n {:#?}", filtered, target_items)
        );
    }
}
