use std::borrow::{Borrow, Cow};
use std::collections::HashMap;
use std::hash::Hash;

use finalfusion::vocab::{
    SimpleVocab as FiFuSimpleVocab, SubwordVocab as FiFuSubwordVocab, VocabWrap,
};

use crate::subword::SubwordIndices;
use crate::{util, SimpleVocabConfig, SubwordVocabConfig};

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

/// A corpus vocabulary with subword lookup.
#[derive(Clone)]
pub struct SubwordVocab {
    config: SubwordVocabConfig,
    words: Vec<Word>,
    subwords: Vec<Vec<u64>>,
    discards: Vec<f32>,
    index: HashMap<String, usize>,
    n_tokens: usize,
}

impl SubwordVocab {
    /// Construct a new vocabulary.
    ///
    /// Normally a `VocabBuilder` should be used. This constructor is used
    /// for deserialization.
    pub(crate) fn new(config: SubwordVocabConfig, words: Vec<Word>, n_tokens: usize) -> Self {
        let index = create_indices(&words);
        let subwords = Self::create_subword_indices(&config, &words);
        let discards = create_discards(config.discard_threshold, &words, n_tokens);
        SubwordVocab {
            config,
            discards,
            words,
            subwords,
            index,
            n_tokens,
        }
    }

    fn create_subword_indices(config: &SubwordVocabConfig, words: &[Word]) -> Vec<Vec<u64>> {
        let mut subword_indices = Vec::new();

        for word in words {
            if word.word() == util::EOS {
                subword_indices.push(Vec::new());
                continue;
            }

            subword_indices.push(
                bracket(word.word())
                    .as_str()
                    .subword_indices(
                        config.min_n as usize,
                        config.max_n as usize,
                        config.buckets_exp as usize,
                    )
                    .into_iter()
                    .map(|idx| idx + words.len() as u64)
                    .collect(),
            );
        }

        assert_eq!(words.len(), subword_indices.len());

        subword_indices
    }

    /// Get the given word.
    pub fn word(&self, word: &str) -> Option<&Word> {
        self.idx(word).map(|idx| &self.words[idx.single_idx()])
    }

    /// Get the subword indices of a word.
    pub fn subword_indices(&self, word: &str) -> Cow<[u64]> {
        if word == util::EOS {
            // Do not create subwords for the EOS marker.
            Cow::Borrowed(&[])
        } else if let Some(&idx) = self.index.get(word) {
            Cow::Borrowed(&self.subwords[idx])
        } else {
            Cow::Owned(
                bracket(word)
                    .as_str()
                    .subword_indices(
                        self.config.min_n as usize,
                        self.config.max_n as usize,
                        self.config.buckets_exp as usize,
                    )
                    .into_iter()
                    .map(|idx| idx + self.words.len() as u64)
                    .collect(),
            )
        }
    }

    pub(crate) fn subword_indices_idx(&self, idx: usize) -> Option<&[u64]> {
        self.subwords.get(idx).map(|v| v.as_slice())
    }

    /// Get all indices of a word, both regular and subword.
    ///
    /// This method copies the subword list for known words into a new Vec.
    pub fn indices(&self, word: &str) -> Vec<u64> {
        let mut indices = self.subword_indices(word).into_owned();
        if let Some(index) = self.idx(word) {
            indices.push(index.single_idx() as u64);
        }

        indices
    }
}

/// Generic corpus vocabulary type.
///
/// Can be used as an input or output lookup.
#[derive(Clone)]
pub struct SimpleVocab<T> {
    config: SimpleVocabConfig,
    types: Vec<CountedType<T>>,
    index: HashMap<T, usize>,
    n_types: usize,
    discards: Vec<f32>,
}

impl<T> SimpleVocab<T>
where
    T: Hash + Eq + Clone + Ord,
{
    /// Constructor only used by the Vocabbuilder
    pub(crate) fn new(
        config: SimpleVocabConfig,
        types: Vec<CountedType<T>>,
        n_types: usize,
    ) -> Self {
        let discards = create_discards(config.discard_threshold, &types, n_types);
        let index = create_indices(&types);
        SimpleVocab {
            config,
            types,
            index,
            n_types,
            discards,
        }
    }

    /// Get a specific context
    pub fn get<Q>(&self, context: &Q) -> Option<&CountedType<T>>
    where
        T: Borrow<Q>,
        Q: Hash + ?Sized + Eq,
    {
        self.idx(context).map(|idx| &self.types[idx.single_idx()])
    }
}

impl From<SimpleVocab<String>> for VocabWrap {
    fn from(vocab: SimpleVocab<String>) -> VocabWrap {
        FiFuSimpleVocab::new(
            vocab
                .types
                .iter()
                .map(|l| l.label().to_owned())
                .collect::<Vec<_>>(),
        )
        .into()
    }
}

impl From<SubwordVocab> for VocabWrap {
    fn from(vocab: SubwordVocab) -> VocabWrap {
        FiFuSubwordVocab::new(
            vocab
                .words
                .iter()
                .map(|l| l.label().to_owned())
                .collect::<Vec<_>>(),
            vocab.config.min_n,
            vocab.config.max_n,
            vocab.config.buckets_exp,
        )
        .into()
    }
}

/// Index trait.
///
/// Trait defining methods on indices stored in the vocabulary.
#[allow(clippy::len_without_is_empty)]
pub trait Idx {
    /// Return the unique index for the Index.
    fn single_idx(&self) -> usize;

    /// Return the number of indices.
    fn len(&self) -> usize;

    /// Return `Self` with only the unique index.
    fn to_single(&self) -> Self;

    /// Return an Iterator over the indices.
    fn indices_iter<'a>(&'a self) -> Box<Iterator<Item = usize> + 'a>;
}

impl Idx for u64 {
    fn single_idx(&self) -> usize {
        *self as usize
    }

    fn len(&self) -> usize {
        1
    }

    fn to_single(&self) -> Self {
        *self
    }

    fn indices_iter<'a>(&'a self) -> Box<Iterator<Item = usize> + 'a> {
        Box::new(Some(*self as usize).into_iter())
    }
}

impl Idx for Vec<u64> {
    fn single_idx(&self) -> usize {
        *self.last().unwrap() as usize
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn to_single(&self) -> Self {
        vec![self.single_idx() as u64]
    }

    fn indices_iter<'a>(&'a self) -> Box<Iterator<Item = usize> + 'a> {
        Box::new(self.iter().map(|i| *i as usize))
    }
}

/// Trait for lookup of indices.
pub trait Vocab {
    type VocabType: Hash + Eq;
    type Config;
    type Idx: Idx;

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
    fn idx<Q>(&self, key: &Q) -> Option<Self::Idx>
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

impl Vocab for SubwordVocab {
    type VocabType = String;
    type Config = SubwordVocabConfig;
    type Idx = Vec<u64>;

    fn config(&self) -> SubwordVocabConfig {
        self.config
    }

    fn idx<Q>(&self, key: &Q) -> Option<Self::Idx>
    where
        Self::VocabType: Borrow<Q>,
        Q: Hash + ?Sized + Eq,
    {
        self.index.get(key).and_then(|idx| {
            self.subword_indices_idx(*idx).map(|v| {
                let mut v = v.to_vec();
                v.push(*idx as u64);
                v
            })
        })
    }

    fn discard(&self, idx: usize) -> f32 {
        self.discards[idx]
    }

    fn n_input_types(&self) -> usize {
        let n_buckets = 2usize.pow(self.config().buckets_exp);
        self.len() + n_buckets
    }

    fn types(&self) -> &[Word] {
        &self.words
    }

    fn n_types(&self) -> usize {
        self.n_tokens
    }
}

impl<T> Vocab for SimpleVocab<T>
where
    T: Hash + Eq,
{
    type VocabType = T;
    type Config = SimpleVocabConfig;
    type Idx = u64;

    fn config(&self) -> SimpleVocabConfig {
        self.config
    }

    fn idx<Q>(&self, key: &Q) -> Option<Self::Idx>
    where
        Self::VocabType: Borrow<Q>,
        Q: Hash + ?Sized + Eq,
    {
        self.index.get(key).cloned().map(|idx| idx as u64)
    }

    fn discard(&self, idx: usize) -> f32 {
        self.discards[idx]
    }

    fn n_input_types(&self) -> usize {
        self.len()
    }

    fn types(&self) -> &[CountedType<Self::VocabType>] {
        &self.types
    }

    fn n_types(&self) -> usize {
        self.n_types
    }
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

/// Constructs a `SimpleVocab<S>` from a `VocabBuilder<T>` where `T: Into<S>`.
impl<T, S> From<VocabBuilder<SimpleVocabConfig, T>> for SimpleVocab<S>
where
    T: Hash + Eq + Into<S>,
    S: Hash + Eq + Clone + Ord,
{
    fn from(builder: VocabBuilder<SimpleVocabConfig, T>) -> Self {
        let min_count = builder.config.min_count;

        let mut types: Vec<_> = builder
            .items
            .into_iter()
            .filter(|(_, count)| *count >= min_count as usize)
            .map(|(item, count)| CountedType::new(item.into(), count))
            .collect();
        types.sort_unstable_by(|w1, w2| w2.cmp(&w1));
        SimpleVocab::new(builder.config, types, builder.n_items)
    }
}

/// Constructs a `SubwordVocab` from a `VocabBuilder<T>` where `T: Into<String>`.
impl<T> From<VocabBuilder<SubwordVocabConfig, T>> for SubwordVocab
where
    T: Hash + Eq + Into<String>,
{
    fn from(builder: VocabBuilder<SubwordVocabConfig, T>) -> Self {
        let config = builder.config;

        let mut words: Vec<_> = builder
            .items
            .into_iter()
            .map(|(word, count)| (word.into(), count))
            .filter(|(word, count)| word == util::EOS || *count >= config.min_count as usize)
            .map(|(word, count)| Word::new(word, count))
            .collect();
        words.sort_unstable_by(|w1, w2| w2.cmp(&w1));
        SubwordVocab::new(config, words, builder.n_items)
    }
}

/// Create discard probabilities based on threshold, specific counts and total counts.
fn create_discards<S>(
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
fn create_indices<S>(types: &[CountedType<S>]) -> HashMap<S, usize>
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
fn bracket(word: &str) -> String {
    let mut bracketed = String::new();
    bracketed.push(BOW);
    bracketed.push_str(word);
    bracketed.push(EOW);

    bracketed
}

#[cfg(test)]
mod tests {
    use super::{bracket, SimpleVocab, SubwordVocab, Vocab, VocabBuilder};
    use crate::subword::SubwordIndices;
    use crate::vocab::Idx;
    use crate::{util, SimpleVocabConfig, SubwordVocabConfig};

    const TEST_SUBWORDCONFIG: SubwordVocabConfig = SubwordVocabConfig {
        buckets_exp: 21,
        discard_threshold: 1e-4,
        min_count: 2,
        max_n: 6,
        min_n: 3,
    };

    const TEST_SIMPLECONFIG: SimpleVocabConfig = SimpleVocabConfig {
        discard_threshold: 1e-4,
        min_count: 2,
    };

    #[test]
    pub fn vocab_is_sorted() {
        let mut config = TEST_SUBWORDCONFIG.clone();
        config.min_count = 1;

        let mut builder: VocabBuilder<SubwordVocabConfig, &str> = VocabBuilder::new(config);
        builder.count("to");
        builder.count("be");
        builder.count("or");
        builder.count("not");
        builder.count("to");
        builder.count("be");
        builder.count("</s>");

        let vocab: SubwordVocab = builder.into();
        let words = vocab.types();

        for idx in 1..words.len() {
            assert!(
                words[idx - 1].count >= words[idx].count,
                "Words are not frequency-sorted"
            );
        }
    }

    #[test]
    pub fn test_vocab_builder() {
        let mut builder: VocabBuilder<SubwordVocabConfig, &str> =
            VocabBuilder::new(TEST_SUBWORDCONFIG.clone());
        builder.count("to");
        builder.count("be");
        builder.count("or");
        builder.count("not");
        builder.count("to");
        builder.count("be");
        builder.count("</s>");

        let vocab: SubwordVocab = builder.into();

        // 'or' and 'not' should be filtered due to the minimum count.
        assert_eq!(vocab.len(), 3);

        assert_eq!(vocab.n_types(), 7);

        // Check expected properties of 'to'.
        let to = vocab.word("to").unwrap();
        assert_eq!("to", to.word());
        assert_eq!(2, to.count);
        assert_eq!(
            &[1141947, 215572, 1324230],
            vocab.subword_indices("to").as_ref()
        );
        assert_eq!(4, vocab.indices("to").len());
        assert!(util::close(
            0.019058,
            vocab.discard(vocab.idx("to").unwrap().single_idx()),
            1e-5
        ));

        // Check expected properties of 'be'.
        let be = vocab.word("be").unwrap();
        assert_eq!("be", be.label);
        assert_eq!(2, be.count);
        assert_eq!(
            &[277351, 1105488, 1482882],
            vocab.subword_indices("be").as_ref()
        );
        assert_eq!(4, vocab.indices("be").len());
        assert!(util::close(
            0.019058,
            vocab.discard(vocab.idx("be").unwrap().single_idx()),
            1e-5
        ));

        // Check expected properties of the end of sentence marker.
        let eos = vocab.word(util::EOS).unwrap();
        assert_eq!(util::EOS, eos.label);
        assert_eq!(1, eos.count);
        assert!(vocab.subword_indices(util::EOS).is_empty());
        assert_eq!(1, vocab.indices(util::EOS).len());
        assert!(util::close(
            0.027158,
            vocab.discard(vocab.idx(util::EOS).unwrap().single_idx()),
            1e-5
        ));

        // Check indices for an unknown word.
        assert_eq!(
            &[1145929, 1737852, 215572, 1187390, 1168229, 858603],
            vocab.indices("too").as_slice()
        );

        // Ensure that the subword indices have the vocab size added.
        assert_eq!(
            bracket("too")
                .subword_indices(
                    TEST_SUBWORDCONFIG.min_n as usize,
                    TEST_SUBWORDCONFIG.max_n as usize,
                    TEST_SUBWORDCONFIG.buckets_exp as usize
                )
                .into_iter()
                .map(|idx| idx + 3)
                .collect::<Vec<_>>(),
            vocab.indices("too").as_slice()
        );
    }

    #[test]
    pub fn types_are_sorted_simple_vocab() {
        let mut builder: VocabBuilder<SimpleVocabConfig, &str> =
            VocabBuilder::new(TEST_SIMPLECONFIG);
        for _ in 0..5 {
            builder.count("a");
        }
        for _ in 0..2 {
            builder.count("b");
        }
        for _ in 0..10 {
            builder.count("d");
        }
        builder.count("c");

        let vocab: SimpleVocab<&str> = builder.into();
        let contexts = vocab.types();
        for idx in 1..contexts.len() {
            assert!(
                contexts[idx - 1].count >= contexts[idx].count,
                "Types are not frequency-sorted"
            );
        }
    }

    #[test]
    pub fn test_simple_vocab_builder() {
        let mut builder: VocabBuilder<SimpleVocabConfig, &str> =
            VocabBuilder::new(TEST_SIMPLECONFIG);
        for _ in 0..5 {
            builder.count("a");
        }
        for _ in 0..2 {
            builder.count("b");
        }
        for _ in 0..10 {
            builder.count("d");
        }
        builder.count("c");

        let vocab: SimpleVocab<&str> = builder.into();

        assert_eq!(vocab.len(), 3);
        assert_eq!(vocab.get("c"), None);

        assert_eq!(vocab.n_types(), 18);
        let a = vocab.get("a").unwrap();
        assert_eq!("a", a.label);
        assert_eq!(5, a.count());
        // 0.0001 / 5/18 + (0.0001 / 5/18).sqrt() = 0.019334
        assert!(util::close(
            0.019334,
            vocab.discard(vocab.idx("a").unwrap().single_idx()),
            1e-5
        ));
    }
}
