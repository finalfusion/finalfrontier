use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;

use finalfusion::prelude::{SimpleVocab as FiFuSimpleVocab, VocabWrap};

use crate::idx::{SingleIdx, WordIdx};
use crate::vocab::{create_discards, create_indices};
use crate::{CountedType, SimpleVocabConfig, Vocab, VocabBuilder};

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
        self.idx(context)
            .map(|idx| &self.types[idx.word_idx() as usize])
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

impl<T> Vocab for SimpleVocab<T>
where
    T: Hash + Eq,
{
    type VocabType = T;
    type IdxType = SingleIdx;
    type Config = SimpleVocabConfig;

    fn config(&self) -> SimpleVocabConfig {
        self.config
    }

    fn idx<Q>(&self, key: &Q) -> Option<Self::IdxType>
    where
        Self::VocabType: Borrow<Q>,
        Q: Hash + ?Sized + Eq,
    {
        self.index
            .get(key)
            .cloned()
            .map(|idx| SingleIdx::from_word_idx(idx as u64))
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

#[cfg(test)]
mod tests {
    use super::{SimpleVocab, Vocab, VocabBuilder};
    use crate::idx::WordIdx;
    use crate::{util, SimpleVocabConfig};

    const TEST_SIMPLECONFIG: SimpleVocabConfig = SimpleVocabConfig {
        discard_threshold: 1e-4,
        min_count: 2,
    };

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
            vocab.discard(vocab.idx("a").unwrap().word_idx() as usize),
            1e-5
        ));
    }
}
