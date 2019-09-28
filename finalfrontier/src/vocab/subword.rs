use std::borrow::{Borrow, Cow};
use std::collections::HashMap;
use std::hash::Hash;

use finalfusion::prelude::{SubwordVocab as FiFuSubwordVocab, VocabWrap};
use finalfusion::subword::{BucketIndexer, FinalfusionHashIndexer, Indexer, SubwordIndices};

use crate::idx::{WordIdx, WordWithSubwordsIdx};
use crate::vocab::{bracket, create_discards, create_indices};
use crate::{util, SubwordVocabConfig, Vocab, VocabBuilder, Word};

/// A corpus vocabulary with subword lookup.
#[derive(Clone)]
pub struct SubwordVocab<I> {
    config: SubwordVocabConfig,
    words: Vec<Word>,
    indexer: I,
    subwords: Vec<Vec<u64>>,
    discards: Vec<f32>,
    index: HashMap<String, usize>,
    n_tokens: usize,
}

impl<I> SubwordVocab<I>
where
    I: Indexer,
{
    /// Construct a new vocabulary.
    pub fn new(config: SubwordVocabConfig, words: Vec<Word>, n_tokens: usize, indexer: I) -> Self {
        let index = create_indices(&words);
        let subwords = Self::create_subword_indices(
            config.min_n as usize,
            config.max_n as usize,
            &indexer,
            &words,
        );
        let discards = create_discards(config.discard_threshold, &words, n_tokens);
        SubwordVocab {
            config,
            discards,
            indexer,
            words,
            subwords,
            index,
            n_tokens,
        }
    }

    fn create_subword_indices(
        min_n: usize,
        max_n: usize,
        indexer: &I,
        words: &[Word],
    ) -> Vec<Vec<u64>> {
        let mut subword_indices = Vec::new();

        for word in words {
            if word.word() == util::EOS {
                subword_indices.push(Vec::new());
                continue;
            }

            subword_indices.push(
                bracket(word.word())
                    .as_str()
                    .subword_indices(min_n, max_n, indexer)
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
        self.idx(word)
            .map(|idx| &self.words[idx.word_idx() as usize])
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
                        &self.indexer,
                    )
                    .into_iter()
                    .map(|idx| idx + self.words.len() as u64)
                    .collect(),
            )
        }
    }

    /// Get all indices of a word, both regular and subword.
    ///
    /// This method copies the subword list for known words into a new Vec.
    pub fn indices(&self, word: &str) -> Vec<u64> {
        let mut indices = self.subword_indices(word).into_owned();
        if let Some(index) = self.idx(word) {
            indices.push(index.word_idx() as u64);
        }

        indices
    }
}

impl<I> SubwordVocab<I> {
    pub(crate) fn subword_indices_idx(&self, idx: usize) -> Option<&[u64]> {
        self.subwords.get(idx).map(|v| v.as_slice())
    }
}

impl<I> Vocab for SubwordVocab<I>
where
    I: Indexer,
{
    type VocabType = String;
    type IdxType = WordWithSubwordsIdx;
    type Config = SubwordVocabConfig;

    fn config(&self) -> SubwordVocabConfig {
        self.config
    }

    fn idx<Q>(&self, key: &Q) -> Option<Self::IdxType>
    where
        Self::VocabType: Borrow<Q>,
        Q: Hash + ?Sized + Eq,
    {
        self.index.get(key).and_then(|idx| {
            self.subword_indices_idx(*idx)
                .map(|v| WordWithSubwordsIdx::new(*idx as u64, v))
        })
    }

    fn discard(&self, idx: usize) -> f32 {
        self.discards[idx]
    }

    fn n_input_types(&self) -> usize {
        self.len() + self.indexer.upper_bound() as usize
    }

    fn types(&self) -> &[Word] {
        &self.words
    }

    fn n_types(&self) -> usize {
        self.n_tokens
    }
}

/// Constructs a `SubwordVocab` from a `VocabBuilder<T>` where `T: Into<String>`.
impl<T> From<VocabBuilder<SubwordVocabConfig, T>> for SubwordVocab<FinalfusionHashIndexer>
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
        SubwordVocab::new(
            config,
            words,
            builder.n_items,
            FinalfusionHashIndexer::new(config.buckets_exp as usize),
        )
    }
}

impl From<SubwordVocab<FinalfusionHashIndexer>> for VocabWrap {
    fn from(vocab: SubwordVocab<FinalfusionHashIndexer>) -> Self {
        let config = vocab.config();
        let words = vocab
            .words
            .into_iter()
            .map(|word| word.label)
            .collect::<Vec<_>>();
        FiFuSubwordVocab::new(words, config.min_n, config.max_n, vocab.indexer).into()
    }
}

#[cfg(test)]
mod tests {
    use super::{bracket, SubwordVocab, Vocab, VocabBuilder};
    use crate::idx::WordIdx;
    use crate::{util, SubwordVocabConfig};
    use finalfusion::subword::SubwordIndices;

    const TEST_SUBWORDCONFIG: SubwordVocabConfig = SubwordVocabConfig {
        buckets_exp: 21,
        discard_threshold: 1e-4,
        min_count: 2,
        max_n: 6,
        min_n: 3,
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

        let vocab: SubwordVocab<_> = builder.into();
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

        let vocab: SubwordVocab<_> = builder.into();

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
            vocab.discard(vocab.idx("to").unwrap().word_idx() as usize),
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
            vocab.discard(vocab.idx("be").unwrap().word_idx() as usize),
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
            vocab.discard(vocab.idx(util::EOS).unwrap().word_idx() as usize),
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
                    &vocab.indexer
                )
                .into_iter()
                .map(|idx| idx + 3)
                .collect::<Vec<_>>(),
            vocab.indices("too").as_slice()
        );
    }
}
