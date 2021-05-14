use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;

use finalfusion::compat::fasttext::FastTextIndexer;
use finalfusion::subword::{
    BucketIndexer, ExplicitIndexer, FinalfusionHashIndexer, Indexer, NGrams, SubwordIndices,
};
use finalfusion::vocab::{SubwordVocab as FiFuSubwordVocab, VocabWrap};

use crate::idx::{WordIdx, WordWithSubwordsIdx};
use crate::vocab::{bracket, create_discards, create_indices};
use crate::{
    BucketConfig, BucketIndexerType, CountedType, NGramConfig, SubwordVocabConfig, Vocab,
    VocabBuilder, Word,
};

/// A corpus vocabulary with subword lookup.
#[derive(Clone)]
pub struct SubwordVocab<C, I> {
    config: SubwordVocabConfig<C>,
    words: Vec<Word>,
    indexer: I,
    subwords: Vec<Vec<u64>>,
    discards: Vec<f32>,
    index: HashMap<String, usize>,
    n_tokens: usize,
}

impl<C, I> SubwordVocab<C, I>
where
    C: Copy + Clone,
    I: Indexer,
{
    /// Construct a new vocabulary.
    pub fn new(
        config: SubwordVocabConfig<C>,
        words: Vec<Word>,
        n_tokens: usize,
        indexer: I,
    ) -> Self {
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
            words,
            indexer,
            subwords,
            discards,
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
            subword_indices.push(
                bracket(word.word())
                    .as_str()
                    .subword_indices(min_n, max_n, indexer)
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
}

impl<C, I> SubwordVocab<C, I> {
    pub(crate) fn subword_indices_idx(&self, idx: usize) -> Option<&[u64]> {
        self.subwords.get(idx).map(|v| v.as_slice())
    }
}

impl<C, I> Vocab for SubwordVocab<C, I>
where
    C: Copy + Clone,
    I: Indexer,
{
    type VocabType = String;
    type IdxType = WordWithSubwordsIdx;
    type Config = SubwordVocabConfig<C>;

    fn config(&self) -> SubwordVocabConfig<C> {
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
impl<I, T> From<VocabBuilder<SubwordVocabConfig<BucketConfig>, T>> for SubwordVocab<BucketConfig, I>
where
    T: Hash + Eq + Into<String>,
    I: BucketIndexer,
{
    fn from(builder: VocabBuilder<SubwordVocabConfig<BucketConfig>, T>) -> Self {
        let config = builder.config;
        let words = config.cutoff.filter(builder.items);
        let buckets = match config.indexer.indexer_type {
            BucketIndexerType::Finalfusion => config.indexer.buckets_exp as usize,
            BucketIndexerType::FastText => 2u64.pow(config.indexer.buckets_exp) as usize,
        };
        SubwordVocab::new(config, words, builder.n_items, I::new(buckets))
    }
}

/// Constructs a `SubwordVocab` from a `VocabBuilder<T>` where `T: Into<String>`.
impl<T> From<VocabBuilder<SubwordVocabConfig<NGramConfig>, T>>
    for SubwordVocab<NGramConfig, ExplicitIndexer>
where
    T: Hash + Eq + Into<String>,
{
    fn from(builder: VocabBuilder<SubwordVocabConfig<NGramConfig>, T>) -> Self {
        let config = builder.config;
        let words: Vec<Word> = builder.config.cutoff.filter(builder.items);
        let mut ngram_counts: HashMap<String, usize> = HashMap::new();
        for word in words.iter() {
            for ngram in NGrams::new(
                &bracket(word.label()),
                config.min_n as usize,
                config.max_n as usize,
            )
            .map(|ngram| ngram.to_string())
            {
                let cnt = ngram_counts.entry(ngram).or_default();
                *cnt += word.count;
            }
        }

        let ngrams: Vec<CountedType<String>> = config.indexer.cutoff.filter(ngram_counts);
        let ngrams = ngrams
            .into_iter()
            .map(|counted| counted.label)
            .collect::<Vec<_>>();
        SubwordVocab::new(config, words, builder.n_items, ExplicitIndexer::new(ngrams))
    }
}

macro_rules! impl_into_vocabwrap (
    ($vocab:ty) => {
        impl From<$vocab> for VocabWrap {
            fn from(vocab: $vocab) -> Self {
                let config = vocab.config;
                let words = vocab
                    .words
                    .into_iter()
                    .map(|word| word.label)
                    .collect::<Vec<_>>();
                FiFuSubwordVocab::new(words, config.min_n, config.max_n, vocab.indexer).into()
            }
        }
    }
);

impl_into_vocabwrap!(SubwordVocab<BucketConfig, FinalfusionHashIndexer>);
impl_into_vocabwrap!(SubwordVocab<BucketConfig, FastTextIndexer>);
impl_into_vocabwrap!(SubwordVocab<NGramConfig, ExplicitIndexer>);

#[cfg(test)]
mod tests {
    use super::{SubwordVocab, Vocab, VocabBuilder};
    use crate::config::SubwordVocabConfig;
    use crate::idx::WordIdx;
    use crate::{util, BucketConfig, Cutoff, NGramConfig};

    use crate::config::BucketIndexerType::Finalfusion;
    use finalfusion::subword::{ExplicitIndexer, FinalfusionHashIndexer, Indexer};

    const TEST_SUBWORDCONFIG: SubwordVocabConfig<BucketConfig> = SubwordVocabConfig {
        discard_threshold: 1e-4,
        cutoff: Cutoff::MinCount(2),
        max_n: 6,
        min_n: 3,
        indexer: BucketConfig {
            buckets_exp: 21,
            indexer_type: Finalfusion,
        },
    };

    const TEST_NGRAMCONFIG: SubwordVocabConfig<NGramConfig> = SubwordVocabConfig {
        discard_threshold: 1e-4,
        cutoff: Cutoff::MinCount(2),
        max_n: 6,
        min_n: 3,
        indexer: NGramConfig {
            cutoff: Cutoff::MinCount(2),
        },
    };

    #[test]
    pub fn vocab_is_sorted() {
        let mut config = TEST_SUBWORDCONFIG;
        config.cutoff = Cutoff::MinCount(1);

        let mut builder: VocabBuilder<_, &str> = VocabBuilder::new(config);
        builder.count("to");
        builder.count("be");
        builder.count("or");
        builder.count("not");
        builder.count("to");
        builder.count("be");
        builder.count("</s>");

        let vocab: SubwordVocab<_, FinalfusionHashIndexer> = builder.into();
        let words = vocab.types();

        for idx in 1..words.len() {
            assert!(
                words[idx - 1].count >= words[idx].count,
                "Words are not frequency-sorted"
            );
        }
    }

    #[test]
    pub fn test_bucket_vocab_builder() {
        let mut builder: VocabBuilder<_, &str> = VocabBuilder::new(TEST_SUBWORDCONFIG);
        builder.count("to");
        builder.count("be");
        builder.count("or");
        builder.count("not");
        builder.count("to");
        builder.count("be");
        builder.count("</s>");

        let vocab: SubwordVocab<_, FinalfusionHashIndexer> = builder.into();

        // 'or' and 'not' should be filtered due to the minimum count.
        assert_eq!(vocab.len(), 2);

        assert_eq!(vocab.n_types(), 7);

        // Check expected properties of 'to'.
        let to = vocab.word("to").unwrap();
        assert_eq!("to", to.word());
        assert_eq!(2, to.count);
        assert_eq!(
            vec![1141946, 215571, 1324229, 0],
            vocab.idx("to").unwrap().into_iter().collect::<Vec<_>>()
        );
        assert!(util::close(
            0.019058,
            vocab.discard(vocab.idx("to").unwrap().word_idx() as usize),
            1e-5,
        ));

        // Check expected properties of 'be'.
        let be = vocab.word("be").unwrap();
        assert_eq!("be", be.label);
        assert_eq!(2, be.count);
        assert_eq!(
            vec![277350, 1105487, 1482881, 1],
            vocab.idx("be").unwrap().into_iter().collect::<Vec<_>>()
        );
        assert!(util::close(
            0.019058,
            vocab.discard(vocab.idx("be").unwrap().word_idx() as usize),
            1e-5,
        ));

        // Check indices for an unknown word.
        assert!(vocab.idx("too").is_none());
    }

    #[test]
    pub fn test_ngram_vocab_builder() {
        let mut builder: VocabBuilder<_, &str> = VocabBuilder::new(TEST_NGRAMCONFIG);
        builder.count("to");
        builder.count("be");
        builder.count("or");
        builder.count("not");
        builder.count("to");
        builder.count("be");
        builder.count("</s>");

        let vocab: SubwordVocab<_, ExplicitIndexer> = builder.into();

        // 'or' and 'not' should be filtered due to the minimum count.
        assert_eq!(vocab.len(), 2);

        assert_eq!(vocab.n_types(), 7);

        // Check expected properties of 'to'.
        let to = vocab.word("to").unwrap();
        assert_eq!("to", to.word());
        assert_eq!(2, to.count);
        // 2x ["<to", "<to>", "to>", "<be", "<be>", "be>"]
        // sorted ["to>", "be>", "<to>", "<to", "<be>", "<be"]
        assert_eq!(6, vocab.indexer.ngrams().len());
        assert_eq!(6, vocab.indexer.upper_bound());
        assert_eq!(
            &["to>", "be>", "<to>", "<to", "<be>", "<be"],
            vocab.indexer.ngrams()
        );
        // subwords have offset of (vocab.len() - 1)
        assert_eq!(
            vec![4, 5, 2, 0],
            vocab.idx("to").unwrap().into_iter().collect::<Vec<_>>()
        );
        assert!(util::close(
            0.019058,
            vocab.discard(vocab.idx("to").unwrap().word_idx() as usize),
            1e-5,
        ));

        // Check expected properties of 'be'.
        let be = vocab.word("be").unwrap();
        assert_eq!("be", be.label);
        assert_eq!(2, be.count);
        // see above explanation
        assert_eq!(
            vec![6, 7, 3, 1],
            vocab.idx("be").unwrap().into_iter().collect::<Vec<_>>()
        );
        assert!(util::close(
            0.019058,
            vocab.discard(vocab.idx("be").unwrap().word_idx() as usize),
            1e-5,
        ));

        // Check indices for an unknown word. Only "<to" is a known ngram.
        assert!(vocab.idx("too").is_none());
    }
}
