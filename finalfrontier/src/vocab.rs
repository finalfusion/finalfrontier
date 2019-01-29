use std::borrow::Cow;
use std::collections::HashMap;

use subword::SubwordIndices;
use {util, Config};

const BOW: char = '<';
const EOW: char = '>';

/// A vocabulary word.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct WordCount {
    word: String,
    count: usize,
}

impl WordCount {
    /// Construct a new word.
    pub(crate) fn new(word: String, count: usize) -> Self {
        WordCount { word, count }
    }

    /// The word count.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The string representation of the word.
    pub fn word(&self) -> &str {
        &self.word
    }
}

/// A corpus vocabulary.
#[derive(Clone)]
pub struct Vocab {
    config: Config,
    words: Vec<WordCount>,
    subwords: Vec<Vec<u64>>,
    discards: Vec<f32>,
    index: HashMap<String, usize>,
    n_tokens: usize,
}

impl Vocab {
    /// Construct a new vocabulary.
    ///
    /// Normally a `VocabBuilder` should be used. This constructor is used
    /// for deserialization.
    pub(crate) fn new(config: Config, words: Vec<WordCount>, n_tokens: usize) -> Self {
        let index = Self::create_word_indices(&words);
        let subwords = Self::create_subword_indices(&config, &words);
        let discards = Self::create_discards(&config, &words, n_tokens);

        Vocab {
            config,
            discards,
            words,
            subwords,
            index,
            n_tokens,
        }
    }

    fn create_discards(config: &Config, words: &[WordCount], n_tokens: usize) -> Vec<f32> {
        let mut discards = Vec::with_capacity(words.len());

        for word in words {
            let p = word.count() as f32 / n_tokens as f32;
            let p_discard = config.discard_threshold / p + (config.discard_threshold / p).sqrt();

            // Not a proper probability, upper bound at 1.0.
            discards.push(1f32.min(p_discard));
        }

        discards
    }

    fn create_subword_indices(config: &Config, words: &[WordCount]) -> Vec<Vec<u64>> {
        let mut subword_indices = Vec::new();

        for word in words {
            if word.word == util::EOS {
                subword_indices.push(Vec::new());
                continue;
            }

            subword_indices.push(
                bracket(&word.word)
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

    fn create_word_indices(words: &[WordCount]) -> HashMap<String, usize> {
        let mut word_indices = HashMap::new();

        for (idx, word) in words.iter().enumerate() {
            word_indices.insert(word.word.clone(), idx);
        }

        // Invariant: The index size should be the same as the number of
        // words.
        assert_eq!(words.len(), word_indices.len());

        word_indices
    }

    /// Get the discard probability of the word with the given index.
    pub(crate) fn discard(&self, idx: usize) -> f32 {
        self.discards[idx]
    }

    /// Get the vocabulary size.
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// Get the given word.
    pub fn word(&self, word: &str) -> Option<&WordCount> {
        self.word_idx(word).map(|idx| &self.words[idx])
    }

    /// Get the index of a word.
    pub(crate) fn word_idx(&self, word: &str) -> Option<usize> {
        self.index.get(word).cloned()
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
        if let Some(index) = self.word_idx(word) {
            indices.push(index as u64);
        }

        indices
    }

    /// Get the number of tokens in the corpus.
    ///
    /// This returns the number of tokens in the corpus that the vocabulary
    /// was constructed from, **before** removing tokens that are below the
    /// minimum count.
    pub fn n_tokens(&self) -> usize {
        self.n_tokens
    }

    /// Get all words in the vocabulary.
    pub fn words(&self) -> &[WordCount] {
        &self.words
    }
}

/// This builder is used to construct a vocabulary.
///
/// Tokens are added to the vocabulary and counted using the `count` method.
/// The final vocabulary is constructed using `build`.
pub struct VocabBuilder {
    config: Config,
    words: HashMap<String, usize>,
    n_tokens: usize,
}

impl VocabBuilder {
    pub fn new(config: Config) -> Self {
        VocabBuilder {
            config,
            words: HashMap::new(),
            n_tokens: 0,
        }
    }

    /// Convert the builder to a vocabulary.
    pub fn build(self) -> Vocab {
        let config = self.config;

        let mut words = Vec::new();
        for (word, count) in self.words.into_iter() {
            if word != util::EOS && count < config.min_count as usize {
                continue;
            }

            words.push(WordCount::new(word, count));
        }

        words.sort_unstable_by(|w1, w2| w2.count.cmp(&w1.count));

        Vocab::new(config, words, self.n_tokens)
    }

    /// Count a word.
    ///
    /// This will have the effect of adding the word to the vocabulary if
    /// it has not been seen before. Otherwise, its count will be updated.
    pub fn count<S>(&mut self, word: S)
    where
        S: Into<String>,
    {
        self.n_tokens += 1;

        let word = self.words.entry(word.into()).or_insert(0);
        *word += 1;
    }
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
    use super::{bracket, VocabBuilder};
    use subword::SubwordIndices;
    use {util, Config, LossType, ModelType};

    const TEST_CONFIG: Config = Config {
        buckets_exp: 21,
        context_size: 5,
        dims: 300,
        discard_threshold: 1e-4,
        epochs: 5,
        loss: LossType::LogisticNegativeSampling,
        lr: 0.05,
        min_count: 2,
        max_n: 6,
        min_n: 3,
        model: ModelType::SkipGram,
        negative_samples: 5,
    };

    #[test]
    pub fn vocab_is_sorted() {
        let mut config = TEST_CONFIG.clone();
        config.min_count = 1;

        let mut builder = VocabBuilder::new(TEST_CONFIG.clone());
        builder.count("to");
        builder.count("be");
        builder.count("or");
        builder.count("not");
        builder.count("to");
        builder.count("be");
        builder.count("</s>");

        let vocab = builder.build();
        let words = vocab.words();

        for idx in 1..words.len() {
            assert!(
                words[idx - 1].count >= words[idx].count,
                "Words are not frequency-sorted"
            );
        }
    }

    #[test]
    pub fn test_vocab_builder() {
        let mut builder = VocabBuilder::new(TEST_CONFIG.clone());
        builder.count("to");
        builder.count("be");
        builder.count("or");
        builder.count("not");
        builder.count("to");
        builder.count("be");
        builder.count("</s>");

        let vocab = builder.build();

        // 'or' and 'not' should be filtered due to the minimum count.
        assert_eq!(vocab.len(), 3);

        assert_eq!(vocab.n_tokens(), 7);

        // Check expected properties of 'to'.
        let to = vocab.word("to").unwrap();
        assert_eq!("to", to.word);
        assert_eq!(2, to.count);
        assert_eq!(
            &[1141947, 215572, 1324230],
            vocab.subword_indices("to").as_ref()
        );
        assert_eq!(4, vocab.indices("to").len());
        assert!(util::close(
            0.019058,
            vocab.discard(vocab.word_idx("to").unwrap()),
            1e-5
        ));

        // Check expected properties of 'be'.
        let be = vocab.word("be").unwrap();
        assert_eq!("be", be.word);
        assert_eq!(2, be.count);
        assert_eq!(
            &[277351, 1105488, 1482882],
            vocab.subword_indices("be").as_ref()
        );
        assert_eq!(4, vocab.indices("be").len());
        assert!(util::close(
            0.019058,
            vocab.discard(vocab.word_idx("be").unwrap()),
            1e-5
        ));

        // Check expected properties of the end of sentence marker.
        let eos = vocab.word(util::EOS).unwrap();
        assert_eq!(util::EOS, eos.word);
        assert_eq!(1, eos.count);
        assert!(vocab.subword_indices(util::EOS).is_empty());
        assert_eq!(1, vocab.indices(util::EOS).len());
        assert!(util::close(
            0.027158,
            vocab.discard(vocab.word_idx(util::EOS).unwrap()),
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
                    TEST_CONFIG.min_n as usize,
                    TEST_CONFIG.max_n as usize,
                    TEST_CONFIG.buckets_exp as usize
                )
                .into_iter()
                .map(|idx| idx + 3)
                .collect::<Vec<_>>(),
            vocab.indices("too").as_slice()
        );
    }
}
