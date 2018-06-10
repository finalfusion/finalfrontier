use std::borrow::Cow;
use std::collections::HashMap;

use {util, Config, SubwordIndices};

const BOW: char = '<';
const EOW: char = '>';

/// A vocabulary token.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Token {
    token: String,
    count: usize,
}

impl Token {
    /// Construct a new token.
    pub(crate) fn new(token: String, count: usize) -> Self {
        Token { token, count }
    }

    /// The token count.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The string representation of the token.
    pub fn token(&self) -> &str {
        &self.token
    }
}

/// A corpus vocabulary.
#[derive(Clone)]
pub struct Vocab {
    config: Config,
    tokens: Vec<Token>,
    subwords: Vec<Vec<u64>>,
    discards: Vec<f32>,
    index: HashMap<String, usize>,
}

impl Vocab {
    /// Construct a new vocabulary.
    ///
    /// Normally a `VocabBuilder` should be used. This constructor is used
    /// for deserialization.
    pub(crate) fn new(config: Config, tokens: Vec<Token>) -> Self {
        let index = Self::create_token_indices(&tokens);
        let subwords = Self::create_subword_indices(&config, &tokens);
        let discards = Self::create_discards(&config, &tokens);

        Vocab {
            config,
            discards,
            tokens,
            subwords,
            index,
        }
    }

    fn create_discards(config: &Config, tokens: &[Token]) -> Vec<f32> {
        let n_tokens: usize = tokens.iter().map(|t| t.count).sum();

        let mut discards = Vec::with_capacity(tokens.len());

        for token in tokens {
            let p = token.count() as f32 / n_tokens as f32;
            let p_discard = config.discard_threshold / p + (config.discard_threshold / p).sqrt();

            // Not a proper probability, upper bound at 1.0.
            discards.push(1f32.min(p_discard));
        }

        discards
    }

    fn create_subword_indices(config: &Config, tokens: &[Token]) -> Vec<Vec<u64>> {
        let mut subword_indices = Vec::new();

        for token in tokens {
            if token.token == util::EOS {
                subword_indices.push(Vec::new());
                continue;
            }

            subword_indices.push(
                bracket(&token.token)
                    .as_str()
                    .subword_indices(
                        config.min_n as usize,
                        config.max_n as usize,
                        config.buckets_exp as usize,
                    )
                    .into_iter()
                    .map(|idx| idx + tokens.len() as u64)
                    .collect(),
            );
        }

        assert_eq!(tokens.len(), subword_indices.len());

        subword_indices
    }

    fn create_token_indices(tokens: &[Token]) -> HashMap<String, usize> {
        let mut token_indices = HashMap::new();

        for (idx, token) in tokens.iter().enumerate() {
            token_indices.insert(token.token.clone(), idx);
        }

        // Invariant: The index size should be the same as the number of
        // tokens.
        assert_eq!(tokens.len(), token_indices.len());

        token_indices
    }

    /// Get the discard probability of the token with the given index.
    pub fn discard(&self, idx: usize) -> f32 {
        self.discards[idx]
    }

    /// Get the vocabulary size.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Get the given token.
    pub fn token(&self, token: &str) -> Option<&Token> {
        self.token_idx(token).map(|idx| &self.tokens[idx])
    }

    /// Get the index of a token.
    pub fn token_idx(&self, token: &str) -> Option<usize> {
        self.index.get(token).cloned()
    }

    /// Get the subword indices of a token.
    pub fn subword_indices(&self, token: &str) -> Cow<[u64]> {
        if token == util::EOS {
            // Do not create subwords for the EOS marker.
            Cow::Borrowed(&[])
        } else if let Some(&idx) = self.index.get(token) {
            Cow::Borrowed(&self.subwords[idx])
        } else {
            Cow::Owned(
                bracket(token)
                    .as_str()
                    .subword_indices(
                        self.config.min_n as usize,
                        self.config.max_n as usize,
                        self.config.buckets_exp as usize,
                    )
                    .into_iter()
                    .map(|idx| idx + self.tokens.len() as u64)
                    .collect(),
            )
        }
    }

    /// Get all indices of a token, both regular and subword.
    ///
    /// This method copies the subword list for known words into a new Vec.
    pub fn indices(&self, token: &str) -> Vec<u64> {
        let mut indices = self.subword_indices(token).into_owned();
        if let Some(index) = self.token_idx(token) {
            indices.push(index as u64);
        }

        indices
    }

    /// Get all tokens in the vocabulary.
    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }
}

/// This builder is used to construct a vocabulary.
///
/// Tokens are added to the vocabulary and counted using the `count` method.
/// The final vocabulary is constructed using `build`.
pub struct VocabBuilder {
    config: Config,
    tokens: HashMap<String, usize>,
}

impl VocabBuilder {
    pub fn new(config: Config) -> Self {
        VocabBuilder {
            config,
            tokens: HashMap::new(),
        }
    }

    /// Convert the builder to a vocabulary.
    pub fn build(self) -> Vocab {
        let config = self.config;

        let mut tokens = Vec::new();
        for (token, count) in self.tokens.into_iter() {
            if token != util::EOS && count < config.min_count as usize {
                continue;
            }

            tokens.push(Token::new(token, count));
        }

        Vocab::new(config, tokens)
    }

    /// Count a token.
    ///
    /// This will have the effect of adding the token to the vocabulary if
    /// it has not been seen before. Otherwise, its count will be updated.
    pub fn count<S>(&mut self, token: S)
    where
        S: Into<String>,
    {
        let token = token.into();

        let token = self.tokens.entry(token.clone()).or_insert(0);

        *token += 1;
    }
}

/// Add begin/end-of-word brackets.
fn bracket(token: &str) -> String {
    let mut bracketed = String::new();
    bracketed.push(BOW);
    bracketed.push_str(token);
    bracketed.push(EOW);

    bracketed
}

#[cfg(test)]
mod tests {
    use super::{bracket, VocabBuilder};
    use {util, Config, LossType, ModelType, SubwordIndices};

    #[test]
    pub fn test_vocab_builder() {
        let config = Config {
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

        let mut builder = VocabBuilder::new(config.clone());
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

        // Check expected properties of 'to'.
        let to = vocab.token("to").unwrap();
        assert_eq!("to", to.token);
        assert_eq!(2, to.count);
        assert_eq!(
            &[1141947, 215572, 1324230],
            vocab.subword_indices("to").as_ref()
        );
        assert_eq!(4, vocab.indices("to").len());
        assert!(util::close(
            0.016061,
            vocab.discard(vocab.token_idx("to").unwrap()),
            1e-5
        ));

        // Check expected properties of 'be'.
        let be = vocab.token("be").unwrap();
        assert_eq!("be", be.token);
        assert_eq!(2, be.count);
        assert_eq!(
            &[277351, 1105488, 1482882],
            vocab.subword_indices("be").as_ref()
        );
        assert_eq!(4, vocab.indices("be").len());
        assert!(util::close(
            0.016061,
            vocab.discard(vocab.token_idx("be").unwrap()),
            1e-5
        ));

        // Check expected properties of the end of sentence marker.
        let eos = vocab.token(util::EOS).unwrap();
        assert_eq!(util::EOS, eos.token);
        assert_eq!(1, eos.count);
        assert!(vocab.subword_indices(util::EOS).is_empty());
        assert_eq!(1, vocab.indices(util::EOS).len());
        println!("{}", vocab.discard(vocab.token_idx(util::EOS).unwrap()));
        assert!(util::close(
            0.022861,
            vocab.discard(vocab.token_idx(util::EOS).unwrap()),
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
                    config.min_n as usize,
                    config.max_n as usize,
                    config.buckets_exp as usize
                )
                .into_iter()
                .map(|idx| idx + 3)
                .collect::<Vec<_>>(),
            vocab.indices("too").as_slice()
        );
    }
}
