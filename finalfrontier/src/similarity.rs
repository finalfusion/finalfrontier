//! Traits and trait implementations for similarity queries.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use ndarray::{Array1, ArrayView1, ArrayView2};
use ordered_float::NotNaN;

use vec_simd::l2_normalize;
use Model;

/// A word with its similarity.
///
/// This data structure is used to store a pair consisting of a word and
/// its similarity to a query word.
#[derive(Debug, Eq, PartialEq)]
pub struct WordSimilarity<'a> {
    pub similarity: NotNaN<f32>,
    pub word: &'a str,
}

impl<'a> Ord for WordSimilarity<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.similarity.cmp(&self.similarity) {
            Ordering::Equal => self.word.cmp(other.word),
            ordering => ordering,
        }
    }
}

impl<'a> PartialOrd for WordSimilarity<'a> {
    fn partial_cmp(&self, other: &WordSimilarity) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Trait for analogy queries.
pub trait Analogy {
    /// Perform an analogy query.
    ///
    /// This method returns words that are close in vector space the analogy
    /// query `word1` is to `word2` as `word3` is to `?`. More concretely,
    /// it searches embeddings that are similar to:
    ///
    /// *embedding(word2) - embedding(word1) + embedding(word3)*
    ///
    /// At most, `limit` results are returned.
    fn analogy(
        &self,
        word1: &str,
        word2: &str,
        word3: &str,
        limit: usize,
    ) -> Option<Vec<WordSimilarity>>;
}

impl Analogy for Model {
    fn analogy(
        &self,
        word1: &str,
        word2: &str,
        word3: &str,
        limit: usize,
    ) -> Option<Vec<WordSimilarity>> {
        self.analogy_by(word1, word2, word3, limit, |embeds, embed| {
            embeds.dot(&embed)
        })
    }
}

/// Trait for analogy queries with a custom similarity function.
pub trait AnalogyBy {
    /// Perform an analogy query using the given similarity function.
    ///
    /// This method returns words that are close in vector space the analogy
    /// query `word1` is to `word2` as `word3` is to `?`. More concretely,
    /// it searches embeddings that are similar to:
    ///
    /// *embedding(word2) - embedding(word1) + embedding(word3)*
    ///
    /// At most, `limit` results are returned.
    fn analogy_by<F>(
        &self,
        word1: &str,
        word2: &str,
        word3: &str,
        limit: usize,
        similarity: F,
    ) -> Option<Vec<WordSimilarity>>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>;
}

impl AnalogyBy for Model {
    fn analogy_by<F>(
        &self,
        word1: &str,
        word2: &str,
        word3: &str,
        limit: usize,
        similarity: F,
    ) -> Option<Vec<WordSimilarity>>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>,
    {
        let embedding1 = self.embedding(word1)?;
        let embedding2 = self.embedding(word2)?;
        let embedding3 = self.embedding(word3)?;

        let mut embedding = (embedding2 - embedding1) + embedding3;
        l2_normalize(embedding.view_mut());

        let skip = [word1, word2, word3].iter().cloned().collect();

        Some(self.similarity_(embedding, &skip, limit, similarity))
    }
}

/// Trait for similarity queries.
pub trait Similarity {
    /// Find words that are similar to the query word.
    ///
    /// The similarity between two words is defined by the dot product of
    /// the embeddings. If the vectors are unit vectors (e.g. by virtue of
    /// calling `normalize`), this is the cosine similarity. At most, `limit`
    /// results are returned.
    fn similarity(&self, word: &str, limit: usize) -> Option<Vec<WordSimilarity>>;
}

impl Similarity for Model {
    fn similarity(&self, word: &str, limit: usize) -> Option<Vec<WordSimilarity>> {
        self.similarity_by(word, limit, |embeds, embed| embeds.dot(&embed))
    }
}

/// Trait for similarity queries with a custom similarity function.
pub trait SimilarityBy {
    /// Find words that are similar to the query word using the given similarity
    /// function.
    ///
    /// The similarity function should return, given the embeddings matrix and
    /// the word vector a vector of similarity scores. At most, `limit` results
    /// are returned.
    fn similarity_by<F>(
        &self,
        word: &str,
        limit: usize,
        similarity: F,
    ) -> Option<Vec<WordSimilarity>>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>;
}

impl SimilarityBy for Model {
    fn similarity_by<F>(
        &self,
        word: &str,
        limit: usize,
        similarity: F,
    ) -> Option<Vec<WordSimilarity>>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>,
    {
        let embed = self.embedding(word)?;
        let mut skip = HashSet::new();
        skip.insert(word);

        Some(self.similarity_(embed, &skip, limit, similarity))
    }
}

trait SimilarityPrivate {
    fn similarity_<F>(
        &self,
        embed: Array1<f32>,
        skip: &HashSet<&str>,
        limit: usize,
        similarity: F,
    ) -> Vec<WordSimilarity>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>;
}

impl SimilarityPrivate for Model {
    fn similarity_<F>(
        &self,
        embed: Array1<f32>,
        skip: &HashSet<&str>,
        limit: usize,
        mut similarity: F,
    ) -> Vec<WordSimilarity>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>,
    {
        let embedding_matrix = self.embedding_matrix();
        let word_embedding_matrix = embedding_matrix.slice(s![0..self.vocab().len(), ..]);
        let sims = similarity(word_embedding_matrix, embed.view());

        let mut results = BinaryHeap::with_capacity(limit);
        for (idx, &sim) in sims.iter().enumerate() {
            let word = self.vocab().words()[idx].word();

            // Don't add words that we are explicitly asked to skip.
            if skip.contains(word) {
                continue;
            }

            let word_similarity = WordSimilarity {
                word: word,
                similarity: NotNaN::new(sim).expect("Encountered NaN"),
            };

            if results.len() < limit {
                results.push(word_similarity);
            } else {
                let mut peek = results.peek_mut().expect("Cannot peek non-empty heap");
                if word_similarity < *peek {
                    *peek = word_similarity
                }
            }
        }

        results.into_sorted_vec()
    }
}
