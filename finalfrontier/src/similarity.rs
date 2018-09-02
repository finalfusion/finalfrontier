use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis};
use ordered_float::NotNaN;

use super::Model;

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

/// A similarity model.
///
/// A similarity model uses an existing `Model`, adding an embedding
/// matrix of l2 normalized word embeddings.
pub struct Similarity<'a> {
    model: &'a Model,
    word_embeds: Array2<f32>,
}

impl<'a> From<&'a Model> for Similarity<'a> {
    fn from(model: &'a Model) -> Self {
        let vocab = model.vocab();

        let mut word_embeds = Array2::zeros((vocab.len(), model.config().dims as usize));
        for word in vocab.words() {
            let mut embed = model
                .embedding(word.word())
                .expect("Word without an embedding");
            l2_normalize_vector(embed.view_mut());
            word_embeds
                .subview_mut(Axis(0), vocab.word_idx(word.word()).unwrap() as usize)
                .assign(&embed);
        }

        Similarity { model, word_embeds }
    }
}

impl<'a> Similarity<'a> {
    /// Perform an analogy query.
    ///
    /// This method returns words that are close in vector space the analogy
    /// query `word1` is to `word2` as `word3` is to `?`. More concretely,
    /// it searches embeddings that are similar to:
    ///
    /// *embedding(word2) - embedding(word1) + embedding(word3)*
    ///
    /// At most, `limit` results are returned.
    pub fn analogy(
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

    /// Perform an analogy query using the given similarity function.
    ///
    /// This method returns words that are close in vector space the analogy
    /// query `word1` is to `word2` as `word3` is to `?`. More concretely,
    /// it searches embeddings that are similar to:
    ///
    /// *embedding(word2) - embedding(word1) + embedding(word3)*
    ///
    /// At most, `limit` results are returned.
    pub fn analogy_by<F>(
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
        let mut embedding1 = self.model.embedding(word1)?;
        l2_normalize_vector(embedding1.view_mut());

        let mut embedding2 = self.model.embedding(word2)?;
        l2_normalize_vector(embedding2.view_mut());

        let mut embedding3 = self.model.embedding(word3)?;
        l2_normalize_vector(embedding3.view_mut());

        let mut embedding = (embedding2 - embedding1) + embedding3;
        l2_normalize_vector(embedding.view_mut());

        let skip = [word1, word2, word3].iter().cloned().collect();

        Some(self.similarity_(embedding, &skip, limit, similarity))
    }

    /// Find words that are similar to the query word.
    ///
    /// The similarity between two words is defined by the dot product of
    /// the embeddings. If the vectors are unit vectors (e.g. by virtue of
    /// calling `normalize`), this is the cosine similarity. At most, `limit`
    /// results are returned.
    pub fn similarity(&self, word: &str, limit: usize) -> Option<Vec<WordSimilarity>> {
        self.similarity_by(word, limit, |embeds, embed| embeds.dot(&embed))
    }

    /// Find words that are similar to the query word using the given similarity
    /// function.
    ///
    /// The similarity function should return, given the embeddings matrix and
    /// the word vector a vector of similarity scores. At most, `limit` results
    /// are returned.
    pub fn similarity_by<F>(
        &self,
        word: &str,
        limit: usize,
        similarity: F,
    ) -> Option<Vec<WordSimilarity>>
    where
        F: FnMut(ArrayView2<f32>, ArrayView1<f32>) -> Array1<f32>,
    {
        let mut embed = self.model.embedding(word)?;
        l2_normalize_vector(embed.view_mut());
        let mut skip = HashSet::new();
        skip.insert(word);

        Some(self.similarity_(embed, &skip, limit, similarity))
    }

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
        let sims = similarity(self.word_embeds.view(), embed.view());

        let mut results = BinaryHeap::with_capacity(limit);
        for (idx, &sim) in sims.iter().enumerate() {
            let word = self.model.vocab().words()[idx].word();

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

fn l2_normalize_vector(mut v: ArrayViewMut1<f32>) {
    let l2norm = v.dot(&v).sqrt();
    if l2norm != 0f32 {
        v /= l2norm;
    }
}
