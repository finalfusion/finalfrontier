use finalfrontier::{BucketConfig, NGramConfig, SimpleVocabConfig, SubwordVocabConfig};

#[derive(Copy, Clone)]
pub enum VocabConfig {
    SubwordVocab(SubwordVocabConfig<BucketConfig>),
    NGramVocab(SubwordVocabConfig<NGramConfig>),
    SimpleVocab(SimpleVocabConfig),
}
