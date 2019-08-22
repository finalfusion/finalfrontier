mod config;
pub use crate::config::{
    CommonConfig, DepembedsConfig, LossType, ModelType, SimpleVocabConfig, SkipGramConfig,
    SubwordVocabConfig, VocabCutoff,
};

mod deps;
pub use crate::deps::{DepIter, Dependency, DependencyIterator};

pub(crate) mod dep_trainer;
pub use crate::dep_trainer::DepembedsTrainer;

pub(crate) mod idx;

mod io;
pub use crate::io::{SentenceIterator, WriteModelBinary, WriteModelText, WriteModelWord2Vec};

pub(crate) mod loss;

pub(crate) mod sampling;

mod sgd;
pub use crate::sgd::SGD;

pub(crate) mod subword;

mod train_model;
pub use crate::train_model::{TrainModel, Trainer};

pub(crate) mod skipgram_trainer;
pub use crate::skipgram_trainer::SkipgramTrainer;

pub(crate) mod util;

pub(crate) mod vec_simd;

mod vocab;
pub use crate::vocab::{CountedType, SimpleVocab, SubwordVocab, Vocab, VocabBuilder, Word};
