extern crate byteorder;

#[macro_use]
extern crate cfg_if;

extern crate failure;

extern crate fnv;

#[macro_use]
#[cfg(test)]
extern crate lazy_static;

#[macro_use]
#[cfg(test)]
extern crate maplit;

extern crate ndarray;

extern crate ndarray_rand;

extern crate ordered_float;

extern crate rand;

extern crate zipf;

mod config;
pub use config::{Config, LossType, ModelType};

mod hogwild;
pub use hogwild::{Hogwild, HogwildArray, HogwildArray1, HogwildArray2, HogwildArray3};

mod io;
pub use io::{ReadModelBinary, SentenceIterator, WriteModelBinary, WriteModelText};

mod loss;
pub use loss::log_logistic_loss;

mod model;
pub use model::{Model, TrainModel};

mod sampling;
pub use sampling::{RangeGenerator, WeightedRangeGenerator, ZipfRangeGenerator};

mod sgd;
pub use sgd::{NegativeSamplingSGD, SGD};

mod similarity;
pub use similarity::{Similarity, WordSimilarity};

mod subword;
pub use subword::{NGrams, SubwordIndices};

pub(crate) mod util;

pub mod vec_simd;

mod vocab;
pub use vocab::{Type, Vocab, VocabBuilder};
