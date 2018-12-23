extern crate byteorder;

#[macro_use]
extern crate cfg_if;

extern crate conllx;

extern crate failure;

extern crate fnv;

extern crate hogwild;

#[macro_use]
#[cfg(test)]
extern crate lazy_static;

#[macro_use]
#[cfg(test)]
extern crate maplit;

extern crate memmap;

#[macro_use]
extern crate ndarray;

extern crate ndarray_rand;

extern crate ordered_float;

extern crate rand;

#[cfg(test)]
extern crate rand_xorshift;

extern crate zipf;

mod config;
pub use config::{Config, LossType, ModelType};

mod deps;
pub(crate) use deps::{DepIter, DependencyIterator, Dependency, PathIter};

mod io;
pub use io::{
    MmapModelBinary, ReadModelBinary, SentenceIterator, WriteModelBinary, WriteModelText,
    WriteModelWord2Vec,
};

pub(crate) mod loss;

pub mod model;

#[doc(inline)]
pub use model::Model;

pub(crate) mod sampling;

mod sgd;
pub use sgd::SGD;

pub mod similarity;

pub(crate) mod subword;

mod train_model;
pub use train_model::TrainModel;

pub(crate) mod util;

pub(crate) mod vec_simd;

mod vocab;
pub use vocab::{Vocab, VocabBuilder, WordCount};
