#![feature(stdsimd)]

#[macro_use]
extern crate cfg_if;

extern crate fnv;

#[macro_use]
#[cfg(test)]
extern crate lazy_static;

#[macro_use]
#[cfg(test)]
extern crate maplit;

extern crate ndarray;

extern crate ndarray_rand;

extern crate rand;

mod io;
pub use io::SentenceIterator;

mod config;
pub use config::{Config, LossType, ModelType};

mod subword;
pub use subword::{NGrams, SubwordIndices};

pub(crate) mod util;

pub mod vec_simd;

mod vocab;
pub use vocab::{Token, Vocab, VocabBuilder};
