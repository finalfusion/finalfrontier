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

mod subword;
pub use subword::NGrams;

pub(crate) mod util;

pub mod vec_simd;
