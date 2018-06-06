extern crate fnv;

#[macro_use]
#[cfg(test)]
extern crate lazy_static;

#[macro_use]
#[cfg(test)]
extern crate maplit;

mod config;
pub use config::{Config, LossType, ModelType};

mod subword;
pub use subword::NGrams;

pub(crate) mod util;
