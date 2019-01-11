extern crate failure;

extern crate indicatif;

extern crate memmap;

mod data;
pub use data::thread_data;

mod progress;
pub use crate::progress::FileProgress;
