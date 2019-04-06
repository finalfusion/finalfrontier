#!/bin/bash

set -ex

cargo build --target ${TARGET}
cargo test --target ${TARGET}

# On Rust 1.31.0, we only care about passing tests.
if [ ! rustc --version | grep "^rustc 1.31.0" ]; then
  cargo fmt --all -- --check
  cargo clippy -- -D warnings
fi


