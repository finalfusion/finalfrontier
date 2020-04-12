#!/bin/bash

set -ex

cargo build --target ${TARGET}
cargo test --target ${TARGET}

# On Rust 1.40.0, we only care about passing tests.
if [ "$TRAVIS_RUST_VERSION" = "stable" ] && [ "$TARGET" = "x86_64-unknown-linux-gnu" ] ; then
  cargo fmt --all -- --check
  cargo clippy -- -D warnings
fi


