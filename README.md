[![Crate](https://img.shields.io/crates/v/finalfrontier.svg)](https://crates.io/crates/finalfrontier)
[![Docs](https://docs.rs/finalfrontier/badge.svg)](https://docs.rs/finalfrontier/)
[![Build Status](https://travis-ci.org/finalfusion/finalfrontier.svg?branch=master)](https://travis-ci.org/finalfusion/finalfrontier)

# finalfrontier

## Introduction

finalfrontier is a Rust library and set of utilities for learning and using
word embeddings. finalfrontier currently has the following features:

  * Models:
    - skip-gram (Mikolov et al., 2013)
    - structured skip-gram (Ling et al., 2015)
    - dependency (Levy and Goldberg, 2014)
  * Noise contrastive estimation (Gutmann and Hyvärinen, 2012)
  * Subword representations (Bojanowski et al., 2016)
  * Hogwild SGD (Recht et al., 2011)

This is an early release of finalfrontier, we are planning to add more
features in the future. The trained embeddings are stored in `finalfusion`
format, which can be read and used with the
[finalfusion](https://github.com/finalfusion/finalfusion-rust) crate and the
[finalfusion](https://github.com/finalfusion/finalfusion-python) Python
module.

## Where to go from here

  * [Installation](docs/INSTALL.md)
  * [Quickstart](docs/QUICKSTART.md)
  * Manual pages:
    - [ff-train-skipgram(1)](man/ff-train-skipgram.1.md) — train word
      embeddings with the (structured) skip-gram model
    - [ff-train-deps(1)](man/ff-train-skipgram.1.md) — train word embeddings with dependency contexts
  * [finalfusion crate](https://github.com/finalfusion/finalfusion-rust)
  * [Python module](https://github.com/finalfusion/finalfusion-python)
