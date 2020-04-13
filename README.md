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
    - directional skip-gram (Song et al., 2018)
    - dependency (Levy and Goldberg, 2014)
  * Noise contrastive estimation (Gutmann and Hyvärinen, 2012)
  * Subword representations (Bojanowski et al., 2016)
  * Hogwild SGD (Recht et al., 2011)
  * Quantized embeddings through the [`finalfusion
    quantize`](https://github.com/finalfusion/finalfusion-utils)
    command.

The trained embeddings are stored in `finalfusion` format, which can
be read and used with the
[finalfusion](https://github.com/finalfusion/finalfusion-rust) crate
and the
[finalfusion](https://github.com/finalfusion/finalfusion-python)
Python module.

The minimum required Rust version is currently 1.40.

## Where to go from here

  * [Installation](docs/INSTALL.md)
  * [Quickstart](docs/QUICKSTART.md)
  * Manual pages:
    - [finalfrontier-skipgram(1)](man/finalfrontier-skipgram.1.md) — train word
      embeddings with the (structured) skip-gram model
    - [finalfrontier-deps(1)](man/finalfrontier-deps.1.md) — train word embeddings with dependency contexts
  * [finalfusion crate](https://github.com/finalfusion/finalfusion-rust)
  * [Python module](https://github.com/finalfusion/finalfusion-python)
