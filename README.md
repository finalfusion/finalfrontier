[![Crate](https://img.shields.io/crates/v/finalfrontier.svg)](https://crates.io/crates/finalfrontier)
[![Docs](https://docs.rs/finalfrontier/badge.svg)](https://docs.rs/finalfrontier/)
[![Build Status](https://travis-ci.org/danieldk/finalfrontier.svg?branch=master)](https://travis-ci.org/danieldk/finalfrontier)

# finalfrontier

## Introduction

finalfrontier is a Rust library and set of utilities for learning and using
word embeddings. finalfrontier currently has the following features:

  * Models:
    - skip-gram (Mikolov et al., 2013)
    - structured skip-gram (Ling et al., 2015)
  * Noise contrastive estimation (Gutmann and Hyvärinen, 2012)
  * Subword representations (Bojanowski et al., 2016)
  * Hogwild SGD (Recht et al., 2011)

This is an early release of finalfrontier, we are planning to add more features
in the future.

## Where to go from here

  * [Installation](docs/INSTALL.md)
  * [Quickstart](docs/QUICKSTART.md)
  * Manual pages:
    - [ff-convert(1)](man/ff-convert.1.md) — convert finalfrontier models to other formats
    - [ff-format(5)](man/ff-format.5.md) — finalfrontier binary format
    - [ff-similar(1)](man/ff-similar.1.md) — word similarity queries
    - [ff-train(1)](man/ff-train.1.md) — train word embeddings with subword representations
  * [Python module](https://github.com/danieldk/finalfrontier-python)
