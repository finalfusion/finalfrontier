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
    - [ff-convert(1)](man/ff-convert.1) — convert finalfrontier models to other formats
    - [ff-similar(1)](man/ff-similar.1) — word similarity queries
    - [ff-train(1)](man/ff-train.1) — train word embeddings with subword representations
