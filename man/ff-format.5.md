% FF-FORMAT(5)
% Daniel de Kok
% Sep 15, 2018

NAME
====

**ff-format** -- finalfrontier binary format

DESCRIPTION
===========

The finalfrontier binary format stores stores word embeddings. The format
contains the following parts:

1. Header
2. Configuration
3. Vocabulary
4. Embedding matrix
5. L2 norms

All numerical values are stored in *little endian* format. The remainder of
this page describes each part of file version *3* in more detail.

HEADER
======

| **Field**    | **Size** | Notes                    |
|--------------|----------|--------------------------|
| Magic number |        2 | The magic number is 'FF' |
| Version      |        4 | File format version (3)  |

CONFIGURATION
=============

| **Field**             | **Type** | **Notes**                  |
|-----------------------|----------|----------------------------|
| Model type            |       u8 | 0: skipgram, 1: structgram |
| Loss type             |       u8 | 0: Logistic NS             |
| Context size          |      u32 |                            |
| Dimensionality        |      u32 |                            |
| Discard threshold     |      f32 |                            |
| Epochs                |      u32 |                            |
| Min count             |      u32 |                            |
| Min N                 |      u32 | Minimal subword length     |
| Max N                 |      u32 | Maximal subword length     |
| Buckets EXP           |      u32 | 2^EXP buckets for subwords |
| Negative samples      |      u32 |                            |
| Initial learning rate |      f32 |                            |

VOCABULARY
==========

| **Field**             | **Type** | **Notes**                  |
|-----------------------|----------|----------------------------|
| No. of corpus tokens  |      u64 |                            |
| Vocabulary size       |      u64 |                            |
| Word length           |      u32 | See below                  |
| Word                  |    UTF-8 | See below                  |
| Word count            |      u64 | See below                  |

* The last three fields are repeated for every word in the vocabulary.
* The vocabulary is ordered by token frequency. The ordering of tokens that are
  equally frequent is undefined.

EMBEDDING MATRIX
================

| **Field**          | **Type**                          |
|--------------------|-----------------------------------|
| Word embeddings    | vocab size * dimensionality * f32 |
| Subword embeddings | 2^EXP * dimensionality * f32      |

* The word embeddings are normalized to unit vectors.
* The word embeddings are in vocabulary order.
* The subword embeddings are unnormalized.

L2 NORMS
========

| **Field** | **Type**        |
|-----------|-----------------|
| l2 norms | vocab size * f32 |

The original unnormalized word embeddings can be obtained by multiplying each
embedding by the corresponding l2 norm.

SEE ALSO
========

ff-convert(1), ff-similar(1), ff-train(1)
