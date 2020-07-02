% FINALFRONTIER-SKIPGRAM(1)
% Daniel de Kok
% Sep 8, 2018

NAME
====

**finalfrontier skipgram** -- train word embeddings with subword representations

SYNOPSIS
========

**finalfrontier skipgram** [*options*] *corpus* *output*

DESCRIPTION
===========

The **finalfrontier skipgram** subcommand trains word embeddings using data
from a *corpus*. The corpus should have tokens separated by spaces and
sentences separated by newlines. After training, the embeddings are written to
*output* in the finalfusion format.

OPTIONS
=======

`--buckets` *EXP*

:   The bucket exponent. finalfrontier uses 2^*EXP* buckets to store subword
    representations. Each subword representation (n-gram) is hashed and
    mapped to a bucket based on this hash. Using more buckets will result
    in fewer bucket collisions between subword representations at the cost
    of memory use. The default bucket exponent is *21* (approximately 2
    million buckets).

`--context` *CONTEXT_SIZE*

:   Words within the *CONTEXT_SIZE* of a focus word will be used to learn
    the representation of the focus word. The default context size is *10*.

`--dims` *DIMENSIONS*

:   The dimensionality of the trained word embeddings. The default
    dimensionality is 300.

`--discard` *THRESHOLD*

:   The discard threshold influences how often frequent words are discarded
    from training. The default discard threshold is *1e-4*.

`--epochs` *N*

:   The number of training epochs. The number of necessary training epochs
    typically decreases with the corpus size. The default number of epochs
    is *15*.

`-f`, `--format` *FORMAT*

:   The output format. This must be one of *fasttext*, *finalfusion*,
    *word2vec*, *text*, and *textdims*.

    All formats, except *finalfusion*, result in a loss of
    information: *word2vec*, *text*, and *textdims* do not store
    subword embeddings, nor hyperparameters. The *fastText* format
    does not store all hyperparemeters.

    The *fasttext* format can only be used in conjunction with
    `--subwords buckets` and `--hash-indexer fasttext`.

`--hash-indexer` *INDEXER*

:   The indexer to use when bucket-based subwords are used (see
    `--subwords`). The possible values are *finalfusion* or
    *fasttext*. Default: finalfusion

    *finalfusion* uses the FNV-1a hasher, whereas *fasttext* emulates
    the (broken) implementation of FNV-1a in fastText. Use of
    *finalfusion* is recommended, unless the resulting embeddings
    should be compatible with fastText.

`--lr` *LEARNING_RATE*

:   The learning rate determines what fraction of a gradient is used for
    parameter updates. The default initial learning rate is *0.05*, the
    learning rate decreases monotonically during training.

`--maxn` *LEN*

:   The maximum n-gram length for subword representations. Default: 6

`--mincount` *FREQ*

:   The minimum count controls discarding of infrequent. Words occuring
    fewer than *FREQ* times are not considered during training. The
    default minimum count is 5.

`--minn` *LEN*

:   The minimum n-gram length for subword representations. Default: 3

`--model` *MODEL*

:   The model to use for training word embeddings. The choices here are:
    *dirgram* for the directional skip-gram model (Song et al., 2018),
    *skipgram* for the skip-gram model (Mikolov et al., 2013), and
    *structgram* for the stuctured skip-gram model (Ling et al. 2015).
    
    The structured skip-gram model takes the position of a context word
    into account and results in embeddings that are typically better
    suited for syntax-oriented tasks.

    The dependency embeddings model is supported by the separate
    `finalfrontier deps`(1) subcommand.

    The default model is *skipgram*.

`--ngram-mincount` *FREQ*

:   The minimum n-gram frequency. n-grams occurring fewer than *FREQ*
    times are excluded from training. This option is only applicable
    with the *ngrams* argument of the `subwords` option.

`--ngram-target-size` *SIZE*

:   The target size for the n-gram vocabulary. At most *SIZE* n-ngrams are
    included for training. Only n-grams appearing more frequently than the
    n-gram at *SIZE* are included. This option is only applicable with the
    *ngrams* argument of the `subwords` option.

`--ns` *FREQ*

:   The number of negatives to sample per positive example. Default: 5

`--subwords` *SUBWORDS*

:   The type of subword embeddings to train. The possible types are
    *buckets*, *ngrams*, and *none*. Subword embeddings are used to
    compute embeddings for unknown words by summing embeddings of
    n-grams within unknown words.

    The *none* type does not use subwords. The resulting model will
    not be able assign an embeddings to unknown words.

    The *ngrams* type stores subword n-grams explicitly. The included
    n-gram lengths are specified using the `minn` and `maxn`
    options. The frequency threshold for n-grams is configured with
    the `ngram-mincount` option.

    The *buckets* type maps n-grams to buckets using the FNV1 hash.
    The considered n-gram lengths are specified using the `minn` and
    `maxn` options.  The number of buckets is controlled with the
    `buckets` option.

`--target-size` *SIZE*

:   The target size for the token vocabulary. At most *SIZE* tokens are
    included for training. Only tokens appearing more frequently than the token
    at *SIZE* are included.

`--threads` *N*

:   The number of thread to use during training for
    parallelization. The default is to use half of the logical CPUs of
    the machine, capped at 20 threads. Increasing the number of
    threads increases the probability of update collisions, requiring
    more epochs to reach the same loss.

`--zipf` *EXP*

:   Exponent *s* used in the Zipf distribution `p(k) = 1 / (k^s H_N)` for
    negative sampling. Default: 0.5

EXAMPLES
========

Train embeddings on *dewiki.txt* using the skip-gram model:

    finalfrontier skipgram dewiki.txt dewiki-skipgram.bin

Train embeddings with dimensionality 200 on *dewiki.txt* using the
structured skip-gram model with a context window of 5 tokens:

    finalfrontier skipgram --model structgram --context 5 --dims 200 \
      dewiki.txt dewiki-structgram.bin

SEE ALSO
========

`finalfrontier`(1), `finalfrontier-deps`(1)
