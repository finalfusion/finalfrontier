% FINALFRONTIER-DEPS(1) % Daniel de Kok, Sebastian Pütz % Apr 6, 2019

NAME
====

**finalfrontier deps** -- train dependency-based word embeddings with subword
representations

SYNOPSIS
========

**finalfrontier deps** [*options*] *corpus* *output*

DESCRIPTION
===========

The **finalfrontier-deps** subcommand trains dependency based word embeddings
(Levy and Goldberg, 2014) using data from a *corpus* in CONLL-U format. The
corpus contains sentences seperated by empty lines. Each sentence needs to be
annotated with a dependency graph. After training, the embeddings are written
to *output* in the finalfusion format.

OPTIONS
=======

`--buckets` *EXP*

:   The bucket exponent. finalfrontier uses 2^*EXP* buckets to store subword
representations. Each subword representation (n-gram) is hashed and mapped to a
bucket based on this hash. Using more buckets will result in fewer bucket
collisions between subword representations at the cost of memory use. The
default bucket exponent is *21* (approximately 2 million buckets).

`--context-discard` *THRESHOLD*

:   The context discard threshold influences how often frequent contexts are
discarded during training. The default context discard threshold is *1e-4*.
    
`--context-mincount` *FREQ*

:   The minimum count controls discarding of infrequent contexts. Contexts
occuring fewer than *FREQ* times are not considered during training.  The
default minimum count is 5.

`--context-target-size` *SIZE*

:   The target size for the context vocabulary. At most *SIZE* contexts are
    included for training. Only contexts appearing more frequently than the
    context at *SIZE* are included.

`--dependency-depth` *DEPTH*

:   Dependency contexts up to *DEPTH* distance from the focus word in the
dependency graph will be used to learn the representation of the focus word. The
default depth is *1*.

`--dims` *DIMS*

:   The dimensionality of the trained word embeddings. The default
dimensionality is 300.

`--discard` *THRESHOLD*

:   The discard threshold influences how often frequent focus words are
discarded from training. The default discard threshold is *1e-4*.

`--epochs` *N*

:   The number of training epochs. The number of necessary training epochs
typically decreases with the corpus size. The default number of epochs is *15*.

`-f`, `--format` *FORMAT*

:   The output format. This must be one of *fasttext*, *finalfusion*,
	*word2vec*, *text*, and *textdims*.

	All formats, except *finalfusion*, result in a loss of
	information: *word2vec*, *text*, and *textdims* do not store
	subword embeddings, nor hyperparameters. The *fastText* format
	does not store all hyperparemeters.

	The *fasttext* format can only be used in conjunction with
    `--subwords buckets` and `--hash-indexer fasttext`.

`--lr` *LEARNING_RATE*

:   The learning rate determines what fraction of a gradient is used for
parameter updates. The default initial learning rate is *0.05*, the learning
rate decreases monotonically during training.

`--maxn` *LEN*

:   The maximum n-gram length for subword representations. Default: 6

`--mincount` *FREQ*

:   The minimum count controls discarding of infrequent focus words. Focus words
occuring fewer than *FREQ* times are not considered during training. The default
minimum count is 5.

`--target-size` *SIZE*

:   The target size for the token vocabulary. At most *SIZE* tokens are
    included for training. Only tokens appearing more frequently than the token
    at *SIZE* are included.

`--minn` *LEN*

:   The minimum n-gram length for subword representations. Default: 3

`--ngram-mincount` *FREQ*

:   The minimum n-gram frequency. n-grams occurring fewer than *FREQ*
    times are excluded from training. This option is only applicable
    with the *ngrams* argument of the `subwords` option.

`--ngram-target-size` *SIZE*

:   The target size for the n-gram vocabulary. At most *SIZE* n-ngrams are
    included for training. Only n-grams appearing more frequently than the
    n-gram at *SIZE* are included. This option is only applicable with the
    *ngrams* argument of the `subwords` option.

`--normalize-contexts`

:   Normalize the attached form in the dependency contexts.

`--ns` *FREQ*

:   The number of negatives to sample per positive example. Default: 5

`--projectivize`

:   Projectivize dependency graphs before training embeddings.

`--threads` *N*

:   The number of thread to use during training for
    parallelization. The default is to use half of the logical CPUs of
    the machine, capped at 20 threads. Increasing the number of
    threads increases the probability of update collisions, requiring
    more epochs to reach the same loss.
    
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

`--untyped-deps`

:   Only use the word of the attached token in the dependency relation as
contexts to train the representation of the focus word.
    
`--use-root`

:   Include the abstract root node in the dependency graph as contexts during
training.

`--zipf` *EXP*

:   Exponent *s* used in the Zipf distribution `p(k) = 1 / (k^s H_N)` for
negative sampling. Default: 0.5

EXAMPLES 
========

Train embeddings on *dewiki.txt* using the dependency model with default
parameters:

    finalfrontier deps dewiki.conll dewiki-deps.bin

Train embeddings with dimensionality 200 on *dewiki.conll* using the dependency
model from contexts with depth up to 2:

    finalfrontier deps --depth 2 --normalize --dims 200 \
      dewiki.conll dewiki-deps.bin

SEE ALSO
========

`finalfrontier`(1), `finalfrontier-skipgram`(1)
