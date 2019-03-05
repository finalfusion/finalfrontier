% FF-TRAIN(1)
% Daniel de Kok
% Sep 8, 2018

NAME
====

**ff-train** -- train word embeddings with subword representations

SYNOPSIS
========

**ff-train** [*options*] *corpus* *output_model*

DESCRIPTION
===========

The **ff-train** trains word embeddings using data from a *corpus*. The corpus
should have tokens separated by spaces and sentences separated by newlines.
After training, the embeddings are written to *output_model*.

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
    the representation of the focus word. The default context size is *5*.

`--discard` *THRESHOLD*

:   The discard threshold influences how often frequent words are discarded
    from training. The default discard threshold is *1e-4*.

`--epochs` *N*

:   The number of training epochs. The number of necessary training epochs
    typically decreases with the corpus size. The default number of epochs
    is *5*.

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

:   The model to use for training word embeddings. The choices here are
    *skipgram* for the skip-gram model (Mikolov et al., 2013) and
    *structgram* for the stuctured skip-gram model (Ling et al. 2015).
    
    The structured skip-gram model takes the position of a context word
    into account and results in embeddings that are typically better
    suited for syntax-oriented tasks.

    The default model is *skipgram*.

`--ns` *FREQ*

:   The number of negatives to sample per positive example. Default: 5

`--threads` *N*

:   The number of thread to use during training for parallelization. The
    default is to use half of the logical CPUs of the machine.

`--zipf` *EXP*

:   Exponent *s* used in the Zipf distribution `p(k) = 1 / (k^s H_N)` for
    negative sampling. Default: 0.5

EXAMPLES
========

Train embeddings on *dewiki.txt* using the skip-gram model:

    ff-train dewiki.txt dewiki-skipgram.bin
	
Train embeddings with dimensionality 100 on *dewiki.txt* using the
structured skip-gram model:

    ff-train --model structgram --dims 100 \
	  dewiki.txt dewiki-structgram.bin

