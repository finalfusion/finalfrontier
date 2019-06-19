% FF-TRAIN-DEPS(1) % Daniel de Kok, Sebastian Pütz % Apr 6, 2019

NAME
====

**ff-train-deps** -- train dependency-based word embeddings with subword
representations

SYNOPSIS
========

**ff-train-deps** [*options*] *corpus* *output*

DESCRIPTION
===========

The **ff-train-deps** trains dependency based word embeddings (Levy and Goldberg,
2014) using data from a *corpus* in CONLL-X format. The corpus contains
sentences seperated by empty lines. Each sentence needs to be annotated with a
dependency graph. After training, the embeddings are written to *output* in the
finalfusion format.

OPTIONS
=======

`--buckets` *EXP*

:   The bucket exponent. finalfrontier uses 2^*EXP* buckets to store subword
representations. Each subword representation (n-gram) is hashed and mapped to a
bucket based on this hash. Using more buckets will result in fewer bucket
collisions between subword representations at the cost of memory use. The
default bucket exponent is *21* (approximately 2 million buckets).

`--context_discard` *THRESHOLD*

:   The context discard threshold influences how often frequent contexts are
discarded during training. The default context discard threshold is *1e-4*.
    
`--context_mincount` *FREQ*

:   The minimum count controls discarding of infrequent contexts. Contexts
occuring fewer than *FREQ* times are not considered during training.  The
default minimum count is 5.
    
`--dims` *DIMS*

:   The dimensionality of the trained word embeddings. The default
dimensionality is 300.

`--dependency_depth` *DEPTH*

:   Dependency contexts up to *DEPTH* distance from the focus word in the
dependency graph will be used to learn the representation of the focus word. The
default depth is *1*.

`--discard` *THRESHOLD*

:   The discard threshold influences how often frequent focus words are
discarded from training. The default discard threshold is *1e-4*.

`--epochs` *N*

:   The number of training epochs. The number of necessary training epochs
typically decreases with the corpus size. The default number of epochs is *15*.

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

`--minn` *LEN*

:   The minimum n-gram length for subword representations. Default: 3

`--normalize_contexts`

:   Normalize the attached form in the dependency contexts.

`--no_subwords`

:   Train embeddings without subword information. This option overrides
arguments for `buckets`, `minn` and `maxn`.

`--ns` *FREQ*

:   The number of negatives to sample per positive example. Default: 5

`--projectivize`

:   Projectivize dependency graphs before training embeddings.

`--threads` *N*

:   The number of thread to use during training for parallelization. The default
is to use half of the logical CPUs of the machine.
    
`--untyped_deps`

:   Only use the word of the attached token in the dependency relation as
contexts to train the representation of the focus word.
    
`--use_root`

:   Include the abstract root node in the dependency graph as contexts during
training.

`--zipf` *EXP*

:   Exponent *s* used in the Zipf distribution `p(k) = 1 / (k^s H_N)` for
negative sampling. Default: 0.5

EXAMPLES 
========

Train embeddings on *dewiki.txt* using the dependency model with default
parameters:

    ff-train-deps dewiki.conll dewiki-deps.bin

Train embeddings with dimensionality 200 on *dewiki.conll* using the dependency
model from contexts with depth up to 2:

    ff-train-deps --depth 2 --normalize --dims 200 \
      dewiki.conll dewiki-deps.bin

