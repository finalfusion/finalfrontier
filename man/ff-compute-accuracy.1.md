% FF-COMPUTE-ACCURACY(1)
% Daniel de Kok
% Jan 27, 2019

NAME
====

**ff-compute-accuracy** -- evaluate a model with analogies

SYNOPSIS
========

**ff-compute-accuracy** [*options*] *model* [*analogies*]

DESCRIPTION
===========

*ff-convert* evaluates a model using analogies of the form '*a* is to *a\**
as *b* is to *b\**'. The model is given *a*, *a\**, and *b* and should
predict *b\**. This utility is similar to word2vec's *compute-accuracy*,
however the rules are more strict:

1. The model's full vocabulary is used during evaluation.
2. Evaluation instances are only skipped when *b\** is absent in the
   vocabulary. If the model is not able to handle one of the query
   tokens, this is counted as an erroneous prediction of the model.
3. The case of a token's characters are reserved.

File format
===========

An analogies file consists of sections followed by analogies. For
example:

~~~
: capital-common-countries
Athens Greece Baghdad Iraq
Athens Greece Bangkok Thailand
: city-in-state
Chicago Illinois Houston Texas
Chicago Illinois Philadelphia Pennsylvania
~~~

Section identifiers are preceded by a colon and a space.

OPTIONS
=======

`--threads` *N*

:   The number of thread to use during evaluation for
    parallelization. The default is to use half of the logical CPUs of
    the machine.


EXAMPLES
========

Evaluate *wikipedia.bin* using *questions-words.txt*:

    ff-compute-accuracy wikipedia.bin questions-words.txt

SEE ALSO
========

ff-convert, ff-format(5), ff-similar(1), ff-train(1)
