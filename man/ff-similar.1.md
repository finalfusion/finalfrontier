% FF-SIMILAR(1)
% Daniel de Kok
% Sep 9, 2018

NAME
====

**ff-similar** -- word similarity queries

SYNOPSIS
========

**ff-similar** [*options*] *model*

DESCRIPTION
===========

*ff-similar* reads an embedding model and then reads words from the standard
input. For each word it will then print the k nearest neighbors and their
cosine similarity to the given word.

OPTIONS
=======

`-k` *K*

:   Print the *K* nearest neighbors. Default: 10

SEE ALSO
========

ff-compute-accuracy(1), ff-convert(1), ff-format(5), ff-train(1)
