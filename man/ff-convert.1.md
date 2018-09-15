% FF-CONVERT(1)
% Daniel de Kok
% Sep 9, 2018

NAME
====

**ff-convert** -- convert finalfrontier models to other formats

SYNOPSIS
========

**ff-convert** [*options*] *input* *output*

DESCRIPTION
===========

*ff-convert* converts an embedding file from the finalfrontier format to
other formats.

OPTIONS
=======

`-f`, `--format` *FORMAT*

:   The format to convert to. Currently, the following formats are
    supported:

    * *text*: text format where each line contains a word followed
      by its embedding.
    * *textdims*: text format where each line contains a word followed
      by its embedding. The first line states the number of words
      in the file and the embedding dimensionality.
    * *word2vec*: word2vec binary format.

    The default format is *textdims*.

EXAMPLES
========

Convert finalfrontier embeddings to text format:

    ff-convert -f text \
      corpus-embedings.bin corpus-embeddings.txt

Convert finalfrontier embeddings to text format with the embedding matrix shape
on the first line:

    ff-convert corpus-embedings.bin corpus-embeddings.txt

Convert finalfrontier embeddings to word2vec format:

    ff-convert -f word2vec \
      corpus-embedings.bin corpus-embeddings.w2v

SEE ALSO
========

ff-format(5), ff-similar(1), ff-train(1)
