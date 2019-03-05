# Quickstart

Train a model with 300-dimensional word embeddings, the structured skip-gram
model, discarding words that occur fewer than 10 times:


    ff-train --dims 300 --model structgram --epochs 10 --mincount 10 \
      --threads 16 corpus.txt corpus-embeddings.fifu

The format of the input file is simple: tokens are separated by spaces,
sentences by newlines (`\n`).

After training, you can use and query the embeddings with
[rust2vec](https://github.com/danieldk/rust2vec) and `rust2vec-utils`:

    r2v-similar corpus-embeddings.fifu
