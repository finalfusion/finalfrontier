# Quickstart

Train a model with 300-dimensional word embeddings, the structured skip-gram
model, discarding words that occur fewer than 10 times:


    ff-train --dims 300 --model structgram --epochs 10 --mincount 10 \
      --threads 16 corpus.txt corpus-embeddings.bin

The format of the input file is simple: tokens are separated by spaces,
sentences by newlines (`\n`).

After training, you can query for similar words:

    ff-similar corpus-embeddings.bin

For interoperability, you can also convert the embedding to various formats:

    ff-convert -f text corpus-embedings.bin corpus-embeddings.txt
    ff-convert -f textdims corpus-embedings.bin corpus-embeddings.txt
    ff-convert -f word2vec corpus-embedings.bin corpus-embeddings.w2v

However, this conversion does not include subword representations, so the
output can only be used to look up embeddings of known words.
