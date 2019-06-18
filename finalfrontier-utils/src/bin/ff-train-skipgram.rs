use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use finalfrontier::{
    SentenceIterator, SkipgramTrainer, SubwordVocab, SubwordVocabConfig, Vocab, VocabBuilder,
    WriteModelBinary, SGD,
};
use finalfrontier_utils::{show_progress, thread_data_text, FileProgress, SkipGramApp};
use rand::{FromEntropy, Rng};
use rand_xorshift::XorShiftRng;
use serde::Serialize;
use stdinout::OrExit;

const PROGRESS_UPDATE_INTERVAL: u64 = 200;

fn main() {
    let app = SkipGramApp::new();
    let corpus = app.corpus();
    let n_threads = app.n_threads();
    let common_config = app.common_config();
    let vocab = build_vocab(app.vocab_config(), corpus);
    let mut output_writer = BufWriter::new(
        File::create(app.output()).or_exit("Cannot open output file for writing.", 1),
    );
    let trainer = SkipgramTrainer::new(
        vocab,
        XorShiftRng::from_entropy(),
        common_config,
        app.skipgram_config(),
    );
    let sgd = SGD::new(trainer.into());

    let mut children = Vec::with_capacity(n_threads);
    for thread in 0..n_threads {
        let corpus = corpus.to_owned();
        let sgd = sgd.clone();

        children.push(thread::spawn(move || {
            do_work(
                corpus,
                sgd,
                thread,
                n_threads,
                common_config.epochs,
                common_config.lr,
            );
        }));
    }

    show_progress(
        &common_config,
        &sgd,
        Duration::from_millis(PROGRESS_UPDATE_INTERVAL),
    );

    // Wait until all threads have finished.
    for child in children {
        let _ = child.join();
    }

    sgd.into_model()
        .write_model_binary(&mut output_writer)
        .or_exit("Cannot write model", 1);
}

fn do_work<P, R, V>(
    corpus_path: P,
    mut sgd: SGD<SkipgramTrainer<R, V>>,
    thread: usize,
    n_threads: usize,
    epochs: u32,
    start_lr: f32,
) where
    P: Into<PathBuf>,
    R: Clone + Rng,
    V: Vocab<VocabType = String>,
    V::Config: Serialize,
    for<'a> &'a V::IdxType: IntoIterator<Item = u64>,
{
    let n_tokens = sgd.model().input_vocab().n_types();

    let f = File::open(corpus_path.into()).or_exit("Cannot open corpus for reading", 1);
    let (data, start) =
        thread_data_text(&f, thread, n_threads).or_exit("Could not get thread-specific data", 1);

    let mut sentences = SentenceIterator::new(&data[start..]);
    while sgd.n_tokens_processed() < epochs as usize * n_tokens {
        let sentence = if let Some(sentence) = sentences.next() {
            sentence
        } else {
            sentences = SentenceIterator::new(&*data);
            sentences
                .next()
                .or_exit("Iterator does not provide sentences", 1)
        }
        .or_exit("Cannot read sentence", 1);

        let lr = (1.0 - (sgd.n_tokens_processed() as f32 / (epochs as usize * n_tokens) as f32))
            * start_lr;

        sgd.update_sentence(&sentence, lr);
    }
}

fn build_vocab<P>(config: SubwordVocabConfig, corpus_path: P) -> SubwordVocab
where
    P: AsRef<Path>,
{
    let f = File::open(corpus_path).or_exit("Cannot open corpus for reading", 1);
    let file_progress = FileProgress::new(f).or_exit("Cannot create progress bar", 1);

    let sentences = SentenceIterator::new(BufReader::new(file_progress));

    let mut builder: VocabBuilder<SubwordVocabConfig, String> = VocabBuilder::new(config);
    for sentence in sentences {
        let sentence = sentence.or_exit("Cannot read sentence", 1);

        for token in sentence {
            builder.count(token);
        }
    }

    builder.into()
}
