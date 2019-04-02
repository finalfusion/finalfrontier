use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use finalfrontier::{
    SentenceIterator, SkipgramTrainer, SubwordVocab, SubwordVocabConfig, Vocab, VocabBuilder,
    WriteModelBinary, SGD,
};
use finalfrontier_utils::*;
use rand::{FromEntropy, Rng};
use rand_xorshift::XorShiftRng;
use stdinout::OrExit;

const PROGRESS_UPDATE_INTERVAL: u64 = 200;

fn main() {
    let matches = AppBuilder::build_with_common_opts("final-frontier")
        .add_skipgram_opts()
        .build()
        .get_matches();

    let common_config = common_config_from_matches(&matches);
    let skipgram_config = skipgram_config_from_matches(&matches);
    let vocab_config = subword_config_from_matches(&matches);

    let n_threads = matches
        .value_of("threads")
        .map(|v| v.parse().or_exit("Cannot parse number of threads", 1))
        .unwrap_or(num_cpus::get() / 2);

    let corpus = matches.value_of(CORPUS).unwrap();
    let vocab = build_vocab(vocab_config, corpus);

    let mut output_writer = BufWriter::new(
        File::create(matches.value_of(OUTPUT).unwrap())
            .or_exit("Cannot open output file for writing.", 1),
    );
    let trainer = SkipgramTrainer::new(
        vocab,
        XorShiftRng::from_entropy(),
        common_config,
        skipgram_config,
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

// Argument constants
static CORPUS: &str = "CORPUS";
static OUTPUT: &str = "OUTPUT";

fn do_work<P, R>(
    corpus_path: P,
    mut sgd: SGD<SkipgramTrainer<R>>,
    thread: usize,
    n_threads: usize,
    epochs: u32,
    start_lr: f32,
) where
    P: Into<PathBuf>,
    R: Clone + Rng,
{
    let n_tokens = sgd.model().input_vocab().n_types();

    let f = File::open(corpus_path.into()).or_exit("Cannot open corpus for reading", 1);
    let (data, start) =
        thread_data(&f, thread, n_threads).or_exit("Could not get thread-specific data", 1);

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
