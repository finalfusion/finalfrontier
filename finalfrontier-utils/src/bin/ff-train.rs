extern crate clap;
extern crate finalfrontier;
extern crate finalfrontier_utils;
extern crate indicatif;
extern crate num_cpus;
extern crate rand;
extern crate rand_xorshift;
extern crate stdinout;

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use clap::{App, AppSettings, Arg, ArgMatches};
use finalfrontier::{
    Config, LossType, ModelType, SentenceIterator, TrainModel, Vocab, VocabBuilder,
    WriteModelBinary, SGD,
};
use finalfrontier_utils::thread_data;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{FromEntropy, Rng};
use rand_xorshift::XorShiftRng;
use stdinout::OrExit;

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

const PROGRESS_UPDATE_INTERVAL: u64 = 200;

fn main() {
    let matches = parse_args();

    let config = config_from_matches(&matches);

    let n_threads = matches
        .value_of("threads")
        .map(|v| v.parse().or_exit("Cannot parse number of threads", 1))
        .unwrap_or(num_cpus::get() / 2);

    let vocab = build_vocab(&config, matches.value_of("CORPUS").unwrap());

    let mut output_writer = BufWriter::new(
        File::create(matches.value_of(OUTPUT).unwrap())
            .or_exit("Cannot open output file for writing.", 1),
    );

    let model = TrainModel::from_vocab(vocab, config.clone());
    let sgd = SGD::new(model, XorShiftRng::from_entropy());

    let corpus = matches.value_of(CORPUS).unwrap();

    let mut children = Vec::with_capacity(n_threads);
    for thread in 0..n_threads {
        let corpus = corpus.to_owned();
        let sgd = sgd.clone();

        children.push(thread::spawn(move || {
            do_work(corpus, sgd, thread, n_threads, config.epochs, config.lr);
        }));
    }

    show_progress(
        &config,
        &sgd,
        Duration::from_millis(PROGRESS_UPDATE_INTERVAL),
    );

    // Wait until all threads have finished.
    for child in children {
        let _ = child.join();
    }

    sgd.model()
        .write_model_binary(&mut output_writer)
        .or_exit("Cannot write model", 1);
}

// Option constants
static BUCKETS: &str = "buckets";
static CONTEXT: &str = "context";
static DIMS: &str = "dims";
static DISCARD: &str = "discard";
static EPOCHS: &str = "epochs";
static LR: &str = "lr";
static MINCOUNT: &str = "mincount";
static MINN: &str = "minn";
static MAXN: &str = "maxn";
static MODEL: &str = "model";
static NS: &str = "ns";
static THREADS: &str = "threads";

// Argument constants
static CORPUS: &str = "CORPUS";
static OUTPUT: &str = "OUTPUT";

fn config_from_matches<'a>(matches: &ArgMatches<'a>) -> Config {
    let buckets_exp = matches
        .value_of(BUCKETS)
        .map(|v| v.parse().or_exit("Cannot parse bucket exponent", 1))
        .unwrap_or(21);
    let context_size = matches
        .value_of(CONTEXT)
        .map(|v| v.parse().or_exit("Cannot parse context size", 1))
        .unwrap_or(5);
    let discard_threshold = matches
        .value_of(DISCARD)
        .map(|v| v.parse().or_exit("Cannot parse discard threshold", 1))
        .unwrap_or(1e-4);
    let dims = matches
        .value_of(DIMS)
        .map(|v| v.parse().or_exit("Cannot parse dimensionality", 1))
        .unwrap_or(100);
    let epochs = matches
        .value_of(EPOCHS)
        .map(|v| v.parse().or_exit("Cannot parse number of epochs", 1))
        .unwrap_or(5);
    let lr = matches
        .value_of(LR)
        .map(|v| v.parse().or_exit("Cannot parse learning rate", 1))
        .unwrap_or(0.05);
    let min_count = matches
        .value_of(MINCOUNT)
        .map(|v| v.parse().or_exit("Cannot parse mincount", 1))
        .unwrap_or(5);
    let min_n = matches
        .value_of(MINN)
        .map(|v| v.parse().or_exit("Cannot parse minimum n-gram length", 1))
        .unwrap_or(3);
    let max_n = matches
        .value_of(MAXN)
        .map(|v| v.parse().or_exit("Cannot parse maximum n-gram length", 1))
        .unwrap_or(6);
    let model = matches
        .value_of(MODEL)
        .map(|v| ModelType::try_from_str(v).or_exit("Cannot parse model type", 1))
        .unwrap_or(ModelType::SkipGram);
    let negative_samples = matches
        .value_of(NS)
        .map(|v| {
            v.parse()
                .or_exit("Cannot parse number of negative samples", 1)
        })
        .unwrap_or(5);

    Config {
        context_size,
        dims,
        discard_threshold,
        epochs,
        loss: LossType::LogisticNegativeSampling,
        model,
        min_count,
        min_n,
        max_n,
        buckets_exp,
        negative_samples,
        lr,
    }
}

fn parse_args() -> ArgMatches<'static> {
    App::new("final-frontier")
        .settings(DEFAULT_CLAP_SETTINGS)
        .arg(
            Arg::with_name(BUCKETS)
                .long("buckets")
                .value_name("EXP")
                .help("Number of buckets: 2^EXP (default: 21)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name(CONTEXT)
                .long("context")
                .value_name("CONTEXT_SIZE")
                .help("Context size (default: 5)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name(DIMS)
                .long("dims")
                .value_name("DIMENSIONS")
                .help("Embedding dimensionality (default: 100)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name(DISCARD)
                .long("discard")
                .value_name("THRESHOLD")
                .help("Discard threshold (default: 1e-4)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name(EPOCHS)
                .long("epochs")
                .value_name("N")
                .help("Number of epochs (default: 5)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name(LR)
                .long("lr")
                .value_name("LEARNING_RATE")
                .help("Initial learning rate (default: 0.05)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name(MINCOUNT)
                .long("mincount")
                .value_name("FREQ")
                .help("Minimum token frequency (default: 5)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name(MINN)
                .long("minn")
                .value_name("LEN")
                .help("Minimum ngram length (default: 3)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name(MAXN)
                .long("maxn")
                .value_name("LEN")
                .help("Maximum ngram length (default: 6)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name(MODEL)
                .long(MODEL)
                .value_name("MODEL")
                .help("Model: skipgram or structgram")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("ns")
                .long("ns")
                .value_name("FREQ")
                .help("Negative samples per word (default: 5)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name(THREADS)
                .long("threads")
                .value_name("N")
                .help("Number of threads (default: logical_cpus / 2)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name(CORPUS)
                .help("Tokenized corpus")
                .index(1)
                .required(true),
        )
        .arg(
            Arg::with_name(OUTPUT)
                .help("Embeddings output")
                .index(2)
                .required(true),
        )
        .get_matches()
}

fn show_progress<R>(config: &Config, sgd: &SGD<R>, update_interval: Duration) {
    let n_tokens = sgd.model().vocab().n_tokens();

    let pb = ProgressBar::new(config.epochs as u64 * n_tokens as u64);
    pb.set_style(
        ProgressStyle::default_bar().template("{bar:40} {percent}% {msg} ETA: {eta_precise}"),
    );

    while sgd.n_tokens_processed() < n_tokens * config.epochs as usize {
        let lr = (1.0
            - (sgd.n_tokens_processed() as f32 / (config.epochs as usize * n_tokens) as f32))
            * config.lr;

        pb.set_position(sgd.n_tokens_processed() as u64);
        pb.set_message(&format!(
            "loss: {:.*} lr: {:.*}",
            5,
            sgd.train_loss(),
            5,
            lr
        ));

        thread::sleep(update_interval);
    }

    pb.finish();
}

fn do_work<P, R>(
    corpus_path: P,
    mut sgd: SGD<R>,
    thread: usize,
    n_threads: usize,
    epochs: u32,
    start_lr: f32,
) where
    P: Into<PathBuf>,
    R: Clone + Rng,
{
    let n_tokens = sgd.model().vocab().n_tokens();

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
        }.or_exit("Cannot read sentence", 1);

        let lr = (1.0 - (sgd.n_tokens_processed() as f32 / (epochs as usize * n_tokens) as f32))
            * start_lr;

        sgd.update_sentence(&sentence, lr);
    }
}

fn build_vocab<P>(config: &Config, corpus_path: P) -> Vocab
where
    P: AsRef<Path>,
{
    let f = File::open(corpus_path).or_exit("Cannot open corpus for reading", 1);

    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::default_spinner().template("{spinner} {msg}"));

    let sentences = SentenceIterator::new(BufReader::new(f));

    let mut token_count = 0usize;
    let mut builder = VocabBuilder::new(config.clone());
    for sentence in sentences {
        let sentence = sentence.or_exit("Cannot read sentence", 1);

        let prev_token_count = token_count;

        for token in sentence {
            token_count += 1;
            builder.count(token);
        }

        if prev_token_count % 1_000_000 > token_count % 1_000_000 {
            pb.set_message(&format!("{}M tokens", token_count / 1_000_000));
        }
    }

    pb.finish();

    builder.build()
}
