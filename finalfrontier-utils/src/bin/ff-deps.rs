use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use conllx::graph::Node;
use conllx::io::{ReadSentence, Reader};
use finalfrontier::{
    DepembedsConfig, DepembedsTrainer, Dependency, DependencyIterator, SimpleVocab,
    SimpleVocabConfig, SubwordVocab, SubwordVocabConfig, Vocab, VocabBuilder, WriteModelBinary,
    SGD,
};
use finalfrontier_utils::{show_progress, thread_data_conllx, DepembedsApp, FileProgress};
use rand::{FromEntropy, Rng};
use rand_xorshift::XorShiftRng;
use stdinout::OrExit;

const PROGRESS_UPDATE_INTERVAL: u64 = 200;

fn main() {
    let app = DepembedsApp::new();
    let corpus = app.corpus();
    let common_config = app.common_config();
    let n_threads = app.n_threads();
    let (input_vocab, output_vocab) = build_vocab(
        app.input_vocab_config(),
        app.output_vocab_config(),
        app.depembeds_config(),
        corpus,
    );

    let mut output_writer = BufWriter::new(
        File::create(app.output()).or_exit("Cannot open output file for writing.", 1),
    );
    let trainer = DepembedsTrainer::new(
        input_vocab,
        output_vocab,
        app.common_config(),
        app.depembeds_config(),
        XorShiftRng::from_entropy(),
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
        &app.common_config(),
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

fn do_work<P, R>(
    corpus_path: P,
    mut sgd: SGD<DepembedsTrainer<R>>,
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
        thread_data_conllx(&f, thread, n_threads).or_exit("Could not get thread-specific data", 1);

    let mut sentences = Reader::new(BufReader::new(&data[start..]));
    while sgd.n_tokens_processed() < epochs as usize * n_tokens {
        let sentence = if let Some(sentence) = sentences
            .read_sentence()
            .or_exit("Could not read sentence", 1)
        {
            sentence
        } else {
            sentences = Reader::new(BufReader::new(&*data));
            sentences
                .read_sentence()
                .unwrap()
                .or_exit("Iterator does not provide sentences", 1)
        };
        let lr = (1.0 - (sgd.n_tokens_processed() as f32 / (epochs as usize * n_tokens) as f32))
            * start_lr;
        sgd.update_sentence(&sentence, lr);
    }
}

fn build_vocab<P>(
    input_config: SubwordVocabConfig,
    output_config: SimpleVocabConfig,
    dep_config: DepembedsConfig,
    corpus_path: P,
) -> (SubwordVocab, SimpleVocab<Dependency>)
where
    P: AsRef<Path>,
{
    let f = File::open(corpus_path).or_exit("Cannot open corpus for reading", 1);
    let file_progress = FileProgress::new(f).or_exit("Cannot create progress bar", 1);
    let mut input_builder: VocabBuilder<_, String> = VocabBuilder::new(input_config);
    let mut output_builder: VocabBuilder<_, Dependency> = VocabBuilder::new(output_config);

    for sentence in Reader::new(BufReader::new(file_progress)) {
        let sentence = sentence.or_exit("Cannot read sentence", 1);
        for token in sentence.iter().filter_map(Node::token) {
            input_builder.count(token.form());
        }

        for (_, context) in DependencyIterator::new_from_config(&sentence.dep_graph(), dep_config) {
            output_builder.count(context);
        }
    }

    (input_builder.into(), output_builder.into())
}