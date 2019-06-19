use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use conllx::graph::{Node, Sentence};
use conllx::io::{ReadSentence, Reader};
use conllx::proj::{HeadProjectivizer, Projectivize};
use finalfrontier::{
    DepembedsConfig, DepembedsTrainer, Dependency, DependencyIterator, SimpleVocab,
    SimpleVocabConfig, SubwordVocab, SubwordVocabConfig, Vocab, VocabBuilder, WriteModelBinary,
    SGD,
};
use finalfrontier_utils::{show_progress, thread_data_conllx, DepembedsApp, FileProgress};
use rand::{FromEntropy, Rng};
use rand_xorshift::XorShiftRng;
use serde::Serialize;
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

    let projectivize = app.depembeds_config().projectivize;
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
                projectivize,
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

fn do_work<P, R, V>(
    corpus_path: P,
    mut sgd: SGD<DepembedsTrainer<R, V>>,
    thread: usize,
    n_threads: usize,
    epochs: u32,
    start_lr: f32,
    projectivize: bool,
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
        thread_data_conllx(&f, thread, n_threads).or_exit("Could not get thread-specific data", 1);
    let projectivizer = if projectivize {
        Some(HeadProjectivizer::new())
    } else {
        None
    };

    let mut sentences = SentenceIter::new(BufReader::new(&data[start..]), projectivizer);
    while sgd.n_tokens_processed() < epochs as usize * n_tokens {
        let sentence = sentences
            .next()
            .or_else(|| {
                sentences = SentenceIter::new(BufReader::new(&*data), projectivizer);
                sentences.next()
            })
            .or_exit("Cannot read sentence.", 1);

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

    let projectivizer = if dep_config.projectivize {
        Some(HeadProjectivizer::new())
    } else {
        None
    };

    for sentence in SentenceIter::new(BufReader::new(file_progress), projectivizer) {
        for token in sentence.iter().filter_map(Node::token) {
            input_builder.count(token.form());
        }

        for (_, context) in DependencyIterator::new_from_config(&sentence.dep_graph(), dep_config) {
            output_builder.count(context);
        }
    }

    (input_builder.into(), output_builder.into())
}

struct SentenceIter<P, R> {
    inner: Reader<R>,
    projectivizer: Option<P>,
}

impl<P, R> SentenceIter<P, R> {
    fn new(read: R, projectivizer: Option<P>) -> Self
    where
        R: BufRead,
    {
        SentenceIter {
            inner: Reader::new(read),
            projectivizer,
        }
    }
}

impl<P, R> Iterator for SentenceIter<P, R>
where
    P: Projectivize,
    R: BufRead,
{
    type Item = Sentence;

    fn next(&mut self) -> Option<Self::Item> {
        let mut sentence = self
            .inner
            .read_sentence()
            .or_exit("Cannot read sentence", 1)?;
        if let Some(proj) = &self.projectivizer {
            proj.projectivize(&mut sentence)
                .or_exit("Cannot projectivize sentence.", 1);
        }
        Some(sentence)
    }
}
