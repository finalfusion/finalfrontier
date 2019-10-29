use std::cmp;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use clap::{App, Arg, ArgMatches};
use conllx::graph::{Node, Sentence};
use conllx::io::{ReadSentence, Reader};
use conllx::proj::{HeadProjectivizer, Projectivize};
use finalfrontier::io::{thread_data_conllx, FileProgress, TrainInfo};
use finalfrontier::{
    CommonConfig, DepembedsConfig, DepembedsTrainer, Dependency, DependencyIterator, SimpleVocab,
    SimpleVocabConfig, SubwordVocab, Vocab, VocabBuilder, WriteModelBinary, SGD,
};
use finalfusion::prelude::VocabWrap;
use rand::{FromEntropy, Rng};
use rand_xorshift::XorShiftRng;
use serde::Serialize;
use stdinout::OrExit;

use crate::subcommands::{show_progress, FinalfrontierApp, VocabConfig};

static CONTEXT_MINCOUNT: &str = "context_mincount";
static CONTEXT_DISCARD: &str = "context_discard";
static DEPENDENCY_DEPTH: &str = "dependency_depth";
static UNTYPED_DEPS: &str = "untyped";
static NORMALIZE_CONTEXT: &str = "normalize";
static PROJECTIVIZE: &str = "projectivize";
static USE_ROOT: &str = "use_root";

const PROGRESS_UPDATE_INTERVAL: u64 = 200;

/// Dependency embeddings subcommand.
pub struct DepsApp {
    train_info: TrainInfo,
    common_config: CommonConfig,
    depembeds_config: DepembedsConfig,
    input_vocab_config: VocabConfig,
    output_vocab_config: SimpleVocabConfig,
}

impl DepsApp {
    fn depembeds_config_from_matches(matches: &ArgMatches) -> DepembedsConfig {
        let depth = matches
            .value_of(DEPENDENCY_DEPTH)
            .map(|v| v.parse().or_exit("Cannot parse dependency depth", 1))
            .unwrap();
        let untyped = matches.is_present(UNTYPED_DEPS);
        let normalize = matches.is_present(NORMALIZE_CONTEXT);
        let projectivize = matches.is_present(PROJECTIVIZE);
        let use_root = matches.is_present(USE_ROOT);
        DepembedsConfig {
            depth,
            untyped,
            normalize,
            projectivize,
            use_root,
        }
    }

    /// Get the corpus path.
    pub fn corpus(&self) -> &str {
        self.train_info().corpus()
    }

    /// Get the output path.
    pub fn output(&self) -> &str {
        self.train_info().output()
    }

    /// Get the number of threads.
    pub fn n_threads(&self) -> usize {
        self.train_info().n_threads()
    }

    /// Get the common config.
    pub fn common_config(&self) -> CommonConfig {
        self.common_config
    }

    /// Get the depembeds config.
    pub fn depembeds_config(&self) -> DepembedsConfig {
        self.depembeds_config
    }

    /// Get the input vocab config.
    pub fn input_vocab_config(&self) -> VocabConfig {
        self.input_vocab_config
    }

    /// Get the output vocab config.
    pub fn output_vocab_config(&self) -> SimpleVocabConfig {
        self.output_vocab_config
    }

    /// Get the train information.
    pub fn train_info(&self) -> &TrainInfo {
        &self.train_info
    }
}

impl FinalfrontierApp for DepsApp {
    fn app() -> App<'static, 'static> {
        Self::common_opts("deps")
            .about("Train a dependency embeddings model")
            .arg(
                Arg::with_name(CONTEXT_DISCARD)
                    .long("context_discard")
                    .value_name("CONTEXT_THRESHOLD")
                    .help("Context discard threshold")
                    .takes_value(true)
                    .default_value("1e-4"),
            )
            .arg(
                Arg::with_name(CONTEXT_MINCOUNT)
                    .long("context_mincount")
                    .value_name("CONTEXT_FREQ")
                    .help("Context mincount")
                    .takes_value(true)
                    .default_value("5"),
            )
            .arg(
                Arg::with_name(DEPENDENCY_DEPTH)
                    .long("dependency_depth")
                    .value_name("DEPENDENCY_DEPTH")
                    .help("Dependency depth")
                    .takes_value(true)
                    .default_value("1"),
            )
            .arg(
                Arg::with_name(UNTYPED_DEPS)
                    .long("untyped_deps")
                    .help("Don't use dependency relation labels."),
            )
            .arg(
                Arg::with_name(NORMALIZE_CONTEXT)
                    .long("normalize_context")
                    .help("Normalize contexts"),
            )
            .arg(
                Arg::with_name(PROJECTIVIZE)
                    .long("projectivize")
                    .help("Projectivize dependency graphs before training."),
            )
            .arg(
                Arg::with_name(USE_ROOT)
                    .long("use_root")
                    .help("Use root when extracting dependency contexts."),
            )
    }

    fn parse(matches: &ArgMatches) -> Self {
        let corpus = matches.value_of(Self::CORPUS).unwrap().into();
        let output = matches.value_of(Self::OUTPUT).unwrap().into();
        let n_threads = matches
            .value_of(Self::THREADS)
            .map(|v| v.parse().or_exit("Cannot parse number of threads", 1))
            .unwrap_or_else(|| cmp::min(num_cpus::get() / 2, 20));

        let discard_threshold = matches
            .value_of(CONTEXT_DISCARD)
            .map(|v| v.parse().or_exit("Cannot parse discard threshold", 1))
            .unwrap();
        let min_count = matches
            .value_of(CONTEXT_MINCOUNT)
            .map(|v| v.parse().or_exit("Cannot parse mincount", 1))
            .unwrap();

        let output_vocab_config = SimpleVocabConfig {
            min_count,
            discard_threshold,
        };
        let train_info = TrainInfo::new(corpus, output, n_threads);

        DepsApp {
            train_info,
            common_config: Self::parse_common_config(&matches),
            depembeds_config: Self::depembeds_config_from_matches(&matches),
            input_vocab_config: Self::parse_vocab_config(&matches),
            output_vocab_config,
        }
    }

    fn run(&self) {
        match self.input_vocab_config() {
            VocabConfig::SimpleVocab(config) => {
                let (input_vocab, output_vocab) = build_vocab::<_, SimpleVocab<String>, _>(
                    config,
                    self.output_vocab_config(),
                    self.depembeds_config(),
                    self.corpus(),
                );
                train(input_vocab, output_vocab, self);
            }
            VocabConfig::SubwordVocab(config) => {
                let (input_vocab, output_vocab) = build_vocab::<_, SubwordVocab<_, _>, _>(
                    config,
                    self.output_vocab_config(),
                    self.depembeds_config(),
                    self.corpus(),
                );
                train(input_vocab, output_vocab, self);
            }
            VocabConfig::NGramVocab(config) => {
                let (input_vocab, output_vocab) = build_vocab::<_, SubwordVocab<_, _>, _>(
                    config,
                    self.output_vocab_config(),
                    self.depembeds_config(),
                    self.corpus(),
                );
                train(input_vocab, output_vocab, self);
            }
        }
    }
}

fn train<V>(input_vocab: V, output_vocab: SimpleVocab<Dependency>, app: &DepsApp)
where
    V: Vocab<VocabType = String> + Into<VocabWrap> + Clone + Send + Sync + 'static,
    V::Config: Serialize,
    for<'a> &'a V::IdxType: IntoIterator<Item = u64>,
{
    let corpus = app.corpus();
    let common_config = app.common_config();
    let n_threads = app.n_threads();

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
        .write_model_binary(&mut output_writer, app.train_info().clone())
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

fn build_vocab<P, V, C>(
    input_config: C,
    output_config: SimpleVocabConfig,
    dep_config: DepembedsConfig,
    corpus_path: P,
) -> (V, SimpleVocab<Dependency>)
where
    P: AsRef<Path>,
    V: Vocab<VocabType = String> + From<VocabBuilder<C, String>>,
    VocabBuilder<C, String>: Into<V>,
{
    let f = File::open(corpus_path).or_exit("Cannot open corpus for reading", 1);
    let file_progress = FileProgress::new(f).or_exit("Cannot create progress bar", 1);
    let mut input_builder = VocabBuilder::new(input_config);
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
