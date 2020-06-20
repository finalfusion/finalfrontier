use std::cmp;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{App, Arg, ArgMatches};
use conllu::graph::{Node, Sentence};
use conllu::io::{ReadSentence, Reader, Sentences};
use conllu::proj::{HeadProjectivizer, Projectivize};
use finalfrontier::io::{thread_data_conllu, FileProgress, TrainInfo};
use finalfrontier::{
    BucketIndexerType, CommonConfig, Cutoff, DepembedsConfig, DepembedsTrainer, Dependency,
    DependencyIterator, SimpleVocab, SimpleVocabConfig, SubwordVocab, Vocab, VocabBuilder,
    WriteModelBinary, SGD,
};
use finalfusion::compat::fasttext::FastTextIndexer;
use finalfusion::prelude::VocabWrap;
use finalfusion::subword::FinalfusionHashIndexer;
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;
use serde::Serialize;

use crate::subcommands::{cutoff_from_matches, show_progress, FinalfrontierApp, VocabConfig};

static CONTEXT_MINCOUNT: &str = "context-mincount";
static CONTEXT_TARGET_SIZE: &str = "context-target-size";
static CONTEXT_DISCARD: &str = "context-discard";
static DEPENDENCY_DEPTH: &str = "dependency-depth";
static UNTYPED_DEPS: &str = "untyped";
static NORMALIZE_CONTEXT: &str = "normalize";
static PROJECTIVIZE: &str = "projectivize";
static USE_ROOT: &str = "use-root";

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
    fn depembeds_config_from_matches(matches: &ArgMatches) -> Result<DepembedsConfig> {
        let depth = matches
            .value_of(DEPENDENCY_DEPTH)
            .map(|v| v.parse().context("Cannot parse dependency depth"))
            .transpose()?
            .unwrap();
        let untyped = matches.is_present(UNTYPED_DEPS);
        let normalize = matches.is_present(NORMALIZE_CONTEXT);
        let projectivize = matches.is_present(PROJECTIVIZE);
        let use_root = matches.is_present(USE_ROOT);

        Ok(DepembedsConfig {
            depth,
            untyped,
            normalize,
            projectivize,
            use_root,
        })
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
                    .long("context-discard")
                    .value_name("CONTEXT_THRESHOLD")
                    .help("Context discard threshold")
                    .takes_value(true)
                    .default_value("1e-4"),
            )
            .arg(
                Arg::with_name(CONTEXT_MINCOUNT)
                    .long("context-mincount")
                    .value_name("CONTEXT_FREQ")
                    .help("Context mincount. Default: 5")
                    .takes_value(true)
                    .conflicts_with(CONTEXT_TARGET_SIZE),
            )
            .arg(
                Arg::with_name(CONTEXT_TARGET_SIZE)
                    .long("context-target-size")
                    .value_name("CONTEXT_TARGET_SIZE")
                    .help("Context vocab target size")
                    .takes_value(true),
            )
            .arg(
                Arg::with_name(DEPENDENCY_DEPTH)
                    .long("dependency-depth")
                    .value_name("DEPENDENCY_DEPTH")
                    .help("Dependency depth")
                    .takes_value(true)
                    .default_value("1"),
            )
            .arg(
                Arg::with_name(UNTYPED_DEPS)
                    .long("untyped-deps")
                    .help("Don't use dependency relation labels."),
            )
            .arg(
                Arg::with_name(NORMALIZE_CONTEXT)
                    .long("normalize-context")
                    .help("Normalize contexts"),
            )
            .arg(
                Arg::with_name(PROJECTIVIZE)
                    .long("projectivize")
                    .help("Projectivize dependency graphs before training."),
            )
            .arg(
                Arg::with_name(USE_ROOT)
                    .long("use-root")
                    .help("Use root when extracting dependency contexts."),
            )
    }

    fn parse(matches: &ArgMatches) -> Result<Self> {
        let corpus = matches.value_of(Self::CORPUS).unwrap().into();
        let output = matches.value_of(Self::OUTPUT).unwrap().into();
        let n_threads = matches
            .value_of(Self::THREADS)
            .map(|v| v.parse().context("Cannot parse number of threads"))
            .transpose()?
            .unwrap_or_else(|| cmp::min(num_cpus::get() / 2, 20));

        let discard_threshold = matches
            .value_of(CONTEXT_DISCARD)
            .map(|v| v.parse().context("Cannot parse discard threshold"))
            .transpose()?
            .unwrap();
        let cutoff = cutoff_from_matches(matches, CONTEXT_MINCOUNT, CONTEXT_TARGET_SIZE)?
            .unwrap_or_else(|| Cutoff::MinCount(5));

        let output_vocab_config = SimpleVocabConfig {
            cutoff,
            discard_threshold,
        };
        let train_info = TrainInfo::new(corpus, output, n_threads);

        Ok(DepsApp {
            train_info,
            common_config: Self::parse_common_config(&matches)?,
            depembeds_config: Self::depembeds_config_from_matches(&matches)?,
            input_vocab_config: Self::parse_vocab_config(&matches)?,
            output_vocab_config,
        })
    }

    fn run(&self) -> Result<()> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        eprintln!("SIMD features: {}", Self::simd_features().join(" "));

        match self.input_vocab_config() {
            VocabConfig::SimpleVocab(config) => {
                let (input_vocab, output_vocab) = build_vocab::<_, SimpleVocab<String>, _>(
                    config,
                    self.output_vocab_config(),
                    self.depembeds_config(),
                    self.corpus(),
                )?;
                train(input_vocab, output_vocab, self)?;
            }
            VocabConfig::SubwordVocab(config) => match config.indexer.indexer_type {
                BucketIndexerType::Finalfusion => {
                    let (input_vocab, output_vocab) =
                        build_vocab::<_, SubwordVocab<_, FinalfusionHashIndexer>, _>(
                            config,
                            self.output_vocab_config(),
                            self.depembeds_config(),
                            self.corpus(),
                        )?;
                    train(input_vocab, output_vocab, self)?
                }
                BucketIndexerType::FastText => {
                    let (input_vocab, output_vocab) =
                        build_vocab::<_, SubwordVocab<_, FastTextIndexer>, _>(
                            config,
                            self.output_vocab_config(),
                            self.depembeds_config(),
                            self.corpus(),
                        )?;
                    train(input_vocab, output_vocab, self)?;
                }
            },
            VocabConfig::NGramVocab(config) => {
                let (input_vocab, output_vocab) = build_vocab::<_, SubwordVocab<_, _>, _>(
                    config,
                    self.output_vocab_config(),
                    self.depembeds_config(),
                    self.corpus(),
                )?;
                train(input_vocab, output_vocab, self)?;
            }
        }

        Ok(())
    }
}

fn train<V>(input_vocab: V, output_vocab: SimpleVocab<Dependency>, app: &DepsApp) -> Result<()>
where
    V: Vocab<VocabType = String> + Into<VocabWrap> + Clone + Send + Sync + 'static,
    V::Config: Serialize,
    for<'a> &'a V::IdxType: IntoIterator<Item = u64>,
{
    let corpus = app.corpus();
    let common_config = app.common_config();
    let n_threads = app.n_threads();

    let mut output_writer =
        BufWriter::new(File::create(app.output()).context("Cannot open output file for writing.")?);
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
            )
        }));
    }

    show_progress(
        &app.common_config(),
        &sgd,
        Duration::from_millis(PROGRESS_UPDATE_INTERVAL),
    );

    // Wait until all threads have finished.
    for child in children {
        child.join().expect("Thread panicked")?;
    }

    sgd.into_model()
        .write_model_binary(&mut output_writer, app.train_info().clone())
        .context("Cannot write model")
}

fn do_work<P, R, V>(
    corpus_path: P,
    mut sgd: SGD<DepembedsTrainer<R, V>>,
    thread: usize,
    n_threads: usize,
    epochs: u32,
    start_lr: f32,
    projectivize: bool,
) -> Result<()>
where
    P: Into<PathBuf>,
    R: Clone + Rng,
    V: Vocab<VocabType = String>,
    V::Config: Serialize,
    for<'a> &'a V::IdxType: IntoIterator<Item = u64>,
{
    let n_tokens = sgd.model().input_vocab().n_types();

    let f = File::open(corpus_path.into()).context("Cannot open corpus for reading")?;
    let (data, start) =
        thread_data_conllu(&f, thread, n_threads).context("Could not get thread-specific data")?;
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
            .transpose()?
            .context("Cannot read sentence")?;

        let lr = (1.0 - (sgd.n_tokens_processed() as f32 / (epochs as usize * n_tokens) as f32))
            * start_lr;
        sgd.update_sentence(&sentence, lr);
    }

    Ok(())
}

fn build_vocab<P, V, C>(
    input_config: C,
    output_config: SimpleVocabConfig,
    dep_config: DepembedsConfig,
    corpus_path: P,
) -> Result<(V, SimpleVocab<Dependency>)>
where
    P: AsRef<Path>,
    V: Vocab<VocabType = String> + From<VocabBuilder<C, String>>,
    VocabBuilder<C, String>: Into<V>,
{
    let f = File::open(corpus_path).context("Cannot open corpus for reading")?;
    let file_progress = FileProgress::new(f).context("Cannot create progress bar")?;
    let mut input_builder = VocabBuilder::new(input_config);
    let mut output_builder: VocabBuilder<_, Dependency> = VocabBuilder::new(output_config);

    let projectivizer = if dep_config.projectivize {
        Some(HeadProjectivizer::new())
    } else {
        None
    };

    for sentence in SentenceIter::new(BufReader::new(file_progress), projectivizer) {
        let sentence = sentence?;

        for token in sentence.iter().filter_map(Node::token) {
            input_builder.count(token.form());
        }

        for (_, context) in DependencyIterator::new_from_config(&sentence.dep_graph(), dep_config) {
            output_builder.count(context);
        }
    }

    Ok((input_builder.into(), output_builder.into()))
}

struct SentenceIter<P, R>
where
    R: ReadSentence,
{
    inner: Sentences<R>,
    projectivizer: Option<P>,
}

impl<P, R> SentenceIter<P, Reader<R>>
where
    R: BufRead,
{
    fn new(read: R, projectivizer: Option<P>) -> Self {
        SentenceIter {
            inner: Reader::new(read).into_iter(),
            projectivizer,
        }
    }
}

impl<P, R> Iterator for SentenceIter<P, R>
where
    P: Projectivize,
    R: ReadSentence,
{
    type Item = Result<Sentence>;

    fn next(&mut self) -> Option<Self::Item> {
        let sentence = self.inner.next()?;
        let mut sentence = match sentence.context("Cannot read sentence") {
            Ok(sentence) => sentence,
            err @ Err(_) => return Some(err),
        };

        if let Some(proj) = &self.projectivizer {
            // Rewrap error.
            if let Err(err) = proj.projectivize(&mut sentence) {
                return Some(Err(err).context("Cannot projectivize sentence."));
            }
        }

        Some(Ok(sentence))
    }
}
