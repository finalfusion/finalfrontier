use clap::{App, AppSettings, Arg, ArgMatches};
use finalfrontier::{
    BucketConfig, CommonConfig, LossType, NGramConfig, SimpleVocabConfig, SubwordVocabConfig,
};
use stdinout::OrExit;

use crate::subcommands::VocabConfig;

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

// Option constants
static BUCKETS: &str = "buckets";
static DIMS: &str = "dims";
static DISCARD: &str = "discard";
static EPOCHS: &str = "epochs";
static LR: &str = "lr";
static MINCOUNT: &str = "mincount";
static MINN: &str = "minn";
static MAXN: &str = "maxn";
static NGRAM_MINCOUNT: &str = "ngram_mincount";
static SUBWORDS: &str = "subwords";
static NS: &str = "ns";
static ZIPF_EXPONENT: &str = "zipf";

pub trait FinalfrontierApp {
    const CORPUS: &'static str = "CORPUS";
    const OUTPUT: &'static str = "OUTPUT";
    const THREADS: &'static str = "THREADS";

    fn app() -> App<'static, 'static>;

    fn parse(matches: &ArgMatches) -> Self;

    fn run(&self);

    fn common_opts<'a, 'b>(name: &str) -> App<'a, 'b> {
        let version = if let Some(git_desc) = option_env!("MAYBE_FINALFRONTIER_GIT_DESC") {
            git_desc
        } else {
            env!("CARGO_PKG_VERSION")
        };
        App::new(name)
            .settings(DEFAULT_CLAP_SETTINGS)
            .version(version)
            .arg(
                Arg::with_name(BUCKETS)
                    .long("buckets")
                    .value_name("EXP")
                    .help("Number of buckets: 2^EXP")
                    .takes_value(true)
                    .default_value("21"),
            )
            .arg(
                Arg::with_name(DIMS)
                    .long("dims")
                    .value_name("DIMENSIONS")
                    .help("Embedding dimensionality")
                    .takes_value(true)
                    .default_value("300"),
            )
            .arg(
                Arg::with_name(DISCARD)
                    .long("discard")
                    .value_name("THRESHOLD")
                    .help("Discard threshold")
                    .takes_value(true)
                    .default_value("1e-4"),
            )
            .arg(
                Arg::with_name(EPOCHS)
                    .long("epochs")
                    .value_name("N")
                    .help("Number of epochs")
                    .takes_value(true)
                    .default_value("15"),
            )
            .arg(
                Arg::with_name(LR)
                    .long("lr")
                    .value_name("LEARNING_RATE")
                    .help("Initial learning rate")
                    .takes_value(true)
                    .default_value("0.05"),
            )
            .arg(
                Arg::with_name(MINCOUNT)
                    .long("mincount")
                    .value_name("FREQ")
                    .help("Minimum token frequency")
                    .takes_value(true)
                    .default_value("5"),
            )
            .arg(
                Arg::with_name(MINN)
                    .long("minn")
                    .value_name("LEN")
                    .help("Minimum ngram length")
                    .takes_value(true)
                    .default_value("3"),
            )
            .arg(
                Arg::with_name(MAXN)
                    .long("maxn")
                    .value_name("LEN")
                    .help("Maximum ngram length")
                    .takes_value(true)
                    .default_value("6"),
            )
            .arg(
                Arg::with_name(SUBWORDS)
                    .long("subwords")
                    .takes_value(true)
                    .value_name("SUBWORDS")
                    .possible_values(&["buckets", "ngrams", "none"])
                    .default_value("buckets")
                    .help("What kind of subwords to use."),
            )
            .arg(
                Arg::with_name(NGRAM_MINCOUNT)
                    .long("ngram_mincount")
                    .value_name("FREQ")
                    .help("Minimum ngram frequency.")
                    .takes_value(true)
                    .default_value("5"),
            )
            .arg(
                Arg::with_name(NS)
                    .long("ns")
                    .value_name("FREQ")
                    .help("Negative samples per word")
                    .takes_value(true)
                    .default_value("5"),
            )
            .arg(
                Arg::with_name(Self::THREADS)
                    .long("threads")
                    .value_name("N")
                    .help("Number of threads (default: min(logical_cpus / 2, 20))")
                    .takes_value(true),
            )
            .arg(
                Arg::with_name(ZIPF_EXPONENT)
                    .long("zipf")
                    .value_name("EXP")
                    .help("Exponent Zipf distribution for negative sampling")
                    .takes_value(true)
                    .default_value("0.5"),
            )
            .arg(
                Arg::with_name(Self::CORPUS)
                    .help("Tokenized corpus")
                    .index(1)
                    .required(true),
            )
            .arg(
                Arg::with_name(Self::OUTPUT)
                    .help("Embeddings output")
                    .index(2)
                    .required(true),
            )
    }

    /// Construct `CommonConfig` from `matches`.
    fn parse_common_config(matches: &ArgMatches) -> CommonConfig {
        let dims = matches
            .value_of(DIMS)
            .map(|v| v.parse().or_exit("Cannot parse dimensionality", 1))
            .unwrap();
        let epochs = matches
            .value_of(EPOCHS)
            .map(|v| v.parse().or_exit("Cannot parse number of epochs", 1))
            .unwrap();
        let lr = matches
            .value_of(LR)
            .map(|v| v.parse().or_exit("Cannot parse learning rate", 1))
            .unwrap();
        let negative_samples = matches
            .value_of(NS)
            .map(|v| {
                v.parse()
                    .or_exit("Cannot parse number of negative samples", 1)
            })
            .unwrap();
        let zipf_exponent = matches
            .value_of(ZIPF_EXPONENT)
            .map(|v| {
                v.parse()
                    .or_exit("Cannot parse exponent zipf distribution", 1)
            })
            .unwrap();

        CommonConfig {
            loss: LossType::LogisticNegativeSampling,
            dims,
            epochs,
            lr,
            negative_samples,
            zipf_exponent,
        }
    }

    /// Construct `SubwordVocabConfig` from `matches`.
    fn parse_vocab_config(matches: &ArgMatches) -> VocabConfig {
        let discard_threshold = matches
            .value_of(DISCARD)
            .map(|v| v.parse().or_exit("Cannot parse discard threshold", 1))
            .unwrap();
        let min_count = matches
            .value_of(MINCOUNT)
            .map(|v| v.parse().or_exit("Cannot parse mincount", 1))
            .unwrap();
        let min_n = matches
            .value_of(MINN)
            .map(|v| v.parse().or_exit("Cannot parse minimum n-gram length", 1))
            .unwrap();
        let max_n = matches
            .value_of(MAXN)
            .map(|v| v.parse().or_exit("Cannot parse maximum n-gram length", 1))
            .unwrap();
        match matches.value_of(SUBWORDS).unwrap() {
            "buckets" => {
                let buckets_exp = matches
                    .value_of(BUCKETS)
                    .map(|v| v.parse().or_exit("Cannot parse bucket exponent", 1))
                    .unwrap();
                VocabConfig::SubwordVocab(SubwordVocabConfig {
                    discard_threshold,
                    min_count,
                    max_n,
                    min_n,
                    indexer: BucketConfig { buckets_exp },
                })
            }
            "ngrams" => {
                let min_ngram_count = matches
                    .value_of(NGRAM_MINCOUNT)
                    .map(|v| v.parse().or_exit("Cannot parse bucket exponent", 1))
                    .unwrap();
                VocabConfig::NGramVocab(SubwordVocabConfig {
                    discard_threshold,
                    min_count,
                    max_n,
                    min_n,
                    indexer: NGramConfig { min_ngram_count },
                })
            }
            "none" => VocabConfig::SimpleVocab(SimpleVocabConfig {
                min_count,
                discard_threshold,
            }),
            // unreachable as long as possible values in clap are in sync with this `VocabConfig`'s
            // variants
            s => unreachable!(format!("Unhandled vocab type: {}", s)),
        }
    }
}
