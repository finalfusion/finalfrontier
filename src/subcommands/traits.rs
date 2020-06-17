use std::convert::TryInto;

use anyhow::{Context, Result};
use clap::{App, AppSettings, Arg, ArgMatches};
use finalfrontier::{
    BucketConfig, CommonConfig, LossType, NGramConfig, SimpleVocabConfig, SubwordVocabConfig,
};

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
static HASH_INDEXER_TYPE: &str = "hash-indexer";
static LR: &str = "lr";
static MINCOUNT: &str = "mincount";
static MINN: &str = "minn";
static MAXN: &str = "maxn";
static NGRAM_MINCOUNT: &str = "ngram-mincount";
static SUBWORDS: &str = "subwords";
static NS: &str = "ns";
static ZIPF_EXPONENT: &str = "zipf";

pub trait FinalfrontierApp
where
    Self: Sized,
{
    const CORPUS: &'static str = "CORPUS";
    const OUTPUT: &'static str = "OUTPUT";
    const THREADS: &'static str = "THREADS";

    fn app() -> App<'static, 'static>;

    fn parse(matches: &ArgMatches) -> Result<Self>;

    fn run(&self) -> Result<()>;

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
                Arg::with_name(HASH_INDEXER_TYPE)
                    .long("hash-indexer")
                    .value_name("INDEXER")
                    .help("Hash indexer type")
                    .takes_value(true)
                    .default_value("finalfusion")
                    .possible_values(&["finalfusion", "fasttext"]),
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
                    .long("ngram-mincount")
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
    fn parse_common_config(matches: &ArgMatches) -> Result<CommonConfig> {
        let dims = matches
            .value_of(DIMS)
            .map(|v| v.parse().context("Cannot parse dimensionality"))
            .transpose()?
            .unwrap();
        let epochs = matches
            .value_of(EPOCHS)
            .map(|v| v.parse().context("Cannot parse number of epochs"))
            .transpose()?
            .unwrap();
        let lr = matches
            .value_of(LR)
            .map(|v| v.parse().context("Cannot parse learning rate"))
            .transpose()?
            .unwrap();
        let negative_samples = matches
            .value_of(NS)
            .map(|v| v.parse().context("Cannot parse number of negative samples"))
            .transpose()?
            .unwrap();
        let zipf_exponent = matches
            .value_of(ZIPF_EXPONENT)
            .map(|v| v.parse().context("Cannot parse exponent zipf distribution"))
            .transpose()?
            .unwrap();

        Ok(CommonConfig {
            loss: LossType::LogisticNegativeSampling,
            dims,
            epochs,
            lr,
            negative_samples,
            zipf_exponent,
        })
    }

    /// Construct `SubwordVocabConfig` from `matches`.
    fn parse_vocab_config(matches: &ArgMatches) -> Result<VocabConfig> {
        let discard_threshold = matches
            .value_of(DISCARD)
            .map(|v| v.parse().context("Cannot parse discard threshold"))
            .transpose()?
            .unwrap();
        let min_count = matches
            .value_of(MINCOUNT)
            .map(|v| v.parse().context("Cannot parse mincount"))
            .transpose()?
            .unwrap();
        let min_n = matches
            .value_of(MINN)
            .map(|v| v.parse().context("Cannot parse minimum n-gram length"))
            .transpose()?
            .unwrap();
        let max_n = matches
            .value_of(MAXN)
            .map(|v| v.parse().context("Cannot parse maximum n-gram length"))
            .transpose()?
            .unwrap();
        match matches.value_of(SUBWORDS).unwrap() {
            "buckets" => {
                let buckets_exp = matches
                    .value_of(BUCKETS)
                    .map(|v| v.parse().context("Cannot parse bucket exponent"))
                    .transpose()?
                    .unwrap();
                let indexer = matches
                    .value_of(HASH_INDEXER_TYPE)
                    .map(|v| v.try_into().context("Unknown subword indexer type"))
                    .transpose()?
                    .unwrap();
                Ok(VocabConfig::SubwordVocab(SubwordVocabConfig {
                    discard_threshold,
                    min_count,
                    max_n,
                    min_n,
                    indexer: BucketConfig {
                        buckets_exp,
                        indexer_type: indexer,
                    },
                }))
            }
            "ngrams" => {
                let min_ngram_count = matches
                    .value_of(NGRAM_MINCOUNT)
                    .map(|v| v.parse().context("Cannot parse bucket exponent"))
                    .transpose()?
                    .unwrap();
                Ok(VocabConfig::NGramVocab(SubwordVocabConfig {
                    discard_threshold,
                    min_count,
                    max_n,
                    min_n,
                    indexer: NGramConfig { min_ngram_count },
                }))
            }
            "none" => Ok(VocabConfig::SimpleVocab(SimpleVocabConfig {
                min_count,
                discard_threshold,
            })),
            // unreachable as long as possible values in clap are in sync with this `VocabConfig`'s
            // variants
            s => unreachable!(format!("Unhandled vocab type: {}", s)),
        }
    }

    /// Get features that will be used by SIMD code paths.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn simd_features() -> Vec<&'static str> {
        let mut features = vec![];

        if is_x86_feature_detected!("sse") {
            features.push("+sse");
        } else {
            features.push("-sse");
        }

        if is_x86_feature_detected!("avx") {
            features.push("+avx");
        } else {
            features.push("-avx");
        }

        if is_x86_feature_detected!("fma") {
            features.push("+fma");
        } else {
            features.push("-fma");
        }

        features
    }
}
