extern crate clap;
extern crate failure;
extern crate finalfrontier;
extern crate stdinout;

use std::fs::File;
use std::io::{BufReader, BufWriter};

use clap::{App, AppSettings, Arg, ArgMatches};
use failure::{err_msg, Error};
use finalfrontier::{Model, ReadModelBinary, WriteModelText, WriteModelWord2Vec};
use stdinout::{OrExit, Output};

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

fn main() {
    let matches = parse_args();
    let config = config_from_matches(&matches);

    let f = File::open(config.model_filename).or_exit("Cannot read model", 1);
    let model = Model::read_model_binary(&mut BufReader::new(f)).or_exit("Cannot load model", 1);

    let output = Output::from(config.output_filename);
    let mut writer = BufWriter::new(output.write().or_exit("Cannot open output for writing", 1));

    match config.output_format {
        OutputFormat::Text => model
            .write_model_text(&mut writer, false)
            .or_exit("Could not write model", 1),
        OutputFormat::TextDims => model
            .write_model_text(&mut writer, true)
            .or_exit("Could not write model", 1),
        OutputFormat::Word2Vec => model
            .write_model_word2vec(&mut writer)
            .or_exit("Could not write model", 1),
    }
}

// Option constants
static OUTPUT_FORMAT: &str = "output_format";

// Argument constants
static INPUT: &str = "INPUT";
static OUTPUT: &str = "OUTPUT";

fn parse_args() -> ArgMatches<'static> {
    App::new("ff-bin2text")
        .settings(DEFAULT_CLAP_SETTINGS)
        .arg(
            Arg::with_name(INPUT)
                .help("FastText Model")
                .index(1)
                .required(true),
        )
        .arg(Arg::with_name(OUTPUT).help("Output file").index(2))
        .arg(
            Arg::with_name(OUTPUT_FORMAT)
                .short("f")
                .long("format")
                .value_name("FORMAT")
                .help("Output format: text, textdims, word2vec (default: textdims)")
                .takes_value(true),
        )
        .get_matches()
}

enum OutputFormat {
    Text,
    TextDims,
    Word2Vec,
}

impl OutputFormat {
    pub fn try_from_str(format: &str) -> Result<OutputFormat, Error> {
        match format {
            "text" => Ok(OutputFormat::Text),
            "textdims" => Ok(OutputFormat::TextDims),
            "word2vec" => Ok(OutputFormat::Word2Vec),
            _ => Err(err_msg(format!("Unknown output format: {}", format))),
        }
    }
}

struct Config {
    model_filename: String,
    output_filename: Option<String>,
    output_format: OutputFormat,
}

fn config_from_matches(matches: &ArgMatches) -> Config {
    let model_filename = matches.value_of(INPUT).unwrap().to_owned();
    let output_filename = matches.value_of(OUTPUT).map(ToOwned::to_owned);
    let output_format = matches
        .value_of(OUTPUT_FORMAT)
        .map(|v| OutputFormat::try_from_str(v).or_exit("Cannot parse output format type", 1))
        .unwrap_or(OutputFormat::TextDims);

    Config {
        model_filename,
        output_filename,
        output_format,
    }
}
