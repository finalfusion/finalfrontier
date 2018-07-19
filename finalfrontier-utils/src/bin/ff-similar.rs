extern crate clap;
extern crate finalfrontier;
extern crate stdinout;

use std::fs::File;
use std::io::{BufRead, BufReader};

use clap::{App, AppSettings, Arg, ArgMatches};
use finalfrontier::{Model, ReadModelBinary, Similarity};
use stdinout::{Input, OrExit};

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

fn main() {
    let matches = parse_args();
    let config = config_from_matches(&matches);

    let f = File::open(config.model_filename).or_exit("Cannot read model", 1);
    let model = Model::read_model_binary(&mut BufReader::new(f)).or_exit("Cannot load model", 1);

    let input = Input::from(matches.value_of("INPUT"));
    let reader = input.buf_read().or_exit("Cannot open input for reading", 1);

    let sim: Similarity = Similarity::from(&model);

    for line in reader.lines() {
        let line = line.or_exit("Cannot read line", 1).trim().to_owned();
        if line.is_empty() {
            continue;
        }

        let results = match sim.similarity(&line, config.k) {
            Some(results) => results,
            None => continue,
        };

        for similar in results {
            println!("{}\t{}", similar.word, similar.similarity);
        }
    }
}

fn parse_args() -> ArgMatches<'static> {
    App::new("ftr-similar")
        .settings(DEFAULT_CLAP_SETTINGS)
        .arg(
            Arg::with_name("neighbors")
                .short("k")
                .value_name("K")
                .help("Return K nearest neighbors (default: 10)")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("MODEL")
                .help("FastText Model")
                .index(1)
                .required(true),
        )
        .arg(Arg::with_name("INPUT").help("Input words").index(2))
        .get_matches()
}

struct Config {
    model_filename: String,
    k: usize,
}

fn config_from_matches<'a>(matches: &ArgMatches<'a>) -> Config {
    let model_filename = matches.value_of("MODEL").unwrap().to_owned();

    let k = matches
        .value_of("neighbors")
        .map(|v| v.parse().or_exit("Cannot parse k", 1))
        .unwrap_or(10);

    Config { model_filename, k }
}
