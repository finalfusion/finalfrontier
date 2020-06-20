use anyhow::{Context, Result};
use clap::ArgMatches;

use finalfrontier::{BucketConfig, Cutoff, NGramConfig, SimpleVocabConfig, SubwordVocabConfig};

#[derive(Copy, Clone)]
pub enum VocabConfig {
    SubwordVocab(SubwordVocabConfig<BucketConfig>),
    NGramVocab(SubwordVocabConfig<NGramConfig>),
    SimpleVocab(SimpleVocabConfig),
}

pub fn cutoff_from_matches(
    matches: &ArgMatches,
    mincount: &str,
    target_size: &str,
) -> Result<Option<Cutoff>> {
    match matches
        .value_of(mincount)
        .map(|v| v.parse().context("Cannot parse mincount"))
        .transpose()?
        .map(Cutoff::MinCount)
    {
        None => Ok(matches
            .value_of(target_size)
            .map(|v| v.parse().context("Cannot parse target size"))
            .transpose()?
            .map(Cutoff::TargetSize)),
        some => Ok(some),
    }
}
