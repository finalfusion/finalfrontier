use git2::{DescribeFormatOptions, DescribeOptions, Repository};

fn main() {
    if let Ok(repo) = Repository::open_from_env() {
        let describe = repo
            .describe(DescribeOptions::new().show_commit_oid_as_fallback(true))
            .expect("Could not get description for git repo.");

        let desc = describe
            .format(Some(DescribeFormatOptions::new().dirty_suffix("-dirty")))
            .expect("Could not format description for git repo.");
        println!(
            "cargo:rustc-env=MAYBE_FINALFRONTIER_GIT_DESC={} {}",
            env!("CARGO_PKG_VERSION"),
            desc
        );
    }
}
