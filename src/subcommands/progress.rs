use std::thread;
use std::time::Duration;

use finalfrontier::{CommonConfig, Trainer, Vocab, SGD};
use indicatif::{ProgressBar, ProgressStyle};

pub fn show_progress<T, V>(config: &CommonConfig, sgd: &SGD<T>, update_interval: Duration)
where
    T: Trainer<InputVocab = V>,
    V: Vocab,
{
    let n_tokens = sgd.model().input_vocab().n_types();

    let pb = ProgressBar::new(u64::from(config.epochs) * n_tokens as u64);
    pb.set_style(
        ProgressStyle::default_bar().template("{bar:30} {percent}% {msg} ETA: {eta_precise}"),
    );

    while sgd.n_tokens_processed() < n_tokens * config.epochs as usize {
        let lr = (1.0
            - (sgd.n_tokens_processed() as f32 / (config.epochs as usize * n_tokens) as f32))
            * config.lr;

        pb.set_position(sgd.n_tokens_processed() as u64);
        pb.set_message(&format!(
            "loss: {:.*} lr: {:.*}",
            5,
            sgd.train_loss(),
            5,
            lr
        ));

        thread::sleep(update_interval);
    }

    pb.finish();
}
