use std::cmp;

use ndarray::{Array1, ArrayView1, ArrayViewMut1};
use rand::Rng;

use hogwild::Hogwild;
use loss::log_logistic_loss;
use sampling::{RangeGenerator, ZipfRangeGenerator};
use vec_simd::scaled_add;

use {ModelType, TrainModel};

/// Stochastic gradient descent
///
/// This data type applies stochastic gradient descent on sentences.
#[derive(Clone)]
pub struct SGD<R> {
    loss: Hogwild<f32>,
    model: TrainModel,
    n_examples: Hogwild<usize>,
    n_tokens_processed: Hogwild<usize>,
    rng: R,
    sgd_impl: NegativeSamplingSGD<ZipfRangeGenerator<R>>,
}

impl<R> SGD<R> {
    /// Get the training model associated with this SGD.
    pub fn model(&self) -> &TrainModel {
        &self.model
    }

    /// Get the number of tokens that are processed by this SGD.
    pub fn n_tokens_processed(&self) -> usize {
        *self.n_tokens_processed
    }

    /// Get the average training loss of this SGD.
    ///
    /// This returns the average training loss over all instances seen by
    /// this SGD instance since its construction.
    pub fn train_loss(&self) -> f32 {
        *self.loss / *self.n_examples as f32
    }
}

impl<R> SGD<R>
where
    R: Clone + Rng,
{
    /// Construct a new SGD instance,
    pub fn new(model: TrainModel, rng: R) -> Self {
        let sgd_impl = NegativeSamplingSGD::new(
            model.config().negative_samples as usize,
            ZipfRangeGenerator::new(rng.clone(), model.vocab().len()),
        );

        SGD {
            loss: Hogwild::default(),
            model,
            n_examples: Hogwild::default(),
            n_tokens_processed: Hogwild::default(),
            rng,
            sgd_impl,
        }
    }

    /// Update the model parameters using the given sentence.
    ///
    /// This applies a gradient descent step on the sentence, with the given
    /// learning rate.
    pub fn update_sentence<S>(&mut self, sentence: &[S], lr: f32)
    where
        S: AsRef<str>,
    {
        let mut rng = self.rng.clone();

        // Convert the sentence into word identifiers, discarding words with
        // the probability indicated by the dictionary.
        let words: Vec<_> = sentence
            .iter()
            .filter_map(|t| self.model.vocab().word_idx(t.as_ref()))
            .filter(|&idx| rng.gen_range(0f32, 1f32) < self.model.vocab().discard(idx))
            .collect();

        for i in 0..words.len() {
            // The input word is represented by its index and subword
            // indices.
            let mut input = self
                .model
                .vocab()
                .subword_indices_idx(words[i])
                .unwrap()
                .to_owned();
            input.push(words[i] as u64);

            let input_embed = self.model.mean_input_embedding(&input);

            // Bojanowski, et al., 2017 uniformly sample the context size between 1 and c.
            let context_size = self.rng.gen_range(1, self.model.config().context_size + 1) as usize;

            let left = i - cmp::min(i, context_size);
            let right = cmp::min(i + context_size + 1, words.len());

            for j in left..right {
                if i != j {
                    let output = self.output_(words[j], i, j);

                    // Update parameters for the token focus token i and the
                    // context token j.
                    *self.loss += self.sgd_impl.sgd_step(
                        &mut self.model,
                        &input,
                        input_embed.view(),
                        output,
                        lr,
                    )
                }
            }

            *self.n_examples += right - left;
        }

        *self.n_tokens_processed += words.len();
    }

    fn output_(&self, token: usize, focus_idx: usize, offset_idx: usize) -> usize {
        match self.model.config().model {
            ModelType::SkipGram => token,
            ModelType::StructuredSkipGram => {
                let context_size = self.model.config().context_size as usize;
                let offset = if offset_idx < focus_idx {
                    (offset_idx + context_size) - focus_idx
                } else {
                    (offset_idx - focus_idx - 1) + context_size
                };

                (token * context_size * 2) + offset
            }
        }
    }
}

/// Log-logistic loss SGD with negative sampling.
///
/// This type implements gradient descent for log-logistic loss with negative
/// sampling (Mikolov, 2013).
///
/// In this approach, word embeddings training is shaped as a
/// prediction task. The word vectors should be parametrized such that
/// words that co-occur with a given input get an estimated probability
/// of 1.0, whereas words that do not co-occur with the input get an
/// estimated probability of 0.0.
///
/// The probability is computed from the inner product of two word
/// vectors by applying the logistic function. The loss is the negative
/// log likelihood.
///
/// Due to the vocabulary sizes, it is not possible to update the vectors
/// for all words that do not co-occur in every step. Instead, such
/// negatives are sampled, weighted by word frequency.
#[derive(Clone)]
pub struct NegativeSamplingSGD<R> {
    negative_samples: usize,
    range_gen: R,
}

impl<R> NegativeSamplingSGD<R>
where
    R: RangeGenerator,
{
    /// Create a new loss function.
    pub fn new(negative_samples: usize, range_gen: R) -> Self {
        NegativeSamplingSGD {
            negative_samples,
            range_gen,
        }
    }

    /// Perform a step of gradient descent.
    ///
    /// This method will estimate the probability of `output` and randomly
    /// chosen negative samples, given the input. It will then update the
    /// embeddings of the positive/negative outputs and the input (and its
    /// subwords).
    ///
    /// The function returns the sum of losses.
    pub fn sgd_step(
        &mut self,
        model: &mut TrainModel,
        input: &[u64],
        input_embed: ArrayView1<f32>,
        output: usize,
        lr: f32,
    ) -> f32 {
        let mut loss = 0.0;
        let mut input_delta = Array1::zeros(model.config().dims as usize);

        // Update the output embedding of the positive instance.
        loss += self.update_output(
            model,
            input_embed.view(),
            input_delta.view_mut(),
            output,
            true,
            lr,
        );

        // Pick the negative examples and update their output embeddings.
        loss += self.negative_samples(model, input_embed, input_delta.view_mut(), output, lr);

        // Update the input embeddings with the accumulated gradient.
        for &idx in input {
            let mut input_embed = model.input_embedding_mut(idx as usize);
            scaled_add(input_embed, input_delta.view(), 1.0);
        }

        loss
    }

    /// Pick, predict and update negative samples.
    fn negative_samples(
        &mut self,
        model: &mut TrainModel,
        input_embed: ArrayView1<f32>,
        mut input_delta: ArrayViewMut1<f32>,
        output: usize,
        lr: f32,
    ) -> f32 {
        let mut loss = 0f32;

        for _ in 0..self.negative_samples {
            let mut negative;
            loop {
                // Cannot panic, since the iterator is endless.
                negative = self.range_gen.next().unwrap();

                // We do not want to use the target word as a negative
                // example.
                if negative != output {
                    break;
                }
            }

            // Update input and output for this negative sample.
            loss += self.update_output(
                model,
                input_embed.view(),
                input_delta.view_mut(),
                negative,
                false,
                lr,
            );
        }

        loss
    }

    /// Update an output embedding.
    ///
    /// This also accumulates an update for the input embedding.
    ///
    /// The method returns the loss for predicting the output.
    fn update_output(
        &mut self,
        model: &mut TrainModel,
        input_embed: ArrayView1<f32>,
        input_delta: ArrayViewMut1<f32>,
        output: usize,
        label: bool,
        lr: f32,
    ) -> f32 {
        let (loss, part_gradient) =
            log_logistic_loss(input_embed.view(), model.output_embedding(output), label);

        // Update the input weight: u_n += lr * u_n' v_n. We are not updating
        // the weight immediately, but accumulating the weight updates in
        // input_delta.
        scaled_add(
            input_delta,
            model.output_embedding(output),
            lr * part_gradient,
        );

        // Update the output weight: v_n += lr * v_n' u_n.
        scaled_add(
            model.output_embedding_mut(output),
            input_embed.view(),
            lr * part_gradient,
        );

        loss
    }
}
