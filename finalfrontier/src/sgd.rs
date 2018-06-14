use ndarray::{Array1, ArrayView1, ArrayViewMut1};

use vec_simd::scaled_add;
use {log_logistic_loss, RangeGenerator, TrainModel};

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
        // the weight immpediately, but accumulating the weight updates in
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
