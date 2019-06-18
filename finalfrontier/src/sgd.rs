use ndarray::{Array1, ArrayView1, ArrayViewMut1};

use crate::loss::log_logistic_loss;
use crate::train_model::{NegativeSamples, TrainIterFrom, Trainer};
use crate::vec_simd::scaled_add;
use hogwild::Hogwild;

use crate::TrainModel;

/// Stochastic gradient descent
///
/// This data type applies stochastic gradient descent on sentences.
#[derive(Clone)]
pub struct SGD<T> {
    loss: Hogwild<f32>,
    model: TrainModel<T>,
    n_examples: Hogwild<usize>,
    n_tokens_processed: Hogwild<usize>,
    sgd_impl: NegativeSamplingSGD,
}

impl<T> SGD<T>
where
    T: Trainer,
{
    pub fn into_model(self) -> TrainModel<T> {
        self.model
    }

    /// Construct a new SGD instance,
    pub fn new(model: TrainModel<T>) -> Self {
        let sgd_impl = NegativeSamplingSGD::new(model.config().negative_samples as usize);

        SGD {
            loss: Hogwild::default(),
            model,
            n_examples: Hogwild::default(),
            n_tokens_processed: Hogwild::default(),
            sgd_impl,
        }
    }
    /// Get the training model associated with this SGD.
    pub fn model(&self) -> &TrainModel<T> {
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

    /// Update the model parameters using the given sentence.
    ///
    /// This applies a gradient descent step on the sentence, with the given
    /// learning rate.
    pub fn update_sentence<S>(&mut self, sentence: &S, lr: f32)
    where
        S: ?Sized,
        T: TrainIterFrom<S> + Trainer + NegativeSamples,
    {
        for (focus, contexts) in self.model.trainer().train_iter_from(sentence) {
            // Update parameters for the token focus token i and the
            // context token j.
            let input_embed = self.model.mean_input_embedding(&focus);

            for context in contexts {
                *self.loss += self.sgd_impl.sgd_step(
                    &mut self.model,
                    &focus,
                    input_embed.view(),
                    context,
                    lr,
                );
                *self.n_examples += 1;
            }
            *self.n_tokens_processed += 1;
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
pub struct NegativeSamplingSGD {
    negative_samples: usize,
}

impl NegativeSamplingSGD {
    /// Create a new loss function.
    pub fn new(negative_samples: usize) -> Self {
        NegativeSamplingSGD { negative_samples }
    }

    /// Perform a step of gradient descent.
    ///
    /// This method will estimate the probability of `output` and randomly
    /// chosen negative samples, given the input. It will then update the
    /// embeddings of the positive/negative outputs and the input (and its
    /// subwords).
    ///
    /// The function returns the sum of losses.
    pub fn sgd_step<T>(
        &mut self,
        model: &mut TrainModel<T>,
        input: impl IntoIterator<Item = u64>,
        input_embed: ArrayView1<f32>,
        output: usize,
        lr: f32,
    ) -> f32
    where
        T: NegativeSamples,
    {
        let mut loss = 0.0;
        let mut input_delta = Array1::zeros(input_embed.shape()[0]);

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
        for idx in input {
            let input_embed = model.input_embedding_mut(idx as usize);
            scaled_add(input_embed, input_delta.view(), 1.0);
        }

        loss
    }

    /// Pick, predict and update negative samples.
    fn negative_samples<T>(
        &mut self,
        model: &mut TrainModel<T>,
        input_embed: ArrayView1<f32>,
        mut input_delta: ArrayViewMut1<f32>,
        output: usize,
        lr: f32,
    ) -> f32
    where
        T: NegativeSamples,
    {
        let mut loss = 0f32;

        for _ in 0..self.negative_samples {
            let negative = model.trainer().negative_sample(output);
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
    fn update_output<T>(
        &mut self,
        model: &mut TrainModel<T>,
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
