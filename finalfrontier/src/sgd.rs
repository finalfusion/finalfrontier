use ndarray::{Array1, ArrayView1, ArrayViewMut1};
use rand::{Rng, SeedableRng};

use hogwild::Hogwild;
use loss::log_logistic_loss;
use sampling::{BandedRangeGenerator, RangeGenerator, ZipfRangeGenerator};
use train_model::TrainIterFrom;
use util::ReseedOnCloneRng;
use vec_simd::scaled_add;

use {ModelType, TrainModel, Vocab};

/// Stochastic gradient descent
///
/// This data type applies stochastic gradient descent on sentences.
#[derive(Clone)]
pub struct SGD<R> {
    loss: Hogwild<f32>,
    model: TrainModel<R>,
    n_examples: Hogwild<usize>,
    n_tokens_processed: Hogwild<usize>,
    sgd_impl: NegativeSamplingSGD<BandedRangeGenerator<R, ZipfRangeGenerator<R>>>,
}

impl<R> SGD<R> {
    pub fn into_model(self) -> TrainModel<R> {
        self.model
    }

    /// Get the training model associated with this SGD.
    pub fn model(&self) -> &TrainModel<R> {
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

impl<R> SGD<ReseedOnCloneRng<R>>
where
    R: Clone + Rng + SeedableRng,
{
    /// Construct a new SGD instance.
    pub fn new(model: TrainModel<ReseedOnCloneRng<R>>, rng: R) -> Self {
        let reseed_on_clone = ReseedOnCloneRng(rng);

        let band_size = match model.config().model {
            ModelType::SkipGram => 1,
            ModelType::StructuredSkipGram => model.config().context_size * 2,
        };

        let range_gen = BandedRangeGenerator::new(
            reseed_on_clone.clone(),
            ZipfRangeGenerator::new_with_exponent(
                reseed_on_clone.clone(),
                model.vocab().len(),
                model.config().zipf_exponent,
            ),
            band_size as usize,
        );

        let sgd_impl =
            NegativeSamplingSGD::new(model.config().negative_samples as usize, range_gen);

        SGD {
            loss: Hogwild::default(),
            model,
            n_examples: Hogwild::default(),
            n_tokens_processed: Hogwild::default(),
            sgd_impl,
        }
    }
}

impl<R> SGD<R>
where
    R: Rng,
{
    /// Update the model parameters using the given sentence.
    ///
    /// This applies a gradient descent step on the sentence, with the given
    /// learning rate.
    pub fn update_sentence<S>(&mut self, sentence: &S, lr: f32)
    where
        S: ?Sized,
        TrainModel<R>: TrainIterFrom<S>,
    {
        for (focus, contexts) in self.model.train_iter_from(sentence) {
            // The input word is represented by its index and subword
            // indices.
            let mut input = self
                .model
                .vocab()
                .subword_indices_idx(focus)
                .unwrap()
                .to_owned();
            input.push(focus as u64);

            let input_embed = self.model.mean_input_embedding(&input);

            for context in contexts {
                *self.loss += self.sgd_impl.sgd_step(
                    &mut self.model,
                    &input,
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
pub struct NegativeSamplingSGD<RS> {
    negative_samples: usize,
    range_gen: RS,
}

impl<RS> NegativeSamplingSGD<RS>
where
    RS: RangeGenerator,
{
    /// Create a new loss function.
    pub fn new(negative_samples: usize, range_gen: RS) -> Self {
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
    pub fn sgd_step<R>(
        &mut self,
        model: &mut TrainModel<R>,
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
    fn negative_samples<R>(
        &mut self,
        model: &mut TrainModel<R>,
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
    fn update_output<R>(
        &mut self,
        model: &mut TrainModel<R>,
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
