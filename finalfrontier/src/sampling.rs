use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::sync::Arc;
use vocab::Vocab;
use zipf::ZipfDistribution;

pub trait RangeGenerator: Iterator<Item = usize> {
    /// Get the upper bound in *[0, upper_bound)*.
    fn upper_bound(&self) -> usize;
}

/// Exponent to use for the Zipf's distribution.
///
/// This is the exponent s in f(k) = 1 / (k^s H_{N, s})
const ZIPF_RANGE_GENERATOR_EXPONENT: f64 = 0.5;

/// An iterator that draws from *[0, n)* with integer weights.
///
/// This iterator returns integers from *[0, n)*, where the probability of
/// each integer is weighted.
///
/// In contrast to `WeightedRangeGenerator`, this data structure
/// creates a table where each index occurs as frequently as its
/// weight. We then sample indices from this table.
#[derive(Clone)]
pub struct SampleWeightedRangeGenerator<R> {
    sample: Arc<Vec<usize>>,
    rng: R,
    vocab_len: usize,
}

impl<R> SampleWeightedRangeGenerator<R> {
    pub fn new(rng: R, vocab: &Vocab, power: f32, table_size: usize) -> Self {
        let mut sample = Vec::with_capacity(table_size);

        let weight_sum = vocab
            .words()
            .iter()
            .map(|w| (w.count() as f32).powf(power))
            .sum::<f32>();

        for (token_idx, token) in vocab.words().iter().enumerate() {
            let token_weight = (token.count() as f32).powf(power);
            let n_table_elems = ((token_weight / weight_sum) * table_size as f32) as usize;

            for _ in 0..n_table_elems {
                sample.push(token_idx);
            }
        }

        SampleWeightedRangeGenerator {
            sample: Arc::new(sample),
            rng,
            vocab_len: vocab.len(),
        }
    }
}

impl<R> Iterator for SampleWeightedRangeGenerator<R>
where
    R: Rng,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let rand_idx = self.rng.gen_range(0, self.sample.len());
        Some(self.sample[rand_idx])
    }
}

impl<R> RangeGenerator for SampleWeightedRangeGenerator<R>
where
    R: Rng,
{
    fn upper_bound(&self) -> usize {
        self.vocab_len
    }
}

/// An iterator that draws from *[0, n)* with integer weights.
///
/// This iterator returns integers from *[0, n)*, where the probability of
/// each integer is weighted.
///
/// See: Geometric Approximation Algorithms, Sariel Har-Peled, pp. 88
#[derive(Clone)]
pub struct WeightedRangeGenerator<R> {
    prefix_sum: Vec<usize>,
    upper: usize,
    rng: R,
}

impl<R> WeightedRangeGenerator<R>
where
    R: Rng,
{
    #[allow(dead_code)]
    pub fn new(rng: R, weights: &[usize]) -> WeightedRangeGenerator<R> {
        assert!(!weights.is_empty(), "Cannot sample from zero elements.");

        let mut prefix_sum = Vec::with_capacity(weights.len());
        let mut sum = 0;
        for &v in weights {
            sum += v;
            prefix_sum.push(sum);
        }

        WeightedRangeGenerator {
            prefix_sum,
            upper: sum + 1,
            rng,
        }
    }
}

impl<R> Iterator for WeightedRangeGenerator<R>
where
    R: Rng,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let val = self.rng.gen_range(1, self.upper);

        let idx = match self.prefix_sum.binary_search(&val) {
            Ok(idx) => idx,
            Err(idx) => idx,
        };

        Some(idx)
    }
}

impl<R> RangeGenerator for WeightedRangeGenerator<R>
where
    R: Rng,
{
    fn upper_bound(&self) -> usize {
        self.prefix_sum.len()
    }
}

/// An iterator that draws from *[0, n)* with a Zipfian distribution.
///
/// This iterator returns integers from *[0, n)*, where the probability of
/// each integer follows a Zipfian distribution.
///
/// This generator can be used to draw words from a vocabulary sorted by
/// descending frequency. Since the token frequencies (presumably) have a
/// Zipfian distribution, this will pick a token with a probability that
/// is proportional to its frequency.
pub struct ZipfRangeGenerator<R> {
    upper_bound: usize,
    rng: R,
    dist: ZipfDistribution,
}

impl<R> Clone for ZipfRangeGenerator<R>
where
    R: Clone,
{
    fn clone(&self) -> Self {
        ZipfRangeGenerator {
            upper_bound: self.upper_bound,
            rng: self.rng.clone(),
            dist: ZipfDistribution::new(self.upper_bound, ZIPF_RANGE_GENERATOR_EXPONENT).unwrap(),
        }
    }
}

impl<R> ZipfRangeGenerator<R>
where
    R: Rng,
{
    #[allow(dead_code)]
    pub fn new(rng: R, upper: usize) -> Self {
        ZipfRangeGenerator {
            upper_bound: upper,
            rng,
            dist: ZipfDistribution::new(upper, ZIPF_RANGE_GENERATOR_EXPONENT).unwrap(),
        }
    }
}

impl<R> Iterator for ZipfRangeGenerator<R>
where
    R: Rng,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let r = self.dist.sample(&mut self.rng);
        Some(r - 1)
    }
}

impl<R> RangeGenerator for ZipfRangeGenerator<R>
where
    R: Rng,
{
    fn upper_bound(&self) -> usize {
        self.upper_bound
    }
}

/// A banded range generator.
///
/// This range generator assumes that the overal range consists of
/// bands with a probability distribution implied by another range
/// generator and items within that band with a uniform distribution.
#[derive(Clone)]
pub struct BandedRangeGenerator<R, G> {
    uniform: Uniform<usize>,
    band_size: usize,
    inner: G,
    rng: R,
}

impl<R, G> BandedRangeGenerator<R, G>
where
    R: Rng,
    G: RangeGenerator,
{
    #[allow(dead_code)]
    pub fn new(rng: R, band_range_gen: G, band_size: usize) -> Self {
        BandedRangeGenerator {
            uniform: Uniform::new(0, band_size),
            band_size,
            inner: band_range_gen,
            rng,
        }
    }
}

impl<R, G> Iterator for BandedRangeGenerator<R, G>
where
    R: Rng,
    G: RangeGenerator,
{
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.band_size == 1 {
            // No banding, use the inner generator.
            self.inner.next()
        } else {
            let band = self.inner.next().unwrap();
            let band_item = self.uniform.sample(&mut self.rng);
            Some(band * self.band_size + band_item)
        }
    }
}

impl<R, G> RangeGenerator for BandedRangeGenerator<R, G>
where
    R: Rng,
    G: RangeGenerator,
{
    fn upper_bound(&self) -> usize {
        self.inner.upper_bound() * self.band_size
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_xorshift::XorShiftRng;

    use super::{BandedRangeGenerator, RangeGenerator, WeightedRangeGenerator, ZipfRangeGenerator};
    use util::{all_close, close};

    const SEED: [u8; 16] = [
        0xe9, 0xfe, 0xf0, 0xfb, 0x6a, 0x23, 0x2a, 0xb3, 0x7c, 0xce, 0x27, 0x9b, 0x56, 0xac, 0xdb,
        0xf8,
    ];

    const SEED2: [u8; 16] = [
        0xc8, 0xae, 0xa3, 0x99, 0x28, 0x5a, 0xbb, 0x27, 0x90, 0xe9, 0x61, 0x60, 0xe5, 0xca, 0xfe,
        0x22,
    ];

    #[test]
    #[should_panic]
    fn empty_weighted_range_generator() {
        let rng = XorShiftRng::from_seed(SEED);
        let _weighted_gen = WeightedRangeGenerator::new(rng, &[]);
    }

    #[test]
    fn weighted_range_generator_test() {
        const DRAWS: usize = 10_000;

        let rng = XorShiftRng::from_seed(SEED);
        let weighted_gen = WeightedRangeGenerator::new(rng, &[4, 1, 3, 2]);

        // Sample using the given weights.
        let mut hits = vec![0; weighted_gen.upper_bound()];
        for idx in weighted_gen.take(DRAWS) {
            hits[idx] += 1;
        }

        // Convert counts to a probability distribution.
        let probs: Vec<_> = hits
            .into_iter()
            .map(|count| count as f32 / DRAWS as f32)
            .collect();

        // Probabilities should be proportional to weights.
        assert!(all_close(&[0.4, 0.1, 0.3, 0.2], &probs, 1e-2));
    }

    #[test]
    fn zipf_range_generator_test() {
        const DRAWS: usize = 20_000;

        let rng = XorShiftRng::from_seed(SEED);
        let weighted_gen = ZipfRangeGenerator::new(rng, 4);

        // Sample using the given weights.
        let mut hits = vec![0; weighted_gen.upper_bound()];
        for idx in weighted_gen.take(DRAWS) {
            hits[idx] += 1;
        }

        // Convert counts to a probability distribution.
        let probs: Vec<_> = hits
            .into_iter()
            .map(|count| count as f32 / DRAWS as f32)
            .collect();

        // Probabilities should be proportional to weights.
        assert!(all_close(
            &[0.4958, 0.2302, 0.1912, 0.0828],
            probs.as_slice(),
            1e-2
        ));
        assert!(close(1.0f32, probs.iter().cloned().sum(), 1e-2));
    }

    #[test]
    fn banded_range_generator_test() {
        const DRAWS: usize = 20_000;

        let rng = XorShiftRng::from_seed(SEED);
        let inner_gen = ZipfRangeGenerator::new(rng, 4);

        let rng = XorShiftRng::from_seed(SEED2);
        let weighted_gen = BandedRangeGenerator::new(rng, inner_gen, 4);

        // Sample using the given weights.
        let mut hits = vec![0; weighted_gen.upper_bound()];
        for idx in weighted_gen.take(DRAWS) {
            hits[idx] += 1;
        }

        // Convert counts to a probability distribution.
        let probs: Vec<_> = hits
            .into_iter()
            .map(|count| count as f32 / DRAWS as f32)
            .collect();

        // Probabilities should be proportional to weights.
        eprintln!("{:?}", probs.as_slice());
        assert!(all_close(
            //&[0.4958, 0.2302, 0.1912, 0.0828],
            &[
                0.1240, 0.1240, 0.1240, 0.1240, 0.0576, 0.0576, 0.0576, 0.0576, 0.0478, 0.0478,
                0.0478, 0.0478, 0.0207, 0.0207, 0.0207, 0.0207
            ],
            probs.as_slice(),
            1e-2
        ));
        assert!(close(1.0f32, probs.iter().cloned().sum(), 1e-2));
    }
}
