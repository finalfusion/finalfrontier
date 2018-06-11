use rand::distributions::Sample;
use rand::Rng;
use zipf::ZipfDistribution;

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

    /// Get the upper bound in *[0, upper_bound)*.
    #[allow(dead_code)]
    pub fn upper_bound(&self) -> usize {
        self.prefix_sum.len()
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
            dist: ZipfDistribution::new(self.upper_bound, 1.0).unwrap(),
        }
    }
}

impl<R> ZipfRangeGenerator<R>
where
    R: Rng,
{
    pub fn new(rng: R, upper: usize) -> Self {
        ZipfRangeGenerator {
            upper_bound: upper,
            rng,
            dist: ZipfDistribution::new(upper, 1.0).unwrap(),
        }
    }

    #[allow(dead_code)]
    pub fn upper_bound(&self) -> usize {
        self.upper_bound
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

#[cfg(test)]
mod tests {
    use super::{WeightedRangeGenerator, ZipfRangeGenerator};
    use util::{all_close, close};

    use rand::XorShiftRng;

    #[test]
    #[should_panic]
    fn empty_weighted_range_generator() {
        let rng = XorShiftRng::new_unseeded();
        let _weighted_gen = WeightedRangeGenerator::new(rng, &[]);
    }

    #[test]
    fn weighted_range_generator_test() {
        const DRAWS: usize = 10_000;

        let rng = XorShiftRng::new_unseeded();
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
        const DRAWS: usize = 10_000;

        let rng = XorShiftRng::new_unseeded();
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
            &[0.6149, 0.1989, 0.1312, 0.055],
            probs.as_slice(),
            1e-2
        ));
        assert!(close(1.0f32, probs.iter().cloned().sum(), 1e-2));
    }
}
