use rand::Rng;

/// An iterator that draws from *[0, n)* with integer weights.
///
/// This iterator returns integers from *[0, n)*, where the probability of
/// each integer is weighted.
///
/// See: Geometric Approximation Algorithms, Sariel Har-Peled, pp. 88
struct WeightedRangeGenerator<R> {
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

#[cfg(test)]
mod tests {
    use rand::XorShiftRng;
    use util::all_close;

    use super::WeightedRangeGenerator;

    #[test]
    #[should_panic]
    fn empty_weighted_range_generator() {
        let mut rng = XorShiftRng::new_unseeded();
        let weighted_gen = WeightedRangeGenerator::new(rng, &[]);
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
        let probs: Vec<_> = hits.into_iter()
            .map(|count| count as f32 / DRAWS as f32)
            .collect();

        // Probabilities should be proportional to weights.
        assert!(all_close(&[0.4, 0.1, 0.3, 0.2], &probs, 1e-2));
    }
}
