/// Compute the logistic function.
///
/// **σ(a) = 1 / (1 + e^{-a})**
fn logistic_function(a: f32) -> f32 {
    1.0 / (1.0 + (-a).exp())
}

#[cfg(test)]
mod tests {
    use util::all_close;

    use super::logistic_function;

    #[test]
    fn logistic_function_test() {
        let activations = &[-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let outputs: Vec<_> = activations.iter().map(|&a| logistic_function(a)).collect();
        assert!(all_close(
            &[
                0.00669, 0.01799, 0.04743, 0.11920, 0.26894, 0.5, 0.73106, 0.88080, 0.95257,
                0.982014, 0.99331
            ],
            outputs.as_slice(),
            1e-5
        ));
    }
}
