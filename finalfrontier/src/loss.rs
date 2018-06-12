use ndarray::ArrayView1;

use util;
use vec_simd::dot;

/// Return the loss and gradient of the co-occurence classification.
///
/// This function returns the negative log likelihood and gradient of
/// a training instance using the probability function *P(1|x) =
/// σ(u·v)*. `u` and `v` are word embeddings and `label` is the
/// target label, where a label of `1` means that the words co-occur
/// and a label of `0` that they do not.
///
/// This model is very resembles logistic regression, except that we
/// optimize both u and v.
///
/// The loss is as follows (y is used as the label):
///
/// log(P(y|x)) =
/// y log(P(1|x)) + (1-y) log(P(0|x)) =
/// y log(P(1|x)) + (1-y) log(1 - P(1|x)) =
/// y log(σ(u·v)) + (1-y) log(1 - σ(u·v)) =
/// y log(σ(u·v)) + (1-y) log(σ(-u·v))
///
/// We can simplify the first term:
///
/// y log(σ(u·v)) =
/// y log(1/e^{-u·v}) =
/// -y log(e^{-u·v})
///
/// Then we find the derivative with respect to v_1:
///
/// ∂/∂v_1 -y log(e^{-u·v}) =
/// -y σ(u·v) ∂/∂v_1(e^{-u·v}) =
/// -y σ(u·v) e^{-u·v} -u_1 =
/// y σ(-u·v) u_1 =
/// y (1 - σ(u·v)) u_1 =
/// (y - yσ(u·v)) u_1
///
/// Iff y = 1, then:
///
/// 1 - σ(u·v)
///
/// For the second term above, we also find the derivative:
///
/// ∂/∂v_1 -(1 - y) log(e^{u·v}) =
/// -(1 - y) σ(-u·v) ∂/∂v_1(e^{u·v}) =
/// -(1 - y) σ(-u·v) e^{u·v} ∂/∂v_1 u·v=
/// -(1 - y) σ(-u·v) e^{u·v} u_1 =
/// -(1 - y) σ(u·v) u_1 =
/// (-σ(u·v) + yσ(u·v)) u_1
///
/// When y = 0 then:
///
/// -σ(u·v)u_1
///
/// Combining both, the partial derivative of v_1 is: y - σ(u·v)u_1
///
/// We return y - σ(u·v) as the gradient, so that the caller can compute
/// the gradient for all components of u and v.
fn log_logistic_loss(u: ArrayView1<f32>, v: ArrayView1<f32>, label: bool) -> (f32, f32) {
    let dp = dot(u, v);
    let lf = logistic_function(dp);
    let grad = (label as usize) as f32 - lf;
    let loss = if label {
        -util::safe_ln(lf)
    } else {
        -util::safe_ln(1.0 - lf)
    };

    (loss, grad)
}

/// Compute the logistic function.
///
/// **σ(a) = 1 / (1 + e^{-a})**
fn logistic_function(a: f32) -> f32 {
    1.0 / (1.0 + (-a).exp())
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use util::{all_close, close};

    use super::{log_logistic_loss, logistic_function};

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

    #[test]
    fn log_logistic_loss_test() {
        let a = Array1::from_shape_vec((6,), vec![1., 1., 1., 0., 0., 0.]).unwrap();
        let a_orth = Array1::from_shape_vec((6,), vec![0., 0., 0., 1., 1., 1.]).unwrap();
        let a_opp = Array1::from_shape_vec((6,), vec![-1., -1., -1., 0., 0., 0.]).unwrap();

        let (loss, gradient) = log_logistic_loss(a.view(), a_orth.view(), true);
        assert!(close(loss, 0.69312, 1e-5));
        assert!(close(gradient, 0.5, 1e-5));

        let (loss, gradient) = log_logistic_loss(a.view(), a_orth.view(), false);
        assert!(close(loss, 0.69312, 1e-5));
        assert!(close(gradient, -0.5, 1e-5));

        let (loss, gradient) = log_logistic_loss(a.view(), a.view(), true);
        assert!(close(loss, 0.04858, 1e-5));
        assert!(close(gradient, 0.04742, 1e-5));

        let (loss, gradient) = log_logistic_loss(a.view(), a_opp.view(), false);
        assert!(close(loss, 0.04858, 1e-5));
        assert!(close(gradient, -0.04743, 1e-5));

        let (loss, gradient) = log_logistic_loss(a.view(), a_opp.view(), true);
        assert!(close(loss, 3.04838, 1e-5));
        assert!(close(gradient, 0.95257, 1e-5));
    }
}
