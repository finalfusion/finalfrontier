#[cfg(test)]
fn close(a: f32, b: f32, eps: f32) -> bool {
    let diff = (a - b).abs();
    if diff > eps {
        return false;
    }

    true
}

#[cfg(test)]
pub fn all_close(a: &[f32], b: &[f32], eps: f32) -> bool {
    for (&av, &bv) in a.iter().zip(b) {
        if !close(av, bv, eps) {
            return false;
        }
    }

    true
}
