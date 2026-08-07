//! Chroma-exact distance kernels, with SIMD acceleration via [`wide`].
//!
//! Semantics are copied from Chroma's distance crate (which itself ports
//! Qdrant's `spaces/simple.rs` and hnswlib's space implementations):
//!
//! - [`l2_squared`]: squared Euclidean distance `Σ(xᵢ − yᵢ)²` — **no** square
//!   root.
//! - [`dot`]: raw dot product.
//! - [`distance`] with [`Distance::Ip`]: `1 − dot(x, y)`.
//! - [`distance`] with [`Distance::Cosine`]: vectors are L2-normalized
//!   (with the `1e-32` epsilon added to the norm), then `1 − dot`.
//!
//! Lower distance always means closer.
//!
//! Kernels operate on `&[f32]` and panic on dimension mismatch. The SIMD path
//! processes 8-wide blocks with fused multiply-add and falls back to scalar for
//! the tail; [`dot_scalar`]/[`l2_squared_scalar`] are the reference scalar
//! implementations kept for tests.

use rekha_core::types::{Distance, Embedding};
use wide::f32x8;

/// Epsilon added to the L2 norm during cosine normalization, matching Chroma.
pub const NORM_EPSILON: f32 = 1e-32;

/// L2-normalizes `v` (`v / (‖v‖ + 1e-32)`). If the norm is `<= NORM_EPSILON`
/// (i.e. a zero vector), `v` is returned unchanged.
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = dot(v, v).sqrt();
    if norm <= NORM_EPSILON {
        return v.to_vec();
    }
    let scale = 1.0 / (norm + NORM_EPSILON);
    v.iter().map(|&x| x * scale).collect()
}

/// Squared Euclidean distance between `a` and `b` (no square root).
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "l2_squared: dimension mismatch: {} != {}",
        a.len(),
        b.len()
    );
    let mut acc = f32x8::splat(0.0);
    let chunks = a.len() / 8;
    for i in 0..chunks {
        let x = f32x8::from(&a[i * 8..i * 8 + 8]);
        let y = f32x8::from(&b[i * 8..i * 8 + 8]);
        let d = x - y;
        acc = d.mul_add(d, acc);
    }
    let mut sum = acc.reduce_add();
    for i in (chunks * 8)..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// Raw dot product of `a` and `b` (NOT `1 − dot`).
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "dot: dimension mismatch: {} != {}",
        a.len(),
        b.len()
    );
    let mut acc = f32x8::splat(0.0);
    let chunks = a.len() / 8;
    for i in 0..chunks {
        let x = f32x8::from(&a[i * 8..i * 8 + 8]);
        let y = f32x8::from(&b[i * 8..i * 8 + 8]);
        acc = x.mul_add(y, acc);
    }
    let mut sum = acc.reduce_add();
    for i in (chunks * 8)..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Distance for `space` against raw (possibly non-normalized) vectors.
///
/// - [`Distance::L2`] → [`l2_squared`]
/// - [`Distance::Ip`] → `1 − dot(a, b)`
/// - [`Distance::Cosine`] → `1 − dot(normalize(a), normalize(b))`
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
pub fn distance(space: Distance, a: &[f32], b: &[f32]) -> f32 {
    match space {
        Distance::L2 => l2_squared(a, b),
        Distance::Ip => 1.0 - dot(a, b),
        Distance::Cosine => 1.0 - dot(&normalize(a), &normalize(b)),
    }
}

/// Precomputes the raw L2 norm of every embedding in a batch. Callers
/// (index/engine) add [`NORM_EPSILON`] themselves when normalizing, so this
/// stays a pure "norm once" helper.
pub fn batch_norms(vectors: &[Embedding]) -> Vec<f32> {
    vectors.iter().map(|v| dot(v, v).sqrt()).collect()
}

/// Reference scalar dot product, kept for cross-checking the SIMD path.
#[cfg(test)]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Reference scalar squared-L2, kept for cross-checking the SIMD path.
#[cfg(test)]
fn l2_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_diff(a: f32, b: f32) -> f32 {
        (a - b).abs() / b.abs().max(1e-12)
    }

    #[test]
    fn normalize_zero_vector_unchanged() {
        assert_eq!(normalize(&[0.0, 0.0, 0.0]), vec![0.0, 0.0, 0.0]);
        assert_eq!(normalize(&[]), Vec::<f32>::new());
    }

    #[test]
    fn normalize_unit_vector_unchanged() {
        let v = normalize(&[1.0, 0.0, 0.0]);
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 0.0);
    }

    #[test]
    fn normalize_three_four() {
        let v = normalize(&[3.0, 4.0]);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_squared_basic() {
        assert_eq!(l2_squared(&[0.0, 0.0], &[3.0, 4.0]), 25.0);
        assert_eq!(l2_squared_scalar(&[0.0, 0.0], &[3.0, 4.0]), 25.0);
    }

    #[test]
    fn dot_basic() {
        assert_eq!(dot(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
        assert_eq!(dot_scalar(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
    }

    #[test]
    fn simd_matches_scalar() {
        for dim in [7, 8, 9, 64, 65, 100] {
            let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 1.7 + 0.3) % 10.0).collect();
            let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 3.1 - 1.5) % 10.0).collect();

            let d_simd = dot(&a, &b);
            let d_scalar = dot_scalar(&a, &b);
            assert!(
                rel_diff(d_simd, d_scalar) < 1e-4,
                "dot dim {dim}: simd {d_simd} vs scalar {d_scalar}"
            );

            let l_simd = l2_squared(&a, &b);
            let l_scalar = l2_squared_scalar(&a, &b);
            assert!(
                rel_diff(l_simd, l_scalar) < 1e-4,
                "l2_squared dim {dim}: simd {l_simd} vs scalar {l_scalar}"
            );
        }
    }

    #[test]
    fn distance_l2_is_squared() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_eq!(distance(Distance::L2, &a, &b), l2_squared(&a, &b));
    }

    #[test]
    fn distance_ip_is_one_minus_dot() {
        let a = [0.5, 1.5, 2.5];
        let b = [1.0, -0.5, 0.25];
        assert_eq!(distance(Distance::Ip, &a, &b), 1.0 - dot(&a, &b));
    }

    #[test]
    fn distance_cosine_orthogonal() {
        assert_eq!(distance(Distance::Cosine, &[1.0, 0.0], &[0.0, 1.0]), 1.0);
    }

    #[test]
    fn distance_cosine_same_vector_is_near_zero() {
        let a = [1.0, 2.0, 3.0];
        assert!(distance(Distance::Cosine, &a, &a) < 1e-6);
    }

    #[test]
    fn batch_norms_matches_individual() {
        let v1: Embedding = vec![3.0, 4.0].into();
        let v2: Embedding = vec![1.0, 2.0, 2.0].into();
        let norms = batch_norms(&[v1, v2]);
        assert!((norms[0] - 5.0).abs() < 1e-6);
        assert!((norms[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn dot_panics_on_dimension_mismatch() {
        let _ = dot(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn l2_squared_panics_on_dimension_mismatch() {
        let _ = l2_squared(&[1.0, 2.0], &[1.0]);
    }
}
