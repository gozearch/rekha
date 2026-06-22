use crate::types::DistanceMetric;

/// Compute distance between two vectors using the specified metric.
pub fn distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::L2 => l2_squared(a, b),
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::InnerProduct => inner_product_distance(a, b),
    }
}

/// Squared L2 distance: sum((a_i - b_i)^2)
///
/// Returns the squared distance. The caller can sqrt if they want Euclidean distance.
/// We keep it squared for early-stop pruning — if the partial squared distance
/// already exceeds the threshold, we can stop early.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dimension mismatch in l2_squared");
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

/// Partial squared L2 distance over a dimension range [start, end).
/// Used for early-stop pruning in dimension-based partitioning.
#[inline]
pub fn l2_squared_partial(a: &[f32], b: &[f32], start: usize, end: usize) -> f32 {
    debug_assert!(
        a.len() >= end && b.len() >= end,
        "dimension range out of bounds"
    );
    let mut sum = 0.0f32;
    for i in start..end {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

/// Cosine distance: 1 - cos_sim
/// cos_sim = dot(a, b) / (||a|| * ||b||)
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dimension mismatch in cosine_distance");
    let dot = dot_product(a, b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    1.0 - (dot / (norm_a * norm_b))
}

/// Inner product distance: -dot(a, b)
/// (We negate so that smaller distance = more similar)
#[inline]
pub fn inner_product_distance(a: &[f32], b: &[f32]) -> f32 {
    -dot_product(a, b)
}

/// Dot product of two vectors.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dimension mismatch in dot_product");
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// L2 norm (magnitude) of a vector.
#[inline]
pub fn l2_norm(a: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &val in a {
        sum += val * val;
    }
    sum.sqrt()
}

/// For early-stop pruning in L2 distance:
/// After computing distance across dimensions [0, computed_end),
/// the remaining dimensions can contribute at minimum 0 to the total.
/// So partial_dist <= total_dist always (monotonicity).
/// We can stop if partial_dist > current_kth_dist.
///
/// For cosine distance, early-stop is approximate: partial dot product
/// doesn't have the same monotonic guarantee.
pub fn can_early_stop(partial_dist: f32, current_kth_dist: f32, metric: DistanceMetric) -> bool {
    match metric {
        DistanceMetric::L2 => partial_dist > current_kth_dist,
        DistanceMetric::Cosine | DistanceMetric::InnerProduct => {
            // For cosine/IP, early stop is approximate
            partial_dist > current_kth_dist * 1.1 // conservative heuristic
        }
    }
}

/// Verify that all vectors in a batch have the same expected dimension.
pub fn validate_dimensions(vectors: &[&[f32]], expected: usize) -> Result<(), crate::RekhaError> {
    for v in vectors.iter() {
        if v.len() != expected {
            return Err(crate::RekhaError::InvalidDimension {
                expected,
                actual: v.len(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod proptests {
    use proptest::prelude::*;
    use crate::distance::*;

    fn valid_vec() -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-1e10f32..1e10f32, 1..10usize)
    }

    proptest! {
        #[test]
        fn l2_squared_nonnegative(a in valid_vec(), b in valid_vec()) {
            if a.len() == b.len() {
                let d = l2_squared(&a, &b);
                assert!(d >= 0.0 || d.is_nan());
            }
        }

        #[test]
        fn l2_squared_symmetric(a in valid_vec(), b in valid_vec()) {
            if a.len() == b.len() {
                let d1 = l2_squared(&a, &b);
                let d2 = l2_squared(&b, &a);
                assert!((d1 - d2).abs() < 1e-6, "l2_squared not symmetric: d1={d1}, d2={d2}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_squared() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27
        assert!((l2_squared(&a, &b) - 27.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_squared_partial() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        // dims 0..2: (1-2)^2 + (2-3)^2 = 1 + 1 = 2
        assert!((l2_squared_partial(&a, &b, 0, 2) - 2.0).abs() < 1e-6);
        // dims 2..4: (3-4)^2 + (4-5)^2 = 1 + 1 = 2
        assert!((l2_squared_partial(&a, &b, 2, 4) - 2.0).abs() < 1e-6);
        // full: 2 + 2 = 4
        assert!((l2_squared_partial(&a, &b, 0, 4) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        // orthogonal → cos_sim = 0 → distance = 1
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![1.0, 0.0];
        // identical → cos_sim = 1 → distance = 0
        assert!(cosine_distance(&a, &c).abs() < 1e-6);
    }

    #[test]
    fn test_early_stop_l2() {
        // L2 partial dist > threshold → can stop
        assert!(can_early_stop(10.0, 5.0, DistanceMetric::L2));
        // L2 partial dist <= threshold → cannot stop
        assert!(!can_early_stop(3.0, 5.0, DistanceMetric::L2));
    }

    #[test]
    fn test_validate_dimensions() {
        let v1 = vec![0.1f32; 128];
        let v2 = vec![0.2f32; 128];
        assert!(validate_dimensions(&[&v1, &v2], 128).is_ok());
        let v3 = vec![0.3f32; 64];
        assert!(validate_dimensions(&[&v1, &v3], 128).is_err());
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_norm() {
        let a = vec![3.0, 4.0];
        // sqrt(9 + 16) = sqrt(25) = 5
        assert!((l2_norm(&a) - 5.0).abs() < 1e-6);
        let zero = vec![0.0, 0.0];
        assert!(l2_norm(&zero).abs() < 1e-6);
    }

    #[test]
    fn test_inner_product_distance() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        // dot = 11, distance = -11
        assert!((inner_product_distance(&a, &b) - (-11.0)).abs() < 1e-6);
    }

    #[test]
    fn test_distance_dispatch() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        // L2: (1-0)^2 + (0-1)^2 = 2
        assert!((distance(&a, &b, DistanceMetric::L2) - 2.0).abs() < 1e-6);
        // Cosine: orthogonal → 1
        assert!((distance(&a, &b, DistanceMetric::Cosine) - 1.0).abs() < 1e-6);
        // InnerProduct: -(1*0 + 0*1) = 0
        assert!(distance(&a, &b, DistanceMetric::InnerProduct).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance_zero_norm() {
        let zero = vec![0.0, 0.0];
        let a = vec![1.0, 0.0];
        // zero vector returns 1.0
        assert!((cosine_distance(&zero, &a) - 1.0).abs() < 1e-6);
        assert!((cosine_distance(&a, &zero) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_early_stop_approximate() {
        // For cosine/IP, threshold is relaxed (1.1x)
        assert!(can_early_stop(12.0, 10.0, DistanceMetric::Cosine));
        assert!(!can_early_stop(10.5, 10.0, DistanceMetric::InnerProduct));
    }
}
