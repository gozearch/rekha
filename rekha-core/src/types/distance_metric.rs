use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum DistanceMetric {
    L2,
    Cosine,
    InnerProduct,
}

impl DistanceMetric {
    pub fn name(&self) -> &'static str {
        match self {
            Self::L2 => "l2",
            Self::Cosine => "cosine",
            Self::InnerProduct => "inner_product",
        }
    }

    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        crate::distance::distance(a, b, *self)
    }
}

impl std::str::FromStr for DistanceMetric {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "l2" | "euclidean" => Ok(Self::L2),
            "cosine" | "cos" => Ok(Self::Cosine),
            "ip" | "inner_product" => Ok(Self::InnerProduct),
            _ => Err(format!("unknown distance metric: {s}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_metric_name() {
        assert_eq!(DistanceMetric::L2.name(), "l2");
        assert_eq!(DistanceMetric::Cosine.name(), "cosine");
        assert_eq!(DistanceMetric::InnerProduct.name(), "inner_product");
    }

    #[test]
    fn test_distance_metric_from_str() {
        assert_eq!(
            "l2".parse::<DistanceMetric>().unwrap(),
            DistanceMetric::L2
        );
        assert_eq!(
            "euclidean".parse::<DistanceMetric>().unwrap(),
            DistanceMetric::L2
        );
        assert_eq!(
            "cosine".parse::<DistanceMetric>().unwrap(),
            DistanceMetric::Cosine
        );
        assert_eq!(
            "cos".parse::<DistanceMetric>().unwrap(),
            DistanceMetric::Cosine
        );
        assert_eq!(
            "ip".parse::<DistanceMetric>().unwrap(),
            DistanceMetric::InnerProduct
        );
        assert!("unknown".parse::<DistanceMetric>().is_err());
    }

    #[test]
    fn test_distance_metric_dispatch() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let d = DistanceMetric::L2.distance(&a, &b);
        assert!((d - 8.0).abs() < 1e-6);
        let d = DistanceMetric::Cosine.distance(&a, &b);
        assert!(d >= 0.0);
    }
}
