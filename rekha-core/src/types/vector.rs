use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: u64,
    pub timestamp: u64,
    pub data: Option<Vec<f32>>,
    pub is_tombstone: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector {
    pub id: u64,
    pub data: Vec<f32>,
    #[serde(default)]
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedVector {
    pub id: u64,
    pub pq_code: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_timestamp_default() {
        let v = Vector { id: 1, data: vec![1.0, 2.0], timestamp: 0 };
        assert_eq!(v.timestamp, 0);
    }

    #[test]
    fn test_vector_record_tombstone() {
        let r = VectorRecord {
            id: 42,
            timestamp: 100,
            data: None,
            is_tombstone: true,
        };
        assert!(r.is_tombstone);
        assert!(r.data.is_none());
    }
}
