use serde::{Deserialize, Serialize};

use crate::Payload;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredPoint {
    pub id: u64,
    pub score: f32,
    pub payload: Option<Payload>,
    #[serde(default)]
    pub timestamp: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scored_point_timestamp_default() {
        let sp = ScoredPoint {
            id: 1,
            score: 0.5,
            payload: None,
            timestamp: 0,
        };
        assert_eq!(sp.timestamp, 0);
    }
}
