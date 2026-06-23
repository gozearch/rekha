use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchParams {
    pub ef_search: usize,
    pub nprobe: usize,
    pub include_payloads: bool,
    pub local_only: bool,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            ef_search: 128,
            nprobe: 16,
            include_payloads: true,
            local_only: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchStats {
    pub total_ms: f64,
    pub nodes_contacted: u32,
    pub vectors_scanned: u64,
    pub warnings: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_params_default() {
        let p = SearchParams::default();
        assert_eq!(p.ef_search, 128);
        assert_eq!(p.nprobe, 16);
        assert!(p.include_payloads);
    }

    #[test]
    fn test_search_stats_default() {
        let s = SearchStats::default();
        assert_eq!(s.total_ms, 0.0);
        assert_eq!(s.nodes_contacted, 0);
        assert_eq!(s.vectors_scanned, 0);
        assert!(s.warnings.is_empty());
    }
}
