use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LwwTimestamp {
    pub timestamp: i64,
    pub node_id: u64,
}

impl LwwTimestamp {
    pub fn new(now: i64, node_id: u64) -> Self {
        LwwTimestamp {
            timestamp: now,
            node_id,
        }
    }

    pub fn resolve(local: i64, remote: i64, local_node: u64, remote_node: u64) -> Ordering {
        match local.cmp(&remote) {
            Ordering::Equal => local_node.cmp(&remote_node),
            other => other,
        }
    }

    pub fn should_keep_local(
        local_ts: i64,
        remote_ts: i64,
        local_node: u64,
        remote_node: u64,
    ) -> bool {
        Self::resolve(local_ts, remote_ts, local_node, remote_node) != Ordering::Less
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newer_timestamp_wins() {
        assert!(LwwTimestamp::should_keep_local(100, 50, 1, 2));
        assert!(!LwwTimestamp::should_keep_local(50, 100, 1, 2));
    }

    #[test]
    fn test_tie_broken_by_node_id() {
        assert!(LwwTimestamp::should_keep_local(100, 100, 5, 3));
        assert!(!LwwTimestamp::should_keep_local(100, 100, 3, 5));
    }

    #[test]
    fn test_equal_all() {
        assert!(LwwTimestamp::should_keep_local(100, 100, 5, 5));
    }
}
