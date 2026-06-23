pub struct LwwResolver;

impl LwwResolver {
    pub fn should_apply(new_ts: u64, existing_ts: u64) -> bool {
        new_ts > existing_ts
    }

    pub fn resolve_timestamp(ts: u64) -> u64 {
        if ts == 0 { rekha_core::now_micros() } else { ts }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newer_wins() {
        assert!(LwwResolver::should_apply(200, 100));
        assert!(!LwwResolver::should_apply(50, 100));
        assert!(!LwwResolver::should_apply(100, 100));
    }

    #[test]
    fn test_zero_timestamp_resolved() {
        let ts = LwwResolver::resolve_timestamp(0);
        assert!(ts > 0);
        assert_eq!(LwwResolver::resolve_timestamp(100), 100);
    }
}
