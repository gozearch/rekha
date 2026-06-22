use std::time::{SystemTime, UNIX_EPOCH};

pub fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn now_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

pub fn quorum(rf: usize) -> usize {
    rf / 2 + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_now_micros_nonzero() {
        let t = now_micros();
        assert!(t > 1_700_000_000_000_000);
    }

    #[test]
    fn test_quorum() {
        assert_eq!(quorum(1), 1);
        assert_eq!(quorum(2), 2);
        assert_eq!(quorum(3), 2);
        assert_eq!(quorum(5), 3);
    }
}
