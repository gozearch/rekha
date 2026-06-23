use rekha_core::ConsistencyLevel;

pub struct ConsistencyGate;

impl ConsistencyGate {
    pub fn required(consistency: ConsistencyLevel, rf: usize) -> usize {
        match consistency {
            ConsistencyLevel::One => 1,
            ConsistencyLevel::Quorum => rf / 2 + 1,
            ConsistencyLevel::All => rf,
        }
    }

    pub fn met(acks: usize, required: usize) -> bool {
        acks >= required
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rekha_core::ConsistencyLevel;

    #[test]
    fn test_required_one() {
        assert_eq!(ConsistencyGate::required(ConsistencyLevel::One, 3), 1);
    }

    #[test]
    fn test_required_quorum() {
        assert_eq!(ConsistencyGate::required(ConsistencyLevel::Quorum, 1), 1);
        assert_eq!(ConsistencyGate::required(ConsistencyLevel::Quorum, 2), 2);
        assert_eq!(ConsistencyGate::required(ConsistencyLevel::Quorum, 3), 2);
        assert_eq!(ConsistencyGate::required(ConsistencyLevel::Quorum, 5), 3);
    }

    #[test]
    fn test_required_all() {
        assert_eq!(ConsistencyGate::required(ConsistencyLevel::All, 5), 5);
    }

    #[test]
    fn test_met() {
        assert!(ConsistencyGate::met(2, 2));
        assert!(ConsistencyGate::met(3, 2));
        assert!(!ConsistencyGate::met(1, 2));
    }
}
