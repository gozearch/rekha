use rekha_core::ConsistencyLevel;

pub struct ConsistencyGate;

impl ConsistencyGate {
    pub fn required_acks(rf: usize, level: ConsistencyLevel) -> usize {
        match level {
            ConsistencyLevel::One => 1,
            ConsistencyLevel::Quorum => (rf / 2) + 1,
            ConsistencyLevel::All => rf,
        }
    }

    pub fn is_quorum_satisfied(acks: usize, rf: usize, level: ConsistencyLevel) -> bool {
        acks >= Self::required_acks(rf, level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_required_acks() {
        assert_eq!(ConsistencyGate::required_acks(3, ConsistencyLevel::One), 1);
        assert_eq!(
            ConsistencyGate::required_acks(3, ConsistencyLevel::Quorum),
            2
        );
        assert_eq!(ConsistencyGate::required_acks(3, ConsistencyLevel::All), 3);
        assert_eq!(
            ConsistencyGate::required_acks(5, ConsistencyLevel::Quorum),
            3
        );
    }

    #[test]
    fn test_quorum_satisfied() {
        assert!(ConsistencyGate::is_quorum_satisfied(
            2,
            3,
            ConsistencyLevel::Quorum
        ));
        assert!(!ConsistencyGate::is_quorum_satisfied(
            1,
            3,
            ConsistencyLevel::Quorum
        ));
        assert!(ConsistencyGate::is_quorum_satisfied(
            1,
            3,
            ConsistencyLevel::One
        ));
        assert!(ConsistencyGate::is_quorum_satisfied(
            3,
            3,
            ConsistencyLevel::All
        ));
        assert!(!ConsistencyGate::is_quorum_satisfied(
            2,
            3,
            ConsistencyLevel::All
        ));
    }
}
