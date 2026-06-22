use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ConsistencyLevel {
    One,
    Quorum,
    All,
}

impl ConsistencyLevel {
    pub fn to_i32(self) -> i32 {
        match self {
            Self::One => 1,
            Self::Quorum => 2,
            Self::All => 3,
        }
    }
}

impl std::str::FromStr for ConsistencyLevel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "one" => Ok(Self::One),
            "quorum" => Ok(Self::Quorum),
            "all" => Ok(Self::All),
            _ => Err(format!("unknown consistency level: {s}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistency_level_debug_clone_copy() {
        let c = ConsistencyLevel::Quorum;
        let _d = format!("{:?}", c);
        let _c = c;
        let _e = c;
    }
}
