use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum RekhaError {
    NotFound(String),
    InvalidArgument(String),
    Storage(String),
    Index(String),
    Timeout(String),
    Unavailable(String),
    Internal(String),
    Serialization(String),
}

impl fmt::Display for RekhaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RekhaError::NotFound(msg) => write!(f, "not found: {}", msg),
            RekhaError::InvalidArgument(msg) => write!(f, "invalid argument: {}", msg),
            RekhaError::Storage(msg) => write!(f, "storage error: {}", msg),
            RekhaError::Index(msg) => write!(f, "index error: {}", msg),
            RekhaError::Timeout(msg) => write!(f, "timeout: {}", msg),
            RekhaError::Unavailable(msg) => write!(f, "unavailable: {}", msg),
            RekhaError::Internal(msg) => write!(f, "internal error: {}", msg),
            RekhaError::Serialization(msg) => write!(f, "serialization error: {}", msg),
        }
    }
}

impl std::error::Error for RekhaError {}

impl From<String> for RekhaError {
    fn from(s: String) -> Self {
        RekhaError::Internal(s)
    }
}

impl From<&str> for RekhaError {
    fn from(s: &str) -> Self {
        RekhaError::Internal(s.to_string())
    }
}

impl From<std::io::Error> for RekhaError {
    fn from(e: std::io::Error) -> Self {
        RekhaError::Storage(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, RekhaError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_display_not_found() {
        let e = RekhaError::NotFound("vector 42".into());
        assert_eq!(e.to_string(), "not found: vector 42");
    }

    #[test]
    fn test_display_invalid_argument() {
        let e = RekhaError::InvalidArgument("bad dims".into());
        assert_eq!(e.to_string(), "invalid argument: bad dims");
    }

    #[test]
    fn test_display_storage() {
        let e = RekhaError::Storage("disk full".into());
        assert_eq!(e.to_string(), "storage error: disk full");
    }

    #[test]
    fn test_display_index() {
        let e = RekhaError::Index("build failed".into());
        assert_eq!(e.to_string(), "index error: build failed");
    }

    #[test]
    fn test_display_timeout() {
        let e = RekhaError::Timeout("search after 5000ms".into());
        assert_eq!(e.to_string(), "timeout: search after 5000ms");
    }

    #[test]
    fn test_display_unavailable() {
        let e = RekhaError::Unavailable("node down".into());
        assert_eq!(e.to_string(), "unavailable: node down");
    }

    #[test]
    fn test_display_internal() {
        let e = RekhaError::Internal("oops".into());
        assert_eq!(e.to_string(), "internal error: oops");
    }

    #[test]
    fn test_display_serialization() {
        let e = RekhaError::Serialization("invalid encoding".into());
        assert_eq!(e.to_string(), "serialization error: invalid encoding");
    }

    #[test]
    fn test_error_trait() {
        let e = RekhaError::NotFound("test".into());
        assert!(e.source().is_none());
    }

    #[test]
    fn test_from_string() {
        let e: RekhaError = "error".into();
        assert!(matches!(e, RekhaError::Internal(_)));
    }

    #[test]
    fn test_from_io_error() {
        let io = std::io::Error::new(std::io::ErrorKind::Other, "io error");
        let e: RekhaError = io.into();
        assert!(matches!(e, RekhaError::Storage(_)));
    }
}
