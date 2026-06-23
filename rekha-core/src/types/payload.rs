use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Payload {
    pub content_type: PayloadType,
    pub data: Vec<u8>,
}

impl Payload {
    pub fn from_text(text: impl Into<String>) -> Self {
        Self {
            content_type: PayloadType::Text,
            data: text.into().into_bytes(),
        }
    }

    pub fn from_json<T: Serialize>(value: &T) -> Result<Self, serde_json::Error> {
        Ok(Self {
            content_type: PayloadType::Json,
            data: serde_json::to_vec(value)?,
        })
    }

    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self {
            content_type: PayloadType::Raw,
            data,
        }
    }

    pub fn as_text(&self) -> Option<String> {
        if matches!(self.content_type, PayloadType::Text) {
            String::from_utf8(self.data.clone()).ok()
        } else {
            None
        }
    }

    pub fn as_json<T: serde::de::DeserializeOwned>(&self) -> Option<T> {
        if matches!(self.content_type, PayloadType::Json) {
            serde_json::from_slice(&self.data).ok()
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PayloadType {
    Text,
    Json,
    Raw,
}

impl std::fmt::Display for PayloadType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text => write!(f, "text"),
            Self::Json => write!(f, "json"),
            Self::Raw => write!(f, "raw"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payload_from_text() {
        let p = Payload::from_text("hello");
        assert_eq!(p.content_type, PayloadType::Text);
        assert_eq!(p.as_text(), Some("hello".into()));
    }

    #[test]
    fn test_payload_from_json() {
        let p = Payload::from_json(&serde_json::json!({"key": "value"})).unwrap();
        assert_eq!(p.content_type, PayloadType::Json);
        let val: serde_json::Value = p.as_json().unwrap();
        assert_eq!(val["key"], "value");
    }

    #[test]
    fn test_payload_from_bytes() {
        let data = vec![1, 2, 3];
        let p = Payload::from_bytes(data.clone());
        assert_eq!(p.content_type, PayloadType::Raw);
        assert!(p.as_text().is_none());
        assert!(p.as_json::<serde_json::Value>().is_none());
    }

    #[test]
    fn test_payload_type_display() {
        assert_eq!(PayloadType::Text.to_string(), "text");
        assert_eq!(PayloadType::Json.to_string(), "json");
        assert_eq!(PayloadType::Raw.to_string(), "raw");
    }
}
