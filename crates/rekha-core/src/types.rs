//! Core data types: embeddings, ids, documents, metadata, and distance spaces.
//!
//! `Embedding` is an `Arc<[f32]>` so vectors can be shared cheaply across the
//! WAL, in-memory buffers, and derived indexes without copying. `Id` and
//! `Document` are plain strings. `MetadataValue` mirrors Chroma's metadata
//! types (string / bool / int / float), and `Distance` names the three Chroma
//! distance spaces with their exact names.

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A dense embedding vector. Shared via [`Arc`] so WAL records, brute-force
/// buffers, and indexes never duplicate the underlying data.
///
/// serde does not implement `Deserialize` for `Arc<[T]>` (or `Serialize`
/// without the `rc` feature), so fields of this type should use
/// `#[serde(with = "rekha_core::types::embedding_serde")]` — see the
/// `embedding_serde` module.
pub type Embedding = Arc<[f32]>;

/// Serde helpers for [`Embedding`]: the value round-trips as a plain array of
/// `f32`. Used via `#[serde(with = "rekha_core::types::embedding_serde")]` on
/// fields of type [`Embedding`] or `Option<Embedding>` — e.g. in
/// [`crate::op::Operation`] and the engine's `Record`. Public so crates outside
/// `rekha-core` can derive serde on their own embedding-carrying types.
pub mod embedding_serde {
    use std::sync::Arc;

    use serde::de::Deserializer;
    use serde::ser::Serializer;
    use serde::{Deserialize, Serialize};

    pub fn serialize<S>(e: &Arc<[f32]>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        e.as_ref().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Arc<[f32]>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = Vec::<f32>::deserialize(deserializer)?;
        Ok(v.into())
    }

    /// Like [`serialize`], but for `Option<Embedding>` fields (e.g. the
    /// engine's `Record`, whose `update` path carries no vector). Wired up via
    /// `#[serde(serialize_with = "...", deserialize_with = "...")]`.
    pub fn serialize_opt<S>(e: &Option<Arc<[f32]>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match e {
            Some(v) => serializer.serialize_some(v.as_ref()),
            None => serializer.serialize_none(),
        }
    }

    /// [`Deserialize`] counterpart to [`serialize_opt`].
    pub fn deserialize_opt<'de, D>(deserializer: D) -> Result<Option<Arc<[f32]>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = <Option<Vec<f32>>>::deserialize(deserializer)?;
        Ok(v.map(Into::into))
    }
}

/// Stable identifier for a vector / record. Strings, as in Chroma.
pub type Id = String;

/// Free-text document payload associated with a vector.
pub type Document = String;

/// A single metadata value, mirroring the JSON-compatible types Chroma stores
/// (`str`, `bool`, `int`, `float`).
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    Str(String),
    Bool(bool),
    Int(i64),
    Float(f64),
}

impl serde::Serialize for MetadataValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if serializer.is_human_readable() {
            match self {
                MetadataValue::Str(s) => serializer.serialize_str(s),
                MetadataValue::Bool(b) => serializer.serialize_bool(*b),
                MetadataValue::Int(i) => serializer.serialize_i64(*i),
                MetadataValue::Float(f) => serializer.serialize_f64(*f),
            }
        } else {
            match self {
                MetadataValue::Str(s) => {
                    serializer.serialize_newtype_variant("MetadataValue", 0, "Str", s)
                }
                MetadataValue::Bool(b) => {
                    serializer.serialize_newtype_variant("MetadataValue", 1, "Bool", b)
                }
                MetadataValue::Int(i) => {
                    serializer.serialize_newtype_variant("MetadataValue", 2, "Int", i)
                }
                MetadataValue::Float(f) => {
                    serializer.serialize_newtype_variant("MetadataValue", 3, "Float", f)
                }
            }
        }
    }
}

impl<'de> serde::Deserialize<'de> for MetadataValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            let v = serde_json::Value::deserialize(deserializer)?;
            match v {
                serde_json::Value::String(s) => Ok(MetadataValue::Str(s)),
                serde_json::Value::Bool(b) => Ok(MetadataValue::Bool(b)),
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        Ok(MetadataValue::Int(i))
                    } else if let Some(f) = n.as_f64() {
                        Ok(MetadataValue::Float(f))
                    } else {
                        Err(serde::de::Error::custom("invalid number for MetadataValue"))
                    }
                }
                _ => Err(serde::de::Error::custom(
                    "invalid metadata value: expected string, bool, or number",
                )),
            }
        } else {
            #[derive(serde::Deserialize)]
            enum Helper {
                Str(String),
                Bool(bool),
                Int(i64),
                Float(f64),
            }
            let h = Helper::deserialize(deserializer)?;
            Ok(match h {
                Helper::Str(s) => MetadataValue::Str(s),
                Helper::Bool(b) => MetadataValue::Bool(b),
                Helper::Int(i) => MetadataValue::Int(i),
                Helper::Float(f) => MetadataValue::Float(f),
            })
        }
    }
}

impl MetadataValue {
    /// Numeric view of the value: `Some(f64)` for [`MetadataValue::Int`] and
    /// [`MetadataValue::Float`], `None` otherwise. Used by numeric comparisons.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetadataValue::Int(i) => Some(*i as f64),
            MetadataValue::Float(f) => Some(*f),
            _ => None,
        }
    }
}

/// Arbitrary metadata keyed by name.
pub type Metadata = HashMap<String, MetadataValue>;

/// The distance space a collection is indexed and searched in. Matches the
/// three Chroma `hnsw:space` options exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Distance {
    /// Squared Euclidean distance (no square root).
    #[default]
    L2,
    /// L2-normalized inner product: `1 - dot(normalize(a), normalize(b))`.
    Cosine,
    /// Raw inner product: `1 - dot(a, b)`.
    Ip,
}

impl Distance {
    /// Canonical lowercase name, as used in collection configs: `"l2"`,
    /// `"cosine"`, `"ip"`.
    pub fn name(&self) -> &'static str {
        match self {
            Distance::L2 => "l2",
            Distance::Cosine => "cosine",
            Distance::Ip => "ip",
        }
    }
}

/// Error returned when a distance space name cannot be parsed.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("unknown distance space `{0}` (expected \"l2\", \"cosine\", or \"ip\")")]
pub struct DistanceParseError(pub String);

impl FromStr for Distance {
    type Err = DistanceParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "l2" => Ok(Distance::L2),
            "cosine" => Ok(Distance::Cosine),
            "ip" => Ok(Distance::Ip),
            _ => Err(DistanceParseError(s.to_owned())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_name_roundtrip() {
        for d in [Distance::L2, Distance::Cosine, Distance::Ip] {
            assert_eq!(Distance::from_str(d.name()).unwrap(), d);
        }
    }

    #[test]
    fn distance_default_is_l2() {
        assert_eq!(Distance::default(), Distance::L2);
    }

    #[test]
    fn distance_parse_errors() {
        assert_eq!(
            Distance::from_str("cos").unwrap_err(),
            DistanceParseError("cos".into())
        );
        assert_eq!(
            Distance::from_str("euclidean").unwrap_err(),
            DistanceParseError("euclidean".into())
        );
    }

    #[test]
    fn metadata_value_as_f64() {
        assert_eq!(MetadataValue::Int(5).as_f64(), Some(5.0));
        assert_eq!(MetadataValue::Float(2.5).as_f64(), Some(2.5));
        assert_eq!(MetadataValue::Str("x".into()).as_f64(), None);
        assert_eq!(MetadataValue::Bool(true).as_f64(), None);
    }
}
