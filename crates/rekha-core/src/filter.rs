//! Chroma-compatible `where` metadata filters.
//!
//! Filters follow Chroma's "where" semantics
//! (<https://cookbook.chromadb.dev/core/filters/>): candidate selection happens
//! against a record's metadata *before* ranking, so a filter only decides
//! eligibility, never similarity.
//!
//! The JSON representation accepts every Chroma form:
//!
//! - operator form: `{"a": {"$eq": 5}}`, `{"a": {"$ne": 5}}`,
//!   `{"a": {"$gt": 1}}`, `{"a": {"$gte": 1}}`, `{"a": {"$lt": 3}}`,
//!   `{"a": {"$lte": 3}}`, `{"a": {"$in": [1, 2]}}`, `{"a": {"$nin": [1, 2]}}`
//! - shorthand form: `{"a": 5}` means `{"a": {"$eq": 5}}`
//! - `{"$and": [ ... ]}` and `{"$or": [ ... ]}` combinators, nestable
//! - multiple field keys at the top level are implicitly ANDed
//!
//! [`WhereFilter`] serializes to the operator form and deserializes from any
//! form, so `serde_json` / [`FromStr`] round-trips are lossless.

use std::str::FromStr;

use serde::de::Deserializer;
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

use crate::types::{Metadata, MetadataValue};

/// A comparison applied to a single metadata key, mirroring Chroma's operators.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComparisonOp {
    /// Equal. Numbers compare numerically across int/float.
    Eq(MetadataValue),
    /// Not equal. Requires the key to be present; absent keys never match.
    Ne(MetadataValue),
    /// Strictly greater than (numeric).
    Gt(f64),
    /// Greater than or equal (numeric).
    Gte(f64),
    /// Strictly less than (numeric).
    Lt(f64),
    /// Less than or equal (numeric).
    Lte(f64),
    /// Value is in the list.
    In(Vec<MetadataValue>),
    /// Value is not in the list; matches when the key is absent as well.
    Nin(Vec<MetadataValue>),
}

/// A filter tree: a comparison, or a conjunction/disjunction of sub-conditions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WhereCondition {
    /// `{"key": {op: value}}`
    Comparison { key: String, op: ComparisonOp },
    /// `{"$and": [...]}` — all sub-conditions must match.
    And(Vec<WhereCondition>),
    /// `{"$or": [...]}` — at least one sub-condition must match.
    Or(Vec<WhereCondition>),
}

/// A whole `where` filter, evaluated against a [`Metadata`] map.
///
/// Serialization is custom (operator form); deserialization accepts every
/// Chroma where shape. [`FromStr`] therefore parses Chroma JSON directly.
#[derive(Debug, Clone, PartialEq)]
pub struct WhereFilter {
    /// Root condition tree.
    pub condition: WhereCondition,
}

impl WhereFilter {
    /// Evaluates the filter against a metadata map.
    ///
    /// `$nin` matches when the key is absent OR its value is not in the list.
    /// Numeric comparisons (`$gt`/`$gte`/`$lt`/`$lte`) are false against
    /// non-numeric or absent values. `$eq`/`$ne`/`$in`/`$nin` compare numbers
    /// numerically (int vs float), matching Chroma's coercion.
    pub fn matches(&self, metadata: &Metadata) -> bool {
        match_condition(&self.condition, metadata)
    }

    /// Renders the filter to its canonical JSON string (operator form), for
    /// storage and logging.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    /// Parses any Chroma where JSON shape into a filter.
    pub fn parse_json(s: &str) -> Result<WhereFilter, WhereParseError> {
        let value: Value =
            serde_json::from_str(s).map_err(|e| WhereParseError::InvalidJson(e.to_string()))?;
        WhereFilter::from_json_value(&value)
    }

    /// Builds a filter from a [`serde_json::Value`] in any accepted Chroma
    /// shape.
    pub fn from_json_value(value: &Value) -> Result<WhereFilter, WhereParseError> {
        Ok(WhereFilter {
            condition: parse_condition(value)?,
        })
    }
}

impl FromStr for WhereFilter {
    type Err = serde_json::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

impl Serialize for WhereFilter {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        condition_to_value(&self.condition).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for WhereFilter {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        WhereFilter::from_json_value(&value).map_err(serde::de::Error::custom)
    }
}

/// Errors produced when parsing a Chroma where JSON shape.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum WhereParseError {
    /// The top level must be a JSON object (a single key map or `$and`/`$or`).
    #[error("where filter must be a JSON object, got {0}")]
    ExpectedObject(&'static str),
    /// `$and`/`$or` values must be arrays.
    #[error("`{op}` must be a JSON array, got {value}")]
    ExpectedArray {
        op: &'static str,
        value: &'static str,
    },
    /// Unknown operator name inside `{"key": {op: value}}`.
    #[error(
        "unknown comparison operator `{0}` (expected $eq, $ne, $gt, $gte, $lt, $lte, $in, or $nin)"
    )]
    UnknownOperator(String),
    /// A numeric operator was given a non-numeric value.
    #[error("operator `{op}` requires a numeric value, got {value}")]
    ExpectedNumber {
        op: &'static str,
        value: &'static str,
    },
    /// A field's operator object was empty.
    #[error("field `{0}` has an empty operator object")]
    EmptyOps(String),
    /// A value that Chroma metadata cannot represent.
    #[error("unsupported metadata value `{0}` (expected string, boolean, integer, or float)")]
    InvalidValue(&'static str),
    /// A JSON integer outside the `i64` range.
    #[error("integer value out of i64 range")]
    NumberOutOfRange,
    /// `$and` and `$or` appeared in the same object.
    #[error("`$and` and `$or` cannot appear in the same object")]
    AndOrConflict,
    /// `$or` cannot be combined with other keys at the same level.
    #[error("`$or` cannot be combined with other conditions")]
    OrWithOtherKeys,
    /// The where object contained no conditions.
    #[error("where filter cannot be empty")]
    Empty,
    /// `$and` or `$or` was given an empty list.
    #[error("`{0}` list cannot be empty")]
    EmptyList(&'static str),
    /// The input was not valid JSON.
    #[error("invalid JSON: {0}")]
    InvalidJson(String),
}

fn parse_condition(value: &Value) -> Result<WhereCondition, WhereParseError> {
    match value {
        Value::Object(map) => parse_object(map),
        other => Err(WhereParseError::ExpectedObject(describe(other))),
    }
}

fn parse_object(map: &Map<String, Value>) -> Result<WhereCondition, WhereParseError> {
    let mut fields: Vec<WhereCondition> = Vec::new();
    let mut explicit_and: Option<&Vec<Value>> = None;
    let mut explicit_or: Option<&Vec<Value>> = None;

    for (key, value) in map {
        match key.as_str() {
            "$and" => explicit_and = Some(as_array(value, "$and")?),
            "$or" => explicit_or = Some(as_array(value, "$or")?),
            field => fields.push(parse_field_condition(field, value)?),
        }
    }

    if explicit_and.is_some() && explicit_or.is_some() {
        return Err(WhereParseError::AndOrConflict);
    }

    if let Some(or_list) = explicit_or {
        if !fields.is_empty() {
            return Err(WhereParseError::OrWithOtherKeys);
        }
        let mut conds = parse_list(or_list)?;
        return match conds.len() {
            0 => Err(WhereParseError::EmptyList("$or")),
            1 => Ok(conds.pop().unwrap()),
            _ => Ok(WhereCondition::Or(conds)),
        };
    }

    if let Some(and_list) = explicit_and {
        let mut conds = parse_list(and_list)?;
        conds.extend(fields);
        return normalize_list(conds, "$and");
    }

    normalize_list(fields, "field")
}

fn parse_list(list: &[Value]) -> Result<Vec<WhereCondition>, WhereParseError> {
    list.iter().map(parse_condition).collect()
}

fn normalize_list(
    conds: Vec<WhereCondition>,
    what: &'static str,
) -> Result<WhereCondition, WhereParseError> {
    match conds.len() {
        0 => Err(WhereParseError::EmptyList(what)),
        1 => Ok(conds.into_iter().next().unwrap()),
        _ => Ok(WhereCondition::And(conds)),
    }
}

fn parse_field_condition(key: &str, value: &Value) -> Result<WhereCondition, WhereParseError> {
    match value {
        Value::Object(ops) => {
            if ops.is_empty() {
                return Err(WhereParseError::EmptyOps(key.to_owned()));
            }
            let conds = ops
                .iter()
                .map(|(op, val)| {
                    Ok(WhereCondition::Comparison {
                        key: key.to_owned(),
                        op: parse_op(op, val)?,
                    })
                })
                .collect::<Result<Vec<_>, WhereParseError>>()?;
            normalize_list(conds, "operator")
        }
        _ => Ok(WhereCondition::Comparison {
            key: key.to_owned(),
            op: ComparisonOp::Eq(metadata_value(value)?),
        }),
    }
}

fn parse_op(op: &str, value: &Value) -> Result<ComparisonOp, WhereParseError> {
    match op {
        "$eq" => Ok(ComparisonOp::Eq(metadata_value(value)?)),
        "$ne" => Ok(ComparisonOp::Ne(metadata_value(value)?)),
        "$gt" => Ok(ComparisonOp::Gt(number(value, "$gt")?)),
        "$gte" => Ok(ComparisonOp::Gte(number(value, "$gte")?)),
        "$lt" => Ok(ComparisonOp::Lt(number(value, "$lt")?)),
        "$lte" => Ok(ComparisonOp::Lte(number(value, "$lte")?)),
        "$in" => Ok(ComparisonOp::In(value_list(value, "$in")?)),
        "$nin" => Ok(ComparisonOp::Nin(value_list(value, "$nin")?)),
        other => Err(WhereParseError::UnknownOperator(other.to_owned())),
    }
}

fn as_array<'a>(value: &'a Value, op: &'static str) -> Result<&'a Vec<Value>, WhereParseError> {
    value
        .as_array()
        .ok_or_else(|| WhereParseError::ExpectedArray {
            op,
            value: describe(value),
        })
}

fn number(value: &Value, op: &'static str) -> Result<f64, WhereParseError> {
    value
        .as_f64()
        .ok_or_else(|| WhereParseError::ExpectedNumber {
            op,
            value: describe(value),
        })
}

fn value_list(value: &Value, op: &'static str) -> Result<Vec<MetadataValue>, WhereParseError> {
    value
        .as_array()
        .ok_or_else(|| WhereParseError::ExpectedArray {
            op,
            value: describe(value),
        })
        .and_then(|items| items.iter().map(metadata_value).collect())
}

fn metadata_value(value: &Value) -> Result<MetadataValue, WhereParseError> {
    match value {
        Value::String(s) => Ok(MetadataValue::Str(s.clone())),
        Value::Bool(b) => Ok(MetadataValue::Bool(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(MetadataValue::Int(i))
            } else if let Some(f) = n.as_f64() {
                Ok(MetadataValue::Float(f))
            } else {
                Err(WhereParseError::NumberOutOfRange)
            }
        }
        other => Err(WhereParseError::InvalidValue(describe(other))),
    }
}

fn describe(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn match_condition(cond: &WhereCondition, metadata: &Metadata) -> bool {
    match cond {
        WhereCondition::Comparison { key, op } => match_op(key, op, metadata),
        WhereCondition::And(conds) => conds.iter().all(|c| match_condition(c, metadata)),
        WhereCondition::Or(conds) => conds.iter().any(|c| match_condition(c, metadata)),
    }
}

fn match_op(key: &str, op: &ComparisonOp, metadata: &Metadata) -> bool {
    match op {
        ComparisonOp::Eq(v) => metadata
            .get(key)
            .is_some_and(|actual| values_equal(actual, v)),
        ComparisonOp::Ne(v) => metadata
            .get(key)
            .is_some_and(|actual| !values_equal(actual, v)),
        ComparisonOp::Gt(n) => numeric(metadata, key).is_some_and(|actual| actual > *n),
        ComparisonOp::Gte(n) => numeric(metadata, key).is_some_and(|actual| actual >= *n),
        ComparisonOp::Lt(n) => numeric(metadata, key).is_some_and(|actual| actual < *n),
        ComparisonOp::Lte(n) => numeric(metadata, key).is_some_and(|actual| actual <= *n),
        ComparisonOp::In(vs) => metadata
            .get(key)
            .is_some_and(|actual| vs.iter().any(|v| values_equal(actual, v))),
        ComparisonOp::Nin(vs) => match metadata.get(key) {
            None => true,
            Some(actual) => !vs.iter().any(|v| values_equal(actual, v)),
        },
    }
}

fn numeric(metadata: &Metadata, key: &str) -> Option<f64> {
    metadata.get(key).and_then(MetadataValue::as_f64)
}

/// Metadata equality with Chroma's numeric coercion: `Int(5) == Float(5.0)`.
fn values_equal(a: &MetadataValue, b: &MetadataValue) -> bool {
    match (a, b) {
        (MetadataValue::Str(x), MetadataValue::Str(y)) => x == y,
        (MetadataValue::Bool(x), MetadataValue::Bool(y)) => x == y,
        (MetadataValue::Int(x), MetadataValue::Int(y)) => x == y,
        (MetadataValue::Float(x), MetadataValue::Float(y)) => x == y,
        (MetadataValue::Int(x), MetadataValue::Float(y)) => *x as f64 == *y,
        (MetadataValue::Float(x), MetadataValue::Int(y)) => *x == *y as f64,
        _ => false,
    }
}

fn condition_to_value(cond: &WhereCondition) -> Value {
    match cond {
        WhereCondition::Comparison { key, op } => {
            let mut obj = Map::new();
            obj.insert(key.clone(), op_to_value(op));
            Value::Object(obj)
        }
        WhereCondition::And(conds) => {
            let mut obj = Map::new();
            obj.insert(
                "$and".to_owned(),
                Value::Array(conds.iter().map(condition_to_value).collect()),
            );
            Value::Object(obj)
        }
        WhereCondition::Or(conds) => {
            let mut obj = Map::new();
            obj.insert(
                "$or".to_owned(),
                Value::Array(conds.iter().map(condition_to_value).collect()),
            );
            Value::Object(obj)
        }
    }
}

fn op_to_value(op: &ComparisonOp) -> Value {
    let (name, value) = match op {
        ComparisonOp::Eq(v) => ("$eq", metadata_value_to_json(v)),
        ComparisonOp::Ne(v) => ("$ne", metadata_value_to_json(v)),
        ComparisonOp::Gt(n) => ("$gt", Value::from(*n)),
        ComparisonOp::Gte(n) => ("$gte", Value::from(*n)),
        ComparisonOp::Lt(n) => ("$lt", Value::from(*n)),
        ComparisonOp::Lte(n) => ("$lte", Value::from(*n)),
        ComparisonOp::In(vs) => (
            "$in",
            Value::Array(vs.iter().map(metadata_value_to_json).collect()),
        ),
        ComparisonOp::Nin(vs) => (
            "$nin",
            Value::Array(vs.iter().map(metadata_value_to_json).collect()),
        ),
    };
    let mut obj = Map::new();
    obj.insert(name.to_owned(), value);
    Value::Object(obj)
}

fn metadata_value_to_json(v: &MetadataValue) -> Value {
    match v {
        MetadataValue::Str(s) => Value::String(s.clone()),
        MetadataValue::Bool(b) => Value::Bool(*b),
        MetadataValue::Int(i) => Value::from(*i),
        MetadataValue::Float(f) => Value::from(*f),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(pairs: &[(&str, MetadataValue)]) -> Metadata {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    fn parse(s: &str) -> WhereFilter {
        s.parse().unwrap()
    }

    #[test]
    fn parse_eq_operator() {
        let f: WhereFilter = serde_json::from_str(r#"{"a": {"$eq": 5}}"#).unwrap();
        assert_eq!(
            f,
            WhereFilter {
                condition: WhereCondition::Comparison {
                    key: "a".into(),
                    op: ComparisonOp::Eq(MetadataValue::Int(5)),
                }
            }
        );
    }

    #[test]
    fn parse_eq_shorthand() {
        let f: WhereFilter = serde_json::from_str(r#"{"a": 5}"#).unwrap();
        assert_eq!(
            f,
            WhereFilter {
                condition: WhereCondition::Comparison {
                    key: "a".into(),
                    op: ComparisonOp::Eq(MetadataValue::Int(5)),
                }
            }
        );
    }

    #[test]
    fn parse_and() {
        let f: WhereFilter =
            serde_json::from_str(r#"{"$and": [{"a": {"$gt": 1}}, {"b": {"$in": ["x", "y"]}}]}"#)
                .unwrap();
        assert_eq!(
            f,
            WhereFilter {
                condition: WhereCondition::And(vec![
                    WhereCondition::Comparison {
                        key: "a".into(),
                        op: ComparisonOp::Gt(1.0),
                    },
                    WhereCondition::Comparison {
                        key: "b".into(),
                        op: ComparisonOp::In(vec![
                            MetadataValue::Str("x".into()),
                            MetadataValue::Str("y".into())
                        ]),
                    },
                ])
            }
        );
    }

    #[test]
    fn parse_or() {
        let f: WhereFilter =
            serde_json::from_str(r#"{"$or": [{"a": 1}, {"b": {"$lt": 3}}]}"#).unwrap();
        assert_eq!(
            f,
            WhereFilter {
                condition: WhereCondition::Or(vec![
                    WhereCondition::Comparison {
                        key: "a".into(),
                        op: ComparisonOp::Eq(MetadataValue::Int(1)),
                    },
                    WhereCondition::Comparison {
                        key: "b".into(),
                        op: ComparisonOp::Lt(3.0),
                    },
                ])
            }
        );
    }

    #[test]
    fn parse_implicit_and_top_level() {
        let f: WhereFilter = serde_json::from_str(r#"{"a": 1, "b": 2}"#).unwrap();
        match &f.condition {
            WhereCondition::And(conds) => assert_eq!(conds.len(), 2),
            other => panic!("expected And, got {other:?}"),
        }
        assert!(f.matches(&metadata(&[
            ("a", MetadataValue::Int(1)),
            ("b", MetadataValue::Int(2))
        ])));
        assert!(!f.matches(&metadata(&[
            ("a", MetadataValue::Int(1)),
            ("b", MetadataValue::Int(3))
        ])));
    }

    #[test]
    fn matches_each_op() {
        let f = parse(r#"{"a": {"$eq": 5}}"#);
        assert!(f.matches(&metadata(&[("a", MetadataValue::Int(5))])));
        assert!(!f.matches(&metadata(&[("a", MetadataValue::Int(6))])));
        assert!(!f.matches(&Metadata::new()));

        // int vs float coercion for $eq
        let f = parse(r#"{"a": {"$eq": 5}}"#);
        assert!(f.matches(&metadata(&[("a", MetadataValue::Float(5.0))])));

        let f = parse(r#"{"a": {"$ne": 5}}"#);
        assert!(f.matches(&metadata(&[("a", MetadataValue::Int(6))])));
        assert!(!f.matches(&metadata(&[("a", MetadataValue::Int(5))])));
        assert!(!f.matches(&Metadata::new()));

        let f = parse(r#"{"a": {"$gt": 1}}"#);
        assert!(f.matches(&metadata(&[("a", MetadataValue::Int(2))])));
        assert!(!f.matches(&metadata(&[("a", MetadataValue::Int(1))])));
        assert!(!f.matches(&metadata(&[("a", MetadataValue::Str("hi".into()))])));

        let f = parse(r#"{"a": {"$gte": 1}}"#);
        assert!(f.matches(&metadata(&[("a", MetadataValue::Float(1.0))])));

        let f = parse(r#"{"a": {"$lt": 3}}"#);
        assert!(f.matches(&metadata(&[("a", MetadataValue::Int(2))])));
        assert!(!f.matches(&metadata(&[("a", MetadataValue::Int(3))])));

        let f = parse(r#"{"a": {"$lte": 3}}"#);
        assert!(f.matches(&metadata(&[("a", MetadataValue::Int(3))])));
        assert!(!f.matches(&metadata(&[("a", MetadataValue::Int(4))])));

        let f = parse(r#"{"a": {"$in": [1, "two"]}}"#);
        assert!(f.matches(&metadata(&[("a", MetadataValue::Int(1))])));
        assert!(f.matches(&metadata(&[("a", MetadataValue::Str("two".into()))])));
        assert!(!f.matches(&metadata(&[("a", MetadataValue::Int(3))])));

        // $nin matches when the key is absent OR not in the list
        let f = parse(r#"{"a": {"$nin": [1, 2]}}"#);
        assert!(f.matches(&Metadata::new()));
        assert!(f.matches(&metadata(&[("a", MetadataValue::Int(3))])));
        assert!(!f.matches(&metadata(&[("a", MetadataValue::Int(1))])));
    }

    #[test]
    fn matches_and_or() {
        let f = parse(r#"{"$and": [{"a": {"$gt": 1}}, {"b": {"$in": ["x", "y"]}}]}"#);
        assert!(f.matches(&metadata(&[
            ("a", MetadataValue::Int(2)),
            ("b", MetadataValue::Str("x".into()))
        ])));
        assert!(!f.matches(&metadata(&[
            ("a", MetadataValue::Int(2)),
            ("b", MetadataValue::Str("z".into()))
        ])));
        assert!(!f.matches(&metadata(&[
            ("a", MetadataValue::Int(0)),
            ("b", MetadataValue::Str("x".into()))
        ])));

        let f = parse(r#"{"$or": [{"a": {"$eq": 1}}, {"b": {"$lt": 3}}]}"#);
        assert!(f.matches(&metadata(&[("a", MetadataValue::Int(1))])));
        assert!(f.matches(&metadata(&[("b", MetadataValue::Int(2))])));
        assert!(!f.matches(&metadata(&[
            ("a", MetadataValue::Int(9)),
            ("b", MetadataValue::Int(9))
        ])));
    }

    #[test]
    fn to_json_roundtrip() {
        let f = parse(r#"{"$and": [{"a": {"$gt": 1}}, {"b": {"$in": ["x", "y"]}}, {"c": 5}]}"#);
        let json = f.to_json();
        let f2: WhereFilter = serde_json::from_str(&json).unwrap();
        assert_eq!(f, f2);
    }

    #[test]
    fn from_json_value_and_parse_json_agree() {
        let f = WhereFilter::parse_json(r#"{"a": {"$eq": 5}}"#).unwrap();
        let g: WhereFilter = serde_json::from_str(r#"{"a": {"$eq": 5}}"#).unwrap();
        assert_eq!(f, g);
    }

    #[test]
    fn parse_errors() {
        assert!(serde_json::from_str::<WhereFilter>(r#"{"a": {"$bogus": 1}}"#).is_err());
        assert!(
            serde_json::from_str::<WhereFilter>(r#"{"$and": [{"a": 1}], "$or": [{"b": 2}]}"#)
                .is_err()
        );
        assert!(serde_json::from_str::<WhereFilter>(r#"{"$or": [{"a": 1}], "b": 2}"#).is_err());
        assert!(serde_json::from_str::<WhereFilter>(r#"{}"#).is_err());
        assert!(serde_json::from_str::<WhereFilter>(r#"[1, 2]"#).is_err());
        assert!(serde_json::from_str::<WhereFilter>(r#"{"a": {"$gt": "x"}}"#).is_err());
        assert!(serde_json::from_str::<WhereFilter>(r#"{"a": null}"#).is_err());
    }
}
