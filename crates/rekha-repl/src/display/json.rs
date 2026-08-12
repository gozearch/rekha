//! JSON output formatting.

use crate::client::QueryResult;

pub fn format_json(result: &QueryResult) -> String {
    serde_json::to_string_pretty(result).unwrap_or_else(|_| "{}".to_string())
}
