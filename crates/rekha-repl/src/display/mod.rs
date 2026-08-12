//! Display formatting for query results.

pub mod json;
pub mod table;
pub mod vertical;

use crate::client::QueryResult;

#[derive(Debug, Clone, PartialEq)]
pub enum OutputFormat {
    Table,
    Json,
    Csv,
    Vertical,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self::Table
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Table => write!(f, "table"),
            Self::Json => write!(f, "json"),
            Self::Csv => write!(f, "csv"),
            Self::Vertical => write!(f, "vertical"),
        }
    }
}

pub fn format_result(result: &QueryResult, format: &OutputFormat) -> String {
    match format {
        OutputFormat::Table => table::format_table(result),
        OutputFormat::Json => json::format_json(result),
        OutputFormat::Csv => format_csv(result),
        OutputFormat::Vertical => vertical::format_vertical(result),
    }
}

fn format_csv(result: &QueryResult) -> String {
    let mut output = String::new();

    let row_count = result.ids.first().map_or(0, |ids| ids.len());
    if row_count == 0 {
        return "Empty set".to_string();
    }

    let has_dist = result.distances.is_some();
    let has_doc = result.documents.is_some();
    let has_meta = result.metadatas.is_some();

    let mut headers = vec!["id".to_string()];
    if has_dist {
        headers.push("distance".to_string());
    }
    if has_doc {
        headers.push("document".to_string());
    }
    if has_meta {
        headers.push("metadata".to_string());
    }
    output.push_str(&headers.join(","));
    output.push('\n');

    for i in 0..row_count {
        if let Some(ids) = result.ids.first() {
            if let Some(id) = ids.get(i) {
                output.push_str(id);
            }
        }
        if let Some(ref dists) = result.distances {
            if let Some(row) = dists.first() {
                if let Some(d) = row.get(i) {
                    output.push_str(&format!(",{d:.4}"));
                }
            }
        }
        if let Some(ref docs) = result.documents {
            if let Some(row) = docs.first() {
                if let Some(Some(doc)) = row.get(i) {
                    output.push_str(&format!(",\"{}\"", doc.replace('"', "\"\"")));
                }
            }
        }
        if let Some(ref metas) = result.metadatas {
            if let Some(row) = metas.first() {
                if let Some(Some(meta)) = row.get(i) {
                    let s = serde_json::to_string(meta).unwrap_or_default();
                    output.push_str(&format!(",\"{}\"", s.replace('"', "\"\"")));
                }
            }
        }
        output.push('\n');
    }

    output
}

pub fn format_count(count: u64, timing: Option<std::time::Duration>) -> String {
    let timing_str = match timing {
        Some(d) => format!(" ({:.3}s)", d.as_secs_f64()),
        None => String::new(),
    };
    format!("{count} rows in set{timing_str}")
}
