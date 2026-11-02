//! Vertical output formatting.

use crate::client::QueryResult;

pub fn format_vertical(result: &QueryResult) -> String {
    let mut output = String::new();

    let row_count = result.ids.first().map_or(0, |ids| ids.len());
    if row_count == 0 {
        return "Empty set".to_string();
    }

    for i in 0..row_count {
        output.push_str(&format!("-[ RESULT {} ]{}\n", i + 1, "-".repeat(50)));

        if let Some(ids) = result.ids.first()
            && let Some(id) = ids.get(i)
        {
            output.push_str(&format!("{:<12} | {id}\n", "id"));
        }

        if let Some(ref dists) = result.distances
            && let Some(row) = dists.first()
            && let Some(d) = row.get(i)
        {
            output.push_str(&format!("{:<12} | {d:.4}\n", "distance"));
        }

        if let Some(ref docs) = result.documents
            && let Some(row) = docs.first()
            && let Some(Some(doc)) = row.get(i)
        {
            output.push_str(&format!("{:<12} | {doc}\n", "document"));
        }

        if let Some(ref metas) = result.metadatas
            && let Some(row) = metas.first()
            && let Some(Some(meta)) = row.get(i)
        {
            let s = serde_json::to_string_pretty(meta).unwrap_or_default();
            for (j, line) in s.lines().enumerate() {
                if j == 0 {
                    output.push_str(&format!("{:<12} | {line}\n", "metadata"));
                } else {
                    output.push_str(&format!("{:<12} | {line}\n", ""));
                }
            }
        }

        output.push('\n');
    }

    output
}
