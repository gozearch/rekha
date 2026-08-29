//! Table output formatting.

use crate::client::QueryResult;

pub fn format_table(result: &QueryResult) -> String {
    let mut output = String::new();

    let row_count = result.ids.first().map_or(0, |ids| ids.len());
    if row_count == 0 {
        return "Empty set".to_string();
    }

    let mut headers = vec!["id".to_string()];
    if result.distances.is_some() {
        headers.push("distance".to_string());
    }
    if result.documents.is_some() {
        headers.push("document".to_string());
    }
    if result.metadatas.is_some() {
        headers.push("metadata".to_string());
    }

    let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();

    for i in 0..row_count {
        if let Some(ids) = result.ids.first()
            && let Some(id) = ids.get(i)
        {
            widths[0] = widths[0].max(id.len());
        }
        if let Some(ref dists) = result.distances
            && let Some(row) = dists.first()
            && let Some(d) = row.get(i)
        {
            let s = format!("{d:.4}");
            widths[1] = widths[1].max(s.len());
        }
        if let Some(ref docs) = result.documents
            && let Some(row) = docs.first()
            && let Some(Some(doc)) = row.get(i)
        {
            let truncated: &str = if doc.len() > 40 { &doc[..40] } else { doc };
            let col_idx = headers.iter().position(|h| h == "document").unwrap_or(2);
            widths[col_idx] = widths[col_idx].max(truncated.len());
        }
    }

    let separator: Vec<String> = widths.iter().map(|w| "-".repeat(*w + 2)).collect();
    let sep_line = format!("+{}+", separator.join("+"));

    let header_line: Vec<String> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| format!(" {:<width$} ", h, width = widths[i]))
        .collect();
    output.push_str(&format!("|{}|\n", header_line.join("|")));
    output.push_str(&format!("{sep_line}\n"));

    for i in 0..row_count {
        let mut cells = Vec::new();

        if let Some(ids) = result.ids.first()
            && let Some(id) = ids.get(i)
        {
            cells.push(format!(" {:<width$} ", id, width = widths[0]));
        }

        if let Some(ref dists) = result.distances
            && let Some(row) = dists.first()
            && let Some(d) = row.get(i)
        {
            cells.push(format!(
                " {:<width$} ",
                format!("{d:.4}"),
                width = widths[1]
            ));
        }

        if let Some(ref docs) = result.documents
            && let Some(row) = docs.first()
            && let Some(Some(doc)) = row.get(i)
        {
            let truncated: &str = if doc.len() > 40 { &doc[..40] } else { doc };
            let col_idx = headers.iter().position(|h| h == "document").unwrap_or(2);
            cells.push(format!(" {:<width$} ", truncated, width = widths[col_idx]));
        }

        if let Some(ref metas) = result.metadatas
            && let Some(row) = metas.first()
            && let Some(Some(meta)) = row.get(i)
        {
            let s = serde_json::to_string(meta).unwrap_or_default();
            let truncated: &str = if s.len() > 40 { &s[..40] } else { &s };
            let col_idx = headers.iter().position(|h| h == "metadata").unwrap_or(3);
            cells.push(format!(" {:<width$} ", truncated, width = widths[col_idx]));
        }

        output.push_str(&format!("|{}|\n", cells.join("|")));
    }
    output.push_str(&format!("{sep_line}\n"));

    output
}
