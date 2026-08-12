//! SQL-like query parser with NEAREST TO support.

#[derive(Debug, Clone)]
pub enum ParsedQuery {
    Select {
        collection: String,
        fields: Vec<String>,
        nearest_to: Option<Vec<f32>>,
        where_filter: Option<serde_json::Value>,
        limit: Option<usize>,
        vertical: bool,
    },
    Count {
        collection: String,
    },
    CreateCollection {
        name: String,
    },
}

pub fn parse_query(input: &str) -> Result<ParsedQuery, String> {
    let input = input.trim().trim_end_matches(';').trim();

    let upper = input.to_uppercase();

    if upper.starts_with("CREATE COLLECTION ") {
        let name = input[18..].trim().to_string();
        if name.is_empty() {
            return Err("Missing collection name".into());
        }
        return Ok(ParsedQuery::CreateCollection { name });
    }

    if upper.starts_with("SELECT COUNT(*)") {
        let collection = extract_from_clause(input)?;
        return Ok(ParsedQuery::Count { collection });
    }

    if input.to_uppercase().starts_with("SELECT ") {
        return parse_select(input);
    }

    Err(format!("Unsupported query: {input}"))
}

fn parse_select(input: &str) -> Result<ParsedQuery, String> {
    let upper = input.to_uppercase();

    let collection = extract_from_clause(input)?;
    let fields = extract_fields(input)?;

    let nearest_to = if let Some(idx) = upper.find("NEAREST TO") {
        let start = idx + "NEAREST TO".len();
        let rest = input[start..].trim();
        Some(parse_vector(rest)?)
    } else {
        None
    };

    let where_filter = if let Some(idx) = upper.find("WHERE ") {
        let start = idx + "WHERE ".len();
        let end = upper
            .find(" NEAREST TO")
            .or_else(|| upper.find(" LIMIT"))
            .or_else(|| upper.find(" HAVING"))
            .or_else(|| upper.find(" VERTICAL"))
            .unwrap_or(input.len());
        let filter_str = input[start..end].trim();
        Some(parse_where(filter_str)?)
    } else {
        None
    };

    let limit = if let Some(idx) = upper.find("LIMIT ") {
        let start = idx + "LIMIT ".len();
        let end = upper
            .find(" VERTICAL")
            .or_else(|| upper.find(" HAVING"))
            .unwrap_or(input.len());
        let num_str = input[start..end].trim();
        num_str.parse().ok()
    } else {
        None
    };

    let vertical = upper.contains("VERTICAL");

    Ok(ParsedQuery::Select {
        collection,
        fields,
        nearest_to,
        where_filter,
        limit,
        vertical,
    })
}

fn extract_from_clause(input: &str) -> Result<String, String> {
    let upper = input.to_uppercase();
    let from_idx = upper.find("FROM ").ok_or("Missing FROM clause")?;
    let after_from = input[from_idx + 5..].trim();

    let end = after_from
        .find(' ')
        .or_else(|| after_from.find('\n'))
        .unwrap_or(after_from.len());

    Ok(after_from[..end].trim().to_string())
}

fn extract_fields(input: &str) -> Result<Vec<String>, String> {
    let upper = input.to_uppercase();
    let select_idx = upper.find("SELECT ").ok_or("Missing SELECT")?;
    let from_idx = upper.find("FROM ").ok_or("Missing FROM")?;

    let fields_str = input[select_idx + 7..from_idx].trim();

    if fields_str == "*" {
        return Ok(vec!["*".to_string()]);
    }

    Ok(fields_str
        .split(',')
        .map(|f| f.trim().to_string())
        .collect())
}

fn parse_vector(input: &str) -> Result<Vec<f32>, String> {
    let input = input.trim();
    let start = input.find('[').ok_or("Missing '[' in vector")?;
    let end = input.find(']').ok_or("Missing ']' in vector")?;
    let inner = &input[start + 1..end];

    inner
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<f32>()
                .map_err(|e| format!("Invalid vector element: {e}"))
        })
        .collect()
}

fn parse_where(input: &str) -> Result<serde_json::Value, String> {
    let input = input.trim();

    if let Ok(val) = serde_json::from_str::<serde_json::Value>(input) {
        return Ok(val);
    }

    let (field, op, value) = parse_condition(input)?;

    match op {
        "=" | "==" => Ok(serde_json::json!({ field: { "$eq": parse_value(value)? } })),
        "!=" | "<>" => Ok(serde_json::json!({ field: { "$ne": parse_value(value)? } })),
        ">" => Ok(serde_json::json!({ field: { "$gt": parse_value(value)? } })),
        ">=" => Ok(serde_json::json!({ field: { "$gte": parse_value(value)? } })),
        "<" => Ok(serde_json::json!({ field: { "$lt": parse_value(value)? } })),
        "<=" => Ok(serde_json::json!({ field: { "$lte": parse_value(value)? } })),
        _ => Err(format!("Unsupported operator: {op}")),
    }
}

fn parse_condition(input: &str) -> Result<(String, &str, &str), String> {
    for op in &["==", "!=", "<=", ">=", "=", ">", "<"] {
        if let Some(idx) = input.find(op) {
            let field = input[..idx].trim().to_string();
            let value = input[idx + op.len()..].trim();
            return Ok((field, op, value));
        }
    }
    Err(format!("Cannot parse condition: {input}"))
}

fn parse_value(input: &str) -> Result<serde_json::Value, String> {
    let input = input.trim();

    if (input.starts_with('"') && input.ends_with('"'))
        || (input.starts_with('\'') && input.ends_with('\''))
    {
        return Ok(serde_json::Value::String(
            input[1..input.len() - 1].to_string(),
        ));
    }

    if input == "true" || input == "false" {
        return Ok(serde_json::Value::Bool(input == "true"));
    }

    if let Ok(n) = input.parse::<i64>() {
        return Ok(serde_json::json!(n));
    }

    if let Ok(f) = input.parse::<f64>() {
        return Ok(serde_json::json!(f));
    }

    if input.starts_with('[') && input.ends_with(']') {
        let inner = &input[1..input.len() - 1];
        let items: Result<Vec<serde_json::Value>, _> =
            inner.split(',').map(|s| parse_value(s.trim())).collect();
        return Ok(serde_json::Value::Array(items?));
    }

    Ok(serde_json::Value::String(input.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_select_all() {
        let q = parse_query("SELECT * FROM my-collection LIMIT 10").unwrap();
        match q {
            ParsedQuery::Select {
                collection, limit, ..
            } => {
                assert_eq!(collection, "my-collection");
                assert_eq!(limit, Some(10));
            }
            _ => panic!("Expected Select"),
        }
    }

    #[test]
    fn test_parse_nearest_to() {
        let q =
            parse_query("SELECT * FROM my-collection NEAREST TO [0.1, 0.2, 0.3] LIMIT 5").unwrap();
        match q {
            ParsedQuery::Select {
                nearest_to, limit, ..
            } => {
                assert_eq!(nearest_to, Some(vec![0.1, 0.2, 0.3]));
                assert_eq!(limit, Some(5));
            }
            _ => panic!("Expected Select"),
        }
    }

    #[test]
    fn test_parse_count() {
        let q = parse_query("SELECT COUNT(*) FROM my-collection").unwrap();
        match q {
            ParsedQuery::Count { collection } => {
                assert_eq!(collection, "my-collection");
            }
            _ => panic!("Expected Count"),
        }
    }

    #[test]
    fn test_parse_where() {
        let q = parse_query("SELECT * FROM my-collection WHERE category = 'tech' LIMIT 5").unwrap();
        match q {
            ParsedQuery::Select { where_filter, .. } => {
                assert!(where_filter.is_some());
            }
            _ => panic!("Expected Select"),
        }
    }

    #[test]
    fn test_parse_vertical() {
        let q = parse_query("SELECT * FROM my-collection NEAREST TO [1.0, 2.0] LIMIT 5 VERTICAL")
            .unwrap();
        match q {
            ParsedQuery::Select { vertical, .. } => {
                assert!(vertical);
            }
            _ => panic!("Expected Select"),
        }
    }

    #[test]
    fn test_parse_create_collection() {
        let q = parse_query("CREATE COLLECTION my-vectors;").unwrap();
        match q {
            ParsedQuery::CreateCollection { name } => {
                assert_eq!(name, "my-vectors");
            }
            _ => panic!("Expected CreateCollection"),
        }
    }

    #[test]
    fn test_parse_create_collection_empty_name() {
        let result = parse_query("CREATE COLLECTION ;");
        assert!(result.is_err());
    }
}
