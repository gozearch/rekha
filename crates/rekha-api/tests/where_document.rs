//! `where_document` filters by document text, or it is refused.
//!
//! The field appeared in no source file before this suite: a Chroma client that
//! sent `where_document` had it dropped by serde, and got the unfiltered result
//! set with a `200`. That is the worst kind of compatibility gap, because the
//! client cannot tell it from "these are all the records that matched".

use std::sync::Arc;

use axum_test::TestServer;
use rekha_api::{AppState, ServerConfig, router};
use rekha_engine::{Engine, EngineConfig};
use rekha_storage::{LocalStorage, RedbCatalog};
use tempfile::TempDir;

const BASE: &str = "/api/v2/tenants/default_tenant/databases/default_database/collections";

fn test_app() -> (TempDir, TestServer) {
    let dir = TempDir::new().unwrap();
    let catalog = Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
    let storage = Arc::new(LocalStorage::new(dir.path().join("objects")));
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    let engine =
        Arc::new(Engine::open(catalog, storage, &wal_dir, EngineConfig::default()).unwrap());
    let state = Arc::new(AppState::new(
        engine,
        None,
        ServerConfig {
            tenant: "default_tenant".into(),
            database: "default_database".into(),
            api_key: None,
            cluster_secret: None,
            pepper: Vec::new(),
            allow_open_mode: true,
        },
    ));
    let server = TestServer::new(router(state));
    (dir, server)
}

/// A collection whose documents are known, so a filter's effect is checkable.
async fn seeded(server: &TestServer) -> String {
    let created = server
        .post(BASE)
        .json(&serde_json::json!({ "name": "docs", "dimension": 3 }))
        .await;
    let body: serde_json::Value = created.json();
    let id = body["id"].as_str().expect("collection id").to_string();

    let resp = server
        .post(&format!("{BASE}/docs/add"))
        .json(&serde_json::json!({
            "ids": ["a", "b", "c", "d"],
            "embeddings": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 0.0, 1.0]
            ],
            "documents": [
                "alpha one",
                "beta two",
                null,
                "alpha three"
            ]
        }))
        .await;
    assert!(
        resp.status_code().is_success(),
        "add failed: {}",
        resp.text()
    );
    id
}

#[tokio::test]
async fn get_returns_only_documents_that_contain_the_substring() {
    let (_dir, server) = test_app();
    seeded(&server).await;

    let resp = server
        .post(&format!("{BASE}/docs/get"))
        .json(&serde_json::json!({
            "where_document": { "$contains": "alpha" }
        }))
        .await;
    assert_eq!(resp.status_code().as_u16(), 200, "{}", resp.text());
    let body: serde_json::Value = resp.json();
    let ids: Vec<&str> = body["ids"]
        .as_array()
        .expect("ids")
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert_eq!(
        ids,
        vec!["a", "d"],
        "only the two alpha documents match: {body}"
    );
}

#[tokio::test]
async fn not_contains_includes_records_with_no_document() {
    let (_dir, server) = test_app();
    seeded(&server).await;

    let resp = server
        .post(&format!("{BASE}/docs/get"))
        .json(&serde_json::json!({
            "where_document": { "$not_contains": "alpha" }
        }))
        .await;
    assert_eq!(resp.status_code().as_u16(), 200, "{}", resp.text());
    let body: serde_json::Value = resp.json();
    let ids: Vec<&str> = body["ids"]
        .as_array()
        .expect("ids")
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert_eq!(
        ids,
        vec!["b", "c"],
        "Chroma's reference server includes records with no document: {body}"
    );
}

#[tokio::test]
async fn a_query_ranks_only_matching_documents() {
    let (_dir, server) = test_app();
    seeded(&server).await;

    // Without the filter, `a` and `c` are the two nearest to this vector.
    let unfiltered: serde_json::Value = server
        .post(&format!("{BASE}/docs/query"))
        .json(&serde_json::json!({
            "query_embeddings": [[1.0, 0.0, 0.0]],
            "n_results": 2
        }))
        .await
        .json();
    let unfiltered_ids: Vec<&str> = unfiltered["ids"][0]
        .as_array()
        .expect("ids")
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert!(
        unfiltered_ids.contains(&"c"),
        "the unfiltered neighbour set must include the near non-alpha record: {unfiltered}"
    );

    let filtered: serde_json::Value = server
        .post(&format!("{BASE}/docs/query"))
        .json(&serde_json::json!({
            "query_embeddings": [[1.0, 0.0, 0.0]],
            "n_results": 2,
            "where_document": { "$contains": "alpha" }
        }))
        .await
        .json();
    let filtered_ids: Vec<&str> = filtered["ids"][0]
        .as_array()
        .expect("ids")
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert_eq!(
        filtered_ids,
        vec!["a", "d"],
        "the filter decides eligibility before ranking, and `k` counts matches: {filtered}"
    );
}

#[tokio::test]
async fn delete_removes_only_matching_documents_and_reports_how_many() {
    let (_dir, server) = test_app();
    seeded(&server).await;

    let resp = server
        .post(&format!("{BASE}/docs/delete"))
        .json(&serde_json::json!({
            "where_document": { "$contains": "alpha" }
        }))
        .await;
    assert_eq!(resp.status_code().as_u16(), 200, "{}", resp.text());
    let body: serde_json::Value = resp.json();
    assert_eq!(body["deleted"], 2, "two alpha documents removed: {body}");

    let count: serde_json::Value = server.get(&format!("{BASE}/docs/count")).await.json();
    assert_eq!(count, serde_json::json!(2), "b and c remain");
}

#[tokio::test]
async fn document_operators_compose_with_and() {
    let (_dir, server) = test_app();
    seeded(&server).await;

    let resp = server
        .post(&format!("{BASE}/docs/get"))
        .json(&serde_json::json!({
            "where_document": {
                "$and": [{ "$contains": "alpha" }, { "$contains": "three" }]
            }
        }))
        .await;
    assert_eq!(resp.status_code().as_u16(), 200, "{}", resp.text());
    let body: serde_json::Value = resp.json();
    let ids: Vec<&str> = body["ids"]
        .as_array()
        .expect("ids")
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert_eq!(ids, vec!["d"], "only `alpha three` satisfies both: {body}");
}

#[tokio::test]
async fn an_unknown_document_operator_is_refused_not_ignored() {
    let (_dir, server) = test_app();
    seeded(&server).await;

    let resp = server
        .post(&format!("{BASE}/docs/get"))
        .json(&serde_json::json!({
            "where_document": { "$regex": "alpha" }
        }))
        .await;
    assert!(
        !resp.status_code().is_success(),
        "an operator this build does not implement must not return an unfiltered \
         200: {}",
        resp.text()
    );
}
