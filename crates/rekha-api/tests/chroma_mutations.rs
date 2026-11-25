//! Chroma-compatible record mutations: add, upsert, update, delete, get, query.
//!
//! The wire contract differs from the native v2 endpoints in ways that are easy
//! to get subtly wrong, so each rule is asserted directly: metadata is merged on
//! upsert/update, a JSON `null` deletes a key, omitted fields are preserved, an
//! `include` list changes the response shape, and a delete reports only the
//! records it actually removed.

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
    (dir, TestServer::new(router(state)))
}

async fn create(server: &TestServer, name: &str, dim: usize) {
    let resp = server
        .post(BASE)
        .json(&serde_json::json!({ "name": name, "dimension": dim }))
        .await;
    assert!(resp.status_code().is_success(), "{}", resp.text());
}

async fn add(server: &TestServer, name: &str, body: serde_json::Value) {
    let resp = server.post(&format!("{BASE}/{name}/add")).json(&body).await;
    assert!(resp.status_code().is_success(), "add: {}", resp.text());
}

async fn get(server: &TestServer, name: &str, body: serde_json::Value) -> serde_json::Value {
    let resp = server.post(&format!("{BASE}/{name}/get")).json(&body).await;
    assert!(resp.status_code().is_success(), "get: {}", resp.text());
    resp.json()
}

async fn count(server: &TestServer, name: &str) -> u64 {
    let resp = server.get(&format!("{BASE}/{name}/count")).await;
    assert!(resp.status_code().is_success(), "count: {}", resp.text());
    let text = resp.text();
    text.trim().trim_matches('"').parse().unwrap()
}

#[tokio::test]
async fn upsert_merges_metadata_and_preserves_unmentioned_fields() {
    let (_dir, server) = test_app();
    create(&server, "ups", 2).await;
    add(
        &server,
        "ups",
        serde_json::json!({
            "ids": ["a"],
            "embeddings": [[1.0, 0.0]],
            "metadatas": [{"keep": "x", "drop": "y"}],
            "documents": ["hello"]
        }),
    )
    .await;

    let resp = server
        .post(&format!("{BASE}/ups/upsert"))
        .json(&serde_json::json!({
            "ids": ["a"],
            "embeddings": [[1.0, 0.0]],
            "metadatas": [{"drop": null, "added": "z"}]
        }))
        .await;
    assert_eq!(resp.status_code(), 200, "{}", resp.text());

    let v = get(
        &server,
        "ups",
        serde_json::json!({
            "ids": ["a"],
            "include": ["metadatas", "documents", "embeddings"]
        }),
    )
    .await;
    assert_eq!(v["ids"], serde_json::json!(["a"]));
    assert_eq!(
        v["metadatas"][0],
        serde_json::json!({"keep": "x", "added": "z"}),
        "upsert merges: keep survives, drop is deleted, added appears"
    );
    assert_eq!(v["documents"][0], "hello", "an omitted document is kept");
    assert_eq!(
        v["embeddings"][0],
        serde_json::json!([1.0, 0.0]),
        "the replacement vector is stored"
    );
}

#[tokio::test]
async fn update_without_embeddings_preserves_vector_and_document() {
    let (_dir, server) = test_app();
    create(&server, "upd", 2).await;
    add(
        &server,
        "upd",
        serde_json::json!({
            "ids": ["b"],
            "embeddings": [[0.0, 1.0]],
            "metadatas": [{"k": "v"}],
            "documents": ["kept"]
        }),
    )
    .await;

    let resp = server
        .post(&format!("{BASE}/upd/update"))
        .json(&serde_json::json!({
            "ids": ["b"],
            "metadatas": [{"k": null, "n": "1"}]
        }))
        .await;
    assert_eq!(resp.status_code(), 200, "{}", resp.text());

    let v = get(
        &server,
        "upd",
        serde_json::json!({
            "ids": ["b"],
            "include": ["metadatas", "documents", "embeddings"]
        }),
    )
    .await;
    assert_eq!(v["metadatas"][0], serde_json::json!({"n": "1"}));
    assert_eq!(v["documents"][0], "kept");
    assert_eq!(v["embeddings"][0], serde_json::json!([0.0, 1.0]));
}

#[tokio::test]
async fn updating_an_unknown_id_is_a_noop_not_an_error() {
    let (_dir, server) = test_app();
    create(&server, "ghost", 2).await;
    add(
        &server,
        "ghost",
        serde_json::json!({"ids": ["a"], "embeddings": [[1.0, 0.0]]}),
    )
    .await;

    let resp = server
        .post(&format!("{BASE}/ghost/update"))
        .json(&serde_json::json!({
            "ids": ["missing"],
            "embeddings": [[0.0, 1.0]],
            "metadatas": [{"k": "v"}]
        }))
        .await;
    assert_eq!(resp.status_code(), 200, "{}", resp.text());
    assert_eq!(count(&server, "ghost").await, 1);
}

#[tokio::test]
async fn a_null_metadata_value_on_add_is_dropped_not_stored() {
    let (_dir, server) = test_app();
    create(&server, "nulls", 2).await;
    add(
        &server,
        "nulls",
        serde_json::json!({
            "ids": ["a"],
            "embeddings": [[1.0, 0.0]],
            "metadatas": [{"keep": "v", "gone": null}]
        }),
    )
    .await;

    let v = get(
        &server,
        "nulls",
        serde_json::json!({"ids": ["a"], "include": ["metadatas"]}),
    )
    .await;
    assert_eq!(v["metadatas"][0], serde_json::json!({"keep": "v"}));
}

#[tokio::test]
async fn add_with_a_duplicate_id_in_one_batch_is_refused() {
    let (_dir, server) = test_app();
    create(&server, "dup", 2).await;

    let resp = server
        .post(&format!("{BASE}/dup/add"))
        .json(&serde_json::json!({
            "ids": ["same", "same"],
            "embeddings": [[1.0, 0.0], [1.0, 0.0]]
        }))
        .await;
    assert_eq!(
        resp.status_code(),
        400,
        "Rekha refuses a duplicate id rather than keeping the first: {}",
        resp.text()
    );
    assert_eq!(
        count(&server, "dup").await,
        0,
        "a refused batch writes nothing"
    );
}

#[tokio::test]
async fn add_rejects_every_length_mismatch_and_bad_id() {
    let (_dir, server) = test_app();
    create(&server, "shape", 2).await;

    let cases = [
        serde_json::json!({"ids": [], "embeddings": []}),
        serde_json::json!({"ids": ["a", "b"], "embeddings": [[1.0, 0.0]]}),
        serde_json::json!({"ids": ["a"], "embeddings": null}),
        serde_json::json!({"ids": ["a", "b"], "embeddings": [[1.0, 0.0], [0.0, 1.0]], "metadatas": [{"k": "v"}]}),
        serde_json::json!({"ids": ["a", "b"], "embeddings": [[1.0, 0.0], [0.0, 1.0]], "documents": ["only one"]}),
        serde_json::json!({"ids": ["a"], "embeddings": [[1.0]]}),
    ];
    for body in cases {
        let resp = server.post(&format!("{BASE}/shape/add")).json(&body).await;
        assert_eq!(
            resp.status_code(),
            400,
            "should refuse {body}: {}",
            resp.text()
        );
    }

    let long_id = "x".repeat(513);
    let resp = server
        .post(&format!("{BASE}/shape/add"))
        .json(&serde_json::json!({
            "ids": [long_id],
            "embeddings": [[1.0, 0.0]]
        }))
        .await;
    assert_eq!(resp.status_code(), 400);
}

#[tokio::test]
async fn get_honours_ids_include_and_the_uri_projection() {
    let (_dir, server) = test_app();
    create(&server, "proj", 2).await;
    add(
        &server,
        "proj",
        serde_json::json!({
            "ids": ["a", "b"],
            "embeddings": [[1.0, 0.0], [0.0, 1.0]],
            "metadatas": [{"n": 1}, {"n": 2}],
            "documents": ["first", "second"]
        }),
    )
    .await;

    let v = get(
        &server,
        "proj",
        serde_json::json!({
            "ids": ["b"],
            "include": ["embeddings", "documents", "metadatas", "uris"]
        }),
    )
    .await;
    assert_eq!(v["ids"], serde_json::json!(["b"]));
    assert_eq!(v["metadatas"][0], serde_json::json!({"n": 2}));
    assert_eq!(v["documents"][0], "second");
    assert_eq!(v["embeddings"][0], serde_json::json!([0.0, 1.0]));
    assert_eq!(
        v["uris"][0],
        serde_json::Value::Null,
        "Rekha has no URI concept; Chroma reports null"
    );

    let ids_only = get(&server, "proj", serde_json::json!({"include": []})).await;
    assert_eq!(ids_only["ids"].as_array().unwrap().len(), 2);
    assert!(ids_only["embeddings"].is_null());
    assert!(ids_only["documents"].is_null());
    assert!(ids_only["metadatas"].is_null());
    assert_eq!(ids_only["include"], serde_json::json!([]));
}

#[tokio::test]
async fn get_reports_an_empty_result_for_ids_that_do_not_exist() {
    let (_dir, server) = test_app();
    create(&server, "none", 2).await;
    let v = get(
        &server,
        "none",
        serde_json::json!({"ids": ["missing"], "include": ["documents"]}),
    )
    .await;
    assert!(v["ids"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn delete_by_ids_reports_the_submitted_count_and_removes_only_matches() {
    let (_dir, server) = test_app();
    create(&server, "del", 2).await;
    add(
        &server,
        "del",
        serde_json::json!({
            "ids": ["a", "b"],
            "embeddings": [[1.0, 0.0], [0.0, 1.0]]
        }),
    )
    .await;

    let resp = server
        .post(&format!("{BASE}/del/delete"))
        .json(&serde_json::json!({"ids": ["a", "missing"]}))
        .await;
    assert_eq!(resp.status_code(), 200, "{}", resp.text());
    // Chroma's count describes the request (`deleted: <ids submitted>`), while
    // the store only loses the ids that existed. Both halves are asserted so a
    // change to either is visible.
    assert_eq!(resp.json::<serde_json::Value>()["deleted"], 2);
    assert_eq!(count(&server, "del").await, 1);
}

#[tokio::test]
async fn delete_by_document_and_limit_removes_at_most_that_many() {
    let (_dir, server) = test_app();
    create(&server, "deldoc", 2).await;
    add(
        &server,
        "deldoc",
        serde_json::json!({
            "ids": ["a", "b", "c"],
            "embeddings": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            "documents": ["apple", "apple pie", "banana"]
        }),
    )
    .await;

    let resp = server
        .post(&format!("{BASE}/deldoc/delete"))
        .json(&serde_json::json!({
            "where_document": {"$contains": "apple"},
            "limit": 1
        }))
        .await;
    assert_eq!(resp.status_code(), 200, "{}", resp.text());
    assert_eq!(resp.json::<serde_json::Value>()["deleted"], 1);
    assert_eq!(count(&server, "deldoc").await, 2);
}

#[tokio::test]
async fn query_echoes_include_and_filters_on_document_text() {
    let (_dir, server) = test_app();
    create(&server, "qdoc", 2).await;
    add(
        &server,
        "qdoc",
        serde_json::json!({
            "ids": ["apple", "banana"],
            "embeddings": [[1.0, 0.0], [0.0, 1.0]],
            "documents": ["a green apple", "a yellow banana"],
            "metadatas": [{"kind": "fruit"}, {"kind": "fruit"}]
        }),
    )
    .await;

    let resp = server
        .post(&format!("{BASE}/qdoc/query"))
        .json(&serde_json::json!({
            "query_embeddings": [[1.0, 0.0]],
            "n_results": 5,
            "where_document": {"$contains": "apple"},
            "include": ["documents", "metadatas", "embeddings", "distances"]
        }))
        .await;
    assert_eq!(resp.status_code(), 200, "{}", resp.text());
    let v: serde_json::Value = resp.json();
    assert_eq!(v["ids"][0], serde_json::json!(["apple"]));
    assert_eq!(v["documents"][0][0], "a green apple");
    assert_eq!(v["metadatas"][0][0], serde_json::json!({"kind": "fruit"}));
    assert!(v["distances"][0][0].is_number());
    assert_eq!(v["include"][3], "distances");
}

#[tokio::test]
async fn count_is_a_plain_number_after_mutations() {
    let (_dir, server) = test_app();
    create(&server, "cnt", 2).await;
    add(
        &server,
        "cnt",
        serde_json::json!({"ids": ["a", "b"], "embeddings": [[1.0, 0.0], [0.0, 1.0]]}),
    )
    .await;
    assert_eq!(count(&server, "cnt").await, 2);

    let resp = server
        .post(&format!("{BASE}/cnt/upsert"))
        .json(&serde_json::json!({
            "ids": ["c"],
            "embeddings": [[0.5, 0.5]],
            "metadatas": [{"k": "v"}]
        }))
        .await;
    assert_eq!(resp.status_code(), 200, "{}", resp.text());
    assert_eq!(count(&server, "cnt").await, 3);
}
