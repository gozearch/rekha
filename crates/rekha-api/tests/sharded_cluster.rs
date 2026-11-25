//! A real two-group cluster, in one process.
//!
//! `leadership_tests` runs one Raft group over an in-process channel network
//! with real HTTP listeners. The intra-collection sharding path needs a second
//! layer: a metadata group **plus** one consensus group per shard, because that
//! is what decides whether a write is prepared, committed, or aborted.
//!
//! Spawning the real binary (as the CLI suites do) is the strongest test, but a
//! single test process that can hold every `Raft` instance makes failure modes
//! reachable without ports, restarts, or timing between processes. The transports
//! differ; the code under test — routing, prepare/commit, materialization — is
//! the same.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use openraft::error::{RPCError, RemoteError};
use rekha_api::{AppState, ServerConfig, router};
use rekha_cluster::network::channel::{ChannelHub, ChannelMessage, ChannelNetworkFactory};
use rekha_cluster::state_machine::ClusterStateMachine;
use rekha_cluster::{
    MemoryLogStore, PeerDirectory, Placement, RaftTypeConfig, ShardCluster, ShardId, parse_peers,
};
use rekha_engine::{Engine, EngineConfig, EngineScope};
use rekha_storage::{LocalStorage, RedbCatalog};
use tokio::sync::mpsc;

const SECRET: &str = "sharded-harness-secret-0123456789";
const SHARDS: u32 = 2;
const DIM: usize = 4;

struct Node {
    id: u64,
    base: String,
    metadata: Arc<openraft::Raft<RaftTypeConfig>>,
    shards: Arc<ShardCluster>,
    peers: PeerDirectory,
    _dir: tempfile::TempDir,
}

impl Node {
    fn shard_raft(&self, shard: u32) -> Arc<openraft::Raft<RaftTypeConfig>> {
        self.shards.raft(ShardId(shard)).expect("hosted shard")
    }
}

#[allow(deprecated)]
fn spawn_group_handler(node_id: u64, hub: &ChannelHub, raft: Arc<openraft::Raft<RaftTypeConfig>>) {
    let (tx, mut rx) = mpsc::channel(512);
    let hub = hub.clone();
    tokio::spawn(async move {
        hub.register(node_id, tx).await;
        while let Some(msg) = rx.recv().await {
            match msg {
                ChannelMessage::AppendEntries { req, resp_tx } => {
                    let result = raft
                        .append_entries(req)
                        .await
                        .map_err(|e| RPCError::RemoteError(RemoteError::new(node_id, e)));
                    let _ = resp_tx.send(result);
                }
                ChannelMessage::Vote { req, resp_tx } => {
                    let result = raft
                        .vote(req)
                        .await
                        .map_err(|e| RPCError::RemoteError(RemoteError::new(node_id, e)));
                    let _ = resp_tx.send(result);
                }
                ChannelMessage::InstallSnapshot { req, resp_tx } => {
                    let result = raft
                        .install_snapshot(req)
                        .await
                        .map_err(|e| RPCError::RemoteError(RemoteError::new(node_id, e)));
                    let _ = resp_tx.send(result);
                }
                _ => {}
            }
        }
    });
}

fn engine_in(dir: &std::path::Path) -> Arc<Engine> {
    let catalog = Arc::new(RedbCatalog::open(dir.join("catalog.redb")).unwrap());
    let storage = Arc::new(LocalStorage::new(dir.join("objects")));
    let wal_dir = dir.join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    Arc::new(
        Engine::open(
            catalog,
            storage,
            &wal_dir,
            EngineConfig {
                apply_mode: rekha_engine::ApplyMode::Committed,
                scope: EngineScope::Shards((0..SHARDS).map(ShardId).collect::<Vec<_>>()),
                ..EngineConfig::default()
            },
        )
        .unwrap(),
    )
}

async fn listeners(n: usize) -> Vec<(tokio::net::TcpListener, u16)> {
    let mut out = Vec::new();
    for _ in 0..n {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        out.push((listener, port));
    }
    out
}

async fn start_cluster(n: usize) -> Vec<Node> {
    start_cluster_inner(n, false, 2).await.nodes
}

async fn start_cluster_with_replication(n: usize, replication_factor: usize) -> Vec<Node> {
    start_cluster_inner(n, false, replication_factor)
        .await
        .nodes
}

/// A cluster whose catalogs already hold a superadmin key on every node.
///
/// The HTTP bootstrap path is itself under test elsewhere; seeding the
/// credential here lets a test that is about promotion and replication start
/// with an authenticated client instead of spending its body on RBAC setup.
async fn start_cluster_with_admin(n: usize) -> (Vec<Node>, String) {
    let seeded = start_cluster_inner(n, true, 2).await;
    (seeded.nodes, seeded.admin_key.expect("admin key"))
}

struct Started {
    nodes: Vec<Node>,
    admin_key: Option<String>,
}

async fn start_cluster_inner(n: usize, seed_admin: bool, rf: usize) -> Started {
    let bound = listeners(n).await;
    let peers_spec = bound
        .iter()
        .enumerate()
        .map(|(i, (_, port))| format!("{}@127.0.0.1:{port}", i + 1))
        .collect::<Vec<_>>()
        .join(",");
    let directory = PeerDirectory::new(&parse_peers(&peers_spec).unwrap());

    let placement = Arc::new(Placement::new(SHARDS, rf, 1..=n as u64).unwrap());
    let metadata_hub = ChannelHub::new();
    let shard_hubs: Vec<ChannelHub> = (0..SHARDS).map(|_| ChannelHub::new()).collect();

    let config = Arc::new(openraft::Config {
        heartbeat_interval: 50,
        election_timeout_min: 200,
        election_timeout_max: 400,
        ..Default::default()
    });

    let mut nodes = Vec::new();
    let mut admin_key = None;
    for (index, (listener, port)) in bound.into_iter().enumerate() {
        let id = (index + 1) as u64;
        let dir = tempfile::tempdir().unwrap();
        let engine = engine_in(dir.path());
        if seed_admin {
            use rekha_core::auth::{generate_api_key, hash_api_key_with_pepper};
            let key = admin_key.get_or_insert_with(generate_api_key).clone();
            let catalog = engine.catalog();
            catalog.ensure_builtin_roles().unwrap();
            let user = rekha_core::auth::UserRecord {
                id: "admin".into(),
                name: "admin".into(),
                enabled: true,
                created_at_ms: 0,
            };
            if catalog.get_user(&user.id).unwrap().is_none() {
                catalog.create_user(&user).unwrap();
                catalog
                    .create_api_key(&user.id, &hash_api_key_with_pepper(&key, &[]), "seed")
                    .unwrap();
                let role = catalog.get_role_by_name("superadmin").unwrap().unwrap();
                catalog.assign_role(&user.id, &role.id, &user.id).unwrap();
            }
        }
        let shards = Arc::new(ShardCluster::new(placement.clone()));

        let mut metadata_sm = ClusterStateMachine::new();
        metadata_sm.set_catalog(engine.catalog());
        metadata_sm.set_engine(engine.clone() as Arc<dyn rekha_cluster::EngineSnapshotProvider>);
        metadata_sm.set_shards(shards.clone(), dir.path().join("placement"));

        let metadata = Arc::new(
            openraft::Raft::<RaftTypeConfig>::new(
                id,
                config.clone(),
                ChannelNetworkFactory::new(metadata_hub.clone()),
                MemoryLogStore::new(),
                metadata_sm,
            )
            .await
            .expect("metadata raft"),
        );
        spawn_group_handler(id, &metadata_hub, metadata.clone());

        for shard in 0..SHARDS {
            let mut shard_sm = ClusterStateMachine::for_shard(id, ShardId(shard));
            shard_sm.set_catalog(engine.catalog());
            shard_sm.set_engine(engine.clone() as Arc<dyn rekha_cluster::EngineSnapshotProvider>);
            let raft = Arc::new(
                openraft::Raft::<RaftTypeConfig>::new(
                    id,
                    config.clone(),
                    ChannelNetworkFactory::new(shard_hubs[shard as usize].clone()),
                    MemoryLogStore::new(),
                    shard_sm,
                )
                .await
                .expect("shard raft"),
            );
            spawn_group_handler(id, &shard_hubs[shard as usize], raft.clone());
            shards.insert(ShardId(shard), raft);
        }

        let state = Arc::new(
            AppState::new(
                engine,
                Some(metadata.clone()),
                ServerConfig {
                    tenant: "default_tenant".into(),
                    database: "default_database".into(),
                    api_key: None,
                    cluster_secret: Some(SECRET.into()),
                    pepper: Vec::new(),
                    allow_open_mode: true,
                },
            )
            .with_peer_directory(directory.clone())
            .with_data_via_log(true)
            .with_node_id(id)
            .with_raft_config(config.clone())
            .with_shards(shards.clone()),
        );

        let base = format!("http://127.0.0.1:{port}");
        let app = router(state);
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        nodes.push(Node {
            id,
            base,
            metadata,
            shards,
            peers: directory.clone(),
            _dir: dir,
        });
    }

    let all: BTreeSet<u64> = (1..=n as u64).collect();
    nodes[0].metadata.initialize(all.clone()).await.unwrap();
    for shard in 0..SHARDS {
        nodes[0]
            .shard_raft(shard)
            .initialize(all.clone())
            .await
            .unwrap();
    }

    for node in &nodes {
        wait_for_leader(&node.metadata, Duration::from_secs(10)).await;
        for shard in 0..SHARDS {
            wait_for_leader(&node.shard_raft(shard), Duration::from_secs(10)).await;
        }
    }
    Started { nodes, admin_key }
}

async fn wait_for_leader(raft: &Arc<openraft::Raft<RaftTypeConfig>>, timeout: Duration) {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if raft.metrics().borrow().current_leader.is_some() {
            return;
        }
        if tokio::time::Instant::now() > deadline {
            panic!("no leader elected within {timeout:?}");
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .no_proxy()
        .build()
        .unwrap()
}

async fn create_partitioned(nodes: &[Node], name: &str) -> String {
    // Send to a follower on purpose: the metadata write must be forwarded to
    // the group that can commit it.
    let target = &nodes[nodes.len() - 1];
    let resp = client()
        .post(format!("{}/api/v2/collections", target.base))
        .json(&serde_json::json!({
            "name": name,
            "dimension": DIM,
            "distance": "l2",
            "partitioned": true
        }))
        .send()
        .await
        .unwrap();
    assert!(
        resp.status().is_success(),
        "create failed: {}",
        resp.text().await.unwrap()
    );
    let body: serde_json::Value = resp.json().await.unwrap();
    body["id"].as_str().unwrap().to_owned()
}

async fn post(node: &Node, path: &str, body: serde_json::Value) -> reqwest::Response {
    client()
        .post(format!("{}{path}", node.base))
        .json(&body)
        .send()
        .await
        .unwrap()
}

fn embeddings(n: usize, seed: f32) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| vec![seed + i as f32, 1.0, 0.0, 0.0])
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_partitioned_collection_commits_across_shard_groups() {
    let nodes = start_cluster(2).await;
    let id = create_partitioned(&nodes, "docs").await;

    // Twenty ids: `shard_for_record` hashes them, so both shard groups are
    // exercised rather than one bucket being hit by luck.
    let ids: Vec<String> = (0..20).map(|i| format!("id-{i}")).collect();
    let resp = post(
        &nodes[0],
        &format!("/api/v2/collections/{id}/add"),
        serde_json::json!({
            "ids": ids,
            "embeddings": embeddings(20, 0.0),
            "metadatas": ids.iter().map(|_| serde_json::json!({"tag": "a"})).collect::<Vec<_>>(),
            "documents": ids.iter().map(|i| format!("doc {i}")).collect::<Vec<_>>()
        }),
    )
    .await;
    assert_eq!(resp.status(), 200, "add: {}", resp.text().await.unwrap());

    // Count goes through every shard group and sums real local counts.
    let resp = post(
        &nodes[1],
        &format!("/api/v2/collections/{id}/count"),
        serde_json::json!({}),
    )
    .await;
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["count"], 20);

    // A read served by the other node sees the same records.
    let resp = post(
        &nodes[1],
        &format!("/api/v2/collections/{id}/get"),
        serde_json::json!({"ids": ["id-0", "id-7", "id-19"]}),
    )
    .await;
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["ids"].as_array().unwrap().len(), 3);
    assert_eq!(body["documents"][0], "doc id-0");

    // Query scatters to both shards and merges the hits.
    let resp = post(
        &nodes[0],
        &format!("/api/v2/collections/{id}/query"),
        serde_json::json!({
            "query_embeddings": [[0.0, 1.0, 0.0, 0.0]],
            "n_results": 5,
            "include": ["documents", "distances"]
        }),
    )
    .await;
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["ids"][0].as_array().unwrap().len(), 5);

    // Delete is also partitioned: it must remove every matching record, not
    // just the ones that live on the node that received the request.
    let resp = post(
        &nodes[0],
        &format!("/api/v2/collections/{id}/delete"),
        serde_json::json!({"ids": ["id-1", "id-2", "id-3"]}),
    )
    .await;
    assert_eq!(resp.status(), 200, "delete: {}", resp.text().await.unwrap());
    let resp = post(
        &nodes[1],
        &format!("/api/v2/collections/{id}/count"),
        serde_json::json!({}),
    )
    .await;
    assert_eq!(resp.json::<serde_json::Value>().await.unwrap()["count"], 17);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn partitioned_upsert_and_update_reach_the_owning_shard() {
    let nodes = start_cluster(2).await;
    let id = create_partitioned(&nodes, "mutate").await;

    let ids: Vec<String> = (0..8).map(|i| format!("m-{i}")).collect();
    let resp = post(
        &nodes[0],
        &format!("/api/v2/collections/{id}/add"),
        serde_json::json!({
            "ids": ids,
            "embeddings": embeddings(8, 0.0),
            "metadatas": ids.iter().map(|_| serde_json::json!({"k": "v"})).collect::<Vec<_>>(),
            "documents": ids.iter().map(|_| "original".to_string()).collect::<Vec<_>>()
        }),
    )
    .await;
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());

    // Upsert replaces the vector and merges metadata for one record.
    let resp = post(
        &nodes[1],
        &format!("/api/v2/collections/{id}/upsert"),
        serde_json::json!({
            "ids": ["m-0"],
            "embeddings": [[9.0, 9.0, 9.0, 9.0]],
            "metadatas": [{"added": "yes"}]
        }),
    )
    .await;
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());

    let resp = post(
        &nodes[0],
        &format!("/api/v2/collections/{id}/get"),
        serde_json::json!({"ids": ["m-0"], "include": ["metadatas", "documents", "embeddings"]}),
    )
    .await;
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body["metadatas"][0],
        serde_json::json!({"k": "v", "added": "yes"})
    );
    assert_eq!(body["documents"][0], "original");
    assert_eq!(
        body["embeddings"][0],
        serde_json::json!([9.0, 9.0, 9.0, 9.0])
    );

    // Metadata-only update preserves the vector and the document.
    let resp = post(
        &nodes[1],
        &format!("/api/v2/collections/{id}/update"),
        serde_json::json!({
            "ids": ["m-1"],
            "metadatas": [{"stage": "second"}]
        }),
    )
    .await;
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let resp = post(
        &nodes[0],
        &format!("/api/v2/collections/{id}/get"),
        serde_json::json!({"ids": ["m-1"], "include": ["metadatas", "documents"]}),
    )
    .await;
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(
        body["metadatas"][0],
        serde_json::json!({"k": "v", "stage": "second"})
    );
    assert_eq!(body["documents"][0], "original");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn both_shard_groups_report_a_leader_and_the_nodes_agree() {
    let nodes = start_cluster(2).await;
    for shard in 0..SHARDS {
        let mut leaders = BTreeSet::new();
        let mut voters = BTreeSet::new();
        for node in &nodes {
            let m = node.shard_raft(shard).metrics().borrow().clone();
            leaders.insert(m.current_leader.expect("shard leader"));
            voters.extend(m.membership_config.membership().voter_ids());
            assert!(voters.contains(&node.id));
        }
        assert_eq!(
            leaders.len(),
            1,
            "both nodes must agree on one leader for shard {shard}"
        );
        assert_eq!(voters, BTreeSet::from([1, 2]));
    }

    for node in &nodes {
        assert_eq!(node.shards.owners(ShardId(0)).len(), 2);
    }
}

// ---------------------------------------------------------------------------
// Internal admin surface
// ---------------------------------------------------------------------------

fn internal(node: &Node, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
    client()
        .request(method, format!("{}{path}", node.base))
        .header("x-rekha-cluster-secret", SECRET)
}

async fn internal_get(node: &Node, path: &str) -> reqwest::Response {
    internal(node, reqwest::Method::GET, path)
        .send()
        .await
        .unwrap()
}

async fn internal_post(node: &Node, path: &str, body: serde_json::Value) -> reqwest::Response {
    internal(node, reqwest::Method::POST, path)
        .json(&body)
        .send()
        .await
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn internal_routes_require_the_cluster_secret() {
    let nodes = start_cluster(2).await;
    let resp = client()
        .get(format!("{}/internal/cluster/membership", nodes[0].base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401, "internal routes fail closed");

    let resp = internal_get(&nodes[0], "/internal/cluster/membership").await;
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn membership_reports_voters_learners_and_per_shard_leaders() {
    let nodes = start_cluster(2).await;
    let body: serde_json::Value = internal_get(&nodes[0], "/internal/cluster/membership")
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(body["voters"], serde_json::json!([1, 2]));
    assert_eq!(body["members"], serde_json::json!([1, 2]));
    assert!(body["learners"].as_array().unwrap().is_empty());
    assert_eq!(body["shards"].as_array().unwrap().len(), SHARDS as usize);
    assert!(body["leader"].as_u64().is_some());
    assert!(body["state"].is_string());
    assert!(body["last_promotion"].is_null());

    let placement: serde_json::Value = internal_get(&nodes[1], "/internal/cluster/placement")
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(placement["shard_count"], SHARDS);
    assert_eq!(placement["replication_factor"], 2);
    assert_eq!(placement["voters"], serde_json::json!([1, 2]));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_learner_is_reported_with_infinite_lag_when_it_never_replicated() {
    let nodes = start_cluster(2).await;
    let resp = internal_post(
        &nodes[1],
        "/internal/cluster/add_learner",
        serde_json::json!({"node_id": 3, "addr": "127.0.0.1:9"}),
    )
    .await;
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());

    let body: serde_json::Value = internal_get(&nodes[0], "/internal/cluster/membership")
        .await
        .json()
        .await
        .unwrap();
    let learners = body["learners"].as_array().unwrap();
    assert_eq!(learners.len(), 1);
    assert_eq!(learners[0]["node"], 3);
    assert!(
        learners[0]["matched_index"].is_null(),
        "no replication yet means infinite lag, not zero: {learners:?}"
    );
    assert!(learners[0]["lag"].is_null());
    assert_eq!(body["members"], serde_json::json!([1, 2, 3]));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn membership_validation_refuses_bad_input_and_loops() {
    let nodes = start_cluster(2).await;

    let missing_id = internal_post(
        &nodes[0],
        "/internal/cluster/add_learner",
        serde_json::json!({}),
    )
    .await;
    assert_eq!(missing_id.status(), 400);

    let missing_addr = internal_post(
        &nodes[0],
        "/internal/cluster/add_learner",
        serde_json::json!({"node_id": 9}),
    )
    .await;
    assert_eq!(missing_addr.status(), 400);

    let looped = internal(
        &nodes[0],
        reqwest::Method::POST,
        "/internal/cluster/add_learner",
    )
    .header("x-rekha-admin-hops", "3")
    .json(&serde_json::json!({"node_id": 9, "addr": "127.0.0.1:9"}))
    .send()
    .await
    .unwrap();
    assert_eq!(looped.status(), 508);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn removing_a_voter_is_refused_while_shards_are_placed() {
    let nodes = start_cluster(2).await;
    let resp = internal_post(
        &nodes[0],
        "/internal/cluster/remove_member",
        serde_json::json!({"node_id": 2}),
    )
    .await;
    assert_eq!(resp.status(), 409);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("membership cannot change"),
        "{body}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn shard_membership_and_counts_answer_from_the_group() {
    let nodes = start_cluster(2).await;
    let id = create_partitioned(&nodes, "counts").await;
    let ids: Vec<String> = (0..6).map(|i| format!("c-{i}")).collect();
    let resp = post(
        &nodes[0],
        &format!("/api/v2/collections/{id}/add"),
        serde_json::json!({"ids": ids, "embeddings": embeddings(6, 0.0)}),
    )
    .await;
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());

    let members: serde_json::Value =
        internal_get(&nodes[0], "/internal/cluster/shard/0/membership")
            .await
            .json()
            .await
            .unwrap();
    assert_eq!(members["voters"], serde_json::json!([1, 2]));

    let counts: serde_json::Value =
        internal_post(&nodes[0], "/internal/stats/counts", serde_json::json!([id]))
            .await
            .json()
            .await
            .unwrap();
    assert_eq!(
        counts[&id], 6,
        "a voter answers for its own partition: {counts}"
    );

    // A group this node does not host has no route.
    let missing = internal_get(&nodes[0], "/internal/cluster/shard/99/membership").await;
    assert_eq!(missing.status(), 404);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn raft_rpc_endpoints_reject_an_unknown_group() {
    let nodes = start_cluster(2).await;
    let resp = internal_post(
        &nodes[0],
        "/internal/raft/99/append_entries",
        serde_json::json!({}),
    )
    .await;
    assert!(
        resp.status() == 404 || resp.status() == 400 || resp.status() == 422,
        "unknown group must not be dispatched: {}",
        resp.status()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn chroma_routes_scatter_over_shard_groups() {
    let nodes = start_cluster(2).await;
    let id = create_partitioned(&nodes, "chroma-docs").await;
    let base = "/api/v2/tenants/default_tenant/databases/default_database/collections/chroma-docs";

    let ids: Vec<String> = (0..10).map(|i| format!("ch-{i}")).collect();
    let resp = post(
        &nodes[0],
        &format!("{base}/add"),
        serde_json::json!({
            "ids": ids,
            "embeddings": embeddings(10, 0.0),
            "metadatas": ids.iter().map(|_| serde_json::json!({"tag": "a"})).collect::<Vec<_>>(),
            "documents": ids.iter().map(|i| format!("doc {i}")).collect::<Vec<_>>()
        }),
    )
    .await;
    assert_eq!(resp.status(), 201, "{}", resp.text().await.unwrap());

    // Upsert one record through the sharded 2PC path.
    let resp = post(
        &nodes[1],
        &format!("{base}/upsert"),
        serde_json::json!({
            "ids": ["ch-0"],
            "embeddings": [[5.0, 5.0, 5.0, 5.0]],
            "metadatas": [{"tag": "b"}]
        }),
    )
    .await;
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());

    // A get with no ids scans every shard and merges by sequence.
    let resp = post(&nodes[0], &format!("{base}/get"), serde_json::json!({})).await;
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["ids"].as_array().unwrap().len(), 10);

    // Update through shards (metadata + document, vector untouched).
    let resp = post(
        &nodes[1],
        &format!("{base}/update"),
        serde_json::json!({
            "ids": ["ch-1"],
            "documents": ["updated"]
        }),
    )
    .await;
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());

    // Query with a metadata filter over both groups.
    let resp = post(
        &nodes[0],
        &format!("{base}/query"),
        serde_json::json!({
            "query_embeddings": [[0.0, 1.0, 0.0, 0.0]],
            "n_results": 3,
            "where": {"tag": {"$eq": "b"}},
            "include": ["metadatas", "documents", "distances"]
        }),
    )
    .await;
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["ids"][0], serde_json::json!(["ch-0"]));

    // Delete by filter is a DeleteWhere prepared on each shard.
    let resp = post(
        &nodes[0],
        &format!("{base}/delete"),
        serde_json::json!({"where": {"tag": {"$eq": "a"}}}),
    )
    .await;
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    assert_eq!(
        resp.json::<serde_json::Value>().await.unwrap()["deleted"],
        9
    );

    let resp = client()
        .get(format!("{}{base}/count", nodes[1].base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let text = resp.text().await.unwrap();
    assert_eq!(
        text.trim().trim_matches('"'),
        "1",
        "only the upserted record remains"
    );
    let _ = id;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn dropping_a_shard_stops_answering_for_it() {
    let nodes = start_cluster(2).await;
    let resp = internal_post(
        &nodes[0],
        "/internal/cluster/shard/1",
        serde_json::json!({}),
    )
    .await;
    // The drop route is a POST to /drop, not the shard root.
    assert_eq!(resp.status(), 404, "no route at the shard root");

    let resp = internal_post(
        &nodes[0],
        "/internal/cluster/shard/1/drop",
        serde_json::json!({}),
    )
    .await;
    assert_eq!(resp.status(), 200);

    let gone = internal_get(&nodes[0], "/internal/cluster/shard/1/membership").await;
    assert_eq!(gone.status(), 404, "a dropped group is not hosted");
    assert!(!nodes[0].shards.hosts(ShardId(1)));

    // The recovery path: a peer that was told this node owns the shard asks it
    // to start the group again, and the node rebuilds it from its own log.
    let restarted = internal_post(
        &nodes[0],
        "/internal/cluster/shard/1/ensure_started",
        serde_json::json!({}),
    )
    .await;
    assert_eq!(restarted.status(), 200);
    assert!(nodes[0].shards.hosts(ShardId(1)));
    let again = internal_get(&nodes[0], "/internal/cluster/shard/1/membership").await;
    assert_eq!(again.status(), 200);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn ensure_started_is_idempotent_for_a_hosted_shard() {
    let nodes = start_cluster(2).await;
    let resp = internal_post(
        &nodes[0],
        "/internal/cluster/shard/0/ensure_started",
        serde_json::json!({}),
    )
    .await;
    assert_eq!(resp.status(), 200);
    assert!(nodes[0].shards.hosts(ShardId(0)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_placed_collection_reports_its_shard_and_deletes_through_the_log() {
    let nodes = start_cluster(2).await;

    let resp = post(
        &nodes[0],
        "/api/v2/collections",
        serde_json::json!({"name": "placed", "dimension": DIM, "distance": "l2"}),
    )
    .await;
    assert!(resp.status().is_success(), "{}", resp.text().await.unwrap());
    let body: serde_json::Value = resp.json().await.unwrap();
    let id = body["id"].as_str().unwrap().to_owned();
    assert!(
        body["shard"].is_u64(),
        "a collection placed by the leader names its shard: {body}"
    );

    // The row is replicated: the other node lists it even though it may not
    // hold the data.
    // The row is committed on the metadata leader and applied on the follower
    // asynchronously, so a single read can legally arrive between the two.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let list: serde_json::Value = loop {
        let list: serde_json::Value = client()
            .get(format!("{}/api/v2/collections", nodes[1].base))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let visible = list
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["id"] == id.as_str());
        if visible {
            break list;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "collection did not replicate: {list}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    assert_eq!(list[0]["count"], 0, "a fresh collection counts zero");

    // A single-shard write goes through the shard group, not the 2PC path.
    let resp = post(
        &nodes[1],
        &format!("/api/v2/collections/{id}/add"),
        serde_json::json!({
            "ids": ["a", "b"],
            "embeddings": embeddings(2, 0.0)
        }),
    )
    .await;
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let count: serde_json::Value = post(
        &nodes[0],
        &format!("/api/v2/collections/{id}/count"),
        serde_json::json!({}),
    )
    .await
    .json()
    .await
    .unwrap();
    assert_eq!(count["count"], 2);

    // Reads on the other node must follow the shard's leader and, when it is
    // already there, still prove leadership before answering.
    for path in ["get", "query"] {
        let body = if path == "get" {
            serde_json::json!({"ids": ["a"]})
        } else {
            serde_json::json!({"query_embeddings": [[1.0, 0.0, 0.0, 0.0]], "n_results": 1})
        };
        let resp = post(
            &nodes[1],
            &format!("/api/v2/collections/{id}/{path}"),
            body.clone(),
        )
        .await;
        assert_eq!(resp.status(), 200, "{path}: {}", resp.text().await.unwrap());

        // `?consistency=local` is the explicit opt-out of the barrier; a
        // follower may answer it or refuse it, but it must not forward.
        let resp = post(
            &nodes[1],
            &format!("/api/v2/collections/{id}/{path}?consistency=local"),
            body,
        )
        .await;
        assert!(
            resp.status() == 200 || resp.status() == 503,
            "{path} local read: {}",
            resp.status()
        );
    }

    let resp = client()
        .delete(format!("{}/api/v2/collections/{id}", nodes[0].base))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success(), "{}", resp.text().await.unwrap());
    let missing = client()
        .get(format!("{}/api/v2/collections/{id}", nodes[0].base))
        .send()
        .await
        .unwrap();
    assert_eq!(missing.status(), 404);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn transaction_outcomes_are_visible_on_the_internal_surface() {
    let nodes = start_cluster(2).await;
    let tx = uuid::Uuid::new_v4();

    let status: serde_json::Value =
        internal_get(&nodes[0], &format!("/internal/cluster/tx/{tx}/status"))
            .await
            .json()
            .await
            .unwrap();
    assert_eq!(status["committed"], false);

    // Commit on whichever node leads the metadata group: a follower answers
    // 503 with the leader hint, which is itself part of the contract.
    let commit = internal_post(
        &nodes[0],
        "/internal/cluster/tx/commit",
        serde_json::json!({"tx_id": tx, "shards": [0]}),
    )
    .await;
    assert!(
        commit.status() == 200 || commit.status() == 503,
        "{}",
        commit.text().await.unwrap()
    );
    if commit.status() == 503 {
        let body: serde_json::Value = commit.json().await.unwrap();
        assert!(body["leader"].is_u64(), "{body}");
    }

    // Abort is unconditional and idempotent.
    let abort = internal_post(
        &nodes[0],
        "/internal/cluster/tx/abort",
        serde_json::json!({"tx_id": tx}),
    )
    .await;
    assert_eq!(abort.status(), 200);
    let status: serde_json::Value =
        internal_get(&nodes[0], &format!("/internal/cluster/tx/{tx}/status"))
            .await
            .json()
            .await
            .unwrap();
    assert_eq!(status["committed"], false);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_shard_write_follows_the_leader_hint_when_the_receiver_lost_the_group() {
    // Three replicas: stopping one node's copy still leaves a quorum that can
    // elect, which is the point of this test.
    let nodes = start_cluster_with_replication(3, 3).await;
    let id = create_partitioned(&nodes, "remote").await;

    // Stop this node's copies of the shard groups. Metadata leadership is
    // untouched, so the node still accepts the request and must forward each
    // prepare/commit to the new shard leaders.
    for shard in 0..SHARDS {
        let _ = nodes[0].shard_raft(shard).shutdown().await;
        // Drop the stopped instance too: a shut-down `Raft` still reports its
        // last known leader, and the dispatcher would take the local path and
        // fail instead of forwarding to a live leader.
        let _ = nodes[0].shards.remove(ShardId(shard));
    }
    tokio::time::sleep(Duration::from_millis(300)).await;

    let ids: Vec<String> = (0..8).map(|i| format!("r-{i}")).collect();
    let resp = post(
        &nodes[0],
        &format!("/api/v2/collections/{id}/add"),
        serde_json::json!({"ids": ids, "embeddings": embeddings(8, 0.0)}),
    )
    .await;
    assert_eq!(
        resp.status(),
        200,
        "a write must be forwarded to the shard leaders: {}",
        resp.text().await.unwrap()
    );

    let count: serde_json::Value = post(
        &nodes[1],
        &format!("/api/v2/collections/{id}/count"),
        serde_json::json!({}),
    )
    .await
    .json()
    .await
    .unwrap();
    assert_eq!(count["count"], 8);

    // Reads from the node that dropped the groups are dispatched to the owners
    // rather than answered from nothing.
    let count: serde_json::Value = post(
        &nodes[0],
        &format!("/api/v2/collections/{id}/count"),
        serde_json::json!({}),
    )
    .await
    .json()
    .await
    .unwrap();
    assert_eq!(count["count"], 8);

    let hits: serde_json::Value = post(
        &nodes[0],
        &format!("/api/v2/collections/{id}/query"),
        serde_json::json!({"query_embeddings": [[0.0, 1.0, 0.0, 0.0]], "n_results": 3}),
    )
    .await
    .json()
    .await
    .unwrap();
    assert_eq!(hits["ids"][0].as_array().unwrap().len(), 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn rbac_and_namespaces_replicate_through_the_metadata_group() {
    let (nodes, key) = start_cluster_with_admin(2).await;

    async fn authed(
        nodes: &[Node],
        path: &str,
        body: serde_json::Value,
        key: &str,
    ) -> reqwest::Response {
        let mut last = None;
        for node in nodes {
            let resp = client()
                .post(format!("{}{path}", node.base))
                .header("x-api-key", key)
                .json(&body)
                .send()
                .await
                .unwrap();
            if resp.status() != 503 {
                return resp;
            }
            last = Some(resp);
        }
        last.expect("at least one node")
    }

    let resp = authed(
        &nodes,
        "/api/v2/auth/users",
        serde_json::json!({"name": "alice"}),
        &key,
    )
    .await;
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let alice: serde_json::Value = resp.json().await.unwrap();
    let alice_id = alice["id"].as_str().unwrap().to_owned();

    let resp = authed(
        &nodes,
        "/api/v2/auth/roles",
        serde_json::json!({"name": "analyst", "description": "read only"}),
        &key,
    )
    .await;
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let role: serde_json::Value = resp.json().await.unwrap();
    let role_id = role["id"].as_str().unwrap().to_owned();

    let resp = authed(
        &nodes,
        &format!("/api/v2/auth/users/{alice_id}/roles"),
        serde_json::json!({"role_id": role_id}),
        &key,
    )
    .await;
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());

    let resp = authed(
        &nodes,
        &format!("/api/v2/auth/roles/{role_id}/permissions"),
        serde_json::json!({"action": "collection.get", "tenant": "default_tenant"}),
        &key,
    )
    .await;
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());

    // Key management travels the same log.
    let resp = authed(
        &nodes,
        &format!("/api/v2/auth/users/{alice_id}/keys"),
        serde_json::json!({"name": "ci"}),
        &key,
    )
    .await;
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    let created: serde_json::Value = resp.json().await.unwrap();
    let hash = created["key_hash_hex"].as_str().unwrap().to_owned();

    let keys: serde_json::Value = client()
        .get(format!(
            "{}/api/v2/auth/users/{alice_id}/keys",
            nodes[0].base
        ))
        .header("x-api-key", &key)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(keys.to_string().contains("ci"), "{keys}");

    let resp = client()
        .delete(format!(
            "{}/api/v2/auth/users/{alice_id}/keys/{hash}",
            nodes[0].base
        ))
        .header("x-api-key", &key)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());

    // The write was a metadata entry: wait for the follower to apply it.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let users: serde_json::Value = client()
            .get(format!("{}/api/v2/auth/users", nodes[1].base))
            .header("x-api-key", &key)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        if users
            .as_array()
            .unwrap()
            .iter()
            .any(|u| u["id"] == alice_id.as_str())
        {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "user did not replicate: {users}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let resp = authed(
        &nodes,
        "/api/v2/tenants",
        serde_json::json!({"name": "acme"}),
        &key,
    )
    .await;
    assert!(
        resp.status().is_success() || resp.status() == 409,
        "{}",
        resp.text().await.unwrap()
    );

    let resp = authed(
        &nodes,
        "/api/v2/tenants/acme/databases",
        serde_json::json!({"name": "vectors"}),
        &key,
    )
    .await;
    assert!(
        resp.status().is_success() || resp.status() == 409,
        "{}",
        resp.text().await.unwrap()
    );

    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let tenant = client()
            .get(format!("{}/api/v2/tenants/acme", nodes[1].base))
            .header("x-api-key", &key)
            .send()
            .await
            .unwrap();
        if tenant.status().is_success() {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "namespace did not replicate: {}",
            tenant.status()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let dbs: serde_json::Value = client()
        .get(format!("{}/api/v2/tenants/acme/databases", nodes[0].base))
        .header("x-api-key", &key)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(dbs.to_string().contains("vectors"), "{dbs}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_rebalancer_tick_records_a_skipped_promotion_and_reconciles_owners() {
    let nodes = start_cluster(2).await;
    // A learner with no reachable process: matched is None, which must be read
    // as infinite lag and never promoted.
    let resp = internal_post(
        &nodes[0],
        "/internal/cluster/add_learner",
        serde_json::json!({"node_id": 3, "addr": "127.0.0.1:9"}),
    )
    .await;
    assert_eq!(resp.status(), 200, "{}", resp.text().await.unwrap());
    tokio::time::sleep(Duration::from_millis(200)).await;

    let leader_index = nodes
        .iter()
        .position(|n| n.metadata.metrics().borrow().state.is_leader())
        .expect("a metadata leader");
    let follower_index = 1 - leader_index.min(1);

    // A follower tick returns before touching shards.
    rekha_cluster::rebalancer::run_rebalancer_tick(
        nodes[follower_index].id,
        &nodes[follower_index].metadata,
        &nodes[follower_index].shards,
        &nodes[follower_index].peers,
        &client(),
        Some(SECRET),
    )
    .await;

    // Make the target owner set disagree with the running shard groups, so the
    // per-shard reconciliation body runs without writing anything (node 3 has
    // no address, so it cannot be reached).
    // Pin shard 0 to the original owners while the voter set grows: the
    // reconciler derives a different target, so `current != target` runs.
    let widened = Placement::new(SHARDS, 2, [1u64, 2, 3])
        .unwrap()
        .with_shard_owners(ShardId(0), &[1, 2])
        .unwrap();
    nodes[leader_index].shards.set_placement(Arc::new(widened));

    for _ in 0..2 {
        rekha_cluster::rebalancer::run_rebalancer_tick(
            nodes[leader_index].id,
            &nodes[leader_index].metadata,
            &nodes[leader_index].shards,
            &nodes[leader_index].peers,
            &client(),
            Some(SECRET),
        )
        .await;
    }

    let body: serde_json::Value = internal_get(&nodes[0], "/internal/cluster/membership")
        .await
        .json()
        .await
        .unwrap();
    let promotion = &body["last_promotion"];
    assert!(
        !promotion.is_null(),
        "a promotion attempt is recorded: {body}"
    );
    assert_eq!(promotion["node"], 3);
    assert_eq!(promotion["outcome"], "skipped_lag");
    assert!(
        promotion["lag"].is_null(),
        "never-replicated is reported as unknown lag: {promotion}"
    );
    assert!(
        body["voters"].as_array().unwrap().len() == 2,
        "a skipped learner is not a voter: {body}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn the_raft_http_endpoints_accept_a_well_formed_rpc() {
    let nodes = start_cluster(2).await;
    let vote = serde_json::json!({
        "vote": {"leader_id": {"term": 1, "node_id": 1}, "committed": false},
        "last_log_id": null
    });
    let append = serde_json::json!({
        "vote": {"leader_id": {"term": 1, "node_id": 1}, "committed": false},
        "prev_log_id": null,
        "entries": [],
        "leader_commit": null
    });

    for group in [0u32, 1] {
        let resp = internal_post(
            &nodes[0],
            &format!("/internal/raft/{group}/vote"),
            vote.clone(),
        )
        .await;
        assert_eq!(
            resp.status(),
            200,
            "vote for group {group}: {}",
            resp.text().await.unwrap()
        );

        let resp = internal_post(
            &nodes[0],
            &format!("/internal/raft/{group}/append_entries"),
            append.clone(),
        )
        .await;
        assert_eq!(
            resp.status(),
            200,
            "append_entries for group {group}: {}",
            resp.text().await.unwrap()
        );
    }
}
