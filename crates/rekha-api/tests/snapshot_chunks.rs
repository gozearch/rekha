//! The chunked snapshot upload's guards.
//!
//! This endpoint repairs a replica that is behind the purged log prefix, which
//! makes its failure modes unusually expensive: a chunk spliced into the wrong
//! offset produces an archive whose CRC check fails somewhere else, and a
//! transfer that silently accepts a short body installs a truncated state
//! machine. So each precondition is asserted here, before the happy path exists
//! anywhere else.
//!
//! The install step itself (a real archive replacing real state) is covered end
//! to end by `rekha-cli`'s `snapshot_rejoin` test. These are the cheap refusals.

use std::collections::BTreeSet;
use std::sync::Arc;

use rekha_api::{AppState, ServerConfig, router};
use rekha_cluster::network::channel::ChannelHub;
use rekha_cluster::state_machine::ClusterStateMachine;
use rekha_cluster::{MemoryLogStore, RaftTypeConfig, SnapshotStore};
use rekha_engine::{Engine, EngineConfig};
use rekha_storage::{LocalStorage, RedbCatalog};

const SECRET: &str = "snapshot-chunk-test-secret-0123456789";

struct Harness {
    base: String,
    _dir: tempfile::TempDir,
    _raft: Arc<openraft::Raft<RaftTypeConfig>>,
}

async fn harness() -> Harness {
    let dir = tempfile::tempdir().expect("tempdir");
    let catalog = Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).expect("catalog"));
    let storage = Arc::new(LocalStorage::new(dir.path().join("objects")));
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&wal_dir).expect("wal dir");
    let engine = Arc::new(
        Engine::open(
            catalog,
            storage,
            &wal_dir,
            EngineConfig {
                apply_mode: rekha_engine::ApplyMode::Committed,
                ..EngineConfig::default()
            },
        )
        .expect("engine"),
    );

    let mut sm = ClusterStateMachine::for_metadata(1);
    sm.set_catalog(engine.catalog());
    sm.set_engine(engine.clone() as Arc<dyn rekha_cluster::EngineSnapshotProvider>);
    sm.set_snapshot_store(Arc::new(
        SnapshotStore::open(wal_dir.join("snapshots").join("metadata"), "metadata").expect("store"),
    ));

    let raft = Arc::new(
        openraft::Raft::<RaftTypeConfig>::new(
            1,
            Arc::new(openraft::Config {
                heartbeat_interval: 50,
                election_timeout_min: 200,
                election_timeout_max: 400,
                ..Default::default()
            }),
            rekha_cluster::network::channel::ChannelNetworkFactory::new(ChannelHub::new()),
            MemoryLogStore::new(),
            sm,
        )
        .await
        .expect("raft"),
    );
    raft.initialize(BTreeSet::from([1u64]))
        .await
        .expect("initialize");

    let state = Arc::new(
        AppState::new(
            engine,
            Some(raft.clone()),
            ServerConfig {
                tenant: "default_tenant".into(),
                database: "default_database".into(),
                api_key: None,
                cluster_secret: Some(SECRET.into()),
                pepper: Vec::new(),
                allow_open_mode: true,
            },
        )
        .with_data_via_log(true),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener");
    let port = listener.local_addr().expect("addr").port();
    let app = router(state);
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });

    Harness {
        base: format!("http://127.0.0.1:{port}"),
        _dir: dir,
        _raft: raft,
    }
}

fn vote_json() -> String {
    serde_json::to_string(&openraft::Vote::new(1, 1u64)).expect("vote json")
}

fn meta_json(id: &str) -> String {
    serde_json::to_string(&openraft::SnapshotMeta::<u64, u64> {
        last_log_id: Some(openraft::LogId::new(
            openraft::CommittedLeaderId::new(1, 1),
            4,
        )),
        last_membership: openraft::StoredMembership::default(),
        snapshot_id: id.to_string(),
    })
    .expect("meta json")
}

struct Chunk<'a> {
    id: &'a str,
    offset: u64,
    total: u64,
    final_chunk: bool,
    body: &'a [u8],
}

impl Harness {
    async fn post(&self, chunk: Chunk<'_>) -> (u16, String) {
        let url = format!("{}/internal/raft/0/snapshot/{}", self.base, chunk.id);
        let resp = reqwest::Client::new()
            .post(&url)
            .header("x-rekha-cluster-secret", SECRET)
            .header("content-type", "application/octet-stream")
            .header("x-rekha-snapshot-offset", chunk.offset.to_string())
            .header("x-rekha-snapshot-total", chunk.total.to_string())
            .header(
                "x-rekha-snapshot-final",
                if chunk.final_chunk { "1" } else { "0" },
            )
            .header("x-rekha-snapshot-vote", vote_json())
            .header("x-rekha-snapshot-meta", meta_json(chunk.id))
            .body(chunk.body.to_vec())
            .send()
            .await
            .expect("request");
        let status = resp.status().as_u16();
        (status, resp.text().await.unwrap_or_default())
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_snapshot_id_that_could_escape_the_staging_directory_is_refused() {
    let h = harness().await;
    let (status, body) = h
        .post(Chunk {
            id: "..%2Fescape",
            offset: 0,
            total: 1,
            final_chunk: true,
            body: b"x",
        })
        .await;
    assert_eq!(status, 400, "a traversal id must be refused: {body}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_chunk_that_starts_past_the_staged_bytes_is_refused() {
    let h = harness().await;
    let (status, body) = h
        .post(Chunk {
            id: "gap",
            offset: 4,
            total: 10,
            final_chunk: false,
            body: b"abcde",
        })
        .await;
    assert_eq!(
        status, 409,
        "a chunk starting past the staged length would leave a hole: {body}"
    );
    assert!(body.contains("restart from zero"), "{body}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_final_chunk_that_does_not_complete_the_snapshot_is_refused() {
    let h = harness().await;
    let (status, body) = h
        .post(Chunk {
            id: "short",
            offset: 0,
            total: 10,
            final_chunk: true,
            body: b"12345",
        })
        .await;
    assert_eq!(status, 400, "a short final chunk must not install: {body}");
    assert!(body.contains("final chunk ends at byte 5"), "{body}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_chunk_beyond_the_declared_total_is_refused() {
    let h = harness().await;
    let (status, body) = h
        .post(Chunk {
            id: "overrun",
            offset: 0,
            total: 3,
            final_chunk: false,
            body: b"12345",
        })
        .await;
    assert_eq!(
        status, 400,
        "a chunk past the total must be refused: {body}"
    );
    assert!(body.contains("beyond the declared total"), "{body}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_intermediate_chunk_is_staged_and_a_restart_at_zero_replaces_it() {
    let h = harness().await;

    // Half the archive: accepted, not installed.
    let (status, body) = h
        .post(Chunk {
            id: "staged",
            offset: 0,
            total: 10,
            final_chunk: false,
            body: b"12345",
        })
        .await;
    assert_eq!(status, 200, "the first chunk must stage: {body}");

    // A continuation at the right offset is accepted, which is what proves the
    // first chunk actually landed rather than being accepted and dropped.
    let (status, body) = h
        .post(Chunk {
            id: "staged",
            offset: 5,
            total: 10,
            final_chunk: false,
            body: b"67890",
        })
        .await;
    assert_eq!(status, 200, "the continuation must append: {body}");

    // Starting over at zero discards the partial transfer instead of 409ing for
    // the rest of the attempt.
    let (status, body) = h
        .post(Chunk {
            id: "staged",
            offset: 0,
            total: 10,
            final_chunk: false,
            body: b"abc",
        })
        .await;
    assert_eq!(status, 200, "a restart at zero must be accepted: {body}");
}
