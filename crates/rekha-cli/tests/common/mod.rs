//! Shared scaffolding for the cluster integration suites.
//!
//! Each test file is its own binary, and the scaffolding is identical: spawn
//! real processes, wait for readiness, talk HTTP. It lives here rather than
//! being copied because two copies of "how to start a cluster" drift, and the
//! copy that drifts starts a cluster the other test's assertions were not
//! written for.

#![allow(dead_code)] // each suite uses a subset

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use rekha_cluster::{Placement, PlacementConfig};

pub const KEY: &str = "shard-test-key-0123456789abcdef0123";
/// At least 32 bytes: `serve` refuses a short cluster secret.
pub const SECRET: &str = "shard-test-cluster-secret-0123456789abcdef";
/// Optional defence-in-depth; tests exercise the peppered verification path.
pub const PEPPER: &str = "shard-test-key-pepper";
pub const DIM: usize = 8;
pub const RECORDS: usize = 40;

/// One cluster test at a time, within a suite.
///
/// Two clusters picking ephemeral ports in the same process can be handed the
/// same port by the OS, and then one test's assertions run against the other
/// test's cluster. This also keeps CPU contention down enough that elections
/// settle predictably.
///
/// Per binary: each test file links its own copy, and cargo runs test binaries
/// one at a time, so this covers the whole run anyway.
pub static SERIAL: std::sync::LazyLock<tokio::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| tokio::sync::Mutex::new(()));

pub struct Node {
    pub id: u64,
    pub port: u16,
    pub data_dir: PathBuf,
    pub child: Child,
}

impl Drop for Node {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Node {
    pub fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.port)
    }
}

pub fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    listener.local_addr().expect("addr").port()
}

/// A fresh data directory that outlives the child processes using it.
///
/// The guard is leaked on purpose: a `tempdir` dropped at the end of the
/// setup expression would delete the directory from under a running server.
pub fn temp_dir() -> PathBuf {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().to_path_buf();
    std::mem::forget(dir);
    path
}

/// Run a `rekha` subcommand against `url`, with the cluster secret.
///
/// `--url` is a global flag, so it precedes the subcommand.
pub fn rekha(args: &[&str], url: &str) -> (i32, String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_rekha"))
        .arg("--url")
        .arg(url)
        .args(args)
        .env("REKHA_API_KEY", KEY)
        .env("REKHA_CLUSTER_SECRET", SECRET)
        .env("RUST_LOG", "error")
        .output()
        .expect("run rekha");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// The parsed output of `rekha membership list` against `url`.
pub fn membership(url: &str) -> serde_json::Value {
    let (code, stdout, stderr) = rekha(&["membership", "list"], url);
    assert_eq!(code, 0, "membership list failed: {stderr}");
    serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("list output `{stdout}`: {e}"))
}

/// The `id@addr` topology string for a set of ports, ids `1..=ports.len()`.
pub fn peers_spec(ports: &[u16]) -> String {
    (0..ports.len())
        .map(|i| format!("{}@127.0.0.1:{}", i + 1, ports[i]))
        .collect::<Vec<_>>()
        .join(",")
}

/// The command that starts one node.
///
/// Shared by [`spawn_cluster`] and by tests that need to *watch* a node fail to
/// start: a refusal asserted against a differently built command could be
/// failing for a reason other than the one under test.
pub fn serve_command(
    dir: &std::path::Path,
    port: u16,
    id: u64,
    peers: &str,
    bootstrap: bool,
) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_rekha"));
    command
        .arg("serve")
        .arg("--data-dir")
        .arg(dir)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--node-id")
        .arg(id.to_string())
        .arg("--peers")
        .arg(peers)
        .env("REKHA_API_KEY", KEY)
        .env("REKHA_CLUSTER_SECRET", SECRET)
        .env("REKHA_KEY_PEPPER", PEPPER)
        // Slower than the defaults: these clusters share a small CI runner with
        // other test binaries, and 100 ms heartbeats thrash into repeated
        // elections there. The assertions use deadlines, not these values.
        .env("REKHA_HEARTBEAT_MS", "250")
        .env("REKHA_ELECTION_MIN_MS", "1000")
        .env("REKHA_ELECTION_MAX_MS", "2000")
        .env("RUST_LOG", "warn");
    if bootstrap {
        command.arg("--bootstrap");
    }
    command
}

/// The command that starts a node which is joining an existing cluster.
///
/// `member` is any running member; the joining node asks it for the cluster's
/// placement. No `--peers`: a joining node that invented a voter set from a flag
/// would compute a placement none of the existing members share.
pub fn join_command(dir: &std::path::Path, port: u16, id: u64, member: &str) -> Command {
    let mut command = serve_command(dir, port, id, "", false);
    command.arg("--join").arg(member);
    command
}

/// Start a joining node and wait for the process to exist.
pub fn spawn_joining_node(dir: &std::path::Path, port: u16, id: u64, member: &str) -> Node {
    let child = join_command(dir, port, id, member)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn rekha");
    Node {
        id,
        port,
        data_dir: dir.to_path_buf(),
        child,
    }
}

/// Start one node and wait for the process to exist, not for it to be ready.
pub fn spawn_node(dir: &std::path::Path, port: u16, id: u64, peers: &str, bootstrap: bool) -> Node {
    let child = serve_command(dir, port, id, peers, bootstrap)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn rekha");
    Node {
        id,
        port,
        data_dir: dir.to_path_buf(),
        child,
    }
}

pub fn spawn_cluster(dirs: &[PathBuf], ports: &[u16]) -> Vec<Node> {
    let peers = peers_spec(ports);
    (0..dirs.len())
        // Exactly one node initializes the metadata group.
        .map(|i| spawn_node(&dirs[i], ports[i], (i + 1) as u64, &peers, i == 0))
        .collect()
}

pub fn client() -> reqwest::Client {
    // One client for the whole test binary: building one per request meant a
    // connection pool per call, and concurrent construction deadlocks in
    // macOS proxy detection. `no_proxy()` also keeps a developer's `HTTP_PROXY`
    // from redirecting requests meant for a node on localhost, which is a
    // genuinely confusing way for a test to fail.
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .no_proxy()
                .build()
                .expect("client")
        })
        .clone()
}

pub async fn wait_ready(nodes: &[Node]) {
    wait_ready_timeout(nodes, Duration::from_secs(120)).await;
}

/// [`wait_ready`] with a caller-chosen deadline.
pub async fn wait_ready_timeout(nodes: &[Node], timeout: Duration) {
    let deadline = Instant::now() + timeout;
    let http = client();
    while Instant::now() < deadline {
        let mut all = true;
        for node in nodes {
            let ok = match http
                .get(node.url("/ready"))
                .header("x-chroma-token", KEY)
                .send()
                .await
            {
                Ok(resp) => resp
                    .text()
                    .await
                    .unwrap_or_default()
                    .contains("\"ready\":true"),
                Err(_) => false,
            };
            all &= ok;
        }
        if all {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("cluster never became ready within {timeout:?}");
}

pub async fn post_json(node: &Node, path: &str, body: serde_json::Value) -> (u16, String) {
    let resp = client()
        .post(node.url(path))
        .header("x-chroma-token", KEY)
        .json(&body)
        .send()
        .await
        .expect("request");
    let status = resp.status().as_u16();
    (status, resp.text().await.unwrap_or_default())
}

/// `POST`, retrying the one status that means "not yet".
///
/// A shard group elects its own leader, so between a collection's creation and
/// the first write there is a window in which the group answers `503` with
/// `Retry-After`. That is the contract — a client retries — and a test that did
/// not retry would be asserting that elections complete faster than they do.
pub async fn post_json_retrying(node: &Node, path: &str, body: serde_json::Value) -> (u16, String) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let (status, body) = post_json(node, path, body.clone()).await;
        if status != 503 || Instant::now() >= deadline {
            return (status, body);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

pub fn placement_of(nodes: &[Node]) -> Placement {
    let bytes = std::fs::read(nodes[0].data_dir.join(PlacementConfig::FILE))
        .expect("the first node must have persisted its placement");
    let config: PlacementConfig =
        serde_json::from_slice(&bytes).expect("placement file must parse");
    // The voters come from the file, not from `nodes`: re-deriving them here
    // would let this helper agree with itself while the cluster disagreed.
    Placement::new(config.shard_count, config.replication_factor, config.voters).expect("placement")
}

/// Ask the cluster which shard a collection was placed on.
///
/// Deliberately not recomputed: the shard is a decision the metadata leader
/// makes for load, so a test that derived it from the id would be comparing its
/// own hash against itself rather than checking where the cluster put the data.
///
/// Retried, because creating a collection returns after the entry **commits**
/// and a node records the row when it **applies** it. Under a loaded CI run the
/// two are milliseconds apart but not the same moment, and a single pass turned
/// that into "the cluster reports no shard".
pub async fn shard_of(nodes: &[Node], collection: uuid::Uuid) -> Option<rekha_cluster::ShardId> {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        for node in nodes {
            let Ok(response) = client()
                .get(node.url(&format!("/api/v2/collections/{collection}")))
                .header("x-chroma-token", KEY)
                .send()
                .await
            else {
                continue;
            };
            let Ok(body) = response.json::<serde_json::Value>().await else {
                continue;
            };
            if let Some(shard) = body["shard"].as_u64() {
                return Some(rekha_cluster::ShardId(shard as u32));
            }
        }
        if Instant::now() >= deadline {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

pub fn has_data(node: &Node, collection: uuid::Uuid) -> bool {
    let wal = node.data_dir.join("wal").join(format!("{collection}.wal"));
    if wal.exists() {
        return true;
    }
    if let Ok(entries) = std::fs::read_dir(node.data_dir.join("wal")) {
        for e in entries.flatten() {
            let name = e.file_name();
            let s = name.to_string_lossy();
            if s.starts_with(&format!("{collection}")) && s.ends_with(".wal") {
                return true;
            }
        }
    }
    false
}

/// Wait until `condition` holds, so the assertion is about placement rather
/// than about how quickly a follower applied a committed entry.
pub async fn eventually(label: &str, mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        if condition() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("{label} did not happen within 20s");
}
