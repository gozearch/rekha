//! End-to-end smoke tests that spawn the real binary.
//!
//! These exist because of a bug that no unit test could see. The write path was
//! chosen with `raft.is_some()`, and a single-node deployment holds a Raft group
//! for metadata even though its data path is the local WAL. So `POST /add`
//! committed a log entry, the engine refused to apply it, `apply` returned `Ok`
//! anyway (openraft requires that), and the client got **201 for a record that
//! was never stored**.
//!
//! Every in-process test missed it: `api_tests` construct an `AppState` with no
//! Raft at all, and the cluster tests set `ApplyMode::Committed`. Only the real
//! wiring in `main()` had the mismatch, so these tests spawn it.

use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::json;

const KEY: &str = "smoke-key-0123456789abcdef0123456789";
const COLLECTIONS: &str = "/api/v2/tenants/default_tenant/databases/default_database/collections";

struct Server {
    child: Child,
    port: u16,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn free_port() -> u16 {
    // Bind and release: the CLI takes a port and cannot report one it chose.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    listener.local_addr().expect("addr").port()
}

fn spawn(data_dir: &std::path::Path, port: u16) -> Server {
    let child = Command::new(env!("CARGO_BIN_EXE_rekha"))
        .arg("serve")
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .env("REKHA_API_KEY", KEY)
        .env("RUST_LOG", "warn")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn rekha");
    Server { child, port }
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .expect("client")
}

async fn wait_ready(server: &Server) {
    let deadline = Instant::now() + Duration::from_secs(20);
    let http = client();
    while Instant::now() < deadline {
        if let Ok(resp) = http
            .get(format!("http://127.0.0.1:{}/ready", server.port))
            .header("x-chroma-token", KEY)
            .send()
            .await
            && let Ok(body) = resp.text().await
            && body.contains("\"ready\":true")
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("server on port {} never became ready", server.port);
}

fn url(server: &Server, path: &str) -> String {
    format!("http://127.0.0.1:{}{path}", server.port)
}

async fn post(server: &Server, path: &str, body: serde_json::Value) -> (u16, String) {
    let resp = client()
        .post(url(server, path))
        .header("x-chroma-token", KEY)
        .json(&body)
        .send()
        .await
        .expect("request");
    let status = resp.status().as_u16();
    (status, resp.text().await.unwrap_or_default())
}

async fn get(server: &Server, path: &str) -> (u16, String) {
    let resp = client()
        .get(url(server, path))
        .header("x-chroma-token", KEY)
        .send()
        .await
        .expect("request");
    let status = resp.status().as_u16();
    (status, resp.text().await.unwrap_or_default())
}

fn create_body(name: &str, dim: usize) -> serde_json::Value {
    json!({
        "name": name,
        "configuration": {"hnsw": {"space": "l2", "dimension": dim}},
    })
}

/// The regression. A single-node server must store what it acknowledges.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn single_node_stores_the_writes_it_acknowledges() {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = spawn(dir.path(), free_port());
    wait_ready(&server).await;

    let (code, body) = post(&server, COLLECTIONS, create_body("smoke", 4)).await;
    assert!(code == 200 || code == 201, "create failed: {code} {body}");

    let (code, body) = post(
        &server,
        &format!("{COLLECTIONS}/smoke/add"),
        json!({"ids": ["a", "b"], "embeddings": [[1,0,0,0],[0,1,0,0]]}),
    )
    .await;
    assert_eq!(code, 201, "add should be accepted, got {code}: {body}");

    // The decisive assertion: acknowledged means stored.
    let (code, body) = get(&server, &format!("{COLLECTIONS}/smoke/count")).await;
    assert_eq!(code, 200, "count failed: {body}");
    assert_eq!(
        body.trim(),
        "2",
        "the server acknowledged the write but does not hold it"
    );

    let (_, ready) = get(&server, "/ready").await;
    assert!(
        ready.contains("\"ready\":true"),
        "a node that applied its own writes must be ready: {ready}"
    );
}

/// A write must survive a restart, through whatever path the deployment uses.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn single_node_write_survives_a_restart() {
    let dir = tempfile::tempdir().expect("tempdir");
    let port = free_port();

    {
        let server = spawn(dir.path(), port);
        wait_ready(&server).await;

        let (code, body) = post(&server, COLLECTIONS, create_body("durable", 3)).await;
        assert!(code == 200 || code == 201, "create failed: {body}");

        let (code, body) = post(
            &server,
            &format!("{COLLECTIONS}/durable/add"),
            json!({"ids": ["x"], "embeddings": [[1,2,3]]}),
        )
        .await;
        assert_eq!(code, 201, "add failed: {code} {body}");
        assert_eq!(
            get(&server, &format!("{COLLECTIONS}/durable/count"))
                .await
                .1
                .trim(),
            "1"
        );
    }

    // The first Server was dropped, which kills the process.
    tokio::time::sleep(Duration::from_millis(300)).await;

    let restarted = spawn(dir.path(), port);
    wait_ready(&restarted).await;

    let (_, count) = get(&restarted, &format!("{COLLECTIONS}/durable/count")).await;
    assert_eq!(count.trim(), "1", "the write did not survive a restart");
}

/// A `SIGKILL` must not lose a write that was acknowledged.
///
/// This is the promise the durability design exists for, and nothing tested it
/// end to end: the restart modes in CI stop the process cleanly, and the fault
/// seam injects failures into checkpoint phases rather than into the process.
/// A kill is the case where there is no chance to flush anything.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_acknowledged_write_survives_sigkill() {
    let dir = tempfile::tempdir().expect("tempdir");
    let port = free_port();

    let mut acknowledged = 0usize;
    {
        let mut server = spawn(dir.path(), port);
        wait_ready(&server).await;

        let (code, body) = post(&server, COLLECTIONS, create_body("crash", 4)).await;
        assert!(code == 200 || code == 201, "create failed: {body}");

        // Each batch is one request, and a 201 means the WAL batch was fsynced
        // before the response was written. Count what was acked, not what was
        // attempted.
        for batch in 0..20 {
            let ids: Vec<String> = (0..5).map(|i| format!("b{batch}-{i}")).collect();
            let embeddings: Vec<[f32; 4]> =
                (0..5).map(|i| [batch as f32, i as f32, 1.0, 0.0]).collect();
            let (code, body) = post(
                &server,
                &format!("{COLLECTIONS}/crash/add"),
                json!({ "ids": ids, "embeddings": embeddings }),
            )
            .await;
            assert_eq!(code, 201, "batch {batch} was not accepted: {body}");
            acknowledged += 5;
        }

        // Deliberately no graceful shutdown: SIGKILL, as the OOM killer or a
        // hard node failure would.
        let _ = server.child.kill();
        let _ = server.child.wait();
    }

    let restarted = spawn(dir.path(), port);
    wait_ready(&restarted).await;

    let (_, count) = get(&restarted, &format!("{COLLECTIONS}/crash/count")).await;
    assert_eq!(
        count.trim().parse::<usize>().unwrap_or(0),
        acknowledged,
        "acked {acknowledged} records but only {} survived SIGKILL",
        count.trim()
    );
}

/// The metrics scrape endpoint must not be exposed unauthenticated.
///
/// It reports cluster topology (term, applied index, quorum age) and the
/// exporter's listener has no way to check a credential, so binding it to a
/// non-loopback address is a decision the process refuses to make for you.
#[test]
fn a_public_metrics_listener_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let port = free_port();

    let output = Command::new(env!("CARGO_BIN_EXE_rekha"))
        .arg("serve")
        .arg("--data-dir")
        .arg(dir.path())
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .env("REKHA_API_KEY", KEY)
        .env("REKHA_METRICS_PORT", "9999")
        .env("REKHA_METRICS_HOST", "0.0.0.0")
        .env("RUST_LOG", "error")
        .output()
        .expect("run rekha");

    assert!(
        !output.status.success(),
        "a node started with the metrics listener on 0.0.0.0 and no authentication"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("REKHA_METRICS_HOST") && stderr.contains("loopback"),
        "the refusal must name the variable and the reason: {stderr}"
    );
}
