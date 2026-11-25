//! The interactive REPL, driven over a pipe.
//!
//! `run_repl` reads stdin and writes stdout directly (it is a terminal front
//! end), so the honest test is the binary itself: start a node, pipe a scripted
//! session into `rekha repl`, and assert on what a user would see. That covers
//! every command handler, the formatters, and the usage/error paths in one run,
//! which a helper-level test of the tokenizer cannot.

mod common;

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

use common::{KEY, SECRET, SERIAL, free_port, spawn_cluster, temp_dir, wait_ready};

struct Session {
    stdout: String,
    stderr: String,
    success: bool,
}

async fn start_node() -> (common::Node, u16) {
    let dirs = [temp_dir()];
    let ports = [free_port()];
    let mut nodes = spawn_cluster(&dirs, &ports);
    wait_ready(&nodes).await;
    let node = nodes.remove(0);
    (node, ports[0])
}

fn run_script(port: u16, script: &str) -> Session {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rekha"))
        .arg("--url")
        .arg(format!("http://127.0.0.1:{port}"))
        .arg("repl")
        .env("REKHA_API_KEY", KEY)
        .env("REKHA_CLUSTER_SECRET", SECRET)
        .env("RUST_LOG", "error")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn repl");

    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(script.as_bytes())
        .expect("write script");

    let output = child.wait_with_output().expect("repl output");
    Session {
        success: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_scripted_session_exercises_every_command() {
    let _guard = SERIAL.lock().await;
    let (_node, port) = start_node().await;

    let script = r#"
help
collections
create demo 4 l2
use demo
describe
add v1 [1,0,0,0] "doc one" {"tag":"a"}
add v2 [0,1,0,0]
add v3 [0,0,1,0] "third"
count
get v1 v2
get
query [1,0,0,0] 2
upsert v1 [1,1,0,0] "doc updated"
delete v2
drop v3
status
cluster
cluster membership
clear
bogus
use missing
describe missing
drop missing
count missing
cluster add 2 127.0.0.1:9
cluster remove 2
cluster weird
exit
"#;
    let session = run_script(port, script);
    let out = &session.stdout;
    let err = &session.stderr;

    assert!(session.success, "repl exited non-zero: {err}");
    assert!(out.contains("RekhaDB Interactive Vector REPL"), "{out}");
    assert!(
        out.contains("Collection 'demo' created"),
        "stdout:\n{out}\nstderr:\n{err}"
    );
    assert!(out.contains("Using collection 'demo'"), "{out}");
    assert!(out.contains("Added record 'v1' into 'demo'"), "{out}");
    assert!(
        out.contains("{\"count\":3} records in collection 'demo'"),
        "{out}"
    );
    assert!(
        out.contains("doc one"),
        "get must format the document: {out}"
    );
    assert!(
        out.contains("(2 records)"),
        "get prints the row count: {out}"
    );
    assert!(
        out.contains("Empty set."),
        "an empty get is explicit: {out}"
    );
    assert!(out.contains("DISTANCE"), "query prints a table: {out}");
    assert!(out.contains("Upserted record 'v1'"), "{out}");
    assert!(
        out.contains("Collection: demo"),
        "describe prints the collection: {out}"
    );
    assert!(
        out.contains("/ready -> HTTP"),
        "status prints readiness: {out}"
    );
    assert!(
        out.contains("Node Status"),
        "cluster prints consensus state: {out}"
    );
    assert!(out.contains("voters"), "membership prints the group: {out}");
    assert!(out.contains("Bye!"), "{out}");

    assert!(
        err.contains("Unknown command: 'bogus'"),
        "an unknown command is reported: {err}"
    );
    assert!(
        err.contains("No active collection") || err.contains("Error"),
        "use of a missing collection is reported: {err}"
    );
    assert!(
        err.contains("network") || err.contains("Error") || err.contains("Usage"),
        "{err}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn usage_errors_are_reported_before_any_request() {
    let _guard = SERIAL.lock().await;
    let (_node, port) = start_node().await;

    // No `use`, so the record commands must refuse locally rather than guess a
    // collection. Each usage line is a different guard.
    let script = "\
add x [1,0,0,0]\n\
upsert x [1,0,0,0]\n\
get x\n\
delete x\n\
query [1,0,0,0]\n\
create\n\
create bad notanumber\n\
use\n\
describe\n\
drop\n\
count\n\
use does-not-exist\n\
exit\n";
    let session = run_script(port, script);
    let err = &session.stderr;
    assert!(session.success, "{err}");
    assert!(err.contains("No active collection"), "{err}");
    assert!(err.contains("Usage: create <name> <dimension>"), "{err}");
    assert!(
        err.contains("dimension must be a positive integer"),
        "{err}"
    );
    assert!(err.contains("Usage: use <collection_name_or_id>"), "{err}");
    assert!(err.contains("Usage: describe"), "{err}");
    assert!(err.contains("Usage: drop"), "{err}");
    assert!(err.contains("Usage: count"), "{err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_bad_initial_collection_is_reported_but_the_repl_starts() {
    let _guard = SERIAL.lock().await;
    let (_node, port) = start_node().await;

    let mut child = Command::new(env!("CARGO_BIN_EXE_rekha"))
        .arg("--url")
        .arg(format!("http://127.0.0.1:{port}"))
        .arg("repl")
        .arg("--collection")
        .arg("absent")
        .env("REKHA_API_KEY", KEY)
        .env("REKHA_CLUSTER_SECRET", SECRET)
        .env("RUST_LOG", "error")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn repl");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(b"exit\n")
        .expect("write");

    let output = child.wait_with_output().expect("output");
    let out = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());
    assert!(
        out.contains("Initial collection 'absent' does not exist yet."),
        "{out}"
    );
    tokio::time::sleep(Duration::from_millis(10)).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn malformed_arguments_are_reported_by_the_command_that_owns_them() {
    let _guard = SERIAL.lock().await;
    let (_node, port) = start_node().await;

    let script = r#"
create demo 4
use demo
add bad [x]
add bad [1,0,0,0] "doc" {not-json}
add ok [1,0,0,0] "doc"
query [1,0,0,0] 2 {not-json}
query [x]
count missing
describe missing
get missing
delete missing
cluster add nope 127.0.0.1:1
cluster remove nope
use demo
exit
"#;
    let session = run_script(port, script);
    let err = &session.stderr;
    assert!(session.success, "{err}");
    assert!(err.contains("Invalid vector number"), "{err}");
    assert!(err.contains("Error parsing metadata JSON"), "{err}");
    assert!(err.contains("Error parsing where JSON filter"), "{err}");
    assert!(err.contains("Error parsing query embedding"), "{err}");
    assert!(err.contains("Error:"), "{err}");
    assert!(err.contains("node_id must be a positive integer"), "{err}");
    assert!(
        session.stdout.contains("Added record 'ok'"),
        "{}",
        session.stdout
    );
}
