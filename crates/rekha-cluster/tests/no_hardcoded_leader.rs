//! Regression guard: no code outside tests may name a node id as the leader.
//!
//! The defect this locks down was not subtle — `peer_map.get(&1)` chose the
//! replication source, `node_id != 1` decided who replicated at all, and a
//! missing peer fell back to `127.0.0.1:8000`. The cluster therefore only
//! worked when every client happened to write to node 1, and any other node
//! acknowledged writes that never left it.
//!
//! A comment saying "don't hardcode the leader" would not have prevented that.
//! This test does, by failing the build when the pattern reappears.
//!
//! Scope and honesty: this scans for the specific shapes the bug took. It is
//! not a type system and it cannot prove the absence of a hardcoded leader — a
//! determined author can dodge it. It is a tripwire for the regression that
//! actually happened, and it is cheap enough to run on every commit.

use std::path::{Path, PathBuf};

/// Files that participate in the data path: identity, leadership, routing,
/// replication, and the server wiring.
const DATA_PATH_FILES: &[&str] = &[
    "crates/rekha-cli/src/main.rs",
    "crates/rekha-api/src/lib.rs",
    "crates/rekha-api/src/chroma.rs",
    "crates/rekha-api/src/leadership.rs",
    "crates/rekha-cluster/src/identity.rs",
    "crates/rekha-cluster/src/leader.rs",
    "crates/rekha-cluster/src/state_machine.rs",
];

/// `(needle, why it is forbidden)`.
const FORBIDDEN: &[(&str, &str)] = &[
    (
        "peer_map.get(&1",
        "the replication source must come from the elected leader, never a fixed id",
    ),
    (
        "node_id != 1",
        "whether a node replicates must not depend on its id",
    ),
    (
        "node_id == 1",
        "whether a node writes must not depend on its id",
    ),
    // Replaced the `owner_nodes: vec![1]` tripwire when ownership stopped being
    // an owner list: the equivalent hazard is a shard written as a literal,
    // which would bypass the leader's load-aware decision.
    (
        "shard: Some(ShardId(",
        "a collection's shard is a leader decision, never a literal",
    ),
    (
        "127.0.0.1:8000",
        "no fallback leader address: with no leader the answer is 'no leader'",
    ),
    (
        "is_leader: bool",
        "leadership is not application state; read it from the consensus view",
    ),
    (
        "is_leader = ",
        "a replicated is_leader flag is a second source of truth",
    ),
];

fn repo_root() -> PathBuf {
    // crates/rekha-cluster -> repo root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("repo root")
        .to_path_buf()
}

/// Drop the test modules at the bottom of a file.
///
/// Convention in this repo is one `#[cfg(test)]` block at the end of each file,
/// so truncating at the first marker excludes test fixtures without needing a
/// parser. A test module in the middle of a file would be scanned, which is the
/// safe direction to be wrong in.
fn strip_test_modules(source: &str) -> &str {
    match source.find("#[cfg(test)]") {
        Some(index) => &source[..index],
        None => source,
    }
}

#[test]
fn the_data_path_names_no_node_as_leader() {
    let root = repo_root();
    let mut violations = Vec::new();

    for relative in DATA_PATH_FILES {
        let path = root.join(relative);
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let production = strip_test_modules(&source);

        for (needle, why) in FORBIDDEN {
            for (line_no, line) in production.lines().enumerate() {
                if line.contains(needle) {
                    violations.push(format!(
                        "{relative}:{}: `{needle}` — {why}\n    {}",
                        line_no + 1,
                        line.trim()
                    ));
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "hardcoded leadership reintroduced:\n{}",
        violations.join("\n")
    );
}

/// There is exactly one replication mechanism: the Raft log.
///
/// A second path — followers polling a leader for WAL deltas — existed and was
/// removed, because per-node WAL sequence numbers made a follower's cursor
/// meaningless across an election. Putting it back would reintroduce that bug,
/// so its absence is asserted rather than assumed.
#[test]
fn only_the_raft_log_replicates_data() {
    let root = repo_root();

    assert!(
        !root
            .join("crates/rekha-cluster/src/replication.rs")
            .exists(),
        "a WAL-shipping replication module reappeared; data must reach followers \
         only as committed log entries"
    );

    for relative in DATA_PATH_FILES {
        let source = std::fs::read_to_string(root.join(relative))
            .unwrap_or_else(|e| panic!("cannot read {relative}: {e}"));
        let production = strip_test_modules(&source);
        for needle in ["WalReplication", "wal_delta", "apply_remote_ops"] {
            assert!(
                !production.contains(needle),
                "{relative} mentions `{needle}`: the WAL-shipping path must not come back"
            );
        }
    }
}

/// The server must not start a cluster without the topology it needs.
#[test]
fn the_cli_takes_identity_and_topology_from_configuration() {
    let source = std::fs::read_to_string(repo_root().join("crates/rekha-cli/src/main.rs"))
        .expect("read main.rs");
    let production = strip_test_modules(&source);

    assert!(
        production.contains("NodeIdentity::open_or_create"),
        "node identity must be read from the data directory"
    );
    assert!(
        production.contains("parse_peers"),
        "peer ids must come from `id@addr`, never list position"
    );
    assert!(
        production.contains("LeaderTracker::spawn"),
        "leadership must be observed from consensus"
    );
}
