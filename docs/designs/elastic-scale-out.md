# Elastic Scale-Out for RekhaDB

*   **Author:** TBD
*   **Status:** Draft
*   **Date:** 2026-11-20

---

## Overview

RekhaDB runs one Raft group per shard plus a metadata group. Operators can add a node as a learner with `--join` and `POST /internal/cluster/add_learner`. The learner copies log entries and snapshots until it holds the same data as voters.

This design adds automatic learner-to-voter promotion as the first elastic step, with no change to shard count. Promotion starts on its own once the learner has caught up; the operator does not run a promote call.

## The Problem

Operators must choose cluster size before first start. When traffic or data grows beyond that choice, there is no supported way to add capacity:

* A new node joins as a learner and stays a learner. It serves no shard traffic and counts toward no quorum.
* Promotion has no API. The shard groups have no membership-change path.
* Shard ownership is recorded in `placement.json` and validated at startup. Changing it by hand moves the record without moving the data.
* The only recovery today is to rebuild the cluster or restart every node with new flags.

The result: sizing is guesswork, failed guesses require rebuilds, and dead replicas are not replaced by new hardware.

The proposed change: once a learner has caught up, the cluster promotes it to voter through joint consensus in each shard group it replicates, and records the new voter set in `placement.json`. Promotion is automatic and runs one shard group at a time, with reads and writes continuing during the change.

## Goals

*   Operator adds a node with `--join` and one admit call, with no restart of existing nodes.
*   A caught-up learner is promoted to voter automatically with no restart and no manual promote call.
*   Promotion keeps write availability: the shard group keeps a quorum through joint consensus, and the metadata group stays available.
*   Placement record matches Raft membership after each promotion. A restart after promotion loads the same voter set.
*   A failed promotion leaves the learner as a learner. No partial voter state and no data loss.
*   Learner lag and promotion state are visible through `/internal/cluster/membership` and metrics. Promotion waits until lag is below the defined threshold.
*   Fresh clusters only. `StoredMembership` and `placement.json` use the new format from the start; no migration from an older format is required.

## Non-Goals

*   Shard movement for load balancing (remove a voter from node A and add one on node B to spread data). Deferred to v2. Automatic promotion of a learner to voter on its existing shards is in scope.
*   Autoscaling from Kubernetes or Prometheus signals. The operator admits each node; promotion after catch-up is automatic.
*   Cross-region or multi-datacenter replication.
*   Heterogeneous node weights (nodes with different disk or CPU counts).
*   Shard split, shard merge, or change of shard count.
*   Change of replication factor as a separate operation.
*   Streaming snapshot transfer. This design reuses the current chunked snapshot path unchanged.
*   Central placement scheduler. The metadata leader remains the single writer.

## Open Questions

*   **Promotion scope**: Does one automatic promotion cover all shard groups the learner replicates, or one group at a time in a fixed order?
*   **Readiness threshold**: What lag number allows promotion (matched index distance, apply lag, or both)?
*   **Replication factor rule**: Is a minimum of 2 voters per shard required before any removal is allowed in v2?
*   **Concurrent promotions**: Is one shard group at a time a hard limit for v1, or can independent groups promote in parallel?

## Critical User Journeys (CUJs)

### CUJ 1: Add a node and reach voter state

*   **Given**: A running cluster with collections and writes.
*   **When**: The operator starts a new node with `--join` and admits it with `add_learner`.
*   **Then**: The node replicates and installs snapshots until lag is below the threshold, the reconciler promotes it per group, `membership` lists it as a voter, and it serves reads and writes.

### CUJ 2: Write during promotion

*   **Given**: A learner within the promotion threshold in at least one shard group.
*   **When**: Clients write and run linearizable reads while joint consensus is in progress.
*   **Then**: Writes commit through quorum, reads wait for the read index on the leader, and no client sees an error caused by the membership change itself.

## Technical Plan (Architecture)

The cluster already runs a reconciler. This design keeps that structure and hardens it for automatic promotion. No second consensus system is added. The metadata Raft group remains the single writer for placement.

### Where promotion happens today

`crates/rekha-cluster/src/rebalancer.rs` runs on every node every 1500ms (`spawn_auto_rebalancer`, started in `crates/rekha-cli/src/main.rs`):

* Each shard leader compares current voters with target owners from `Placement::new(shard_count, replication_factor, voters)`. Missing owners are added as learners with `add_learner`. When every target owner is within 50 log entries of the leader, the shard leader calls `change_membership(target_set)`.
* The metadata leader does the same for the metadata group: learners within 50 entries are promoted with `change_membership(new_voters)` followed by `ClusterOperation::SetVoters` so `placement.json` records the new voter set.
* For shard moves, the metadata leader asks new owners to `ensure_started`, polls `membership`, then writes `ClusterOperation::SetShardOwners` and asks dropped owners to `drop`.

What v1 changes: the threshold, ordering, and visibility around this loop are defined and tested. No new Raft protocol is introduced. openraft 0.9 joint consensus is reused as-is.

```text
new node                 shard leader / metadata leader          placement record
   |                                   |                                   |
   |--join, fetch placement----------->|                                   |
   |<--hosts nothing, replicates log---|                                   |
   |---snapshot chunks if behind------>|                                   |
   |                                   |--add_learner (already a learner)-->|
   |---replicate, matched_index grows->|                                   |
   |                                   |--tick: lag <= 50?----------------->|
   |                                   |--change_membership (joint)------->|
   |                                   |--SetVoters / SetShardOwners------>|--placement.json updated
   |<--voter, serves reads/writes------|                                   |
```

### How the pieces fit

* **Catch-up path (unchanged):** Learner replicates from the leader. If it fell behind the purge point, it installs a chunked snapshot (group identity checked) and replays from the boundary. Promotion never runs before this completes because the lag check reads `matched_index` from `raft.metrics()`.
* **Promotion trigger (defined):** One tick handles one learner per group. Promotion requires `matched >= last_log - 50` on the leader that owns the tick. Shard groups are decided by their own leader; the metadata group is decided by the metadata leader. No cross-group atomicity is attempted. A partial result (some groups promoted, others not) is valid and converges on later ticks.
* **Placement write (unchanged mechanism, stricter order):** `SetVoters` and `SetShardOwners` are committed log entries, applied by `ClusterStateMachine`. `placement.json` is rewritten from the committed entry, not from the membership call. A restart loads voters from `placement.json` because fresh clusters write the new format from the start.
* **New voter startup:** A promoted node already runs its shard groups as a learner, so no new initialization path is needed. If a target owner has no group running, the metadata leader calls `ensure_started` before promotion is attempted.
* **Read path during promotion:** Linearizable reads go to the shard leader and wait for the read index. Joint consensus keeps quorum across old and new configs, so reads and writes continue. `?consistency=local` may answer from a learner that has not yet applied the latest entry; the `x-rekha-served-by` header names the node that answered.

### What changes in code

* `rebalancer.rs`: name the 50-entry threshold, order shard groups in a fixed order, skip groups without a local leader, record per-group promotion attempts in logs and metrics. No new background task.
* `placement.rs` / `state_machine.rs`: keep `SetVoters` and `SetShardOwners` as the only writers of the voter set and owner table. No direct file edits from the reconciler.
* `rekha-api`: `GET /internal/cluster/membership` reports learners with `matched_index` and last promotion result. No new promote endpoint for v1 because promotion is automatic.
* Metrics: learner lag per group, promotion attempts and outcomes, time from admit to voter. These reuse the existing Prometheus exporter.

### Failure handling

* Promotion call fails or times out: learner stays a learner. Next tick retries. No rollback beyond leaving membership unchanged.
* Leader changes mid-promotion: new leader reads current voters and learners from its own metrics and resumes. Joint consensus state comes from openraft, not from the reconciler.
* New node crashes mid-catch-up: learner entry remains. Operator restarts the node with the same `--node-id`; it resumes replication. A learner that never catches up is never promoted and never counts toward quorum.
* Metadata leader fails mid-`SetVoters`: the membership change may commit without the placement write, or neither commits. The next metadata leader re-reads voters from Raft metrics and proposes the missing `SetVoters` entry. Idempotent because the entry carries the full voter set.

## Alternatives Considered

### Alternative A: Manual promote call per group

*   **Pros**: Operator controls exact timing. No background decision loop. Easy to test: one request, one membership change.
*   **Cons**: Operator must poll lag and time the call. A mistimed call fails and must be retried. More API surface to maintain.
*   **Reason for Rejection**: Owner chose automatic promotion. The reconciler already polls lag every 1500ms, so a manual call adds work without adding safety.

### Alternative B: Central placement scheduler

*   **Pros**: One place decides all moves, like TiKV PD. Can balance load across shards in one plan.
*   **Cons**: A second leader election and second source of truth beside the metadata Raft group. Must handle scheduler failure, split-brain with Raft, and version skew. Large new component for v1.
*   **Reason for Rejection**: The metadata leader already orders placement writes. A separate scheduler fights with Raft membership changes and is not needed to promote learners.

### Alternative C: Move shards to balance load in v1

*   **Pros**: Adding a node would spread data at once instead of only adding a replica.
*   **Cons**: Each move is two membership changes (add on target, remove from source) plus a data copy. Partial failure leaves some groups moved and others not, with no atomic multi-group commit. Large shards need disk and network headroom during copy.
*   **Reason for Rejection**: Deferred to v2. Promotion alone adds capacity and quorum without moving data. Load balancing needs the copy path and drop path proven first.

### Alternative D: Change shard count to add capacity

*   **Pros**: More shards means smaller groups and finer placement.
*   **Cons**: Shard count is stored in `placement.json` and validated at startup. Changing it rehashes every collection onto different nodes. That is a full reshard, not an incremental add.
*   **Reason for Rejection**: Out of scope. Shard count stays fixed for the life of the data directory.

### Alternative E: Manual membership only, no reconciler

*   **Pros**: No background loop. Every change is an explicit operator action, like etcd.
*   **Cons**: Learner that catches up stays a learner until someone acts. Shard groups drift from target placement with no correction. Operator must watch every group.
*   **Reason for Rejection**: The reconciler already exists and runs. Removing it returns to manual tracking with no gain.

---

## Non-Functional Requirements

*   **Rollout**: Fresh clusters only. No migration path for older placement formats. The reconciler change ships with the binary; no feature flag is defined in this draft.
*   **Rollback**: A failed promotion leaves membership unchanged. The learner stays a learner and retries on later ticks. No data movement means no cleanup path is needed for v1.
*   **Observability**: Promotion attempts, lag at decision, and voter state must be visible per group through `membership` and metrics so a stalled promotion can be told apart from slow catch-up.

---

### Additional Resources

*   Predecessor brief: `./concepts_brief.md`
*   Reconciler: `crates/rekha-cluster/src/rebalancer.rs`
*   Placement and membership: `crates/rekha-cluster/src/placement.rs`, `crates/rekha-cluster/src/state_machine.rs`
*   Membership routes: `crates/rekha-api/src/lib.rs` (`raft_membership`, `shard_membership_handler`, `shard_ensure_started`, `shard_drop`)
*   Operator docs: `docs/operations.md`

### Future Improvements

*   **v2 shard moves for load balancing**: add on target, verify catch-up, remove from source, with throttle and progress reporting.
*   **Threshold and throttle tuning**: make the lag threshold and per-tick group limit explicit settings once measured on large shards.
*   **Richer promotion status**: per-group last attempt, failure cause, and time from admit to voter in `membership` output.

---

## Detailed Implementation Plan

All paths are relative to the workspace root. No production code is included here; names below are API shapes and file scopes.

### Target Files to Create/Modify

#### 1. `crates/rekha-cluster/src/rebalancer.rs` (Modify)

*   **Type / Scope**: Rust library, background reconciler loop. A shard group is one Raft group; the metadata group is group 0.
*   **Proposed Changes**: Add `const PROMOTION_LAG_THRESHOLD: u64 = 50` in `rebalancer.rs` (entries behind `last_log_index`) and use it in both the shard-leader path and the metadata-leader path. Sort shard groups by `ShardId.0` ascending before visiting. At most one promotion attempt per group per tick; groups proceed independently, so partial promotion across groups is normal. Skip groups where this node is not the leader. Keep `retain=false` on the shard path and `retain=true` on the metadata path. Treat `matched == None` as infinite lag (skip promotion). Keep the 1500ms tick and the existing cross-node `ensure_started` / `shard membership` / `drop` HTTP calls and their routes unchanged. Emit metric `rekha_promotion_attempts_total{group,node,outcome}` with outcome `promoted | skipped_lag | error`, gauge `rekha_learner_lag_entries{group,node}`, and a log line with group, node, lag, and outcome per attempt.

#### 2. `crates/rekha-api/src/lib.rs` (Modify)

*   **Type / Scope**: Rust service, internal cluster routes.
*   **Proposed Changes**: `GET /internal/cluster/membership` returns `{"voters":[],"learners":[{"node":4,"matched_index":n,"last_log_index":m,"lag":k}],"last_promotion":{"group":1,"node":4,"at":"...","outcome":"promoted | skipped_lag | error"}}` and keeps `members` as a deprecated alias (voters followed by learners) until the CLI migrates. `GET /internal/cluster/shard/{shard}/membership` returns the same shape for one group. No new promote route for v1; `POST /internal/cluster/add_learner` stays the admit path. Leave `refuse_membership_change` on the HTTP routes (`remove_member` keeps 409 while shards are placed); the reconciler calls `Raft::change_membership` directly and is the sole bypass. Last promotion result is an in-memory `Mutex<HashMap<(group,node), LastAttempt>>` on `AppState`, populated by the leader tick, cleared on success, empty on non-leaders. Update the CLI `membership list` parser and the `members.len()` assertion in `auto_rebalance.rs` to the new shape.

#### 3. `crates/rekha-cli/tests/auto_rebalance.rs` (Modify)

*   **Type / Scope**: Integration test, real processes.
*   **Proposed Changes**: After admitting a learner, poll `membership` every 500ms for up to 60s until it lists the node as a voter in the metadata group and each hosted shard group; fail with the last `membership` payload on expiry. Fixture: collection `rebalance-test` with docs `a` and `b` before the join (as today). After voter state, write doc `c` through a survivor and assert `count==3` with linearizable reads (no `?consistency=local`) on all 4 nodes, asserting `x-rekha-served-by` names the leader. This replaces the current write-before-join-only coverage with a write-after-promotion check.

#### 4. `crates/rekha-cli/tests/join_node.rs` (Modify)

*   **Type / Scope**: Integration test, real processes.
*   **Proposed Changes**: The voter-set assertion already allows the auto-promoted set (original voters or original plus the joined node); keep it. Add: poll `membership` every 500ms for up to 60s until the joined node moves from learner to voter, then `GET /api/v2/collections/{id}` on the joined node asserts 200 with the collection row, shard id, and `count==2`.

#### 5. `docs/operations.md` (Modify)

*   **Type / Scope**: Operator docs.
*   **Proposed Changes**: Rewrite the Disk constraint (~L69-73) and the alerts row (~L50, apply lag `applied` vs `last_log_index` stays distinct from promotion lag `matched` vs `last_log_index`). Document the join-then-auto-promote flow in runbook order: admit, watch `learners[].lag` fall below the promotion lag threshold of 50 entries, expect one promotion attempt per group per 1500ms tick, then verify voter state. What to check when promotion stalls: lag above threshold, no leader, snapshot transfer errors.

#### 6. `README.md` (Modify)

*   **Type / Scope**: Product docs, production checklist.
*   **Proposed Changes**: Replace `README.md:283-286` keeping the shards and replication-factor-fixed clause. New text: learner admission plus automatic promotion once lag is below 50 entries; shard movement for load balancing deferred to v2.

#### 7. `CHANGELOG.md` (Modify)

*   **Type / Scope**: Release notes.
*   **Proposed Changes**: Append under `## [Unreleased]` an entry describing automatic learner-to-voter promotion, the 50-entry promotion lag threshold, the `voters`/`learners` membership shape with deprecated `members` alias, and the v2 deferral of shard movement.

### Verification Plan

*   **Compilation/Build**: `cargo build --workspace --locked` and `cargo build --release --locked --bin rekha`.
*   **Unit Tests**: `cargo test --workspace --locked` including `rekha-cluster` membership and placement tests.
*   **Static Analysis & Linting**: `cargo fmt --check` and `cargo clippy --workspace --all-targets -- -D warnings`.
*   **Cluster tests**: `cargo test -p rekha-cli --test auto_rebalance --test join_node` must pass.
*   **Failover unchanged**: `bench/cluster-failover.sh` must still pass.
*   **Manual 4-node check**: spawn 3 nodes, create `rebalance-test` with 2 docs, join node 4 with `--join`, admit via `add_learner`, wait up to 60s for voter state in `membership`, write 1 doc through a survivor, assert `count==3` with linearizable reads on all 4 nodes.
*   **API regression**: `tests/conformance/run.py` must show no Chroma API regression (single-node scope only).

---

*Drafted with EGM*
