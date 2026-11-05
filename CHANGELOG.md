# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
for [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Reads are linearizable. `query`, `get` and `count` are routed to the shard's
  leader and answered only after a barrier that confirms leadership with a
  quorum and waits for the leader's applied state to reach the read index. Before
  this, any replica answered them from whatever it had applied, so a client that
  wrote a record and immediately read it from another node could see the old
  value with no way to know it was stale. `?consistency=local` opts out for
  workloads that prefer a fast, possibly stale answer, and `x-rekha-served-by`
  reports which node actually answered.
- A leader with no persisted snapshot builds one on demand when replication asks
  for it. openraft treats "no snapshot" as a storage error and ends that
  replica's replication stream, so a learner could previously stop making
  progress and never catch up. The build is gated on having applied state, so
  `Raft::new`'s snapshot probe — which happens before any replay — still costs
  nothing.

- Raft snapshots are now persisted, served and transferred, so a replica that
  falls behind the purged log prefix can rejoin. Previously `get_current_snapshot`
  returned `None` unconditionally and `build_snapshot`'s archive was dropped when
  the call returned, so openraft's replication stopped that follower with
  "snapshot not found" and it never caught up. Snapshots are written atomically
  under `<data-dir>/wal/snapshots/<group>/`, kept two deep, and returned from
  disk.
- Chunked snapshot transfer: 3 MiB raw `application/octet-stream` bodies staged
  at an exact offset under `<data-dir>/wal/snapshots/.incoming`, instead of one
  JSON body carrying a base64 archive under a 32 MiB limit. A retried or
  reordered chunk is refused rather than appended into the wrong place, and a
  chunk at offset zero discards a partial transfer so a retry starts clean.
  Note the disk cost: two archives are retained per group, and an archive is a
  copy of the collection data, so a compacting node needs up to three times its
  data size. Retention drops to one when the transfer becomes a stream.
- `install_snapshot` refuses an archive produced by a different group (a metadata
  snapshot cannot be installed into a shard, and vice versa), checked from the
  archive header before anything is written.
- `install_snapshot_timeout` is 10 minutes instead of openraft's 200 ms default.
- A three-node test where a replica is stopped, the cluster writes past the purge
  point, and the replica restarts and recovers by snapshot — including the
  assertion that it installed one rather than catching up by replay.
- `docs/operations.md`: readiness semantics, alerts worth having, disk sizing
  (including the snapshot multiplier), what to do when a replica falls behind,
  restart and backup procedures, and which log lines mean "this node's state is
  wrong" rather than "slow".
- `LICENSE` (Apache-2.0, matching the `license` field the workspace has been
  declaring without the file).
- A test gate in the release and Docker workflows: both jobs now depend on
  `cargo test --workspace --locked`, so a tag or image cannot be published from
  a commit whose suite does not pass.
- A `cluster-smoke` CI job that starts the documented three-node compose file,
  writes through one node, reads through another, and keeps serving after the
  bootstrap node stops.
- A required changelog check in the release workflow.
- Production checklist and expanded configuration/endpoint documentation in the
  README.
- `REKHA_SNAPSHOT_EVERY_N_LOGS`: compact the Raft log every N entries, keeping N
  entries of history. `0` (the default) disables snapshotting, so the log is
  never purged; operators raising `raft_log.redb` size alerts can opt in.
- Persisted Raft commit point and purge watermark, including the leader that
  wrote them. Both are restored on startup.
- A restart test that kills a node after its log has been purged and asserts it
  becomes ready and can still commit, plus a test that the default
  configuration never purges.
- A four-node test that restarts a shard owner across two purge cycles and
  asserts no records are lost.
- `rekha_raft_log_bytes{group}` Prometheus gauge for the metadata log and each
  shard log. With compaction disabled by default the logs grow, and unmeasured
  growth is a disk-full outage.

### Removed

- `RedbSnapshotBuilder`. It returned an empty body with a default membership,
  which is a *valid* snapshot to openraft: a follower would have installed an
  empty state machine and reported success. The log store cannot see the engine,
  so no implementation there could ever be correct, and leaving it in the public
  API was one wiring mistake away from silent data loss on a replica.

### Changed

- Enabling `REKHA_SNAPSHOT_EVERY_N_LOGS` in a cluster now warns that membership
  is reconstructed from the placement document after a purge, which is correct
  only while nodes are not added or removed; the default remains "never snapshot"
  until promotion and removal land.
- A replica asking for a snapshot it cannot have logs the cause (behind a purged
  prefix, rebuild from backup) instead of leaving openraft's generic
  "snapshot not found" as the only signal.

- Clustered startup no longer requires `REKHA_KEY_PEPPER`. A bootstrap
  `REKHA_API_KEY` must be at least 32 characters instead: the stored hash is
  unsalted, so the key itself has to be high-entropy. The pepper remains
  available as optional defence-in-depth.
- `REKHA_CLUSTER_SECRET` must be at least 32 bytes in cluster mode.
- Committed Raft batches are applied on the blocking pool. A slow HNSW flush can
  no longer delay heartbeats and trigger an election under write load.
- `/ready` waits for the cluster's bootstrap credentials to be applied, and
  reports `bootstrap_ready` in its body. A client that waits for readiness can
  no longer receive a 403 from the window before the bootstrap role assignment
  lands.
- Internal membership commands follow the metadata leader. A command that lands
  on a follower, or on a node that was the leader a moment ago, is replayed on
  the current leader instead of returning a 503.
- Admitting a learner is non-blocking: the command returns once the membership
  entry commits, and catch-up progress is reported through `/ready` and the
  replication metrics. An unreachable learner can no longer hold an operator's
  request open indefinitely.
- A public/internal port collision is refused before startup, instead of
  failing on the second bind after Raft has already initialized.
- Unused dependencies were removed from `rekha-api`, `rekha-cli`, and
  `rekha-storage`, and `scripts/dead-deps.sh` checks only the workspace members
  so the deliberately excluded crates do not produce false positives.
- The three-node compose quickstart uses `id@addr` peers, an internal listener,
  a cluster secret, and exactly one bootstrap node.
- The single-node compose file requires `REKHA_API_KEY`, because a container
  bound to `0.0.0.0` does not run in open mode.
- The published image is built from `Containerfile`.
- `rekha-cluster` no longer depends on the unused `rekha-wal` crate.

### Fixed

- Cluster HTTP clients no longer consult a system proxy, and there is one per
  process rather than one per `AppState` and one per replication stream. Two
  reasons: `reqwest::Client::new()` reads the system proxy configuration, which
  on macOS deadlocks when several threads construct a client at once (observed as
  a test binary that reports every test passed and then never exits); and
  internal Raft/forwarding traffic must never be routed through `HTTP_PROXY`,
  which cannot speak the protocol and would cost the cluster quorum.
- A replica that fails to open a collection it owns now marks itself not-ready
  instead of continuing to serve empty reads for records it is supposed to hold
  and to build snapshots that silently omit the collection (the snapshot scope
  is filtered by what the engine has loaded). A metadata row that cannot be
  recorded stays a logged warning: the engine returns a string, so a refusal
  cannot be distinguished from a store error, and it is loud at routing and
  snapshot time anyway.
- An RBAC request referencing a user or role that does not exist committed a
  Raft entry that failed to apply on every node, and the apply path treated that
  as divergence — so one authenticated call marked the whole cluster
  `ready:false` until it was restarted. Application-level rejections
  (`NotFound`, `InvalidArgument`) are deterministic and are no longer counted as
  divergence; only store failures are. The handlers now also validate before
  proposing, so the caller gets a 404/409 instead of a 200 for work that was
  never done.
- Installing a snapshot no longer clears a node's apply-failure and
  bootstrap-ready flags. A snapshot cannot prove an earlier divergence is gone,
  so the failure stays visible until an operator restarts the node.
- A checkpoint after any delete or flushed-vector upsert failed forever: the
  staged-generation validator required the segment vector count to equal the
  live count, but incremental generations hard-link earlier segments that still
  hold deleted and superseded vectors. Every later write on that collection then
  returned 500 after already committing to the WAL, and the WAL was never
  pruned. Validation now checks that each live record resolves to a staged
  vector instead of comparing totals, and a failed checkpoint no longer fails
  the write that triggered it.
- Restarting a node whose Raft log had been purged panicked inside openraft
  (`purge_upto(...) <= committed(None)`) and left the process serving every
  request with 503 forever. The commit point is now persisted, purge writes the
  entries, watermark, and commit point in one transaction, and a state machine
  that resumes from a compaction boundary seeds its membership and placement
  from durable state instead of replaying a prefix that no longer exists.
- `read_log_entries` silently skipped a missing or undecodable entry in the
  middle of the Raft log. A hole is now an error, while a purged prefix and an
  unwritten tail are still legitimately absent.
- A clustered checkpoint failed with "record has seq past the snapshot
  boundary". The boundary was taken from the collection's local WAL, which the
  cluster write path never appends to, so it was always 0 while records carried
  Raft indexes. The collection now tracks the highest sequence applied through
  either path, restores it from the checkpoint manifest, and uses it as the
  boundary.
- A corrupted `batch_len` in the middle of the WAL was classified as a torn tail
  and truncated, discarding every acknowledged batch behind it while `verify`
  reported the directory clean. Classification now asks whether the sequence
  continues behind the damage: a continuation means a hole, which is refused and
  never truncated. A torn tail at the end of the file is still discardable.
- Chroma's `delete` silently ignored `limit`, so a request to remove one
  matching record removed every match. It is now honoured for both the
  filter/id selection and the explicit id list, under the same collection lock
  as the selection itself.
- The documented `docker compose -f docker-compose.cluster.yml up -d` command
  could not start the current binary.
- Stale "WAL shipping" documentation in the cluster and API crates and in the
  README architecture diagram.
- A transient 403 during cluster bootstrap: readiness now waits for the
  bootstrap credentials.
- A membership command failing with 503 when leadership moved between the
  caller's lookup and the command; it now forwards to the leader.
- A membership command hanging while a learner caught up; admission is now
  non-blocking.
- `GET /api/v2/auth/identity` returned 404, so a `chromadb` client could not
  connect at all. The allow-list classified it as an unknown route.
- A fresh server reported its own configured tenant and database as missing, and
  database responses lacked the `id` the Chroma client requires.
- The 10-node balance suite timed out at 60 s on a 2-core CI runner (measured
  71 s); it now has a 180 s deadline.
- `where_document` now supports `$and` and `$or`, and `$not_contains` includes
  records with no document, matching the reference server.

### Removed

- `Dockerfile`, a duplicate of `Containerfile` with a weaker runtime user and no
  healthcheck.
