# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
for [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A `cluster-smoke` CI job that starts the documented three-node compose file,
  writes through one node, reads through another, and keeps serving after the
  bootstrap node stops.
- A required changelog check in the release workflow.
- Production checklist and expanded configuration/endpoint documentation in the
  README.

### Changed

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
