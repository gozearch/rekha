# Operating RekhaDB

What to watch, what each signal means, and what to do about it. This describes
behaviour that is implemented and tested, not the roadmap; where a capability is
missing it says so explicitly.

## Readiness

`GET /ready` returns `200` only when this node can serve a write:

- the engine is open;
- in a cluster, a leader exists and this node can reach it;
- the cluster's bootstrap credentials have been applied here;
- no committed entry has failed to apply on this node.

The last one is the important one. A node that returns `503` with
`apply_failure` set has refused an entry the log committed, so its state
disagrees with the cluster. It stays not-ready until it is restarted; installing
a snapshot does not clear it, because a snapshot cannot prove the divergence is
gone.

`200` means "can serve", not "is caught up". A replica that is still installing a
snapshot reports ready. Use the lag metrics for progress, not readiness.

## Read consistency

Data reads (`query`, `get`, `count`) are served by the shard's leader, after it
confirms leadership with a quorum and waits for its own applied state to cover
the read. A write that returned 200 is therefore visible to the next read, from
any node.

Two consequences worth planning for:

- read traffic lands on one node per shard, like write traffic;
- if the leader cannot confirm leadership within five seconds, reads return
  `503` rather than a stale `200`.

`?consistency=local` asks the receiving replica to answer immediately from what
it has applied. It is appropriate for analytics and backfills, where a stale
answer is fine. It is not appropriate for read-your-writes. The response header
`x-rekha-served-by` names the node that actually answered.

## Alerts worth having

| Metric / signal | Why | Threshold |
|---|---|---|
| `/ready` != 200 for > 1 min | the node cannot serve writes | page |
| `rekha_raft_quorum_age_ms` > 3 × heartbeat | isolated from quorum, or the disk stalled | page |
| `rekha_raft_log_bytes{group}` growing without bound | snapshots disabled or failing, so the log never purges | warn, then set `REKHA_SNAPSHOT_EVERY_N_LOGS` |
| apply lag (`applied` vs `last_log_index` in `/internal/cluster/membership`) | a replica is not keeping up | warn at minutes, page at hours |
| free disk | two snapshot archives per group plus the data | page below 40% |

Apply lag that persists while `/ready` is `200` is the shape of the failure this
guide exists for: the node answers reads from stale state and never catches up.

## Disk

Three things consume disk, and they are sized separately.

- **Collection data** — WAL plus checkpoint generations per collection.
- **Raft logs** — `raft_log.redb` (metadata) and `raft_log/shard-N.redb`. These
  are the only monotonically growing files when snapshots are off, which is the
  default.
- **Snapshots** — `wal/snapshots/<group>/`, the last two archives per group. An
  archive is a copy of that group's collection data, so a compacting node needs
  up to **three times** its data size. Retention drops to one when the transfer
  becomes a stream.

`REKHA_SNAPSHOT_EVERY_N_LOGS=<n>` compacts every `n` entries and keeps `n` of
history. It is safe to enable, with one constraint: membership is reconstructed
from the placement document after a purge, so **do not change the peer list or
`--shards` while compaction is on**. Promotion and removal are not implemented
yet; the warning printed at startup says the same thing.

## A replica that fell behind

Symptom: one node's apply index is far behind, it is not ready, and the others
are healthy.

What happens automatically:

1. the leader notices `next_index` is below its purge point and asks its state
   machine for a snapshot;
2. it streams the archive in 3 MiB chunks; the receiver stages them under
   `wal/snapshots/.incoming` and installs on the last one;
3. the receiver persists the installed snapshot, so it can serve the next node
   that needs one;
4. replication resumes from the snapshot's boundary.

What to check if it does not converge:

- **`install_snapshot` returning 404** — the chunked route is not mounted on the
  internal listener; this build's transport and this build's API must match.
- **"no snapshot store is attached"** — this node cannot build one at all. Only
  single-node deployments without compaction are configured that way.
- **"the snapshot store is unreadable"** — corruption under
  `wal/snapshots/<group>/`. The install is refused before any write; recover the
  directory from a backup rather than deleting it (deleting it makes the next
  request rebuild the archive, which is slow but not wrong).
- **"snapshot is for group `shard:2`, but this node asked for `metadata`"** — a
  snapshot addressed to the wrong group. The install is refused before any write;
  nothing is corrupted. This indicates a routing bug, so keep the log line.
- **`/ready` is 200 but the count is still short** — the install finished and
  replay is still running. Watch the apply index, not the count.

## Restarting a node

1. Stop it (`SIGTERM`). It shuts down its Raft groups before exiting.
2. Keep the data directory. Deleting it makes the node a new member: it will
   need a full snapshot, and if the cluster has purged below its old position
   that is the only way it comes back — which works, but it is slower.
3. Start it with the same `--node-id`, `--peers`, `--shards` and
   `--replication-factor`. A mismatch is refused at startup rather than
   silently producing a node that votes differently from its peers.

A restart with a purged log is safe: the applied state is in durable storage and
the node resumes from the purge watermark. It does not replay the purged prefix.

## Backup and restore

```sh
rekha backup  --data-dir /var/lib/rekha --out /backups/rekha-$(date +%F)
rekha restore --from /backups/rekha-2026-01-01 --data-dir /var/lib/rekha
```

- Run `rekha verify --data-dir ...` on a **stopped** directory first. It exits
  non-zero on error findings, so it can gate a backup job.
- Restore refuses a target directory that already holds data. Move the old one
  aside instead of deleting it; a rollback needs it.
- A backup of a running cluster is per-node and not a cluster snapshot. To
  restore a cluster, restore every node from a backup taken at roughly the same
  point, then restart them together.

## When a node is not ready

Read the last lines of its stderr. The messages that mean "this node's state is
wrong, not slow" are:

- an entry could not be applied (`did not apply`);
- a collection it owns could not be opened;
- the snapshot store is unreadable.

Each of those sets the not-ready flag deliberately. Restarting clears the flag;
it does not fix the cause, so capture the log first.
