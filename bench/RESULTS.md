# Benchmark results

These are engineering measurements used to choose the next milestone, not a
competitive benchmark. One machine, one run per configuration, no CI gating.

- **Machine**: Apple M1 Max, 10 cores, macOS 26.5.2, local APFS.
- **Base commit**: `ca6e983`, the commit before this harness was added. The
  numbers were produced by `crates/rekha-bench` in the commit that introduced
  it.
- **Workload**: 128-dimensional vectors, one collection, one writer thread
  unless stated, `hnsw.sync_threshold = 1000` (the default), fresh data
  directory per run.

Every configuration was produced by:

```sh
target/debug/rekha-bench --data-dir /tmp/bench-<name> --out bench/results/<name>.json \
  --durability <durable|relaxed> --records <n> --dim 128 \
  --batch <b> --threads 1 --collections 1 [--skip-optimize]
```

Raw JSON is committed alongside this file in `bench/results/`.

## Durability cost (40k records, batch 100, 400 writes)

| Level | WAL fsyncs | p50 | p99 | Data dir |
|---|---:|---:|---:|---:|
| `durable` | 400 | 22.1 ms | 435.4 ms | 74.8 MB |
| `relaxed` | 0 | 17.3 ms | 423.0 ms | 74.8 MB |

An fsync costs about **5 ms per 100-record batch** — roughly 22% of p50. The
p99 is *unchanged*, because the tail is not durability: it is the automatic
checkpoint that runs every `sync_threshold = 1000` writes (one batch in ten).

## Batch size (durable)

| Records | Batch | Writes | fsyncs | p50 | p99 |
|---:|---:|---:|---:|---:|---:|
| 2,000 | 1 | 2,000 | 2,000 | 4.2 ms | 12.7 ms |
| 40,000 | 100 | 400 | 400 | 22.1 ms | 435.4 ms |
| 20,000 | 1,000 | 20 | 20 | 406.0 ms | 503.8 ms |

At batch 1 the per-write cost is the fsync itself (~4 ms). At batch 1,000 the
per-record cost is 0.41 ms, so batching already amortises the sync by ~10×; a
further group-commit layer would target a cost that is no longer dominant.

## Checkpoint phases (`optimize` at the end of ingest)

| Records | flush | rebuild | checkpoint | records | vectors | index | manifest | cleanup | publish |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0 ms | 1,688 ms | **176 ms** | 26 | 68 | 29 | 22 | 17 | 10 |
| 40,000 | 0 ms | 6,491 ms | **431 ms** | 53 | 217 | 81 | 36 | 17 | 13 |

All values in milliseconds; the phase columns are components of the checkpoint
column.

## What this says

1. **The checkpoint is the tail.** At 40k records one batch in ten pays ~430 ms
   while holding the collection mutex, so reads block too. Durable and relaxed
   p99 match, which rules out fsync as the cause.
2. **Checkpoint cost tracks corpus size.** 176 ms → 431 ms for 4× the records.
   `vectors` is the largest and fastest-growing phase (68 → 217 ms) because
   every checkpoint rewrites every vector; `records` and `index` grow too.
3. **`optimize` is 15× the checkpoint**, dominated by a full HNSW rebuild
   (6.5 s at 40k). That is a separate, explicitly-invoked operation, but it is
   the second scaling wall.
4. **Group commit is not the next win.** fsync is ~22% of p50 and none of the
   tail. It stays on the list, below the storage work.

Consequence for the roadmap: checkpoint must become proportional to *changed*
records and must publish without holding the collection lock. That is the next
milestone; the numbers above are its before-picture.

## P1: incremental checkpoints (after)

Routine checkpoints append only the vectors written since the last one and
hard-link the segments already on disk; `optimize` and every eighth checkpoint
compact.

Same workload (40k records, dim 128, batch 100, durable):

| Measurement | Before | After |
|---|---:|---:|
| p50 | 22.1 ms | 21.5 ms |
| p95 | 292.1 ms | 238.1 ms |
| p99 | 435.4 ms | **360.6 ms** |
| Ingest (40k records) | 19.8 s (~2,019 rec/s) | **18.3 s (~2,186 rec/s)** |

Checkpoint phases at 40k after the change (`incremental-40k-b100.json`):

| Kind | total | records | vector segments | id→offset map | graph | manifest | validate | publish | cleanup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| incremental (100 new records) | 323.1 ms | 43.7 | 92.7 | 21.9 | 78.0 | 33.0 | 23.5 | 12.6 | 16.4 |
| compaction (segments ≥ 8) | 436.5 ms | 48.1 | 192.3 | 24.1 | 78.0 | 33.4 | 22.3 | 12.7 | 16.7 |

What is still whole-corpus in an incremental checkpoint: `records.bin`, the
id→offset map, `order.bin`, validation, and the graph. Those are the next
target — measured, not assumed.

Two trades measured and recorded:

- **Deferring the derived graph.** Skipping `index.bin` except when compacting
  gives a better write path (p99 277.5 ms, ingest 16.5 s) but forces a
  whole-corpus graph rebuild on any restart that did not land on a compaction
  (6.5 s at 40k, growing with the corpus). Rejected: it moves a bounded cost
  into an unbounded restart. It stays a candidate if the graph can be saved
  asynchronously and safely.
- **Bounding segment fan-out at 8.** Without a bound, every checkpoint re-opens
  and checksum-verifies every segment, and the improvement disappears. The
  bound is what keeps verification proportional to a fixed number of segments.

Only one run per configuration; treat the ratios as directional.

## Cluster write latency after C0b (Raft log as the data log)

`bench/cluster-latency.py` against a real three-process cluster, three nodes,
dim 128, batch 50, default `hnsw.sync_threshold = 1000`.

**These are debug-build numbers** (`target/debug/rekha`), so absolute latency is
pessimistic by a large constant. Only the comparison between rows is meaningful.

| Scenario | Writes | p50 | p99 | max |
|---|---:|---:|---:|---:|
| Writes to the leader, no checkpoint fires (900 records) | 18 | 40.3 ms | 52.5 ms | 52.5 ms |
| Writes to a follower, no checkpoint fires (900 records) | 18 | 35.9 ms | 55.5 ms | 55.5 ms |
| Writes to a follower, checkpoints fire (3,000 records) | 60 | 37.1 ms | 240.9 ms | 240.9 ms |

**Forwarding is not the cost.** Writing to a follower (one extra HTTP hop) is
within noise of writing to the leader, because the log round trip dominates.

**C0b did not remove the p99 tail, and the claim that it would was wrong.** The
tail fell from ~435 ms at 40k records to ~240 ms at 3k, but that is a smaller
corpus, not a smaller spike: the background checkpoint still takes the
collection mutex, so a write that arrives during one waits for it. What C0b
changed is that the write *triggering* a checkpoint no longer pays for it, and
the apply path no longer stalls behind it.

Removing the tail requires the checkpoint to stop taking the write path's lock,
which is P1c's immutable-generation work: checkpoint cost proportional to change
rather than to corpus. Until then the tail is bounded by checkpoint duration and
grows with the collection.

## Release profile and lock discipline (2026-10-31, `c7fc660`+)

H10.1 swapped the engine's `std::sync` locks for `parking_lot`, removed
poisoning, and set the release profile (`lto = "fat"`, `codegen-units = 1`,
`panic = "abort"`, `strip = true`). Both changes are in the same measurement, so
the split between them is **not** attributed — only the combined effect is
claimed.

Release build, 40k records, batch 100, `sync_threshold = 1000`, one run each:

| | p50 | p99 | ingest | data dir |
|---|---:|---:|---:|---:|
| before profile + locks | 20.0 ms | 233.7 ms | 15.88 s | — |
| after | — | **175.5 ms** | **13.41 s** | — |

p99 improved 24.9%, ingest 15.6%. `bench/results/release-40k-b100.json` is the
after-measurement and is the reference the nightly gate compares against.

**Caveat that matters more than the numbers:** that baseline was measured on an
M1 Max, and the nightly gate runs on a GitHub `ubuntu-latest` runner. A fixed
baseline from different hardware measures the machines, not the code, so
`bench.yml` compares the base revision and HEAD *on the same runner in the same
job*. This file is a record, not the threshold.

## Ten-node placement (2026-11-01)

`bench/cluster-scale.sh`, one machine, ten processes, 40 shards x 3 replicas
(the `4 x nodes` default), ephemeral local ports.

| Run | Collections | Data files | Expected | Per-node share |
|---|---:|---:|---:|---|
| 10 nodes, 40 shards | 40 | 120 | 120 | 12% – 40% (ideal 30%) |
| 10 nodes, 40 shards | 200 | 600 | 600 | **8% – 46%** |
| 10 nodes, 80 shards | 40 | 120 | 120 | 18% – 55% |
| 10 nodes, 160 shards | 40 | 120 | 120 | 12% – 52% |

**Correctness holds:** every collection exists on exactly three nodes — the
total is exactly `3 x collections` in every run — and killing the metadata
leader left the cluster accepting new collections and writes.

**Balance does not hold.** At 200 collections one node holds 46% of the corpus
while another holds 8%. That is not noise from too few collections: a
simulation of the same placement function over 2,000 trials gives, for 40
collections:

| shards | max per node (mean / p95) | min per node (mean) | spread (mean) |
|---|---:|---:|---:|
| 40 | 19.4 / 23 | 4.6 | 14.8 |
| 80 | 18.2 / 22 | 6.0 | 12.2 |
| 160 | 17.6 / 21 | 6.9 | 10.8 |

The cause is structural, not a bad hash: rendezvous assignment of a **fixed
shard set** to nodes has variance that scales as `1/sqrt(shards)`, and the
number of groups per node is `shards x R / nodes`, so buying balance with more
shards buys it with more consensus groups (160 shards at ten nodes is 48
groups per node). More shards also did not help at 40 *collections*, because
each shard then holds about one collection and the skew in shard ownership is
inherited directly.

**Conclusion:** collection-level balance needs the placement decision made per
collection rather than inherited from a fixed shard set — which means recording
the shard on the collection's catalog row and letting the metadata leader pick
it, instead of deriving it from the id. That is a routing source-of-truth
change and is not made here.

## Placement is a decision, not a hash (2026-11-01)

The run above exposed the balance problem. The fix was to stop deriving a
collection's shard from its id and have the metadata leader choose it for node
load, recording it on the collection's catalog row.

Measured with `bench/cluster-scale.sh`, ten processes, 40 shards x 3 replicas,
on one machine:

| Placement | Collections | Per-node share | Spread |
|---|---:|---|---:|
| id-derived (before) | 40 | 11 – 13 (ideal 12) | 2 |
| id-derived (before) | 200 | **17 – 92 (8% – 46%)** | **75** |
| leader-chosen (after) | 40 | 12 – 12 (ideal 12) | **0** |
| leader-chosen (after) | 200 | **59 – 61 (30% each)** | **2** |
| leader-chosen (after) | 1000 | 299 – 301 (ideal 300) | **2** |

Each of these runs also kills the metadata leader and requires the cluster to
keep creating and writing afterwards. 3-node failover: 3.47 s, all checks passed.

Correctness is unchanged in every run: the number of collection data files is
exactly `3 x collections`, no node holds the corpus, and the cluster keeps
creating and writing after the metadata leader is killed.

The 200-collection row is now asserted by two gates, both of which were shown
failing on the old behaviour:

- `crates/rekha-cli/tests/balance.rs` — ten real processes, 200 collections,
  asserts `max - min <= 2`. With id-derived placement it fails at 24 – 86.
- `crates/rekha-cluster/src/placement.rs::placement_follows_node_load_not_shard_count`
  — the decision function alone, against the measured shard-ownership skew
  (`9, 13, 12, 17, 12, 5, 16, 9, 16, 11` shards per node).

### Small-cluster limit, stated rather than hidden

Per-collection placement can only choose among the replica sets that exist. With
`R = 3` over four nodes there are exactly four distinct owner sets, and
rendezvous does not hand them out evenly, so a four-node cluster can be lopsided
no matter how collections are assigned — measured: 53 – 80 of 80 collections.
Ten nodes have 120 possible owner sets and balance to within two. The harness
therefore asserts the bound for six or more nodes; the strict gate is at ten.

### Cost

Creating `n` collections must not be `O(n^2)`. Reading the load from the catalog
on every creation measured 82 s for 200 collections. The load is now cached per
leadership term — only the leader places, so the cache is exact while it leads
and is rebuilt once per election, not once per creation.

The per-request cost was checked against the previous commit built from `eada0ed`
in a separate worktree, ten nodes, one persistent HTTP client, 30 samples:

| Request | `eada0ed` | this change |
|---|---|---|
| `GET /health` | 1.1 ms | 0.8 ms |
| `GET` one collection | 6.0 ms | 1.1 ms |
| `POST .../add` | 53.0 ms | 5.5 ms |
| `POST /api/v2/collections` | 61.4 ms | 62.5 ms |

Creation is unchanged, which is the number that had to stay flat: it is one
metadata round trip. The other rows improved, but the cause was not isolated, so
they are recorded as observations rather than as a claimed result.

