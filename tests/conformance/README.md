# Conformance suites

Three scripts, each answering a different question. A capability is only
"supported" when all three agree.

| Script | Question it answers |
|---|---|
| `run.py` | Does the **real `chromadb` Python client** work against Rekha? |
| `restart.py` | Does that state survive a **process restart**? |
| `differential.py` | Does Rekha behave the **same as a real Chroma server**? |

All three exit non-zero on failure.

## Run locally

```sh
# Rekha on a scratch data dir (localhost => open mode; give it a key anyway so
# bootstrap auth is exercised like production).
cargo run --bin rekha -- serve \
    --data-dir /tmp/rekha-conf --host 127.0.0.1 --port 8123

REKHA_API_KEY=conformance-key-0123456789abcdef cargo run --bin rekha -- serve \
    --data-dir /tmp/rekha-conf --host 127.0.0.1 --port 8123
```

```sh
uv run --no-project --with chromadb python tests/conformance/run.py \
    --host 127.0.0.1 --port 8123 --token conformance-key-0123456789abcdef
```

### Restart durability

Recovery has three distinct paths and a test of one says nothing about the
others, so each is exercised separately:

```sh
bash tests/conformance/restart-mode.sh checkpoint   # published generation, WAL pruned
bash tests/conformance/restart-mode.sh tail         # generation + post-checkpoint tail
bash tests/conformance/restart-mode.sh wal          # no checkpoint at all
```

The wrapper writes a known state, stops the server, restarts it on the same data
directory, and verifies. `--data-dir` (set by the wrapper) makes the write phase
assert the on-disk shape it claims: that a generation really was published with
an inventory, or — for `wal` — that none exists, so the mode cannot pass while
silently testing the wrong path.

A checkpoint can also be forced over HTTP:

```sh
curl -X POST -H "x-chroma-token: conformance-key-0123456789abcdef" \
    http://127.0.0.1:8123/api/v2/collections/<uuid>/optimize
```

`optimize` force-flushes the buffer, rebuilds the ANN graph, publishes a
checkpoint generation, and prunes the WAL.

### Differential (needs a reference Chroma)

```sh
# Terminal 1: reference server. `chroma run` binds localhost over IPv6 here, so
# use `localhost` rather than 127.0.0.1 when targeting it.
uv run --no-project --with chromadb chroma run --path /tmp/chroma-ref --port 8899

# Terminal 2
python3 tests/conformance/differential.py \
    --target http://127.0.0.1:8123 \
    --reference http://localhost:8899 \
    --token conformance-key-0123456789abcdef
```

Raw HTTP is used here on purpose: it compares the wire contract directly,
including status codes and response field presence, which a client library
normalises away.

## Verified against

- Rekha: this working tree (`cargo build --bin rekha`).
- `chromadb` Python client `1.5.9` (pinned in CI as `CHROMADB_VERSION`).
- Reference Chroma server `1.0.0` (from `chroma run` in the same environment).

The differential suite reports both servers' `/api/v2/version` before comparing
them, so a stale process squatting on the reference port is caught instead of
being mistaken for a behavioural difference.

Record new versions here when they change; an unpinned comparison is not a
regression test.

## What the differential suite covers

`create/config`, `metric` (cosine distances), `include-defaults`,
`get-filters` (`$eq`, ranges, `$in`, `limit`), `delete-selectors`,
`include-empty`, `ordering`, `mutations`, `dimension`, `unknown-include`.

The `include-empty` group pins the difference between an **omitted** `include`
(use the endpoint defaults) and an explicit `[]` (ids only). Both used to be
treated as "empty", so an explicit empty list silently returned metadatas and
documents.

The `ordering` group pins retrieval order:

- Records come back in **insertion** order, not sorted by id.
- An explicit `ids` list is filtered but **not** re-ordered.
- Delete then re-add moves a record to the end.
- `update`/`upsert` keep their position.
- `limit`/`offset` walk that same order.

Rekha additionally guarantees that this order survives a restart, which the
restart suite asserts (it is not something the reference contract promises).

The `mutations` group pins the semantics that are easy to get wrong and were:

- `update` **merges** metadata and preserves every field it does not mention.
- A `null` metadata value **deletes** the key.
- A document-only or vector-only update leaves the other fields alone.
- Updating an unknown id is a silent no-op (200), not a batch error.
- `upsert` merges metadata into an existing record.

## Recovery contract

Checkpoints are immutable generations published through a `CURRENT` pointer.
The manifest records the format version, the owning collection, the vector
width and metric it was written for, an inventory of every authoritative file
(length and CRC), and the WAL sequence the snapshot covers.

Recovery refuses to start rather than presenting a partial collection when:

- the manifest version is unknown,
- the inventory is missing, incomplete, duplicated, or fails its checksum,
- the generation belongs to another collection, or was written for a different
  width or metric,
- a record's sequence is beyond the snapshot boundary,
- a record is covered by the checkpoint but its vector does not resolve to an
  in-bounds vector of the expected width,
- the segment index references a record that no longer exists,
- the catalog records durable progress but no generation is readable.

A corrupt latest generation is **not** downgraded to a WAL replay: the snapshot
covers the prefix it publishes and the log below it has been pruned, so the
remaining log cannot reconstruct the acknowledged state.

### WAL repair

The log is inspected **read-only** before anything opens it for writing, and the
inspection distinguishes three outcomes:

| Tail | Meaning | Recovery |
|---|---|---|
| `Clean` | every byte is a verified record | replay normally |
| `Suspect` | the invalid region runs to end-of-file | may be discarded |
| `Corrupt` | damage followed by more data | always an error |

A missing commit marker is proof that the final batch never completed. A batch
whose marker is present but whose checksum fails is ambiguous — a torn write that
persisted the marker, or bit rot — but because nothing follows it, discarding it
is still the only safe repair. A `Corrupt` interior is never truncated: doing so
shortens an acknowledged collection and reports it as a successful open.

Recovery also refuses to start when the log resumes *after* the checkpoint
boundary (a gap in the middle) or when there is no checkpoint and the log does
not start at sequence 1 (a lost prefix).

The vector graph is derived state. If it is missing or was written for a
different configuration it is rebuilt from the authoritative vectors on open,
rather than left empty — an empty graph would make the ANN path return nothing
once a collection is large enough to use it.

## WAL format and acknowledgement

The log uses format **v2** (`RKW2`), which frames each request as one batch with
a trailing commit marker and a checksum covering the batch header and entries. A
batch is acknowledged only after it is fsynced (`--wal-durability durable`, the
default), so a crash either preserves the whole batch or none of it.

`GET /ready` reports the active durability level.

## Metrics

Setting `REKHA_METRICS_PORT` starts a Prometheus endpoint. The engine records:

| Metric | Type | Meaning |
|---|---|---|
| `rekha_checkpoint_phase_us` | histogram | checkpoint time by `phase` |
| `rekha_checkpoint_total_us` | histogram | whole checkpoint |
| `rekha_checkpoint_total{outcome}` | counter | `ok` / `failed` |
| `rekha_write_batches_total` | counter | committed WAL batches |
| `rekha_wal_ops_total` | counter | operations appended |
| `rekha_wal_syncs_total` | counter | fsyncs issued for batches (durable only) |
| `rekha_wal_bytes` | gauge | WAL size of the most recent write |
| `rekha_records` | gauge | live records after the most recent write |

## Checkpoint manifest

A published `checkpoint.json` is format **3** and carries the file inventory, the
owning collection, the vector configuration, and the `snapshot_seq` boundary.
A generation may reference several vector segments: routine checkpoints append
only the vectors written since the last one and hard-link the rest, while
`optimize` (and every eighth checkpoint) compacts into a single segment. The
derived ANN graph is persisted only when compacting; recovery rebuilds it when
it is absent.
There is no older manifest format to fall back to: a manifest that does not match
is rejected rather than guessed at, and `order.bin` is required because it is
always written.

Checkpoint files placed directly under the collection's checkpoint directory,
with no `CURRENT` naming a generation, are the pre-generation layout. They are
rejected rather than read or treated as absent.

## Deliberate extensions (not Chroma behaviour)

These are Rekha additions. They are asserted on Rekha and reported as expected
differences, never silently passed off as parity:

- **`configuration.hnsw.dimension` / top-level `dimension`.** Chroma has no
  create-time dimension field; its `configuration.hnsw` accepts only
  `space`, `ef_construction`, `ef_search`, `max_neighbors`, `num_threads`,
  `resize_factor`, `sync_threshold`, `batch_size`, and the width is inferred
  from the first write. Chroma answers 422 for `hnsw.dimension`. Rekha accepts
  it so a caller can pin the width up front — useful, but **not** parity.
- **UUID addressing of collection-level `GET`/`DELETE`.** Chroma addresses those
  by name and returns 404 for a UUID. Rekha accepts either, with the owner's
  tenant/database enforced (see `chroma_uuid_does_not_bypass_tenant_scope`).
- **`POST /api/v2/collections/{id}/optimize` reports timings.** The response
  adds `flush_us`, `rebuild_us`, and a `checkpoint` breakdown
  (`records_us`, `index_us`, `vectors_us`, `manifest_us`, `validate_us`,
  `publish_us`, `cleanup_us`, `total_us`). Chroma's optimize has no equivalent
  fields. These exist so checkpoint cost is measurable in production rather
  than inferred from a benchmark.
- **Duplicate ids in `add` are rejected.** Probed against Chroma 1.0.0: adding
  an id that already exists, or repeating an id inside one batch, returns 201
  and silently keeps the first record. Rekha rejects the batch with 400. This is
  a deliberate, stricter divergence: a duplicate `add` is almost always a caller
  bug, and silently discarding the new vector makes it invisible. Documented
  rather than matched.

## Defects these suites found and now guard

- `hnsw:space` was hardcoded to L2, changing reported distances for cosine.
- `get` parsed `where` and discarded it.
- `delete` required `ids`, so filter-only deletes were impossible.
- `delete` with neither selector returned 400; Chroma treats it as a no-op
  returning `{"deleted": 0}` and deletes nothing.
- `delete` returned `{}`; Chroma returns `{"deleted": n}` (ids submitted, for an
  id delete; records matched, for a `where` delete).
- `add` returned 200; Chroma returns 201.
- A duplicate collection name returned 500; Chroma returns 409.
- `update` 422'd on `"embeddings": null`, which `chromadb >= 1.5` always sends.
- `configuration.hnsw.dimension` was ignored, silently falling back to
  inference.
- `include` defaults returned ids only instead of Chroma's documented defaults.

## Not covered yet

`fork`, `search`, `indexing_status`, conditional get/commit, attached functions,
server-side embedding functions, sparse/hybrid retrieval. They are listed in
`UNSUPPORTED` in `run.py` so the gap stays visible rather than implied.
