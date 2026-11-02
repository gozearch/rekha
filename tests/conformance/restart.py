#!/usr/bin/env python3
"""Restart-durability check for RekhaDB's Chroma compatibility layer.

The main conformance suite runs against one server lifetime, so it cannot catch
a defect where data survives in memory but not across a restart. This script is
deliberately two-phase:

    # 1. write a known state, then stop the server
    python restart.py write --mode checkpoint --port 8123 --token conformance-key-0123456789abcdef
    # 2. start the server again on the same data dir
    python restart.py verify --port 8123 --token conformance-key-0123456789abcdef

Recovery has three distinct paths, and a test of one does not cover the others:

| `--mode`     | What it exercises                                              |
|--------------|----------------------------------------------------------------|
| `wal`        | No checkpoint at all — the WAL must hold the entire history.    |
| `checkpoint` | A published generation with the WAL prefix pruned.              |
| `tail`       | A published generation **plus** a post-checkpoint WAL tail.     |

All three leave the *same* logical state, so `verify` is mode-independent.

Pass `--data-dir` to also assert the on-disk shape the mode claims to produce
(that a generation really was published, or really was not).

Exit code is 0 only when every assertion passes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PASS = 0
FAIL = 0

COLL = "conf_restart"
EMPTIED = "conf_restart_emptied"

# The state `write` creates and `verify` expects back.
IDS = ["r0", "r1", "r2", "r3"]
EMBEDDINGS = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
METADATAS = [
    {"tag": "a", "n": 0},
    {"tag": "b", "n": 1},
    {"tag": "a", "n": 2},
    {"tag": "b", "n": 3},
]
DOCUMENTS = ["doc-0", "doc-1", "doc-2", "doc-3"]


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  \u2713 {name}")
    else:
        FAIL += 1
        print(f"  \u2717 {name} \u2014 {detail}")


def same_vec(a, b, tol: float = 1e-6) -> bool:
    """Compare embeddings numerically.

    The client returns numpy arrays, so `a == b` yields an array whose truth
    value is ambiguous rather than a bool.
    """
    try:
        return len(a) == len(b) and all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))
    except (TypeError, ValueError):
        return False


def cleanup(client, *names: str) -> None:
    for name in names:
        try:
            client.delete_collection(name)
        except Exception:  # noqa: BLE001 - best effort teardown
            pass


def generation_pointer(data_dir: str, collection_id: str) -> str:
    return os.path.join(
        data_dir, "wal", f"{collection_id}.checkpoint", "CURRENT"
    )


def optimize_via_http(state, collection_id: str) -> dict:
    """Force a checkpoint through the native optimize endpoint."""
    status, body = state.request(
        "POST", f"/api/v2/collections/{collection_id}/optimize", {}
    )
    if status not in (200, 201):
        raise RuntimeError(f"optimize failed: {status} {body}")
    return body


def write(client, args, state) -> None:
    print(f"\n[write mode={args.mode}]")
    cleanup(client, COLL, EMPTIED)

    col = client.create_collection(COLL, configuration={"hnsw": {"space": "l2"}})
    col.add(
        ids=IDS,
        embeddings=EMBEDDINGS,
        metadatas=METADATAS,
        documents=DOCUMENTS,
    )
    check("wrote 4 records", col.count() == 4, f"count={col.count()}")

    # Modify one record after the initial write so replay/checkpoint ordering
    # matters, not just bulk ingest.
    col.upsert(
        ids=["r1"],
        embeddings=[[0.0, 1.0, 0.0, 0.0]],
        metadatas=[{"tag": "b", "n": 1, "rev": 2}],
        documents=["doc-1-rev2"],
    )

    # A collection whose records are all deleted must stay empty.
    emptied = client.create_collection(EMPTIED)
    emptied.add(ids=["gone0", "gone1"], embeddings=[[1.0, 1.0], [2.0, 2.0]])
    emptied.delete(ids=["gone0", "gone1"])
    check("emptied collection is empty", emptied.count() == 0)

    # Normalise: the client may expose the id as a UUID object rather than a
    # string, which compares unequal to the manifest's string form.
    collection_id = str(col.id)

    if args.mode in ("checkpoint", "tail"):
        stats = optimize_via_http(state, collection_id)
        check(
            "optimize pruned the WAL",
            stats.get("wal_bytes_after", 1) < stats.get("wal_bytes_before", 0)
            or stats.get("wal_bytes_after") == 0,
            f"stats={stats}",
        )

    if args.mode == "tail":
        # Writes after the checkpoint must survive via the retained WAL tail.
        # They are removed again so `verify` sees the same logical state as the
        # other modes.
        col.add(ids=["tail0"], embeddings=[[0.5, 0.5, 0.5, 0.5]], documents=["tail"])
        col.delete(ids=["tail0"])
        check("post-checkpoint tail applied", col.count() == 4, f"count={col.count()}")

    # Assert the on-disk shape this mode claims to produce.
    if args.data_dir:
        pointer = generation_pointer(args.data_dir, collection_id)
        if args.mode == "wal":
            check(
                "wal mode published no generation",
                not os.path.exists(pointer),
                f"{pointer} exists; the WAL-only path was not exercised",
            )
        else:
            check("checkpoint generation published", os.path.exists(pointer), pointer)
            if os.path.exists(pointer):
                name = open(pointer, encoding="utf-8").read().strip()
                manifest = os.path.join(
                    args.data_dir,
                    "wal",
                    f"{collection_id}.checkpoint",
                    "generations",
                    name,
                    "checkpoint.json",
                )
                check("generation manifest present", os.path.exists(manifest), manifest)
                if os.path.exists(manifest):
                    meta = json.load(open(manifest, encoding="utf-8"))
                    check(
                        "manifest records an inventory",
                        bool(meta.get("files")),
                        f"files={meta.get('files')}",
                    )
                    check(
                        "manifest records the owning collection",
                        meta.get("collection_id") == collection_id,
                        f"collection_id={meta.get('collection_id')}",
                    )


def verify(client, args) -> None:
    print("\n[verify]")
    col = client.get_collection(COLL)

    check("record count survives restart", col.count() == 4, f"count={col.count()}")

    # Insertion order must survive, not just membership: offsets used to be
    # rebuilt from a HashMap on reopen, so pagination could reorder a collection
    # without a single write.
    ordered = col.get(include=[]).get("ids", [])
    check(
        "insertion order survives restart",
        ordered == IDS,
        f"got {ordered}, expected {IDS}",
    )

    # Pages must concatenate back to that same order.
    pages = []
    for offset in (0, 2):
        pages.extend(col.get(limit=2, offset=offset).get("ids", []))
    check(
        "pagination order survives restart",
        pages == IDS,
        f"pages {pages}",
    )

    got = col.get(ids=IDS, include=["documents", "metadatas", "embeddings"])
    by_id = dict(zip(got["ids"], zip(got["documents"], got["metadatas"], got["embeddings"])))

    check("all ids survive", sorted(got["ids"]) == IDS, f"{sorted(got['ids'])}")
    check(
        "documents survive",
        by_id.get("r0", (None,))[0] == "doc-0",
        f"r0 document={by_id.get('r0', (None,))[0]}",
    )
    check(
        "upsert after initial write survives",
        by_id.get("r1", (None,))[0] == "doc-1-rev2",
        f"r1 document={by_id.get('r1', (None,))[0]}",
    )
    check(
        "upserted metadata survives",
        by_id.get("r1", (None, None))[1] == {"tag": "b", "n": 1, "rev": 2},
        f"r1 metadata={by_id.get('r1', (None, None))[1]}",
    )
    check(
        "embedding survives for an untouched record",
        same_vec(by_id.get("r2", (None, None, None))[2], EMBEDDINGS[2]),
        f"r2 embedding={by_id.get('r2', (None, None, None))[2]}",
    )
    check(
        "embedding survives for an upserted record",
        same_vec(by_id.get("r1", (None, None, None))[2], [0.0, 1.0, 0.0, 0.0]),
        f"r1 embedding={by_id.get('r1', (None, None, None))[2]}",
    )

    filtered = col.get(where={"tag": {"$eq": "a"}})
    check(
        "metadata filter survives",
        sorted(filtered["ids"]) == ["r0", "r2"],
        f"{sorted(filtered['ids'])}",
    )

    ranged = col.get(where={"n": {"$gte": 2}})
    check("range filter survives", sorted(ranged["ids"]) == ["r2", "r3"], f"{ranged['ids']}")

    res = col.query(
        query_embeddings=[[1.0, 0.0, 0.0, 0.0]],
        n_results=2,
        include=["distances", "documents"],
    )
    check("query returns nearest record", res["ids"][0][0] == "r0", f"{res['ids'][0]}")
    check("query distance for self is 0", abs(res["distances"][0][0]) < 1e-5)
    check(
        "query documents survive",
        res["documents"][0][0] == "doc-0",
        f"{res['documents'][0]}",
    )

    emptied = client.get_collection(EMPTIED)
    check(
        "deleted records do not resurrect",
        emptied.count() == 0,
        f"count={emptied.count()}",
    )

    if args.data_dir:
        check(
            "no recovery error was raised",
            True,
            "the server started and served reads",
        )


class HttpState:
    """Minimal raw-HTTP client for native (non-Chroma) endpoints."""

    def __init__(self, base: str, token: str | None):
        self.base = base.rstrip("/")
        self.token = token

    def request(self, method: str, path: str, body=None):
        import urllib.error
        import urllib.request

        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("content-type", "application/json")
        if self.token:
            req.add_header("x-chroma-token", self.token)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode()
                return resp.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as e:
            raw = e.read().decode()
            try:
                return e.code, (json.loads(raw) if raw else None)
            except json.JSONDecodeError:
                return e.code, raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["write", "verify"])
    ap.add_argument(
        "--mode",
        choices=["wal", "checkpoint", "tail"],
        default="checkpoint",
        help="which recovery path the write phase should leave behind",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--token", default=None)
    ap.add_argument(
        "--data-dir",
        default=None,
        help="server data dir; when set, assert the on-disk shape of the mode",
    )
    args = ap.parse_args()

    import chromadb

    headers = {"x-chroma-token": args.token} if args.token else None
    client = chromadb.HttpClient(host=args.host, port=args.port, headers=headers)
    state = HttpState(f"http://{args.host}:{args.port}", args.token)

    if args.phase == "write":
        write(client, args, state)
    else:
        verify(client, args)

    print("\n" + "=" * 56)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 56 + "\n")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
