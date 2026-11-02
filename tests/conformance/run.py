#!/usr/bin/env python3
"""Chroma client conformance suite for RekhaDB.

Runs the **real** `chromadb` Python client against a live Rekha server and
asserts the behaviours the compatibility layer is supposed to provide. This is
the Phase 0 deliverable: every capability the server claims must have a case
here, or an explicit entry in `UNSUPPORTED`.

Usage:
    # start the server first (see tests/conformance/README.md), then:
    uv run --no-project --with chromadb python tests/conformance/run.py \\
        --host 127.0.0.1 --port 8000

Exit code is 0 only when every case passes.
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback

UNSUPPORTED: dict[str, str] = {
    "fork": "collection forks are not implemented yet",
    "search": "hybrid search is not implemented yet",
    "indexing_status": "not implemented yet",
}

PASS = 0
FAIL = 0
SKIP = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  \u2713 {name}")
    else:
        FAIL += 1
        print(f"  \u2717 {name} \u2014 {detail}")


def case(name: str, fn) -> None:
    """Run one grouped case, reporting a failure instead of aborting."""
    print(f"\n[{name}]")
    try:
        fn()
    except Exception:  # noqa: BLE001 - a failing case must not stop the suite
        global FAIL
        FAIL += 1
        print(f"  \u2717 {name} raised:")
        traceback.print_exc()


def close(a: float, b: float, tol: float = 1e-4) -> bool:
    return math.isclose(a, b, rel_tol=0, abs_tol=tol)


def cleanup(client, *names: str) -> None:
    for name in names:
        try:
            client.delete_collection(name)
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--token", default=None, help="sent as x-chroma-token")
    args = ap.parse_args()

    import chromadb

    headers = {"x-chroma-token": args.token} if args.token else None
    client = chromadb.HttpClient(host=args.host, port=args.port, headers=headers)

    print(f"chromadb client {chromadb.__version__} -> {args.host}:{args.port}")

    # -- meta -------------------------------------------------------------
    def meta():
        check("heartbeat returns nanoseconds", client.heartbeat() > 0)
        check("version is a string", isinstance(client.get_version(), str))

    case("meta", meta)

    # -- collection lifecycle ---------------------------------------------
    def collections():
        cleanup(client, "conf_meta")
        col = client.create_collection("conf_meta", metadata={"purpose": "conformance"})
        check("create returns a collection", col is not None)
        check("name round-trips", col.name == "conf_meta")
        check("metadata round-trips", col.metadata.get("purpose") == "conformance")

        again = client.get_or_create_collection("conf_meta")
        check("get_or_create reuses the collection", again.id == col.id)

        listed = [c.name for c in client.list_collections()]
        check("list_collections includes it", "conf_meta" in listed)

        client.delete_collection("conf_meta")
        check(
            "delete_collection removes it",
            "conf_meta" not in [c.name for c in client.list_collections()],
        )

        try:
            client.get_collection("conf_meta")
            check("get missing collection raises", False, "no error raised")
        except Exception:  # noqa: BLE001
            check("get missing collection raises", True)

    case("collections", collections)

    # -- metric semantics --------------------------------------------------
    def metric():
        cleanup(client, "conf_cos")
        col = client.create_collection(
            "conf_cos", metadata={"hnsw:space": "cosine"}
        )
        col.add(
            ids=["x", "y"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            metadatas=[{"k": 1}, {"k": 2}],
        )
        res = col.query(
            query_embeddings=[[1.0, 0.0]], n_results=2, include=["distances"]
        )
        d = res["distances"][0]
        check("cosine self-distance is 0", close(d[0], 0.0), f"got {d[0]}")
        check(
            "cosine orthogonal distance is 1",
            close(d[1], 1.0),
            f"got {d[1]} (L2-squared would be 2)",
        )
        cleanup(client, "conf_cos")

    case("metric", metric)

    # -- records -----------------------------------------------------------
    def records():
        cleanup(client, "conf_rec")
        col = client.create_collection("conf_rec")
        col.add(
            ids=["a0", "b0", "a1", "b1", "a2"],
            embeddings=[
                [1.0, 0.0],
                [0.0, 1.0],
                [1.1, 0.0],
                [0.0, 1.1],
                [1.2, 0.0],
            ],
            metadatas=[
                {"tag": "a"},
                {"tag": "b"},
                {"tag": "a"},
                {"tag": "b"},
                {"tag": "a"},
            ],
            documents=["d0", "d1", "d2", "d3", "d4"],
        )
        check("count after add", col.count() == 5, f"count={col.count()}")

        col.upsert(
            ids=["a0"],
            embeddings=[[1.0, 0.0]],
            metadatas=[{"tag": "a", "v": 2}],
            documents=["updated"],
        )
        check("upsert keeps count", col.count() == 5)
        got = col.get(ids=["a0"], include=["documents", "metadatas"])
        check("upsert replaces document", got["documents"] == ["updated"])
        check("upsert merges metadata", got["metadatas"][0].get("v") == 2)

        col.update(ids=["a1"], metadatas=[{"tag": "a", "v": 9}])
        got = col.get(ids=["a1"], include=["metadatas"])
        check("update changes metadata", got["metadatas"][0].get("v") == 9)

        got = col.get(ids=["zzz"])
        check("get unknown id returns empty", got["ids"] == [])

        col.delete(ids=["a2"])
        check("delete by id", col.count() == 4, f"count={col.count()}")

        cleanup(client, "conf_rec")

    case("records", records)

    # -- filters -----------------------------------------------------------
    def filters():
        cleanup(client, "conf_flt")
        col = client.create_collection("conf_flt")
        col.add(
            ids=["a0", "b0", "a1", "b1", "a2"],
            embeddings=[
                [1.0, 0.0],
                [0.0, 1.0],
                [1.1, 0.0],
                [0.0, 1.1],
                [1.2, 0.0],
            ],
            metadatas=[
                {"tag": "a", "n": 0},
                {"tag": "b", "n": 1},
                {"tag": "a", "n": 2},
                {"tag": "b", "n": 3},
                {"tag": "a", "n": 4},
            ],
        )

        got = col.get(where={"tag": {"$eq": "a"}})
        check("get with where", sorted(got["ids"]) == ["a0", "a1", "a2"], f"{got['ids']}")

        got = col.get(where={"n": {"$gte": 3}})
        check("get with range", sorted(got["ids"]) == ["a2", "b1"], f"{got['ids']}")

        got = col.get(where={"tag": {"$in": ["b"]}})
        check("get with $in", sorted(got["ids"]) == ["b0", "b1"], f"{got['ids']}")

        page = col.get(limit=2)
        check("get honours limit", len(page["ids"]) == 2, f"{len(page['ids'])}")
        page2 = col.get(limit=2, offset=2)
        check("get honours offset", page["ids"] != page2["ids"])

        res = col.query(
            query_embeddings=[[1.0, 0.0]],
            n_results=3,
            where={"tag": {"$eq": "a"}},
            include=["distances", "metadatas"],
        )
        check(
            "query applies where",
            sorted(res["ids"][0]) == ["a0", "a1", "a2"],
            f"{res['ids'][0]}",
        )

        col.delete(where={"tag": {"$eq": "b"}})
        check("delete by where", col.count() == 3, f"count={col.count()}")

        cleanup(client, "conf_flt")

    case("filters", filters)

    # -- include semantics -------------------------------------------------
    def includes():
        cleanup(client, "conf_inc")
        col = client.create_collection("conf_inc")
        col.add(
            ids=["i0"],
            embeddings=[[1.0, 0.0]],
            metadatas=[{"tag": "a"}],
            documents=["doc"],
        )

        got = col.get(ids=["i0"], include=["embeddings", "metadatas", "documents"])
        check("get incl embeddings", len(got["embeddings"][0]) == 2)
        check("get incl metadatas", got["metadatas"][0] == {"tag": "a"})
        check("get incl documents", got["documents"] == ["doc"])

        res = col.query(
            query_embeddings=[[1.0, 0.0]], n_results=1, include=["distances", "embeddings"]
        )
        check("query incl embeddings", res["embeddings"] is not None)
        check("query incl distances", res["distances"][0][0] == 0.0)

        res = col.query(query_embeddings=[[1.0, 0.0]], n_results=1)
        check(
            "query default include has distances+documents+metadatas",
            all(k in res and res[k] is not None for k in ("distances", "documents", "metadatas")),
            f"keys={sorted(res)}",
        )

        cleanup(client, "conf_inc")

    case("includes", includes)

    # -- dimension handling ------------------------------------------------
    def dimensions():
        cleanup(client, "conf_dim")
        # chromadb >= 1.5 has no `dimension=` kwarg; the width goes in the
        # HNSW configuration object.
        col = client.create_collection(
            "conf_dim", configuration={"hnsw": {"dimension": 4}}
        )
        # The *first* write must be validated against the configured width.
        # Writing a matching vector first would also pass if the server ignored
        # the configuration and inferred the width from that write, so the
        # mismatch is probed before any valid write establishes it.
        try:
            col.add(ids=["d_bad"], embeddings=[[0.1] * 8])
            check(
                "configured dimension rejects a mismatched first write",
                False,
                "accepted an 8-dim vector into a 4-dim collection (config ignored)",
            )
            # Leave the collection usable for the remaining checks.
            col.delete(ids=["d_bad"])
        except Exception:  # noqa: BLE001
            check("configured dimension rejects a mismatched first write", True)

        col.add(ids=["d0"], embeddings=[[0.1] * 4])
        check("explicit dimension accepts matching width", col.count() == 1)
        try:
            col.add(ids=["d1"], embeddings=[[0.1] * 8])
            check("wrong dimension rejected", False, "no error raised")
        except Exception:  # noqa: BLE001
            check("wrong dimension rejected", True)
        cleanup(client, "conf_dim")

        cleanup(client, "conf_auto")
        auto = client.create_collection("conf_auto")
        auto.add(ids=["a"], embeddings=[[0.1] * 6])
        check("inferred dimension accepts first write", auto.count() == 1)
        cleanup(client, "conf_auto")

    case("dimensions", dimensions)

    # -- explicit unsupported capabilities --------------------------------
    print("\n[unsupported]")
    for capability, reason in UNSUPPORTED.items():
        global SKIP
        SKIP += 1
        print(f"  - {capability}: {reason}")

    print("\n" + "=" * 56)
    print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} declared unsupported")
    print("=" * 56 + "\n")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
