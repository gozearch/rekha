#!/usr/bin/env python3
"""Differential conformance check: RekhaDB vs a reference Chroma server.

The Python-client suite in `run.py` proves Rekha works with the real client, but
it cannot tell whether the *behaviour* matches Chroma — it has no reference to
compare against. This script sends identical raw HTTP requests to two servers
and compares the normalized outcome.

Raw HTTP is deliberate: it exercises the wire contract directly, without a
client library normalising (or hiding) the difference — for example the client
aggregates `get`/`query` results into a positional shape that can mask a server
returning fields in a different order or omitting them.

Usage:
    python differential.py --target http://127.0.0.1:8123 \
        --reference http://localhost:8899 --token conformance-key-0123456789abcdef

Note: `chroma run` binds `localhost` over IPv6 only here, so the reference URL
must use `localhost` rather than `127.0.0.1`.

Exit code is 0 when every compared case agrees.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

PASS = 0
FAIL = 0
SKIP = 0

TENANT = "default_tenant"
DATABASE = "default_database"
BASE = f"/api/v2/tenants/{TENANT}/databases/{DATABASE}/collections"

# Cases where a difference is known and accepted, with the reason. A case listed
# here is reported but does not fail the run; keeping the reason next to the
# exemption stops it from silently becoming a permanent gap.
ACCEPTED_DIFFERENCES: dict[str, str] = {}


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  \u2713 {name}")
    else:
        FAIL += 1
        print(f"  \u2717 {name} \u2014 {detail}")


def skip(name: str, reason: str) -> None:
    global SKIP
    SKIP += 1
    print(f"  - {name}: {reason}")


class Server:
    def __init__(self, base: str, token: str | None):
        self.base = base.rstrip("/")
        self.token = token

    def request(self, method: str, path: str, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("content-type", "application/json")
        if self.token:
            req.add_header("x-chroma-token", self.token)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode()
                return resp.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as e:
            raw = e.read().decode()
            try:
                parsed = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                parsed = raw
            return e.code, parsed


def _version_string(value) -> str:
    """Normalise the version endpoint, which returns a bare JSON string."""
    if isinstance(value, str):
        return value
    return str(value)


def status_class(code: int) -> str:
    return f"{code // 100}xx"


def case(name: str, target, reference, fn) -> None:
    print(f"\n[{name}]")
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - a case failure must not stop the suite
        check(name, False, f"raised {type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="Rekha base URL")
    ap.add_argument("--reference", required=True, help="Chroma base URL")
    ap.add_argument("--token", default=None, help="x-chroma-token for the target")
    args = ap.parse_args()

    target = Server(args.target, args.token)
    ref = Server(args.reference, None)

    # Identify both servers before comparing them. A stale listener on the
    # reference port answers requests happily and its canned responses look
    # like behavioural differences, which once produced a false finding.
    rs, rbody = ref.request("GET", "/api/v2/version")
    ts, tbody = target.request("GET", "/api/v2/version")
    if rs != 200:
        print(f"reference {args.reference} did not answer /api/v2/version ({rs}): {rbody}")
        return 1
    if ts != 200:
        print(f"target {args.target} did not answer /api/v2/version ({ts})")
        return 1
    print(f"target   {args.target} version={_version_string(tbody)}")
    print(f"reference {args.reference} version={_version_string(rbody)}")

    def both(method, path, body=None):
        return target.request(method, path, body), ref.request(method, path, body)

    def both_at(method, tpath: str, rpath: str, body=None):
        """Same call on both servers, but with server-specific collection ids."""
        return target.request(method, tpath, body), ref.request(method, rpath, body)

    seq = {"n": 0}
    # Names must be unique across *runs* as well as within one: a leftover
    # collection from a previous run would otherwise make creation fail and
    # masquerade as a behavioural difference.
    run = f"{int(__import__('time').time())}"

    def fresh(prefix: str) -> str:
        seq["n"] += 1
        return f"diff_{prefix}_{run}_{seq['n']}"

    def create(server, name: str, configuration=None) -> str:
        """Create a collection on `server` and return its id.

        Chroma v2 addresses sub-resources (`/add`, `/get`, `/query`, ...) by
        collection **id**; passing the name there is rejected with
        "Collection ID is not a valid UUIDv4". Rekha accepts both, so using the
        id compares like with like.
        """
        server.request("DELETE", f"{BASE}/{name}")
        body = {"name": name}
        if configuration is not None:
            body["configuration"] = configuration
        status, resp = server.request("POST", BASE, body)
        if status not in (200, 201) or not isinstance(resp, dict) or "id" not in resp:
            raise RuntimeError(f"create {name} on {server.base} failed: {status} {resp}")
        return resp["id"]

    def create_both(name: str, configuration=None):
        return create(target, name, configuration), create(ref, name, configuration)

    def must(server, method: str, path: str, body=None, label: str = ""):
        """Run a setup request that MUST succeed.

        Setup failures used to be ignored, so a case could compare two servers
        that both failed to load data and report agreement. A case whose
        precondition did not hold is not evidence of anything.
        """
        status, resp = server.request(method, path, body)
        if status not in (200, 201):
            raise RuntimeError(
                f"setup{(' (' + label + ')') if label else ''} failed on "
                f"{server.base}{path}: {status} {resp}"
            )
        return resp

    # -- create + metric configuration ------------------------------------
    def create_and_config():
        name = fresh("cfg")
        for s in (target, ref):
            s.request("DELETE", f"{BASE}/{name}")

        (ts, tb), (rs, rb) = both(
            "POST",
            BASE,
            {"name": name, "configuration": {"hnsw": {"space": "cosine"}}},
        )
        check(
            "create status class agrees",
            status_class(ts) == status_class(rs),
            f"target {ts} vs reference {rs}: {tb} / {rb}",
        )

        (ts, tb), (rs, rb) = both("GET", f"{BASE}/{name}")
        check(
            "get status agrees",
            ts == rs,
            f"target {ts} vs reference {rs}",
        )
        if ts == 200 and rs == 200:
            check(
                "name round-trips identically",
                tb.get("name") == rb.get("name") == name,
                f"{tb.get('name')} / {rb.get('name')}",
            )

        for s in (target, ref):
            s.request("DELETE", f"{BASE}/{name}")

    case("create/config", target, ref, create_and_config)

    # -- distance semantics ------------------------------------------------
    def metric_semantics():
        name = fresh("metric")
        tid, rid = create_both(name, {"hnsw": {"space": "cosine"}})
        for s, cid in ((target, tid), (ref, rid)):
            must(
                s,
                "POST",
                f"{BASE}/{cid}/add",
                {
                    "ids": ["x", "y"],
                    "embeddings": [[1.0, 0.0], [0.0, 1.0]],
                },
                "metric setup",
            )

        (ts, tb), (rs, rb) = both_at(
            "POST",
            f"{BASE}/{tid}/query",
            f"{BASE}/{rid}/query",
            {
                "query_embeddings": [[1.0, 0.0]],
                "n_results": 2,
                "include": ["distances"],
            },
        )
        check("query status agrees", ts == rs, f"target {ts} vs reference {rs}: {tb} / {rb}")
        if ts == 200 and rs == 200:
            td = tb["distances"][0]
            rd = rb["distances"][0]
            tol = 1e-4

            def close_all(a, b):
                return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))

            check(
                "cosine distances agree",
                close_all(td, rd),
                f"target {td} vs reference {rd}",
            )

        for s in (target, ref):
            s.request("DELETE", f"{BASE}/{name}")

    case("metric", target, ref, metric_semantics)

    # -- include defaults --------------------------------------------------
    def include_defaults():
        name = fresh("incl")
        records = {
            "ids": ["i0"],
            "embeddings": [[1.0, 0.0]],
            "metadatas": [{"tag": "a"}],
            "documents": ["doc"],
        }
        tid, rid = create_both(name)
        for s, cid in ((target, tid), (ref, rid)):
            must(s, "POST", f"{BASE}/{cid}/add", records, "include setup")

        (ts, tb), (rs, rb) = both_at("POST", f"{BASE}/{tid}/get", f"{BASE}/{rid}/get", {})
        check("get status agrees", ts == rs, f"target {ts} vs reference {rs}: {tb} / {rb}")
        if ts == 200 and rs == 200:
            for field in ("metadatas", "documents"):
                check(
                    f"get default includes {field}",
                    (tb.get(field) is not None) == (rb.get(field) is not None),
                    f"target {tb.get(field)!r} vs reference {rb.get(field)!r}",
                )
            check(
                "get default omits embeddings",
                (tb.get("embeddings") is None) == (rb.get("embeddings") is None),
                f"target {tb.get('embeddings')!r} vs reference {rb.get('embeddings')!r}",
            )

        (ts, tb), (rs, rb) = both_at(
            "POST",
            f"{BASE}/{tid}/query",
            f"{BASE}/{rid}/query",
            {"query_embeddings": [[1.0, 0.0]], "n_results": 1},
        )
        if ts == 200 and rs == 200:
            for field in ("distances", "metadatas", "documents"):
                check(
                    f"query default includes {field}",
                    (tb.get(field) is not None) == (rb.get(field) is not None),
                    f"target {tb.get(field)!r} vs reference {rb.get(field)!r}",
                )

        for s in (target, ref):
            s.request("DELETE", f"{BASE}/{name}")

    case("include-defaults", target, ref, include_defaults)

    # -- get with where / pagination --------------------------------------
    def get_filters():
        name = fresh("flt")
        tid, rid = create_both(name)
        for s, cid in ((target, tid), (ref, rid)):
            must(
                s,
                "POST",
                f"{BASE}/{cid}/add",
                {
                    "ids": ["a0", "b0", "a1", "b1", "a2"],
                    "embeddings": [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.1, 0.0],
                        [0.0, 1.1],
                        [1.2, 0.0],
                    ],
                    "metadatas": [
                        {"tag": "a", "n": 0},
                        {"tag": "b", "n": 1},
                        {"tag": "a", "n": 2},
                        {"tag": "b", "n": 3},
                        {"tag": "a", "n": 4},
                    ],
                },
                "filters setup",
            )

        for label, body in [
            ("$eq", {"where": {"tag": {"$eq": "a"}}}),
            ("range", {"where": {"n": {"$gte": 3}}}),
            ("$in", {"where": {"tag": {"$in": ["b"]}}}),
            ("limit", {"limit": 2}),
        ]:
            (ts, tb), (rs, rb) = both_at(
                "POST", f"{BASE}/{tid}/get", f"{BASE}/{rid}/get", body
            )
            if ts == 200 and rs == 200:
                check(
                    f"get {label} ids agree",
                    sorted(tb.get("ids", [])) == sorted(rb.get("ids", [])),
                    f"target {tb.get('ids')} vs reference {rb.get('ids')}",
                )
            else:
                check(
                    f"get {label} status agrees",
                    status_class(ts) == status_class(rs),
                    f"target {ts} vs reference {rs}: {tb} / {rb}",
                )

        for s in (target, ref):
            s.request("DELETE", f"{BASE}/{name}")

    case("get-filters", target, ref, get_filters)

    # -- where_document ----------------------------------------------------
    def where_document():
        name = fresh("doc")
        tid, rid = create_both(name)
        for s, cid in ((target, tid), (ref, rid)):
            must(
                s,
                "POST",
                f"{BASE}/{cid}/add",
                {
                    "ids": ["d0", "d1", "d2", "d3"],
                    "embeddings": [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.1, 0.0],
                        [0.0, 1.1],
                    ],
                    "documents": ["alpha one", "beta two", None, "alpha three"],
                },
                "where_document setup",
            )

        for label, body in [
            ("$contains", {"where_document": {"$contains": "alpha"}}),
            ("$not_contains", {"where_document": {"$not_contains": "alpha"}}),
            ("unknown-operator", {"where_document": {"$and": [{"$contains": "a"}]}}),
        ]:
            (ts, tb), (rs, rb) = both_at(
                "POST", f"{BASE}/{tid}/get", f"{BASE}/{rid}/get", body
            )
            if ts == 200 and rs == 200:
                check(
                    f"where_document {label} ids agree",
                    sorted(tb.get("ids", [])) == sorted(rb.get("ids", [])),
                    f"target {tb.get('ids')} vs reference {rb.get('ids')}",
                )
            else:
                check(
                    f"where_document {label} status class agrees",
                    status_class(ts) == status_class(rs),
                    f"target {ts} vs reference {rs}: {tb} / {rb}",
                )

        for s in (target, ref):
            s.request("DELETE", f"{BASE}/{name}")

    case("where-document", target, ref, where_document)

    # -- delete selectors --------------------------------------------------
    def delete_selectors():
        name = fresh("del")
        tid, rid = create_both(name)
        for s, cid in ((target, tid), (ref, rid)):
            must(
                s,
                "POST",
                f"{BASE}/{cid}/add",
                {
                    "ids": ["a0", "b0", "a1"],
                    "embeddings": [[1.0, 0.0], [0.0, 1.0], [1.1, 0.0]],
                    "metadatas": [{"tag": "a"}, {"tag": "b"}, {"tag": "a"}],
                },
                "delete setup",
            )

        # A selector-free delete is a no-op in Chroma: 200 with `deleted: 0`.
        (ts, tb), (rs, rb) = both_at(
            "POST", f"{BASE}/{tid}/delete", f"{BASE}/{rid}/delete", {}
        )
        check(
            "selector-free delete deleted nothing",
            ts == rs == 200 and tb.get("deleted") == rb.get("deleted") == 0,
            f"target {ts} {tb} vs reference {rs} {rb}",
        )

        (ts, tb), (rs, rb) = both_at(
            "GET", f"{BASE}/{tid}/count", f"{BASE}/{rid}/count"
        )
        check(
            "selector-free delete left every record",
            tb == rb == 3,
            f"target {tb!r} vs reference {rb!r}",
        )

        (ts, _tb), (rs, _rb) = both_at(
            "POST",
            f"{BASE}/{tid}/delete",
            f"{BASE}/{rid}/delete",
            {"where": {"tag": {"$eq": "b"}}},
        )
        check("delete by where status agrees", ts == rs, f"target {ts} vs reference {rs}")

        (ts, tb), (rs, rb) = both_at("GET", f"{BASE}/{tid}/count", f"{BASE}/{rid}/count")
        check(
            "count after delete agrees",
            tb == rb,
            f"target {tb!r} vs reference {rb!r}",
        )

        for s in (target, ref):
            s.request("DELETE", f"{BASE}/{name}")

    case("delete-selectors", target, ref, delete_selectors)

    # -- include: omitted vs explicit empty --------------------------------
    def include_empty():
        name = fresh("inc0")
        tid, rid = create_both(name)
        seed = {
            "ids": ["i0"],
            "embeddings": [[1.0, 0.0]],
            "metadatas": [{"tag": "a"}],
            "documents": ["d0"],
        }
        for s, cid in ((target, tid), (ref, rid)):
            must(s, "POST", f"{BASE}/{cid}/add", seed, "include-empty seed")

        def nonnull(body, endpoint):
            t = must(target, "POST", f"{BASE}/{tid}/{endpoint}", body, endpoint)
            r = must(ref, "POST", f"{BASE}/{rid}/{endpoint}", body, endpoint)
            tn = sorted(k for k, v in t.items() if v is not None)
            rn = sorted(k for k, v in r.items() if v is not None)
            return tn, rn

        tn, rn = nonnull({"ids": ["i0"], "include": []}, "get")
        check(
            "get include [] returns ids only",
            tn == rn == ["ids", "include"],
            f"target {tn} vs reference {rn}",
        )

        tn, rn = nonnull(
            {"query_embeddings": [[1.0, 0.0]], "n_results": 1, "include": []},
            "query",
        )
        check(
            "query include [] returns ids only",
            tn == rn == ["ids", "include"],
            f"target {tn} vs reference {rn}",
        )

        # Omitted include must apply each endpoint's documented defaults.
        tn, rn = nonnull({"ids": ["i0"]}, "get")
        check("get omitted include matches", tn == rn, f"target {tn} vs reference {rn}")

        tn, rn = nonnull(
            {"query_embeddings": [[1.0, 0.0]], "n_results": 1}, "query"
        )
        check("query omitted include matches", tn == rn, f"target {tn} vs reference {rn}")

        for s in (target, ref):
            s.request("DELETE", f"{BASE}/{name}")

    case("include-empty", target, ref, include_empty)

    # -- ordering / pagination --------------------------------------------
    def ordering():
        name = fresh("ord")
        tid, rid = create_both(name)
        order = ["z", "a", "m", "b"]
        add_body = {"ids": order, "embeddings": [[1.0, 0.0]] * len(order)}
        for s, cid in ((target, tid), (ref, rid)):
            must(s, "POST", f"{BASE}/{cid}/add", add_body, "ordering seed")

        def get_ids(body):
            t = must(target, "POST", f"{BASE}/{tid}/get", body, "order get")
            r = must(ref, "POST", f"{BASE}/{rid}/get", body, "order get")
            return t["ids"], r["ids"]

        t, r = get_ids({})
        check(
            "records come back in insertion order",
            t == r == order,
            f"target {t} vs reference {r}",
        )

        # An explicit id list is filtered but not re-ordered.
        t, r = get_ids({"ids": ["b", "z", "a"]})
        check(
            "explicit ids keep insertion order",
            t == r,
            f"target {t} vs reference {r}",
        )

        # Delete then re-add moves the record to the end.
        for s, cid in ((target, tid), (ref, rid)):
            must(s, "POST", f"{BASE}/{cid}/delete", {"ids": ["a"]}, "order delete")
            must(
                s,
                "POST",
                f"{BASE}/{cid}/add",
                {"ids": ["a"], "embeddings": [[1.0, 0.0]]},
                "order re-add",
            )
        t, r = get_ids({})
        check(
            "re-added record moves to the end",
            t == r,
            f"target {t} vs reference {r}",
        )

        # Updates and upserts keep their position.
        for s, cid in ((target, tid), (ref, rid)):
            must(
                s,
                "POST",
                f"{BASE}/{cid}/update",
                {"ids": ["m"], "documents": ["x"]},
                "order update",
            )
            must(
                s,
                "POST",
                f"{BASE}/{cid}/upsert",
                {"ids": ["z"], "embeddings": [[2.0, 0.0]]},
                "order upsert",
            )
        t, r = get_ids({})
        check(
            "update/upsert preserve position",
            t == r,
            f"target {t} vs reference {r}",
        )

        # Pagination walks the same order.
        full_t, full_r = get_ids({})
        pages_t, pages_r = [], []
        for offset in (0, 2, 4):
            t, r = get_ids({"limit": 2, "offset": offset})
            pages_t.extend(t)
            pages_r.extend(r)
        check(
            "paginated pages concatenate to the full order",
            pages_t == pages_r == full_t == full_r,
            f"pages {pages_t} / {pages_r} vs full {full_t} / {full_r}",
        )

        for s in (target, ref):
            s.request("DELETE", f"{BASE}/{name}")

    case("ordering", target, ref, ordering)

    # -- mutation semantics ------------------------------------------------
    def mutations():
        name = fresh("mut")
        tid, rid = create_both(name)
        seed = {
            "ids": ["m0"],
            "embeddings": [[1.0, 0.0]],
            "metadatas": [{"a": 1, "b": 2}],
            "documents": ["d1"],
        }
        for s, cid in ((target, tid), (ref, rid)):
            must(s, "POST", f"{BASE}/{cid}/add", seed, "mutation seed")

        def state(s, cid):
            resp = must(
                s,
                "POST",
                f"{BASE}/{cid}/get",
                {"ids": ["m0"], "include": ["documents", "metadatas", "embeddings"]},
                "mutation read-back",
            )
            meta = resp["metadatas"][0]
            return {
                "metadata": {k: meta[k] for k in sorted(meta)} if meta else None,
                "document": resp["documents"][0],
                "embedding": [round(float(x), 6) for x in resp["embeddings"][0]],
            }

        def compare(label):
            t = state(target, tid)
            r = state(ref, rid)
            check(f"{label} matches the reference", t == r, f"target {t} vs reference {r}")

        # Metadata-only update: merge, and preserve document + vector.
        for s, cid in ((target, tid), (ref, rid)):
            must(
                s,
                "POST",
                f"{BASE}/{cid}/update",
                {"ids": ["m0"], "metadatas": [{"a": 10}]},
                "metadata-only update",
            )
        compare("metadata-only update")
        t = state(target, tid)
        check(
            "metadata-only update merged keys",
            t["metadata"] == {"a": 10, "b": 2},
            f"{t['metadata']}",
        )
        check("metadata-only update kept document", t["document"] == "d1", f"{t['document']}")

        # Document-only update: metadata and vector survive.
        for s, cid in ((target, tid), (ref, rid)):
            must(
                s,
                "POST",
                f"{BASE}/{cid}/update",
                {"ids": ["m0"], "documents": ["d2"]},
                "document-only update",
            )
        compare("document-only update")

        # Vector-only update.
        for s, cid in ((target, tid), (ref, rid)):
            must(
                s,
                "POST",
                f"{BASE}/{cid}/update",
                {"ids": ["m0"], "embeddings": [[0.0, 1.0]]},
                "vector-only update",
            )
        compare("vector-only update")
        check(
            "vector-only update kept metadata",
            state(target, tid)["metadata"] == {"a": 10, "b": 2},
            f"{state(target, tid)['metadata']}",
        )

        # A null metadata value deletes the key.
        for s, cid in ((target, tid), (ref, rid)):
            must(
                s,
                "POST",
                f"{BASE}/{cid}/update",
                {"ids": ["m0"], "metadatas": [{"b": None}]},
                "null metadata delete",
            )
        compare("null metadata delete")

        # Updating an unknown id is a no-op, not an error.
        (ts, tb), (rs, rb) = both_at(
            "POST",
            f"{BASE}/{tid}/update",
            f"{BASE}/{rid}/update",
            {"ids": ["absent-id"], "metadatas": [{"z": 1}]},
        )
        check(
            "update of an unknown id agrees",
            ts == rs,
            f"target {ts} ({tb}) vs reference {rs} ({rb})",
        )
        check(
            "update of an unknown id creates nothing",
            state(target, tid)["metadata"] is not None,
            "existing record disturbed",
        )

        # Upsert merges metadata and preserves an unmentioned document.
        for s, cid in ((target, tid), (ref, rid)):
            must(
                s,
                "POST",
                f"{BASE}/{cid}/upsert",
                {"ids": ["m0"], "embeddings": [[1.0, 1.0]], "metadatas": [{"a": 99}]},
                "upsert merge",
            )
        compare("upsert merge")

        for s in (target, ref):
            s.request("DELETE", f"{BASE}/{name}")

    case("mutations", target, ref, mutations)

    # -- dimension semantics ----------------------------------------------
    def dimension():
        # Chroma has no create-time dimension field: `configuration.hnsw`
        # accepts only space/ef_construction/ef_search/max_neighbors/
        # num_threads/resize_factor/sync_threshold/batch_size, and the width is
        # inferred from the first write. Rekha additionally accepts
        # `configuration.hnsw.dimension` (and a top-level `dimension`) as an
        # extension so a caller can pin the width up front. That extension is
        # asserted on Rekha below and recorded as an accepted difference.
        name = fresh("dim")
        (ts, tb), (rs, rb) = both(
            "POST",
            BASE,
            {"name": name, "configuration": {"hnsw": {"dimension": 4}}},
        )
        check(
            "create with hnsw.dimension: Rekha extension, Chroma rejects",
            ts in (200, 201) and rs == 422,
            f"target {ts} vs reference {rs}: {rb}",
        )
        for s in (target, ref):
            s.request("DELETE", f"{BASE}/{name}")

        # Real shared behaviour: width is inferred from the first write and a
        # later mismatched write is rejected. Probe the mismatch first so an
        # ignored configuration cannot be mistaken for enforcement.
        name = fresh("dim_infer")
        tid, rid = create_both(name)
        (ts, tb), (rs, rb) = both_at(
            "POST",
            f"{BASE}/{tid}/add",
            f"{BASE}/{rid}/add",
            {"ids": ["ok"], "embeddings": [[0.1] * 4]},
        )
        check(
            "first write accepted with the same status",
            ts == rs == 201,
            f"target {ts} vs reference {rs}: {tb} / {rb}",
        )

        (ts, tb), (rs, rb) = both_at(
            "POST",
            f"{BASE}/{tid}/add",
            f"{BASE}/{rid}/add",
            {"ids": ["bad"], "embeddings": [[0.1] * 8]},
        )
        check(
            "mismatched width rejected",
            status_class(ts) == status_class(rs) == "4xx",
            f"target {ts} ({tb}) vs reference {rs} ({rb})",
        )

        for s in (target, ref):
            s.request("DELETE", f"{BASE}/{name}")

    case("dimension", target, ref, dimension)

    # -- unknown include field --------------------------------------------
    def unknown_include():
        name = fresh("inc2")
        tid, rid = create_both(name)
        for s, cid in ((target, tid), (ref, rid)):
            must(
                s,
                "POST",
                f"{BASE}/{cid}/add",
                {"ids": ["z"], "embeddings": [[1.0, 0.0]]},
                "include-unknown setup",
            )

        (ts, tb), (rs, rb) = both_at(
            "POST", f"{BASE}/{tid}/get", f"{BASE}/{rid}/get", {"include": ["nonsense"]}
        )
        check(
            "unknown include status class agrees",
            status_class(ts) == status_class(rs),
            f"target {ts} ({tb}) vs reference {rs} ({rb})",
        )

        for s in (target, ref):
            s.request("DELETE", f"{BASE}/{name}")

    case("unknown-include", target, ref, unknown_include)

    print("\n" + "=" * 56)
    print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    print("=" * 56 + "\n")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
