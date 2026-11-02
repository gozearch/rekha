#!/usr/bin/env python3
"""Write latency through a real three-node cluster.

Validates the claim that moving checkpoints off the write path removes the p99
tail. Before C0b a checkpoint ran inside the write call and held the collection
mutex, which is what made p50 ~21ms and p99 ~435ms at 40k records.

Usage: run against an already-running cluster; see the shell wrapper.
"""

import json
import statistics
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18201"
KEY = "smoke-test-key"
TENANT = "default_tenant"
DATABASE = "default_database"
COLL = "latency"
DIM = 128
RECORDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
BATCH = 50
# The default sync_threshold, so checkpoints actually fire during the run.
SYNC_THRESHOLD = 1000
# Write to the last node on purpose: it is a follower, so every request is
# forwarded to the leader and the number includes the forwarding hop.
WRITE_PORT = sys.argv[3] if len(sys.argv) > 3 else "18202"


def call(method, path, body=None, port=None):
    url = f"http://127.0.0.1:{port or WRITE_PORT}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("x-chroma-token", KEY)
    req.add_header("content-type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else None


def collection_path():
    return f"/api/v2/tenants/{TENANT}/databases/{DATABASE}/collections"


def main():
    try:
        call("POST", collection_path(), {
            "name": COLL,
            "configuration": {"hnsw": {"space": "l2", "dimension": DIM}},
        })
    except urllib.error.HTTPError as e:
        if e.code != 409:
            raise

    cid = COLL
    latencies = []
    written = 0
    while written < RECORDS:
        n = min(BATCH, RECORDS - written)
        ids = [f"r{written + i}" for i in range(n)]
        # Vectors are cheap to build but not degenerate, so the distance work is
        # representative of a real write.
        embs = [[((hash(i) + j) % 97) / 97.0 for j in range(DIM)] for i in range(n, n + 1)]
        embs = [[((written + i + j) % 89) / 89.0 for j in range(DIM)] for i in range(n)]
        body = {"ids": ids, "embeddings": embs}

        started = time.perf_counter()
        call("POST", f"{collection_path()}/{cid}/add", body)
        latencies.append((time.perf_counter() - started) * 1000.0)
        written += n

    latencies.sort()

    def pct(p):
        idx = min(len(latencies) - 1, int(len(latencies) * p))
        return latencies[idx]

    count = None
    try:
        count = call("GET", f"{collection_path()}/{cid}/count", port=WRITE_PORT)
    except Exception:
        pass

    print(json.dumps({
        "records": written,
        "batch": BATCH,
        "writes": len(latencies),
        "sync_threshold": SYNC_THRESHOLD,
        "p50_ms": round(pct(0.50), 2),
        "p95_ms": round(pct(0.95), 2),
        "p99_ms": round(pct(0.99), 2),
        "max_ms": round(latencies[-1], 2),
        "mean_ms": round(statistics.fmean(latencies), 2),
        "count_reported": count,
    }, indent=2))


if __name__ == "__main__":
    main()
