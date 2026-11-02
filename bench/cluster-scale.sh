#!/usr/bin/env bash
#
# Ten-node placement qualification.
#
# The Rust integration test proves placement exactly on four nodes, which is the
# smallest cluster where it distributes. This script answers the other question:
# does the same design hold when the metadata group has ten voters and every
# node runs a dozen shard groups?
#
# What it asserts, all from outside the processes:
#
#   1. Every node is ready and every shard has an elected leader.
#   2. For C collections, the total number of collection data files across the
#      cluster is exactly 3C — i.e. every collection exists on exactly three
#      nodes, not on all ten. No hash is reimplemented here: the sum is exact
#      without knowing which nodes.
#   3. No node holds every collection (the whole point of placement).
#   4. Killing the metadata leader does not stop the cluster: a new collection
#      can still be created and written.
#
#   5. No node holds materially more of the corpus than another, once the
#      cluster is large enough for placement to have a choice (see below).
#
# The balance bound is asserted only for NODES >= 6. With three replicas the
# possible replica sets are the shard count's owner triples; at four nodes there
# are exactly four of them, and rendezvous does not hand them out evenly, so a
# small cluster can be legitimately lopsided no matter how collections are
# assigned. The exact per-collection gate is the Rust test at ten nodes.
#
# Usage: ./bench/cluster-scale.sh [node_count] [collection_count]
set -euo pipefail

NODES=${1:-10}
COLLECTIONS=${2:-120}
BASE=${REKHA_SCALE_DIR:-/tmp/rekha-scale}
BIN=${REKHA_BIN:-./target/debug/rekha}
KEY=scale-key-0123456789abcdef0123456
SECRET=scale-secret-0123456789abcdef0123456789
DIM=8
PORT_BASE=${REKHA_SCALE_PORT_BASE:-21100}
# Shard count is what every collection id was hashed by, so it is a creation
# parameter, not a tuning knob. Exposed here to measure how balance improves
# with more shards at a fixed node count.
SHARDS=${REKHA_SCALE_SHARDS:-$((NODES * 4))}

rm -rf "$BASE"
mkdir -p "$BASE"

pids=()
data_dirs=()
ports=()

cleanup() {
  for pid in "${pids[@]:-}"; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

peers=""
for i in $(seq 1 "$NODES"); do
  port=$((PORT_BASE + i))
  peers="${peers}${peers:+,}${i}@127.0.0.1:${port}"
done

echo "== starting $NODES nodes, $COLLECTIONS collections, shards=$SHARDS =="
for i in $(seq 1 "$NODES"); do
  port=$((PORT_BASE + i))
  dir="$BASE/n$i"
  args=(serve --data-dir "$dir" --host 127.0.0.1 --port "$port" --node-id "$i" --peers "$peers" --shards "$SHARDS")
  [ "$i" = 1 ] && args+=(--bootstrap)
  REKHA_API_KEY="$KEY" REKHA_CLUSTER_SECRET="$SECRET" \
  REKHA_HEARTBEAT_MS=200 REKHA_ELECTION_MIN_MS=800 REKHA_ELECTION_MAX_MS=1600 \
  RUST_LOG=error "$BIN" "${args[@]}" >"$BASE/n$i.log" 2>&1 &
  pids+=($!)
  data_dirs+=("$dir")
  ports+=("$port")
done

wait_for() {
  local label=$1 deadline=$((SECONDS + ${2:-120}))
  shift 2
  until "$@" >/dev/null 2>&1; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "FAIL: timed out waiting for $label"
      exit 1
    fi
    sleep 0.5
  done
}

node_ready() {
  curl -sf -H "x-chroma-token: $KEY" "http://127.0.0.1:$1/ready" | grep -q '"ready":true'
}

# Every node reports the shards it hosts; the union must cover every shard with
# a leader. A shard is an integer 0..shards-1.
all_shards_led() {
  local shards=$SHARDS i port
  local -A led=()
  for i in $(seq 1 "$NODES"); do
    port=$((PORT_BASE + i))
    while IFS=$'\t' read -r shard leader; do
      [ -n "$shard" ] && [ "$leader" != "null" ] && led[$shard]=1
    done < <(curl -sf -H "x-rekha-cluster-secret: $SECRET" \
      "http://127.0.0.1:$port/internal/cluster/membership" 2>/dev/null |
      python3 -c '
import json,sys
try:
    d=json.load(sys.stdin)
except Exception:
    sys.exit(0)
for s in (d.get("shards") or []):
    print("%s\t%s" % (s.get("shard"), s.get("leader")))
' 2>/dev/null || true)
  done
  [ "${#led[@]}" -eq "$shards" ]
}

for i in $(seq 1 "$NODES"); do
  wait_for "node $i ready" 120 node_ready "$((PORT_BASE + i))"
done
wait_for "every shard to elect" 180 all_shards_led

echo "== all $NODES nodes ready, $((NODES * 4)) shards led =="
grep -h "^Placement:" "$BASE"/n*.log | head -1
grep -h "^Shard groups:" "$BASE"/n1.log | head -1

started=$SECONDS
fail=0
for i in $(seq 0 $((COLLECTIONS - 1))); do
  body=$(curl -sf -X POST "http://127.0.0.1:$((PORT_BASE + 1))/api/v2/collections" \
    -H "x-chroma-token: $KEY" -H 'content-type: application/json' \
    -d "{\"name\":\"c$i\",\"dimension\":$DIM,\"distance\":\"l2\"}") || { fail=$((fail + 1)); continue; }
  id=$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
  curl -sf -o /dev/null -X POST "http://127.0.0.1:$((PORT_BASE + 1))/api/v2/collections/$id/add" \
    -H "x-chroma-token: $KEY" -H 'content-type: application/json' \
    -d "{\"ids\":[\"r0\"],\"embeddings\":[[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]]}" \
    || fail=$((fail + 1))
done
elapsed=$((SECONDS - started))
[ "$fail" -eq 0 ] || { echo "FAIL: $fail collection operations failed"; exit 1; }
echo "== created and wrote $COLLECTIONS collections in ${elapsed}s =="

# Every collection is one .wal file on each replica.
total=0
for dir in "${data_dirs[@]}"; do
  n=$(find "$dir/wal" -maxdepth 1 -name '*.wal' 2>/dev/null | wc -l | tr -d ' ')
  total=$((total + n))
done
expected=$((COLLECTIONS * 3))
echo "== collection data files across the cluster: $total (expected $expected) =="
if [ "$total" -ne "$expected" ]; then
  echo "FAIL: placement is not 3 replicas per collection"
  for i in "${!data_dirs[@]}"; do
    echo "  node $((i + 1)): $(find "${data_dirs[$i]}/wal" -maxdepth 1 -name '*.wal' 2>/dev/null | wc -l | tr -d ' ')"
  done
  exit 1
fi

counts=()
for i in "${!data_dirs[@]}"; do
  n=$(find "${data_dirs[$i]}/wal" -maxdepth 1 -name '*.wal' 2>/dev/null | wc -l | tr -d ' ')
  counts+=("$n")
  echo "  node $((i + 1)): $n collections ($(python3 -c "print(f'{$n * 100 / $COLLECTIONS:.0f}%')") of the corpus)"
  if [ "$n" -eq "$COLLECTIONS" ] && [ "$NODES" -gt 3 ]; then
    echo "FAIL: node $((i + 1)) holds every collection; placement is not distributing"
    exit 1
  fi
done

ideal=$((COLLECTIONS * 3 / NODES))
read -r lowest highest spread <<< "$(python3 -c 'import sys
c = list(map(int, sys.argv[1:]))
print(min(c), max(c), max(c) - min(c))' "${counts[@]}")"
echo "== per-node share: min $lowest / ideal $ideal / max $highest (spread $spread) =="
if [ "$NODES" -ge 6 ] && [ "$spread" -gt 2 ]; then
  echo "FAIL: placement spread $spread exceeds 2 at $NODES nodes ($COLLECTIONS collections);"
  echo "      the leader is not placing by node load"
  exit 1
fi

# Kill the metadata leader and require the cluster to keep accepting writes.
meta_leader=$(curl -sf -H "x-rekha-cluster-secret: $SECRET" \
  "http://127.0.0.1:$((PORT_BASE + 2))/internal/cluster/membership" |
  python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["leader"])') || {
  echo "FAIL: could not read the metadata leader"; exit 1; }
echo "== killing metadata leader (node $meta_leader) =="
kill -9 "${pids[$((meta_leader - 1))]}" 2>/dev/null || true
sleep 4

survivor=$((PORT_BASE + 1))
[ "$meta_leader" = 1 ] && survivor=$((PORT_BASE + 3))

# A metadata write during an election legitimately answers 503 with
# Retry-After; retrying is the contract, and the same rule the Rust tests follow.
# Without this the check measures how fast an election completes at this cluster
# size rather than whether the cluster recovers.
failover_deadline=$((SECONDS + 30))
body=""
while :; do
  body=$(curl -sf -X POST "http://127.0.0.1:$survivor/api/v2/collections" \
    -H "x-chroma-token: $KEY" -H 'content-type: application/json' \
    -d '{"name":"after-failover","dimension":8,"distance":"l2"}') && break
  if [ "$SECONDS" -ge "$failover_deadline" ]; then
    echo "FAIL: the cluster did not accept a metadata write within 30s of the leader dying"
    exit 1
  fi
  sleep 0.5
done
id=$(printf '%s' "$body" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
while ! curl -sf -o /dev/null -X POST "http://127.0.0.1:$survivor/api/v2/collections/$id/add" \
  -H "x-chroma-token: $KEY" -H 'content-type: application/json' \
  -d '{"ids":["r0"],"embeddings":[[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]]}'; do
  if [ "$SECONDS" -ge "$failover_deadline" ]; then
    echo "FAIL: could not write after the metadata leader died"; exit 1
  fi
  sleep 0.5
done
echo "== writes continue after losing the metadata leader =="

echo "ALL CHECKS PASSED"
