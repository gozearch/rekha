#!/bin/zsh
# End-to-end failover check for a real three-process cluster.
#
# Asserts the C0a contract with actual processes and actual HTTP:
#   1. a collection created on a follower exists on every node
#   2. a write sent to a follower is applied by the leader and replicates
#   3. killing the leader elects a new one
#   4. writes continue through a survivor, and every survivor converges
set -u

BIN=/Users/sathyanarrayanan/gozearch/rekha-db/target/debug/rekha
BASE=/tmp/rekha-cluster-failover
K=smoke-test-key-0123456789abcdef012345
SECRET=smoke-cluster-secret-0123456789abcdef
P1=18101; P2=18102; P3=18103
PEERS="1@127.0.0.1:$P1,2@127.0.0.1:$P2,3@127.0.0.1:$P3"
C="api/v2/tenants/default_tenant/databases/default_database/collections"
FAIL=0

# SIGKILL and wait: since the server drains on SIGTERM, a polite kill leaves the
# previous run holding its ports while the next one tries to bind them.
cleanup() {
  pkill -9 -f "rekha serve --data-dir $BASE" 2>/dev/null
  while pgrep -f "rekha serve --data-dir $BASE" >/dev/null 2>&1; do sleep 0.05; done
}
trap cleanup EXIT
cleanup; sleep 0.5; rm -rf "$BASE"; mkdir -p "$BASE"

start() {
  REKHA_API_KEY=$K REKHA_CLUSTER_SECRET=$SECRET \
    "$BIN" serve --data-dir "$BASE/n$1" --host 127.0.0.1 --port "$2" \
      --node-id "$1" --peers "$PEERS" ${3:-} > "$BASE/n$1.log" 2>&1 &
}

port() { case $1 in 1) echo $P1;; 2) echo $P2;; 3) echo $P3;; esac }
count() { curl -s -m 3 -H "x-chroma-token: $K" "http://127.0.0.1:$(port $1)/$C/docs/count" 2>/dev/null; }
leader_of() {
  curl -s -m 2 -H "x-chroma-token: $K" "http://127.0.0.1:$(port $1)/ready" 2>/dev/null |
    python3 -c 'import sys,json
try: print((json.load(sys.stdin).get("cluster") or {}).get("leader") or "")
except Exception: print("")'
}
check() { # name expected actual
  if [ "$2" = "$3" ]; then echo "  PASS $1 ($3)"; else echo "  FAIL $1: expected $2, got $3"; FAIL=1; fi
}

start 1 $P1 --bootstrap; start 2 $P2; start 3 $P3

for i in $(seq 1 100); do
  L=$(leader_of 1); [ -n "$L" ] && break; sleep 0.1
done
echo "elected leader=$L"
check "leader exists" 1 "$L"

# Every node must agree on the same leader before the failover case means
# anything. A half-formed cluster makes every later check pass for the wrong
# reason.
for n in 1 2 3; do
  seen=$(leader_of $n)
  check "node $n agrees on the leader" "$L" "$seen"
done

echo "create collection on follower 2"
curl -s -o /dev/null -X POST -H "x-chroma-token: $K" -H 'content-type: application/json' \
  -d '{"name":"docs","configuration":{"hnsw":{"space":"l2","dimension":4}}}' \
  "http://127.0.0.1:$P2/$C"
sleep 1
for n in 1 2 3; do
  seen=$(curl -s -H "x-chroma-token: $K" "http://127.0.0.1:$(port $n)/$C" | python3 -c 'import sys,json;print(len(json.load(sys.stdin)))' 2>/dev/null)
  check "collection visible on node $n" 1 "$seen"
done

echo "write via follower 3"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST -H "x-chroma-token: $K" -H 'content-type: application/json' \
  -d '{"ids":["a","b"],"embeddings":[[1,0,0,0],[0,1,0,0]]}' \
  "http://127.0.0.1:$P3/$C/docs/add")
check "add accepted" 201 "$code"
sleep 2
for n in 1 2 3; do check "count on node $n after write" 2 "$(count $n)"; done

OLD=$L
echo "killing leader $OLD with SIGKILL"
KILLED_AT=$(python3 -c 'import time;print(time.time())')
# SIGKILL, not SIGTERM: since the server now drains on SIGTERM, a polite signal
# would no longer exercise an abrupt death.
# Resolve from the command line rather than a captured $!: a pid that silently
# points elsewhere turns every later check into a false pass.
LEADER_PID=$(pgrep -f "rekha serve --data-dir $BASE/n$OLD" | head -1)
if [ -z "$LEADER_PID" ]; then echo "  FAIL could not resolve the leader's pid"; FAIL=1; fi
kill -9 "$LEADER_PID" 2>/dev/null || true
# A kill that silently failed would leave the old leader serving, and every
# check after this point would pass for the wrong reason.
for _ in $(seq 1 100); do kill -0 "$LEADER_PID" 2>/dev/null || break; sleep 0.02; done
if kill -0 "$LEADER_PID" 2>/dev/null; then echo "  FAIL leader $OLD ($LEADER_PID) survived SIGKILL"; FAIL=1; fi
# A killed process whose port still answers means this is not the node the
# survivors were talking to, and the rest of the run would be meaningless.
if curl -s -m 1 -o /dev/null "http://127.0.0.1:$(port $OLD)/ready"; then
  echo "  FAIL port $(port $OLD) still serves after killing node $OLD"
  FAIL=1
fi

NEW=""
for i in $(seq 1 120); do
  for n in 1 2 3; do
    [ "$n" = "$OLD" ] && continue
    l=$(leader_of $n); [ -n "$l" ] && [ "$l" != "$OLD" ] && { NEW=$l; break 2; }
  done
  sleep 0.25
done
echo "new leader=$NEW"
check "a new leader was elected" 1 "$([ -n "$NEW" ] && echo 1 || echo 0)"
if [ -z "$NEW" ]; then
  for n in 1 2 3; do
    [ "$n" = "$OLD" ] && continue
    echo "  --- node $n last log lines ---"
    tail -3 "$BASE/n$n.log"
  done
fi
python3 -c "import time;print(f'  failover latency: {time.time()-$KILLED_AT:.2f}s')"

SURV=()
for n in 1 2 3; do [ "$n" = "$OLD" ] || SURV+=($n); done
TARGET=$SURV[1]
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST -H "x-chroma-token: $K" -H 'content-type: application/json' \
  -d '{"ids":["c"],"embeddings":[[0,0,1,0]]}' \
  "http://127.0.0.1:$(port $TARGET)/$C/docs/add")
check "write after failover accepted via node $TARGET" 201 "$code"

# Convergence: every survivor must reach 3 records.
for i in $(seq 1 40); do
  ok=1
  for n in $SURV; do [ "$(count $n)" = "3" ] || ok=0; done
  [ $ok -eq 1 ] && break
  sleep 0.5
done
for n in $SURV; do check "survivor $n converged" 3 "$(count $n)"; done

# The gap C0a could not close: a node that was down while another node led had
# missed writes that existed only in the deposed leader's WAL. Now it replays the
# log and converges with no operator action.
echo "restarting the node that was killed, with no catch-up command"
start $OLD $(port $OLD)
converged=0
for i in $(seq 1 120); do
  if [ "$(count $OLD)" = "3" ]; then converged=1; break; fi
  sleep 0.25
done
check "restarted node $OLD caught up from the log" 1 "$converged"

echo
[ $FAIL -eq 0 ] && echo "ALL CHECKS PASSED" || echo "FAILURES PRESENT"
exit $FAIL
