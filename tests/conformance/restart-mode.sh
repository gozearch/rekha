#!/usr/bin/env bash
# Restart the server on the same data directory and verify one recovery path.
#
#   bash tests/conformance/restart-mode.sh {wal|checkpoint|tail}
#
# Splitting the modes is the point: a run that only ever restarts after a
# replay proves nothing about the checkpoint publication path, and vice versa.
set -euo pipefail

MODE="${1:?usage: restart-mode.sh wal|checkpoint|tail}"
DATA_DIR="${DATA_DIR:-/tmp/rekha-conf}"
PIDFILE="${PIDFILE:-/tmp/rekha.pid}"
PORT="${PORT:-8123}"
KEY="${REKHA_KEY:-conformance-key-0123456789abcdef}"
BIN="${BIN:-./target/debug/rekha}"
CLIENT="chromadb==${CHROMADB_VERSION:-1.5.9}"

wait_ready() {
  for _ in $(seq 1 60); do
    if curl -sf -H "x-chroma-token: $KEY" \
        "http://127.0.0.1:$PORT/api/v2/heartbeat" > /dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "server did not become ready" >&2
  tail -50 "$LOG" >&2 || true
  return 1
}

stop_server() {
  if [ -f "$PIDFILE" ]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break
      sleep 1
    done
  fi
}

LOG="/tmp/rekha-server-restart-${MODE}.log"

echo "== restart mode: $MODE =="
wait_ready

uv run --no-project --with "$CLIENT" \
  python tests/conformance/restart.py write \
  --mode "$MODE" --host 127.0.0.1 --port "$PORT" --token "$KEY" \
  --data-dir "$DATA_DIR"

stop_server

REKHA_API_KEY="$KEY" "$BIN" serve \
  --data-dir "$DATA_DIR" --host 127.0.0.1 --port "$PORT" \
  > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

wait_ready

uv run --no-project --with "$CLIENT" \
  python tests/conformance/restart.py verify \
  --host 127.0.0.1 --port "$PORT" --token "$KEY" \
  --data-dir "$DATA_DIR"
