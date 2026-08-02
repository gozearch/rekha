#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "================================================"
echo " Rekha E2E Test — Docker Compose 3-node cluster"
echo "================================================"

cleanup() {
    echo ""; echo "Cleaning up..."
    docker compose down -v 2>/dev/null || true
}
trap cleanup EXIT

echo ""; echo "[1/5] Building Docker image..."
docker build -t rekha:latest -f Dockerfile .

echo ""; echo "[2/5] Starting 3-node cluster..."
docker compose up -d --wait --wait-timeout 120

echo ""; echo "[3/5] Checking node health..."
for i in 1 2 3; do
    echo -n "  node-$i: "
    if docker exec rekha-node-$i rekha health; then
        echo "       healthy"
    else
        echo "       unhealthy"
        docker logs rekha-node-$i --tail 20
        exit 1
    fi
done

DIM=8
echo ""; echo "[4/5] Inserting 100 vectors (dim=$DIM) into 'default' collection..."
for i in $(seq 1 100); do
    VEC=$(python3 -c "import random; print(' '.join(str(round(random.random(),6)) for _ in range($DIM)))")
    echo "$VEC" | docker exec -i rekha-node-1 rekha insert -c default
done
echo "  Inserted 100 vectors"

echo ""; echo "[5/5] Searching from node-2..."
QUERY=$(python3 -c "print(' '.join('0.5' for _ in range($DIM)))")
RESULTS=$(echo "$QUERY" | docker exec -i rekha-node-2 rekha search -k 5 -c default)
echo "$RESULTS"

RESULT_COUNT=$(echo "$RESULTS" | grep -cP '^\s+\d+\.' || true)
if [ "$RESULT_COUNT" -ge 1 ]; then
    echo ""; echo "================================================"
    echo " SUCCESS: Retrieved $RESULT_COUNT results"
    echo "================================================"
else
    echo "ERROR: No search results returned"
    exit 1
fi
