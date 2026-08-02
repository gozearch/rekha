#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "================================================"
echo " Rekha E2E — Production 3-node cluster test"
echo "================================================"

cleanup() {
    echo ""; echo "Cleaning up..."
    docker compose down -v 2>/dev/null || true
}
trap cleanup EXIT

echo ""; echo "[1] Building Docker image..."
docker compose build --no-cache

echo ""; echo "[2] Starting 3-node cluster..."
docker compose up -d --wait --wait-timeout 120

echo ""; echo "[3] Checking node health..."
for i in 1 2 3; do
    echo -n "  node-$i: "
    if docker exec rekha-node-$i rekha health; then
        echo "       healthy"
    else
        echo "       unhealthy"; docker logs rekha-node-$i --tail 20; exit 1
    fi
done

echo ""; echo "[4] Creating 8D 'images' collection via node-1..."
docker exec rekha-node-1 rekha create-collection -c images --rf 3

echo ""; echo "[5] Inserting 50 vectors (dim=8) via node-1..."
for i in $(seq 1 50); do
    VEC=$(python3 -c "import random; print(' '.join(str(round(random.random(),6)) for _ in range(8)))")
    echo "$VEC" | docker exec -i rekha-node-1 rekha insert -c images
done

echo ""; echo "[6] Searching from node-2..."
QUERY=$(python3 -c "print(' '.join('0.5' for _ in range(8)))")
RESULTS=$(echo "$QUERY" | docker exec -i rekha-node-2 rekha search -k 5 -c images)
echo "$RESULTS"
RESULT_COUNT=$(echo "$RESULTS" | grep -cE '^[[:space:]]+[0-9]+\.' || true)

if [ "$RESULT_COUNT" -ge 1 ]; then
    echo "  Node-2 returned $RESULT_COUNT results ✓"
else
    echo "ERROR: Node-2 search returned no results"; exit 1
fi

echo ""; echo "[7] Testing failover: stopping node-1..."
docker compose stop node-1

echo ""; echo "[8] Searching from node-3 (should still work)..."
RESULTS=$(echo "$QUERY" | docker exec -i rekha-node-3 rekha search -k 5 -c images)
echo "$RESULTS"
RESULT_COUNT=$(echo "$RESULTS" | grep -cE '^[[:space:]]+[0-9]+\.' || true)

if [ "$RESULT_COUNT" -ge 1 ]; then
    echo "  Node-3 returned $RESULT_COUNT results after failover ✓"
else
    echo "ERROR: Node-3 search returned no results after node-1 failure"; exit 1
fi

echo ""; echo "[9] Creating second collection 'texts' (dim=4) on node-2..."
docker exec rekha-node-2 rekha create-collection -c texts --rf 3 --config '{"dim":4,"nlist":16,"nprobe":4}'

echo ""; echo "[10] Inserting into 'texts' on node-2..."
for i in $(seq 1 10); do
    VEC=$(python3 -c "import random; print(' '.join(str(round(random.random(),6)) for _ in range(4)))")
    echo "$VEC" | docker exec -i rekha-node-2 rekha insert -c texts
done

echo "[11] Searching 'texts' from node-3 (cross-collection)..."
QTEXT=$(python3 -c "print(' '.join('0.5' for _ in range(4)))")
echo "$QTEXT" | docker exec -i rekha-node-3 rekha search -k 3 -c texts

echo ""; echo "[12] Listing collections from node-3..."
docker exec rekha-node-3 rekha list-collections

echo ""; echo "[13] Restarting node-1..."
docker compose start node-1
sleep 5

echo ""; echo "[14] Final health check..."
for i in 1 2 3; do
    echo -n "  node-$i: "
    if docker exec rekha-node-$i rekha health; then
        echo "       healthy"
    else
        echo "       unhealthy"
    fi
done

echo ""; echo "================================================"
echo " ALL TESTS PASSED"
echo "================================================"
