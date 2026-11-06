# RekhaDB Helm Chart

Production-grade Helm chart for deploying a distributed, high-availability [RekhaDB](https://github.com/rekha-db/rekha) vector database cluster on Kubernetes.

## Architecture

This chart deploys RekhaDB as a Kubernetes `StatefulSet` with:
- **Dual-Listener Network Isolation**:
  - Public Client API listener (`port 8000`) for vector queries, inserts, and management.
  - Internal Raft/WAL/2PC listener (`port 9000`) on a headless Kubernetes Service (`publishNotReadyAddresses: true`) protected by Kubernetes `NetworkPolicy`.
- **Raft Quorum Bootstrapping**:
  - Automatically derives node IDs from StatefulSet ordinals (`1..N`).
  - Bootstraps the initial Raft quorum (`REKHA_PEERS`) on pod 0 and forms consensus across initial cluster nodes.
- **Zero-Downtime Dynamic Scale-Out**:
  - Scale up replicas via `helm upgrade --set replicaCount=5`.
  - New nodes automatically start in `--join` mode and register as learners with the cluster leader; the auto-rebalancer promotes them to voters and migrates shard replicas automatically.
- **Production Hardening**:
  - Non-root user (`UID 10001`), read-only root filesystem, dropped Linux capabilities (`drop: [ALL]`).
  - PodDisruptionBudget (`maxUnavailable: 1`) protecting Raft quorums during maintenance and node drains.
  - Inter-pod anti-affinity across nodes and failure domains.
  - Startup, Liveness, and Readiness probes tuned for WAL replay and Raft quorum.
  - Optional Prometheus `ServiceMonitor`, alerting rules matching `docs/operations.md`, and automated sealed backup `CronJob`.

## Quick Start

### 1. Add and Install the Chart

```bash
# Generate high-entropy secrets (>= 32 bytes each)
export REKHA_API_KEY="$(openssl rand -hex 16)"
export REKHA_CLUSTER_SECRET="$(openssl rand -hex 32)"

# Install a 3-node HA cluster
helm install rekha charts/rekha \
  --set auth.apiKey="${REKHA_API_KEY}" \
  --set auth.clusterSecret="${REKHA_CLUSTER_SECRET}" \
  --set persistence.size=50Gi \
  --set persistence.storageClass="fast-nvme"
```

### 2. Verify Cluster Status

```bash
# Retrieve generated API key (if not explicitly provided above)
export REKHA_API_KEY=$(kubectl get secret rekha-auth -o jsonpath="{.data.REKHA_API_KEY}" | base64 --decode)

# Port-forward to the client API
kubectl port-forward svc/rekha 8000:8000 &

# Check cluster readiness (returns {"ready": true})
curl -H "x-chroma-token: $REKHA_API_KEY" http://127.0.0.1:8000/ready

# Inspect live cluster membership and leader
rekha --url http://127.0.0.1:8000 membership list
```

### 3. Scaling Out

To add 2 nodes to an active cluster:

```bash
helm upgrade rekha charts/rekha --set replicaCount=5
```

The new pods will join, replicate the log, and receive their share of placement shards with zero downtime.

---

## Configuration Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of cluster nodes (minimum 3 for HA) | `3` |
| `cluster.initialClusterSize` | Initial bootstrap quorum node count | `3` |
| `cluster.shards` | Number of placement shards (`0` = 4 shards/node) | `12` |
| `cluster.replicationFactor` | Replicas per shard | `3` |
| `cluster.walDurability` | WAL fsync mode (`durable` or `relaxed`) | `durable` |
| `cluster.heartbeatMs` | Raft heartbeat interval in ms | `500` |
| `cluster.electionMinMs` | Raft election timeout floor in ms | `1500` |
| `cluster.electionMaxMs` | Raft election timeout ceiling in ms | `3000` |
| `cluster.snapshotEveryNLogs`| Log compaction and purge interval | `10000` |
| `auth.existingSecret` | Name of existing Kubernetes Secret for auth keys | `""` |
| `auth.apiKey` | Admin API key (min 32 chars; auto-generated if empty)| `""` |
| `auth.clusterSecret` | Inter-node secret (min 32 bytes; auto-generated if empty)| `""` |
| `persistence.enabled` | Enable PersistentVolumeClaim storage | `true` |
| `persistence.storageClass` | StorageClass for WAL & data (NVMe recommended) | `""` |
| `persistence.size` | Size of data volume per node | `20Gi` |
| `resources.requests.cpu` | Requested CPU | `1000m` |
| `resources.requests.memory`| Requested Memory | `2Gi` |
| `podDisruptionBudget.enabled` | Protect Raft quorum with PDB | `true` |
| `networkPolicy.enabled` | Isolate internal Raft port (`9000`) | `true` |
| `metrics.enabled` | Enable Prometheus metrics scraping | `false` |
| `backup.enabled` | Scheduled automated sealed backups | `false` |
