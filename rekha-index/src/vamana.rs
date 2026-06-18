use fnv::FnvHashSet;
use rand::Rng;
use rekha_core::{distance::l2_squared, IndexError, RekhaError};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// DiskANN-style Vamana graph with flat adjacency arrays (u32 positions).
///
/// Memory-efficient: adjacency stored as a flat `Vec<u32>` of node positions
/// (not IDs), with offset indices per node. This halves edge memory vs `Vec<Vec<u64>>`
/// and eliminates hash lookups during search (positions are direct indices).
pub struct VamanaGraph {
    pub r: usize,
    /// Flat adjacency array: node positions (index into `vectors`/`ids`).
    edges: Vec<u32>,
    /// Start offset in `edges` for each node's neighbor list.
    /// `edges[offsets[i]..offsets[i+1]]` are node i's neighbors.
    offsets: Vec<usize>,
    /// All vector IDs in the graph (in insertion order).
    ids: Vec<u64>,
    /// ID -> position mapping (position is u32 for compact storage).
    id_to_pos: fnv::FnvHashMap<u64, u32>,
    /// Medoid vector ID — the optimal entry point for graph search.
    medoid_id: Option<u64>,
    built: bool,
}

impl VamanaGraph {
    pub fn new(r: usize) -> Self {
        Self {
            r,
            edges: Vec::new(),
            offsets: Vec::new(),
            ids: Vec::new(),
            id_to_pos: FnvHashMap::default(),
            medoid_id: None,
            built: false,
        }
    }

    /// Build the Vamana graph from a set of vectors.
    #[allow(clippy::needless_range_loop)]
    pub fn build(&mut self, vectors: &[(u64, &[f32])]) -> Result<(), RekhaError> {
        if vectors.is_empty() {
            return Err(IndexError::EmptyIndex.into());
        }

        let n = vectors.len();
        self.ids = vectors.iter().map(|(id, _)| *id).collect();
        self.offsets = vec![0usize; n + 1]; // +1 for end sentinel
        self.edges = Vec::with_capacity(n * self.r);
        self.id_to_pos.clear();

        for (i, (id, _)) in vectors.iter().enumerate() {
            self.id_to_pos.insert(*id, i as u32);
        }

        // Phase 1: Build RNG via medoid insertion.
        let medoid_idx = self.find_medoid(vectors);
        self.medoid_id = Some(vectors[medoid_idx].0);

        let mut edge_buf: Vec<Vec<u32>> = vec![Vec::new(); n];

        for i in 0..n {
            if i == medoid_idx {
                continue;
            }

            let query = vectors[i].1;
            let candidates =
                self.greedy_search_build(&edge_buf, query, vectors, medoid_idx, self.r);
            let pruned = self.robust_prune(i, &candidates, query, vectors);
            edge_buf[i] = pruned;

            // Add reverse edges (with pruning).
            let current_neighbors: Vec<usize> = edge_buf[i]
                .iter()
                .map(|p| *p as usize)
                .filter(|p| *p != i)
                .collect();

            for &neighbor_pos in &current_neighbors {
                let neighbor_query = vectors[neighbor_pos].1;
                let mut all_dists: Vec<(f32, u32)> = edge_buf[neighbor_pos]
                    .iter()
                    .map(|p| {
                        let dist = l2_squared(neighbor_query, vectors[*p as usize].1);
                        (dist, *p)
                    })
                    .collect();
                all_dists.push((l2_squared(neighbor_query, query), i as u32));

                all_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                edge_buf[neighbor_pos] = all_dists
                    .iter()
                    .take(self.r)
                    .map(|(_, pos)| *pos)
                    .filter(|p| *p != neighbor_pos as u32)
                    .collect();
            }
        }

        // Phase 2: Re-prune all nodes.
        for i in 0..n {
            let query = vectors[i].1;
            let mut neighbors: Vec<(f32, u32)> = edge_buf[i]
                .iter()
                .map(|p| {
                    let dist = l2_squared(query, vectors[*p as usize].1);
                    (dist, *p)
                })
                .collect();
            neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            edge_buf[i] = neighbors.iter().take(self.r).map(|(_, p)| *p).collect();
        }

        // Flatten edge_buf into self.edges + self.offsets.
        let mut offset = 0usize;
        for i in 0..n {
            self.offsets[i] = offset;
            self.edges.extend_from_slice(&edge_buf[i]);
            offset += edge_buf[i].len();
        }
        self.offsets[n] = offset;

        self.built = true;
        Ok(())
    }

    pub fn search(
        &self,
        query: &[f32],
        vectors: &[(u64, Vec<f32>)],
        k: usize,
        l: usize,
    ) -> Result<(Vec<u64>, Vec<f32>), RekhaError> {
        if !self.built || self.ids.is_empty() {
            return Err(IndexError::EmptyIndex.into());
        }

        let medoid_id = self.medoid_id.unwrap_or(self.ids[0]);
        let start_pos = self.id_to_pos[&medoid_id] as usize;
        let mut visited = FnvHashSet::default();
        let mut candidates: BinaryHeap<SearchNode> = BinaryHeap::new();
        let mut kth_best = KthBest::new(k);
        let ef_search = l.max(k * 2);

        let start_dist = l2_squared(query, &vectors[start_pos].1);
        candidates.push(SearchNode {
            dist: start_dist,
            id: medoid_id,
            pos: start_pos,
        });
        visited.insert(medoid_id);
        kth_best.insert(start_dist);

        while let Some(node) = candidates.pop() {
            if node.dist > kth_best.threshold() && visited.len() > ef_search {
                break;
            }

            let start = self.offsets[node.pos];
            let end = self.offsets[node.pos + 1];
            let neighbors = &self.edges[start..end];
            for &neighbor_pos in neighbors {
                let neighbor_id = self.ids[neighbor_pos as usize];
                if visited.contains(&neighbor_id) {
                    continue;
                }
                visited.insert(neighbor_id);

                let dist = l2_squared(query, &vectors[neighbor_pos as usize].1);
                if dist < kth_best.threshold() {
                    candidates.push(SearchNode {
                        dist,
                        id: neighbor_id,
                        pos: neighbor_pos as usize,
                    });
                }
                kth_best.insert(dist);
            }
        }

        let mut all: Vec<(f32, u64)> = visited
            .iter()
            .filter_map(|&vid| {
                self.id_to_pos
                    .get(&vid)
                    .map(|&p| (l2_squared(query, &vectors[p as usize].1), vid))
            })
            .collect();
        all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        all.truncate(k);

        let ids: Vec<u64> = all.iter().map(|(_, id)| *id).collect();
        let dists: Vec<f32> = all.iter().map(|(d, _)| *d).collect();

        Ok((ids, dists))
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn is_built(&self) -> bool {
        self.built
    }

    pub fn contains_id(&self, id: u64) -> bool {
        self.id_to_pos.contains_key(&id)
    }

    // ── Internal helpers ──────────────────────────────────────

    fn find_medoid(&self, vectors: &[(u64, &[f32])]) -> usize {
        let n = vectors.len();
        if n == 0 {
            return 0;
        }

        let sample_size = n.min(100);
        let mut rng = rand::thread_rng();
        let sample: Vec<usize> = (0..sample_size).map(|_| rng.gen_range(0..n)).collect();

        let mut best_idx = 0;
        let mut best_sum = f32::MAX;

        for &i in &sample {
            let mut sum = 0.0f32;
            for &j in &sample {
                if i != j {
                    sum += l2_squared(vectors[i].1, vectors[j].1);
                }
            }
            if sum < best_sum {
                best_sum = sum;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Greedy search using the build-time edge buffer (edge_buf).
    #[allow(clippy::too_many_arguments)]
    fn greedy_search_build(
        &self,
        edge_buf: &[Vec<u32>],
        query: &[f32],
        vectors: &[(u64, &[f32])],
        start: usize,
        l: usize,
    ) -> Vec<(f32, u32)> {
        let mut visited = FnvHashSet::default();
        let mut candidates: BinaryHeap<SearchNode> = BinaryHeap::new();
        let mut kth_best = f32::MAX;

        let start_dist = l2_squared(query, vectors[start].1);
        candidates.push(SearchNode {
            dist: start_dist,
            id: vectors[start].0,
            pos: start,
        });
        visited.insert(vectors[start].0);

        while let Some(node) = candidates.pop() {
            if node.dist > kth_best {
                break;
            }

            for &npos in &edge_buf[node.pos] {
                let nid = self.ids[npos as usize];
                if visited.contains(&nid) {
                    continue;
                }
                visited.insert(nid);

                let dist = l2_squared(query, vectors[npos as usize].1);
                if dist < kth_best {
                    candidates.push(SearchNode {
                        dist,
                        id: nid,
                        pos: npos as usize,
                    });

                    let mut all: Vec<f32> = visited
                        .iter()
                        .filter_map(|vid| self.id_to_pos.get(vid))
                        .map(|&p| l2_squared(query, vectors[p as usize].1))
                        .collect();
                    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    if all.len() >= l {
                        kth_best = all[l - 1];
                    }
                }
            }
        }

        let mut all_results: Vec<(f32, u32)> = visited
            .iter()
            .filter_map(|&vid| {
                self.id_to_pos
                    .get(&vid)
                    .map(|&p| (l2_squared(query, vectors[p as usize].1), p))
            })
            .collect();
        all_results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        all_results.truncate(l);

        all_results
    }

    /// Robust prune from DiskANN. Works with positions internally.
    fn robust_prune(
        &self,
        _node_pos: usize,
        candidates: &[(f32, u32)],
        query: &[f32],
        vectors: &[(u64, &[f32])],
    ) -> Vec<u32> {
        let mut pruned: Vec<u32> = Vec::new();

        let mut sorted: Vec<(f32, u32)> = candidates.to_vec();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for &(_, npos) in &sorted {
            if pruned.len() >= self.r {
                break;
            }

            let nv = vectors[npos as usize].1;

            let mut dominated = false;
            for &ppos in &pruned {
                let pv = vectors[ppos as usize].1;

                let dist_to_p = l2_squared(nv, pv);
                let dist_to_query = l2_squared(nv, query);
                if dist_to_p < dist_to_query {
                    dominated = true;
                    break;
                }
            }

            if !dominated {
                pruned.push(npos);
            }
        }

        pruned
    }
}

    /// Min-heap node for search priority queue.
    /// Smaller distances are higher priority (reversed Ord for BinaryHeap).
    #[derive(Debug, Clone)]
    struct SearchNode {
        dist: f32,
        #[allow(dead_code)]
        id: u64,
        pos: usize,
    }

impl PartialEq for SearchNode {
    fn eq(&self, other: &Self) -> bool {
        self.dist.total_cmp(&other.dist) == Ordering::Equal
    }
}

impl Eq for SearchNode {}

impl PartialOrd for SearchNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.dist.total_cmp(&self.dist)
    }
}

/// Max-heap for tracking the k-th best distance.
#[derive(Debug, Clone)]
struct KthBest {
    heap: BinaryHeap<KthNode>,
    k: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct KthNode(f32);

impl Eq for KthNode {}

impl PartialOrd for KthNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for KthNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl KthBest {
    fn new(k: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k + 1),
            k,
        }
    }

    fn insert(&mut self, dist: f32) {
        self.heap.push(KthNode(dist));
        if self.heap.len() > self.k {
            self.heap.pop();
        }
    }

    fn threshold(&self) -> f32 {
        if self.heap.len() >= self.k {
            self.heap.peek().unwrap().0
        } else {
            f32::MAX
        }
    }
}

use fnv::FnvHashMap;

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vectors(n: usize, dim: usize) -> Vec<(u64, Vec<f32>)> {
        (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|d| (i * dim + d) as f32 / 100.0).collect();
                (i as u64, v)
            })
            .collect()
    }

    fn make_refs(vectors: &[(u64, Vec<f32>)]) -> Vec<(u64, &[f32])> {
        vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect()
    }

    #[test]
    fn test_vamana_new() {
        let g = VamanaGraph::new(32);
        assert_eq!(g.r, 32);
        assert!(!g.built);
        assert_eq!(g.len(), 0);
    }

    #[test]
    fn test_vamana_build_empty_fails() {
        let mut g = VamanaGraph::new(8);
        let result = g.build(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_vamana_build_small() {
        let vectors = make_vectors(20, 8);
        let refs = make_refs(&vectors);
        let mut g = VamanaGraph::new(4);
        g.build(&refs).unwrap();
        assert!(g.built);
        assert_eq!(g.len(), 20);
    }

    #[test]
    fn test_vamana_search_empty_fails() {
        let g = VamanaGraph::new(8);
        let vectors = make_vectors(5, 4);
        let result = g.search(&[0.0; 4], &vectors, 3, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_vamana_search_returns_results() {
        let vectors = make_vectors(30, 4);
        let refs = make_refs(&vectors);
        let mut g = VamanaGraph::new(6);
        g.build(&refs).unwrap();

        let query: Vec<f32> = vec![0.5; 4];
        let result = g.search(&query, &vectors, 5, 20).unwrap();
        let (ids, dists) = result;
        assert!(!ids.is_empty());
        assert_eq!(ids.len(), dists.len());
        assert!(ids.len() <= 5);
        for i in 1..dists.len() {
            assert!(dists[i - 1] <= dists[i] || (dists[i - 1] - dists[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_vamana_search_k_equals_n() {
        let vectors = make_vectors(10, 4);
        let refs = make_refs(&vectors);
        let mut g = VamanaGraph::new(4);
        g.build(&refs).unwrap();

        let (ids, _) = g.search(&[0.0; 4], &vectors, 10, 20).unwrap();
        assert_eq!(ids.len(), 10);
    }

    #[test]
    fn test_vamana_exact_search_small() {
        let vectors = vec![
            (1, vec![10.0, 10.0, 10.0, 10.0]),
            (2, vec![1.0, 1.0, 1.0, 1.0]),
            (3, vec![0.5, 0.5, 0.5, 0.5]),
        ];
        let refs = make_refs(&vectors);
        let mut g = VamanaGraph::new(2);
        g.build(&refs).unwrap();

        let (ids, _) = g.search(&[0.0; 4], &vectors, 1, 10).unwrap();
        assert_eq!(ids[0], 3);
    }

    #[test]
    fn test_medoid_stored() {
        let vectors = make_vectors(20, 8);
        let refs = make_refs(&vectors);
        let mut g = VamanaGraph::new(4);
        g.build(&refs).unwrap();
        assert!(g.medoid_id.is_some());
        assert!(g.ids.contains(&g.medoid_id.unwrap()));
    }

    #[test]
    fn test_vamana_find_medoid_single() {
        let g = VamanaGraph::new(4);
        let idx = g.find_medoid(&[(1, &[1.0, 2.0, 3.0])]);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_is_built_flag() {
        let mut g = VamanaGraph::new(6);
        assert!(!g.is_built());
        let vectors = make_vectors(10, 4);
        let refs = make_refs(&vectors);
        g.build(&refs).unwrap();
        assert!(g.is_built());
    }

    #[test]
    fn test_contains_id() {
        let vectors = make_vectors(10, 4);
        let refs = make_refs(&vectors);
        let mut g = VamanaGraph::new(4);
        g.build(&refs).unwrap();
        assert!(g.contains_id(0));
        assert!(!g.contains_id(999));
    }

    #[test]
    fn test_len_with_vectors() {
        let vectors = make_vectors(25, 4);
        let refs = make_refs(&vectors);
        let mut g = VamanaGraph::new(6);
        assert_eq!(g.len(), 0);
        g.build(&refs).unwrap();
        assert_eq!(g.len(), 25);
    }

    #[test]
    fn test_is_empty_on_new() {
        let g = VamanaGraph::new(8);
        assert!(g.is_empty());
    }
}
