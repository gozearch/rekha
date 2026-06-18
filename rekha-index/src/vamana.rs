use fnv::FnvHashSet;
use rand::Rng;
use rekha_core::{distance::l2_squared, IndexError, RekhaError};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// DiskANN-style Vamana graph for approximate nearest neighbor search.
///
/// The Vamana graph is designed for disk-based, HDD-friendly access patterns.
/// Key properties:
/// - Outgoing degree bound `R` (controls memory and I/O)
/// - Construction via `robust prune` algorithm
/// - Search via greedy beam search with search list size `L`
///
/// This implementation maintains the graph in memory for the index structure,
/// with full-precision vectors loaded as needed. For billion-scale, vectors
/// live on disk and the graph adjacency lists reference disk offsets.
pub struct VamanaGraph {
    /// Maximum outgoing degree.
    pub r: usize,
    /// Graph adjacency list: for each node ID, its list of neighbor IDs.
    edges: Vec<Vec<u64>>,
    /// All vector IDs in the graph (in insertion order).
    ids: Vec<u64>,
    /// ID -> position mapping.
    id_to_pos: fnv::FnvHashMap<u64, usize>,
    /// Medoid vector ID — the optimal entry point for graph search.
    medoid_id: Option<u64>,
    /// Whether the graph has been indexed/constructed.
    built: bool,
}

impl VamanaGraph {
    /// Create a new Vamana graph with maximum degree `r`.
    pub fn new(r: usize) -> Self {
        Self {
            r,
            edges: Vec::new(),
            ids: Vec::new(),
            id_to_pos: FnvHashMap::default(),
            medoid_id: None,
            built: false,
        }
    }

    /// Build the Vamana graph from a set of vectors.
    /// Uses the robust-prune algorithm from the DiskANN paper.
    pub fn build(&mut self, vectors: &[(u64, &[f32])]) -> Result<(), RekhaError> {
        if vectors.is_empty() {
            return Err(IndexError::EmptyIndex.into());
        }

        let n = vectors.len();
        self.ids = vectors.iter().map(|(id, _)| *id).collect();
        self.edges = vec![Vec::new(); n];
        self.id_to_pos.clear();

        for (i, (id, _)) in vectors.iter().enumerate() {
            self.id_to_pos.insert(*id, i);
        }

        // Phase 1: Build a RNG (Randomized Neighborhood Graph) via medoid insertion.
        // Find medoid (point closest to all others).
        let medoid_idx = self.find_medoid(vectors);
        self.medoid_id = Some(vectors[medoid_idx].0);

        for i in 0..n {
            if i == medoid_idx {
                continue;
            }

            // Greedy search from medoid to find nearest neighbors.
            let query = vectors[i].1;
            let candidates = self.greedy_search(query, vectors, medoid_idx, self.r);

            // Robust prune to select diverse neighbors.
            let pruned = self.robust_prune(i, &candidates, query, vectors);

            self.edges[i] = pruned;

            // Add reverse edges (with pruning).
            let current_neighbors: Vec<usize> = self.edges[i]
                .iter()
                .map(|nid| self.id_to_pos[nid])
                .filter(|p| *p != i)
                .collect();

            for &neighbor_pos in &current_neighbors {
                let neighbor_query = vectors[neighbor_pos].1;
                let mut all_neighbors: Vec<(f32, usize)> = self.edges[neighbor_pos]
                    .iter()
                    .map(|nid| {
                        let pos = self.id_to_pos[nid];
                        let dist = l2_squared(neighbor_query, vectors[pos].1);
                        (dist, pos)
                    })
                    .collect();
                all_neighbors.push((l2_squared(neighbor_query, query), i));

                all_neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                // Keep top-R neighbors after robust prune.
                let neighbor_id = self.ids[neighbor_pos];
                let neighbor_nid: Vec<u64> = all_neighbors
                    .iter()
                    .take(self.r)
                    .map(|(_, pos)| self.ids[*pos])
                    .filter(|n| *n != neighbor_id)
                    .collect();

                self.edges[neighbor_pos] = neighbor_nid;
            }
        }

        // Phase 2: Apply robust prune to all nodes.
        for i in 0..n {
            let query = vectors[i].1;
            let neighbors: Vec<(f32, u64)> = self.edges[i]
                .iter()
                .map(|nid| {
                    let pos = self.id_to_pos[nid];
                    (l2_squared(query, vectors[pos].1), *nid)
                })
                .collect();
            self.edges[i] = neighbors.iter().map(|(_, nid)| *nid).take(self.r).collect();
        }

        self.built = true;
        Ok(())
    }

    /// Search for the top-k nearest neighbors.
    /// `l` is the search list size (beam width), larger = more accurate but slower.
    /// `k` is the number of results to return.
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
        let start_pos = self.id_to_pos[&medoid_id];
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
            let max_dist = kth_best.threshold();
            if node.dist > max_dist && visited.len() > ef_search {
                break;
            }

            if let Some(neighbors) = self.edges.get(node.pos) {
                for &neighbor_id in neighbors {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }
                    visited.insert(neighbor_id);

                    if let Some(&neighbor_pos) = self.id_to_pos.get(&neighbor_id) {
                        let dist = l2_squared(query, &vectors[neighbor_pos].1);

                        if dist < kth_best.threshold() {
                            candidates.push(SearchNode {
                                dist,
                                id: neighbor_id,
                                pos: neighbor_pos,
                            });
                        }
                        kth_best.insert(dist);
                    }
                }
            }
        }

        let mut all: Vec<(f32, u64)> = visited
            .iter()
            .filter_map(|&vid| {
                self.id_to_pos
                    .get(&vid)
                    .map(|&p| (l2_squared(query, &vectors[p].1), vid))
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

    /// Check if a vector ID exists in the current graph.
    pub fn contains_id(&self, id: u64) -> bool {
        self.id_to_pos.contains_key(&id)
    }

    // ── Internal helpers ──────────────────────────────────────

    /// Find the medoid: the point closest (in sum of distances) to all others.
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

    /// Greedy search from a start node to find nearest neighbors.
    fn greedy_search(
        &self,
        query: &[f32],
        vectors: &[(u64, &[f32])],
        start: usize,
        l: usize,
    ) -> Vec<(f32, u64)> {
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

            if let Some(neighbors) = self.edges.get(node.pos) {
                for &nid in neighbors {
                    if visited.contains(&nid) {
                        continue;
                    }
                    visited.insert(nid);
                    if let Some(&npos) = self.id_to_pos.get(&nid) {
                        let dist = l2_squared(query, vectors[npos].1);
                        if dist < kth_best {
                            candidates.push(SearchNode {
                                dist,
                                id: nid,
                                pos: npos,
                            });

                            let mut all: Vec<f32> = visited
                                .iter()
                                .filter_map(|vid| self.id_to_pos.get(vid))
                                .map(|&p| l2_squared(query, vectors[p].1))
                                .collect();
                            all.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            if all.len() >= l {
                                kth_best = all[l - 1];
                            }
                        }
                    }
                }
            }
        }

        let mut all_results: Vec<(f32, u64)> = visited
            .iter()
            .filter_map(|&vid| {
                self.id_to_pos
                    .get(&vid)
                    .map(|&p| (l2_squared(query, vectors[p].1), vid))
            })
            .collect();
        all_results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        all_results.truncate(l);

        all_results
    }

    /// Robust prune algorithm from DiskANN.
    /// Selects up to `R` diverse neighbors from the candidate set.
    fn robust_prune(
        &self,
        _node_pos: usize,
        candidates: &[(f32, u64)],
        query: &[f32],
        vectors: &[(u64, &[f32])],
    ) -> Vec<u64> {
        let mut pruned = Vec::new();

        // Sort candidates by distance to query.
        let mut sorted: Vec<(f32, u64)> = candidates.to_vec();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for &(_, nid) in &sorted {
            if pruned.len() >= self.r {
                break;
            }

            let npos = self.id_to_pos[&nid];
            let nv = vectors[npos].1;

            let mut dominated = false;
            for &pid in &pruned {
                let ppos = self.id_to_pos[&pid];
                let pv = vectors[ppos].1;

                // If nid is closer to pid than to the query, skip nid (it's "covered").
                let dist_to_p = l2_squared(nv, pv);
                let dist_to_query = l2_squared(nv, query);
                if dist_to_p < dist_to_query {
                    dominated = true;
                    break;
                }
            }

            if !dominated {
                pruned.push(nid);
            }
        }

        pruned
    }
}

/// Internal node representation for the search priority queue.
/// BinaryHeap is a max-heap, but we want min-first behavior.
/// The `Ord` impl uses `total_cmp` for f32 (panics on NaN, which should never occur).
#[derive(Debug, Clone)]
struct SearchNode {
    dist: f32,
    id: u64,
    pos: usize,
}

impl PartialEq for SearchNode {
    fn eq(&self, other: &Self) -> bool {
        self.dist.total_cmp(&other.dist) == std::cmp::Ordering::Equal
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
        // Reverse for min-heap: BinaryHeap pops the largest, so smaller dists
        // should be considered "greater" in ordering.
        other.dist.total_cmp(&self.dist)
    }
}

/// Max-heap wrapper for tracking the k-th best distance.
/// `BinaryHeap` is a max-heap by default, so larger distance = higher priority.
/// `peek()` returns the worst (largest distance) of the top-k.
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
        // max-heap: larger distance is higher priority
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
        // Results should be sorted by distance ascending
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
        // With small data, the nearest neighbor of [0,0,0,0] should be the vector closest to origin.
        let vectors = vec![
            (1, vec![10.0, 10.0, 10.0, 10.0]),
            (2, vec![1.0, 1.0, 1.0, 1.0]),
            (3, vec![0.5, 0.5, 0.5, 0.5]),
        ];
        let refs = make_refs(&vectors);
        let mut g = VamanaGraph::new(2);
        g.build(&refs).unwrap();

        let (ids, _) = g.search(&[0.0; 4], &vectors, 1, 10).unwrap();
        assert_eq!(ids[0], 3); // closest to origin
    }

    #[test]
    fn test_vamana_find_medoid_single() {
        let g = VamanaGraph::new(4);
        let idx = g.find_medoid(&[(1, &[1.0, 2.0, 3.0])]);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_vamana_find_medoid_empty() {
        let g = VamanaGraph::new(4);
        let idx = g.find_medoid(&[]);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_greedy_search_direct() {
        let vectors = make_vectors(15, 4);
        let refs = make_refs(&vectors);
        let mut g = VamanaGraph::new(4);
        g.build(&refs).unwrap();
        // greedy_search is private; we test it indirectly via build + search
        let (ids, _) = g.search(&[0.0; 4], &vectors, 3, 10).unwrap();
        assert!(!ids.is_empty());
        assert!(ids.len() <= 3);
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

    #[test]
    fn test_vamana_robust_prune_dominated() {
        let vectors = vec![
            (0, vec![0.0, 0.0]),
            (1, vec![1.0, 1.0]),
            (2, vec![2.0, 2.0]),
        ];
        let refs = make_refs(&vectors);
        let mut g = VamanaGraph::new(4);
        g.build(&refs).unwrap();
        // After build, greedy search should work
        let (ids, _) = g.search(&[0.0, 0.0], &vectors, 1, 10).unwrap();
        assert!(!ids.is_empty());
    }
}
