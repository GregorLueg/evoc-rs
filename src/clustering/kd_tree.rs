//! KD-tree for component-aware nearest neighbour queries in Boruvka's MST.
//!
//! Flattened binary space partition tree. Splits on the widest-spread
//! dimension at the median. Accelerates the per-round "find nearest point
//! in a different component" query via same-component subtree pruning.

use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::prelude::*;

//////////////////////
// Candidate search //
//////////////////////

/// A neighbour candidate for the max-heap used in k-NN queries.
/// Ordered by distance descending so `BinaryHeap` (max-heap) evicts
/// the furthest candidate first.
#[derive(Clone, Debug)]
struct KnnCandidate<T> {
    /// Squarted distance
    dist_sq: T,
    /// Idx of the candidate
    idx: usize,
}

impl<T: PartialOrd> PartialEq for KnnCandidate<T> {
    fn eq(&self, other: &Self) -> bool {
        self.dist_sq == other.dist_sq
    }
}

impl<T: PartialOrd> Eq for KnnCandidate<T> {}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl<T: PartialOrd> PartialOrd for KnnCandidate<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.dist_sq.partial_cmp(&other.dist_sq) {
            Some(Ordering::Equal) => Some(self.idx.cmp(&other.idx)),
            ord => ord,
        }
    }
}

impl<T: PartialOrd> Ord for KnnCandidate<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

////////////
// KdTree //
////////////

/// Axis-aligned KD-tree stored as a flattened binary tree.
///
/// Children of node `i` live at `2i + 1` and `2i + 2`. Each node stores an
/// index range into a permuted point array and an axis-aligned bounding box.
/// Leaf sizes are governed by the `leaf_size` parameter at build time.
pub struct KdTree<T> {
    /// Permuted point indices; subtree of node `i` owns
    /// `idx_array[idx_start[i]..idx_end[i]]`
    idx_array: Vec<usize>,
    /// Per-node start offset into `idx_array`
    idx_start: Vec<usize>,
    /// Per-node end offset into `idx_array` (exclusive)
    idx_end: Vec<usize>,
    /// Whether the node is a leaf
    is_leaf: Vec<bool>,
    /// Lower AABB corners, flat `[n_nodes * dim]`
    lower: Vec<T>,
    /// Upper AABB corners, flat `[n_nodes * dim]`
    upper: Vec<T>,
    /// Total number of nodes in the flattened tree
    n_nodes: usize,
    /// Dimensionality of the point data
    dim: usize,
}

impl<T: EvocFloat> KdTree<T> {
    //////////////
    // Building //
    //////////////

    /// Build a KD-tree over flat point data.
    ///
    /// ### Params
    ///
    /// * `data` - Flat point coordinates; point `i` is `data[i*dim..(i+1)*dim]`
    /// * `dim`  - Dimensionality of each point
    /// * `leaf_size` - Maximum points per leaf
    pub fn build(data: &[T], dim: usize, leaf_size: usize) -> Self {
        let n = if dim == 0 { 0 } else { data.len() / dim };
        let leaf_size = leaf_size.max(1);

        if n == 0 {
            return Self {
                idx_array: vec![],
                idx_start: vec![],
                idx_end: vec![],
                is_leaf: vec![],
                lower: vec![],
                upper: vec![],
                n_nodes: 0,
                dim,
            };
        }

        let n_levels = if n <= leaf_size {
            1
        } else {
            ((n - 1) as f64 / leaf_size as f64).log2() as usize + 1
        };
        let n_nodes = (1usize << n_levels) - 1;

        let mut tree = Self {
            idx_array: (0..n).collect(),
            idx_start: vec![0; n_nodes],
            idx_end: vec![0; n_nodes],
            is_leaf: vec![true; n_nodes],
            lower: vec![T::zero(); n_nodes * dim],
            upper: vec![T::zero(); n_nodes * dim],
            n_nodes,
            dim,
        };
        tree.build_node(data, 0, n, 0);
        tree
    }

    /// Number of nodes in the flattened tree.
    ///
    /// ### Returns
    ///
    /// Number of nodes
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Helper function to return the data of a point
    ///
    /// ### Params
    ///
    /// * `data` - The flattened data
    /// * `i` - Index of the data point
    ///
    /// ### Returns
    ///
    /// Slice of the data of point `i`.
    #[inline]
    fn pt<'a>(&self, data: &'a [T], i: usize) -> &'a [T] {
        &data[i * self.dim..(i + 1) * self.dim]
    }

    /// Recursively partition points and populate node metadata.
    ///
    /// Computes the AABB for the range `start..end` of `idx_array`, records it,
    /// then splits on the widest axis at the median via introselect and
    /// recurses into the left (`2 * node + 1`) and right (`2 * node + 2`)
    /// children. Recursion stops when the node would fall outside the flattened
    /// tree or the range contains fewer than two points, at which point the
    /// node is marked as a leaf.
    ///
    /// ### Params
    ///
    /// * `data` - Full point array passed to `build` (flattened version)
    /// * `start` - Inclusive start offset into `idx_array` for this node
    /// * `end` - Exclusive end offset into `idx_array` for this node
    /// * `node` - Index of the current node in the flattened tree
    fn build_node(&mut self, data: &[T], start: usize, end: usize, node: usize) {
        let dim = self.dim;
        let bo = node * dim;

        let first = self.pt(data, self.idx_array[start]);
        self.lower[bo..bo + dim].copy_from_slice(first);
        self.upper[bo..bo + dim].copy_from_slice(first);
        for i in (start + 1)..end {
            let pt = self.pt(data, self.idx_array[i]);
            for d in 0..dim {
                self.lower[bo + d] = self.lower[bo + d].min(pt[d]);
                self.upper[bo + d] = self.upper[bo + d].max(pt[d]);
            }
        }

        self.idx_start[node] = start;
        self.idx_end[node] = end;

        let left = 2 * node + 1;
        if left >= self.n_nodes || end - start < 2 {
            self.is_leaf[node] = true;
            return;
        }
        self.is_leaf[node] = false;

        let mut split_dim = 0;
        let mut max_spread = T::zero();
        for d in 0..dim {
            let s = self.upper[bo + d] - self.lower[bo + d];
            if s > max_spread {
                max_spread = s;
                split_dim = d;
            }
        }

        let mid = start + (end - start) / 2;
        self.idx_array[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
            data[a * dim + split_dim]
                .partial_cmp(&data[b * dim + split_dim])
                .unwrap()
        });

        self.build_node(data, start, mid, left);
        self.build_node(data, mid, end, left + 1);
    }

    /// Squared distance from a point to a node's AABB (lower bound on the
    /// squared distance to any point inside the node).
    ///
    /// ### Params
    ///
    /// * `node` - Node index
    /// * `point` - Query point coordinates
    ///
    /// ### Returns
    ///
    /// Squared Euclidean distance from `point` to the closest face of the
    /// node's bounding box, or zero if the point is inside.
    #[inline]
    fn aabb_sq(&self, node: usize, point: &[T]) -> T {
        let bo = node * self.dim;
        let mut r = T::zero();
        for d in 0..self.dim {
            let lo = self.lower[bo + d];
            let hi = self.upper[bo + d];
            let p = point[d];
            let diff = if p < lo {
                lo - p
            } else if p > hi {
                p - hi
            } else {
                T::zero()
            };
            r += diff * diff;
        }
        r
    }

    /// Find the nearest point in a different component under squared mutual
    /// reachability distance.
    ///
    /// Three pruning rules accelerate the search:
    ///
    /// 1. **AABB bound**: if the lower-bound distance to a node exceeds the
    ///    best found so far, skip it.
    /// 2. **Core distance bound**: if the query point's core distance already
    ///    exceeds the best found, no candidate can improve things.
    /// 3. **Same-component pruning**: if every point under a node belongs to
    ///    the same component as the query, skip the entire subtree.
    ///
    /// ### Params
    ///
    /// * `data` - Full point array (same one passed to `build` - flattened)
    /// * `qi` - Index of the query point
    /// * `core_sq` - Squared core distances for all points
    /// * `pt_comp` - Component label per point
    /// * `nd_comp` - Component label per node (`-1` = mixed)
    ///
    /// ### Returns
    ///
    /// `(index, squared_mutual_reachability_distance)` of the nearest point
    /// in a different component, or `(qi, T::max_value())` if none exists.
    pub fn nearest_other_component(
        &self,
        data: &[T],
        qi: usize,
        core_sq: &[T],
        pt_comp: &[usize],
        nd_comp: &[i64],
    ) -> (usize, T) {
        let mut best_j = qi;
        let mut best_d = T::max_value();
        let lb = self.aabb_sq(0, self.pt(data, qi));
        self.noc_recurse(
            data,
            qi,
            core_sq,
            pt_comp,
            nd_comp,
            0,
            lb,
            &mut best_j,
            &mut best_d,
        );
        (best_j, best_d)
    }

    /// Recursive traversal for `nearest_other_component`.
    ///
    /// Applies three pruning rules before visiting children or scanning leaves.
    /// At leaves, computes squared mutual reachability distance for each
    /// candidate and updates `best_j` and `best_d` if a closer point is found.
    /// At internal nodes, the closer child (by AABB lower bound) is visited
    /// first so that `best_d` tightens as quickly as possible.
    ///
    /// ### Params
    ///
    /// * `data` - Full point array passed to `build`
    /// * `qi` - Index of the query point
    /// * `core_sq` - Squared core distances for all points
    /// * `pt_comp` - Component label per point
    /// * `nd_comp` - Component label per node (`-1` = mixed)
    /// * `node` - Current node index in the flattened tree
    /// * `lb` - Precomputed AABB lower-bound distance for `node`
    /// * `best_j` - Running nearest neighbour index, updated in place
    /// * `best_d` - Running best squared mutual reachability distance, updated
    ///   in place
    #[allow(clippy::too_many_arguments)]
    fn noc_recurse(
        &self,
        data: &[T],
        qi: usize,
        core_sq: &[T],
        pt_comp: &[usize],
        nd_comp: &[i64],
        node: usize,
        lb: T,
        best_j: &mut usize,
        best_d: &mut T,
    ) {
        if lb > *best_d {
            return;
        }
        if core_sq[qi] > *best_d {
            return;
        }
        let qc = pt_comp[qi] as i64;
        if nd_comp[node] == qc {
            return;
        }

        if self.is_leaf[node] {
            let qi_pt = self.pt(data, qi);
            for i in self.idx_start[node]..self.idx_end[node] {
                let j = self.idx_array[i];
                if pt_comp[j] as i64 == qc {
                    continue;
                }
                if core_sq[j] > *best_d {
                    continue;
                }
                let sq = T::euclidean_simd(qi_pt, self.pt(data, j));
                let mr = sq.max(core_sq[qi]).max(core_sq[j]);
                if mr < *best_d {
                    *best_d = mr;
                    *best_j = j;
                }
            }
            return;
        }

        let left = 2 * node + 1;
        let right = left + 1;
        let qi_pt = self.pt(data, qi);
        let lb_l = self.aabb_sq(left, qi_pt);
        let lb_r = self.aabb_sq(right, qi_pt);

        if lb_l <= lb_r {
            self.noc_recurse(
                data, qi, core_sq, pt_comp, nd_comp, left, lb_l, best_j, best_d,
            );
            self.noc_recurse(
                data, qi, core_sq, pt_comp, nd_comp, right, lb_r, best_j, best_d,
            );
        } else {
            self.noc_recurse(
                data, qi, core_sq, pt_comp, nd_comp, right, lb_r, best_j, best_d,
            );
            self.noc_recurse(
                data, qi, core_sq, pt_comp, nd_comp, left, lb_l, best_j, best_d,
            );
        }
    }

    /// Bottom-up sweep to label each node with its component.
    ///
    /// A node receives a component label if all points in its subtree share
    /// the same component. Otherwise it is labelled `-1` (mixed). This
    /// enables same-component pruning in `nearest_other_component`.
    ///
    /// ### Params
    ///
    /// * `pt_comp` - Current component label per point
    /// * `nd_comp` - Output buffer of length `n_nodes`, filled in-place
    pub fn update_node_components(&self, pt_comp: &[usize], nd_comp: &mut [i64]) {
        for node in (0..self.n_nodes).rev() {
            if self.is_leaf[node] {
                let s = self.idx_start[node];
                let e = self.idx_end[node];
                if s >= e {
                    nd_comp[node] = -1;
                    continue;
                }
                let c = pt_comp[self.idx_array[s]] as i64;
                nd_comp[node] = if (s..e).all(|i| pt_comp[self.idx_array[i]] as i64 == c) {
                    c
                } else {
                    -1
                };
            } else {
                let l = 2 * node + 1;
                let r = l + 1;
                nd_comp[node] = if r < self.n_nodes && nd_comp[l] >= 0 && nd_comp[l] == nd_comp[r] {
                    nd_comp[l]
                } else {
                    -1
                };
            }
        }
    }

    //////////////////
    // kNN querying //
    //////////////////

    /// Find the `k` nearest neighbours of `query` in the tree (squared
    /// Euclidean distances).
    ///
    /// Standard best-first KD-tree search: maintains a max-heap of `k`
    /// candidates and prunes subtrees whose AABB lower-bound exceeds the
    /// current k-th best distance.
    ///
    /// Find the `k` nearest neighbours of `query`.
    ///
    /// ### Params
    ///
    /// * `data` - Flat point data used to build the tree
    /// * `query` - Query point coordinates (length `dim`)
    /// * `k` - Number of neighbours
    /// * `exclude` - Optional point index to skip (for self-queries)
    pub fn knn_query(
        &self,
        data: &[T],
        query: &[T],
        k: usize,
        exclude: Option<usize>,
    ) -> Vec<(usize, T)> {
        if self.n_nodes == 0 || k == 0 {
            return Vec::new();
        }
        let mut heap: BinaryHeap<KnnCandidate<T>> = BinaryHeap::with_capacity(k);
        self.knn_recurse(data, query, k, exclude, 0, &mut heap);

        let mut result: Vec<(usize, T)> = heap.into_iter().map(|c| (c.idx, c.dist_sq)).collect();
        result.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
        result
    }

    /// Parallel batch k-NN query: find the `k` nearest neighbours for every
    /// point in `data`, excluding self.
    ///
    /// ### Params
    ///
    /// * `data` - Point coordinates (same as used to build the tree -
    ///   flattened).
    /// * `k` - Number of neighbours per point
    ///
    /// ### Returns
    ///
    /// `(indices, sq_distances)` where each inner `Vec` has length `k`,
    /// sorted by ascending distance.
    pub fn knn_query_batch(&self, data: &[T], k: usize) -> (Vec<Vec<usize>>, Vec<Vec<T>>) {
        let n = data.len() / self.dim;
        let results: Vec<Vec<(usize, T)>> = (0..n)
            .into_par_iter()
            .map(|i| self.knn_query(data, self.pt(data, i), k, Some(i)))
            .collect();

        let mut indices = Vec::with_capacity(n);
        let mut distances = Vec::with_capacity(n);
        for r in results {
            let (idx, dist): (Vec<usize>, Vec<T>) = r.into_iter().unzip();
            indices.push(idx);
            distances.push(dist);
        }
        (indices, distances)
    }

    /// Recursive k-NN traversal.
    fn knn_recurse(
        &self,
        data: &[T],
        query: &[T],
        k: usize,
        exclude: Option<usize>,
        node: usize,
        heap: &mut BinaryHeap<KnnCandidate<T>>,
    ) {
        let lb = self.aabb_sq(node, query);
        if heap.len() == k && lb > heap.peek().unwrap().dist_sq {
            return;
        }

        if self.is_leaf[node] {
            for i in self.idx_start[node]..self.idx_end[node] {
                let j = self.idx_array[i];
                if exclude == Some(j) {
                    continue;
                }
                let d = T::euclidean_simd(self.pt(data, j), query);
                if heap.len() < k {
                    heap.push(KnnCandidate { dist_sq: d, idx: j });
                } else {
                    let worst = heap.peek().unwrap();
                    if d < worst.dist_sq || (d == worst.dist_sq && j < worst.idx) {
                        heap.pop();
                        heap.push(KnnCandidate { dist_sq: d, idx: j });
                    }
                }
            }
            return;
        }

        let left = 2 * node + 1;
        let right = left + 1;
        let lb_l = self.aabb_sq(left, query);
        let lb_r = self.aabb_sq(right, query);

        if lb_l <= lb_r {
            self.knn_recurse(data, query, k, exclude, left, heap);
            self.knn_recurse(data, query, k, exclude, right, heap);
        } else {
            self.knn_recurse(data, query, k, exclude, right, heap);
            self.knn_recurse(data, query, k, exclude, left, heap);
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(nested: &[&[f64]]) -> Vec<f64> {
        nested.iter().flat_map(|v| v.iter().copied()).collect()
    }

    #[test]
    fn test_build_empty() {
        let data: Vec<f64> = vec![];
        let tree = KdTree::build(&data, 2, 10);
        assert_eq!(tree.n_nodes(), 0);
    }

    #[test]
    fn test_build_single() {
        let data = flat(&[&[1.0, 2.0]]);
        let tree = KdTree::build(&data, 2, 10);
        assert!(tree.is_leaf[0]);
        assert_eq!(tree.idx_start[0], 0);
        assert_eq!(tree.idx_end[0], 1);
    }

    #[test]
    fn test_build_covers_all_points() {
        let data: Vec<f64> = (0..100).flat_map(|i| [i as f64, 0.0]).collect();
        let tree = KdTree::build(&data, 2, 10);
        let mut seen = [false; 100];
        for &idx in &tree.idx_array {
            assert!(!seen[idx]);
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn test_aabb_sq_inside() {
        let data = flat(&[&[0.0, 0.0], &[10.0, 10.0]]);
        let tree = KdTree::build(&data, 2, 10);
        assert_eq!(tree.aabb_sq(0, &[5.0, 5.0]), 0.0);
    }

    #[test]
    fn test_aabb_sq_outside() {
        let data = flat(&[&[0.0, 0.0], &[1.0, 1.0]]);
        let tree = KdTree::build(&data, 2, 10);
        let d: f64 = tree.aabb_sq(0, &[3.0, 0.5]);
        assert!((d - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_nearest_other_component_simple() {
        let data = flat(&[&[0.0, 0.0], &[1.0, 0.0], &[10.0, 0.0]]);
        let tree = KdTree::build(&data, 2, 10);
        let core_sq = vec![0.0f64; 3];
        let pt_comp = vec![0, 0, 1];
        let mut nd_comp = vec![-1i64; tree.n_nodes()];
        tree.update_node_components(&pt_comp, &mut nd_comp);

        let (j, _) = tree.nearest_other_component(&data, 0, &core_sq, &pt_comp, &nd_comp);
        assert_eq!(j, 2);
        let (j, _) = tree.nearest_other_component(&data, 1, &core_sq, &pt_comp, &nd_comp);
        assert_eq!(j, 2);
        let (j, _) = tree.nearest_other_component(&data, 2, &core_sq, &pt_comp, &nd_comp);
        assert_eq!(j, 1);
    }

    #[test]
    fn test_nearest_other_component_with_core_distances() {
        let data = flat(&[&[0.0, 0.0], &[1.0, 0.0], &[2.0, 0.0]]);
        let tree = KdTree::build(&data, 2, 10);
        let core_sq = vec![0.0, 100.0, 0.0];
        let pt_comp = vec![0, 0, 1];
        let mut nd_comp = vec![-1i64; tree.n_nodes()];
        tree.update_node_components(&pt_comp, &mut nd_comp);

        let (_, d0) = tree.nearest_other_component(&data, 0, &core_sq, &pt_comp, &nd_comp);
        let (_, d1) = tree.nearest_other_component(&data, 1, &core_sq, &pt_comp, &nd_comp);
        assert!(d0 < d1);
    }

    #[test]
    fn test_node_components_all_same() {
        let data = flat(&[&[0.0], &[1.0], &[2.0]]);
        let tree = KdTree::build(&data, 1, 10);
        let pt_comp = vec![5, 5, 5];
        let mut nd_comp = vec![-1i64; tree.n_nodes()];
        tree.update_node_components(&pt_comp, &mut nd_comp);
        assert_eq!(nd_comp[0], 5);
    }

    #[test]
    fn test_node_components_mixed() {
        let data = flat(&[&[0.0], &[1.0], &[2.0]]);
        let tree = KdTree::build(&data, 1, 10);
        let pt_comp = vec![0, 0, 1];
        let mut nd_comp = vec![-1i64; tree.n_nodes()];
        tree.update_node_components(&pt_comp, &mut nd_comp);
        assert_eq!(nd_comp[0], -1);
    }

    #[test]
    fn test_agrees_with_brute_force() {
        let pts: &[&[f64]] = &[
            &[0.0, 0.0],
            &[1.0, 0.0],
            &[5.0, 5.0],
            &[6.0, 5.0],
            &[10.0, 10.0],
        ];
        let data = flat(pts);
        let core_sq = vec![0.0f64; 5];
        let pt_comp = vec![0, 0, 1, 1, 2];

        let tree = KdTree::build(&data, 2, 2);
        let mut nd_comp = vec![-1i64; tree.n_nodes()];
        tree.update_node_components(&pt_comp, &mut nd_comp);

        for qi in 0..5 {
            let (tj, td) = tree.nearest_other_component(&data, qi, &core_sq, &pt_comp, &nd_comp);

            let mut bf_j = qi;
            let mut bf_d = f64::MAX;
            for j in 0..5 {
                if pt_comp[j] == pt_comp[qi] {
                    continue;
                }
                let sq: f64 = pts[qi]
                    .iter()
                    .zip(pts[j])
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                let mr = sq.max(core_sq[qi]).max(core_sq[j]);
                if mr < bf_d {
                    bf_d = mr;
                    bf_j = j;
                }
            }

            assert_eq!(tj, bf_j, "Mismatch for query point {qi}");
            assert!(
                (td - bf_d).abs() < 1e-10,
                "Distance mismatch for point {qi}"
            );
        }
    }

    #[test]
    fn test_knn_query_basic() {
        let data: Vec<f64> = (0..5).flat_map(|i| [i as f64]).collect();
        let tree = KdTree::build(&data, 1, 2);

        let result = tree.knn_query(&data, &[2.0], 2, Some(2));
        assert_eq!(result.len(), 2);
        let indices: Vec<usize> = result.iter().map(|r| r.0).collect();
        assert!(indices.contains(&1));
        assert!(indices.contains(&3));
        for &(_, d) in &result {
            assert!((d - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_knn_query_without_exclude() {
        let data = flat(&[&[0.0, 0.0], &[1.0, 0.0], &[3.0, 0.0]]);
        let tree = KdTree::build(&data, 2, 10);
        let result = tree.knn_query(&data, &[0.0, 0.0], 1, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0);
        assert!(result[0].1.abs() < 1e-10);
    }

    #[test]
    fn test_knn_query_sorted_ascending() {
        let data: Vec<f64> = vec![0.0, 1.0, 5.0, 10.0, 20.0];
        let tree = KdTree::build(&data, 1, 2);
        let result = tree.knn_query(&data, &[0.0], 4, Some(0));
        assert_eq!(result.len(), 4);
        for i in 1..result.len() {
            assert!(result[i].1 >= result[i - 1].1);
        }
    }

    #[test]
    fn test_knn_query_batch_sizes() {
        let data: Vec<f64> = (0..20).flat_map(|i| [i as f64, 0.0]).collect();
        let tree = KdTree::build(&data, 2, 5);
        let (indices, distances) = tree.knn_query_batch(&data, 3);
        assert_eq!(indices.len(), 20);
        assert_eq!(distances.len(), 20);
        for idx in &indices {
            assert_eq!(idx.len(), 3);
        }
    }

    #[test]
    fn test_knn_query_batch_excludes_self() {
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let tree = KdTree::build(&data, 1, 5);
        let (indices, _) = tree.knn_query_batch(&data, 2);
        for (i, idx) in indices.iter().enumerate() {
            assert!(
                !idx.contains(&i),
                "Point {i} found itself in its own k-NN result"
            );
        }
    }

    #[test]
    fn test_knn_agrees_with_brute_force() {
        let pts: &[&[f64]] = &[
            &[0.0, 0.0],
            &[1.0, 0.0],
            &[0.0, 1.0],
            &[5.0, 5.0],
            &[6.0, 5.0],
            &[5.0, 6.0],
            &[10.0, 10.0],
        ];
        let data = flat(pts);
        let tree = KdTree::build(&data, 2, 2);
        let k = 3;

        for qi in 0..pts.len() {
            let tree_result = tree.knn_query(&data, pts[qi], k, Some(qi));

            let mut all: Vec<(usize, f64)> = (0..pts.len())
                .filter(|&j| j != qi)
                .map(|j| {
                    let d: f64 = pts[qi]
                        .iter()
                        .zip(pts[j])
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    (j, d)
                })
                .collect();
            all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
            let bf: Vec<(usize, f64)> = all.into_iter().take(k).collect();

            assert_eq!(
                tree_result.len(),
                bf.len(),
                "Length mismatch for query {qi}"
            );
            for (t, b) in tree_result.iter().zip(bf.iter()) {
                assert_eq!(t.0, b.0, "Index mismatch for query {qi}");
                assert!(
                    (t.1 - b.1).abs() < 1e-10,
                    "Distance mismatch for query {qi}: tree={}, brute={}",
                    t.1,
                    b.1
                );
            }
        }
    }

    #[test]
    fn test_knn_query_k_larger_than_data() {
        let data = flat(&[&[0.0], &[1.0], &[2.0]]);
        let tree = KdTree::build(&data, 1, 10);
        let result = tree.knn_query(&data, &[0.0], 5, Some(0));
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_knn_query_empty_tree() {
        let data: Vec<f64> = vec![];
        let tree = KdTree::build(&data, 2, 10);
        let result = tree.knn_query(&data, &[1.0, 2.0], 3, None);
        assert!(result.is_empty());
    }
}
