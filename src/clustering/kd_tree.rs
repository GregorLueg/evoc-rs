//! KD-tree for component-aware nearest neighbour queries in Boruvka's MST.
//!
//! Flattened binary space partition tree. Splits on the widest-spread
//! dimension at the median. Accelerates the per-round "find nearest point
//! in a different component" query via same-component subtree pruning.

use crate::prelude::*;

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
    /// Build a KD-tree over `data`.
    ///
    /// ### Params
    ///
    /// * `data` - Point coordinates; `data[i]` is the vector for point `i`
    /// * `leaf_size` - Maximum points per leaf. Tree depth is chosen so leaves
    ///   contain between `leaf_size` and `2 * leaf_size` points.
    ///
    /// ### Returns
    ///
    /// A `KdTree` ready for queries
    pub fn build(data: &[Vec<T>], leaf_size: usize) -> Self {
        let n = data.len();
        let dim = data.first().map_or(0, |v| v.len());
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
        if n > 0 {
            tree.build_node(data, 0, n, 0);
        }
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
    /// * `data` - Full point array passed to `build`
    /// * `start` - Inclusive start offset into `idx_array` for this node
    /// * `end` - Exclusive end offset into `idx_array` for this node
    /// * `node` - Index of the current node in the flattened tree
    fn build_node(&mut self, data: &[Vec<T>], start: usize, end: usize, node: usize) {
        let dim = self.dim;
        let bo = node * dim;

        // AABB from first point, then expand
        let first = &data[self.idx_array[start]];
        self.lower[bo..bo + dim].copy_from_slice(first);
        self.upper[bo..bo + dim].copy_from_slice(first);
        for i in (start + 1)..end {
            let pt = &data[self.idx_array[i]];
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

        // split on widest dimension
        let mut split_dim = 0;
        let mut max_spread = T::zero();
        for d in 0..dim {
            let s = self.upper[bo + d] - self.lower[bo + d];
            if s > max_spread {
                max_spread = s;
                split_dim = d;
            }
        }

        // partition around median via introselect
        let mid = start + (end - start) / 2;
        self.idx_array[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
            data[a][split_dim].partial_cmp(&data[b][split_dim]).unwrap()
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
            r = r + diff * diff;
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
    /// * `data` - Full point array (same one passed to `build`)
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
        data: &[Vec<T>],
        qi: usize,
        core_sq: &[T],
        pt_comp: &[usize],
        nd_comp: &[i64],
    ) -> (usize, T) {
        let mut best_j = qi;
        let mut best_d = T::max_value();
        let lb = self.aabb_sq(0, &data[qi]);
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
    /// * `best_d` - Running best squared mutual reachability distance, updated in place
    #[allow(clippy::too_many_arguments)]
    fn noc_recurse(
        &self,
        data: &[Vec<T>],
        qi: usize,
        core_sq: &[T],
        pt_comp: &[usize],
        nd_comp: &[i64],
        node: usize,
        lb: T,
        best_j: &mut usize,
        best_d: &mut T,
    ) {
        // Prune 1: AABB lower bound
        if lb > *best_d {
            return;
        }
        // Prune 2: query core distance
        if core_sq[qi] > *best_d {
            return;
        }
        // Prune 3: entire node is same component as query
        let qc = pt_comp[qi] as i64;
        if nd_comp[node] == qc {
            return;
        }

        if self.is_leaf[node] {
            for i in self.idx_start[node]..self.idx_end[node] {
                let j = self.idx_array[i];
                if pt_comp[j] as i64 == qc {
                    continue;
                }
                if core_sq[j] > *best_d {
                    continue;
                }
                let sq = T::euclidean_simd(&data[qi], &data[j]);
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
        let lb_l = self.aabb_sq(left, &data[qi]);
        let lb_r = self.aabb_sq(right, &data[qi]);

        // Visit closer child first for tighter early pruning
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_empty() {
        let data: Vec<Vec<f64>> = vec![];
        let tree = KdTree::build(&data, 10);
        assert_eq!(tree.n_nodes(), 0);
    }

    #[test]
    fn test_build_single() {
        let data = vec![vec![1.0, 2.0]];
        let tree = KdTree::build(&data, 10);
        assert!(tree.is_leaf[0]);
        assert_eq!(tree.idx_start[0], 0);
        assert_eq!(tree.idx_end[0], 1);
    }

    #[test]
    fn test_build_covers_all_points() {
        let data: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64, 0.0]).collect();
        let tree = KdTree::build(&data, 10);

        // Every point index should appear exactly once in idx_array
        let mut seen = [false; 100];
        for &idx in &tree.idx_array {
            assert!(!seen[idx]);
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn test_aabb_sq_inside() {
        let data = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let tree = KdTree::build(&data, 10);
        // Point inside the root AABB should have zero distance
        let d = tree.aabb_sq(0, &[5.0, 5.0]);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn test_aabb_sq_outside() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let tree = KdTree::build(&data, 10);
        // Point at (3, 0): distance to box [0,1]x[0,1] is 2 in x, 0 in y
        let d: f64 = tree.aabb_sq(0, &[3.0, 0.5]);
        assert!((d - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_nearest_other_component_simple() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![10.0, 0.0]];
        let tree = KdTree::build(&data, 10);
        let core_sq = vec![0.0; 3]; // zero core distances
        let pt_comp = vec![0, 0, 1]; // points 0,1 in comp 0; point 2 in comp 1
        let mut nd_comp = vec![-1i64; tree.n_nodes()];
        tree.update_node_components(&pt_comp, &mut nd_comp);

        // From point 0, nearest in a different component is point 2
        let (j, _) = tree.nearest_other_component(&data, 0, &core_sq, &pt_comp, &nd_comp);
        assert_eq!(j, 2);

        // From point 1, also point 2
        let (j, _) = tree.nearest_other_component(&data, 1, &core_sq, &pt_comp, &nd_comp);
        assert_eq!(j, 2);

        // From point 2, nearest other-component is point 1 (closer than 0)
        let (j, _) = tree.nearest_other_component(&data, 2, &core_sq, &pt_comp, &nd_comp);
        assert_eq!(j, 1);
    }

    #[test]
    fn test_nearest_other_component_with_core_distances() {
        // Points: 0 at origin, 1 at (1,0), 2 at (2,0)
        // Components: 0={0,1}, 1={2}
        // Core sq for point 1 = 100 (inflated), so MR from 1->2 = max(1, 100, core[2])
        let data = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
        let tree = KdTree::build(&data, 10);
        let core_sq = vec![0.0, 100.0, 0.0];
        let pt_comp = vec![0, 0, 1];
        let mut nd_comp = vec![-1i64; tree.n_nodes()];
        tree.update_node_components(&pt_comp, &mut nd_comp);

        // From point 0: MR(0,2) = max(4, 0, 0) = 4
        // From point 1: MR(1,2) = max(1, 100, 0) = 100
        // So point 0 should have the cheaper cross-edge
        let (_, d0) = tree.nearest_other_component(&data, 0, &core_sq, &pt_comp, &nd_comp);
        let (_, d1) = tree.nearest_other_component(&data, 1, &core_sq, &pt_comp, &nd_comp);
        assert!(d0 < d1);
    }

    #[test]
    fn test_node_components_all_same() {
        let data = vec![vec![0.0], vec![1.0], vec![2.0]];
        let tree = KdTree::build(&data, 10);
        let pt_comp = vec![5, 5, 5]; // all same component
        let mut nd_comp = vec![-1i64; tree.n_nodes()];
        tree.update_node_components(&pt_comp, &mut nd_comp);

        // Root should be labelled 5
        assert_eq!(nd_comp[0], 5);
    }

    #[test]
    fn test_node_components_mixed() {
        let data = vec![vec![0.0], vec![1.0], vec![2.0]];
        let tree = KdTree::build(&data, 10);
        let pt_comp = vec![0, 0, 1];
        let mut nd_comp = vec![-1i64; tree.n_nodes()];
        tree.update_node_components(&pt_comp, &mut nd_comp);

        // Root should be mixed
        assert_eq!(nd_comp[0], -1);
    }

    #[test]
    fn test_agrees_with_brute_force() {
        // Verify tree-based query matches naive scan
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![5.0, 5.0],
            vec![6.0, 5.0],
            vec![10.0, 10.0],
        ];
        let core_sq = vec![0.0; 5];
        let pt_comp = vec![0, 0, 1, 1, 2];

        let tree = KdTree::build(&data, 2);
        let mut nd_comp = vec![-1i64; tree.n_nodes()];
        tree.update_node_components(&pt_comp, &mut nd_comp);

        for qi in 0..5 {
            let (tj, td) = tree.nearest_other_component(&data, qi, &core_sq, &pt_comp, &nd_comp);

            // Brute force
            let mut bf_j = qi;
            let mut bf_d = f64::MAX;
            for j in 0..5 {
                if pt_comp[j] == pt_comp[qi] {
                    continue;
                }
                let sq: f64 = data[qi]
                    .iter()
                    .zip(&data[j])
                    .map(|(a, b)| (a - b) * (a - b))
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
}
