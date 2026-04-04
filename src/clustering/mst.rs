//! Boruvka's MST over mutual reachability distance.
//!
//! Builds the minimum spanning tree used as input to HDBSCAN's cluster
//! hierarchy. Edge weights are mutual reachability distances: the Euclidean
//! distance between two points, inflated to the larger of their core
//! distances. This pushes sparse points further apart and makes cluster
//! boundaries more robust to noise.
//!
//! The KD-tree from [`crate::clustering::kd_tree`] accelerates the
//! per-round nearest-other-component search via same-component subtree
//! pruning. Internal arithmetic uses squared distances; weights are
//! square-rooted to Euclidean before returning.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::prelude::*;
use crate::utils::disjoint_set::DisjointSet;

use crate::clustering::kd_tree::KdTree;

/// An edge in the minimum spanning tree.
#[derive(Clone, Debug)]
pub struct MstEdge<T> {
    /// Index of the first endpoint
    pub u: usize,
    /// Index of the second endpoint
    pub v: usize,
    /// Mutual reachability distance (Euclidean, not squared)
    pub weight: T,
}

/// Compute the mutual reachability distance MST via Boruvka's algorithm,
/// accelerated by a KD-tree with same-component subtree pruning.
///
/// Builds the KD-tree first, then uses a parallel batch k-NN query to
/// compute squared core distances (k-th nearest neighbour distance per
/// point) in O(n log n) expected time, replacing the previous O(n^2)
/// brute-force scan.
///
/// Boruvka's proceeds in O(log n) rounds. Each round finds, for every
/// connected component, the minimum-weight cross-edge to a different
/// component, then merges all such edges simultaneously.
///
/// All internal comparisons use squared Euclidean distances. Final MST edge
/// weights are converted to Euclidean (sqrt) before returning.
///
/// ### Params
///
/// * `data` - Point coordinates; `data[i]` is the embedding vector for point
///   i. Typically 4-16 dimensional from the EVoC embedding stage.
/// * `min_samples` - Controls density estimation: core distance for point i is
///   the distance to its `min_samples`-th nearest neighbour.
///
/// ### Returns
///
/// `Vec<MstEdge<T>>` with `n - 1` edges forming the MST under mutual
/// reachability distance.
pub fn build_mst<T>(data: &[Vec<T>], min_samples: usize) -> Vec<MstEdge<T>>
where
    T: EvocFloat,
{
    let n = data.len();
    if n <= 1 {
        return Vec::new();
    }

    // build the KD-tree first -used for both core distances and Boruvka
    let tree = KdTree::build(data, 40);

    // core distances via parallel k-NN query on the tree: O(n log n)
    let k = min_samples.min(n - 1);
    let core_sq = if k == 0 {
        vec![T::zero(); n]
    } else {
        let (_, dists) = tree.knn_query_batch(data, k);
        dists
            .into_iter()
            .map(|d| *d.last().unwrap_or(&T::zero()))
            .collect()
    };

    let mut ds = DisjointSet::new(n);
    let mut mst = Vec::with_capacity(n - 1);
    let mut pt_comp = vec![0usize; n];
    let mut nd_comp = vec![-1i64; tree.n_nodes()];

    loop {
        for i in 0..n {
            pt_comp[i] = ds.find(i);
        }
        tree.update_node_components(&pt_comp, &mut nd_comp);

        let best: Vec<(usize, T)> = (0..n)
            .into_par_iter()
            .map(|i| tree.nearest_other_component(data, i, &core_sq, &pt_comp, &nd_comp))
            .collect();

        let mut best_per_comp: FxHashMap<usize, (usize, usize, T)> = FxHashMap::default();
        for (i, &(j, d)) in best.iter().enumerate() {
            if j == i {
                continue;
            }
            let c = pt_comp[i];
            best_per_comp
                .entry(c)
                .and_modify(|e| {
                    if d < e.2 {
                        *e = (i, j, d);
                    }
                })
                .or_insert((i, j, d));
        }

        if best_per_comp.is_empty() {
            break;
        }

        let mut merged = false;
        for &(u, v, w_sq) in best_per_comp.values() {
            if ds.union(u, v) {
                mst.push(MstEdge { u, v, weight: w_sq });
                merged = true;
                if mst.len() == n - 1 {
                    for e in &mut mst {
                        e.weight = e.weight.sqrt();
                    }
                    return mst;
                }
            }
        }

        if !merged {
            break;
        }
    }

    for e in &mut mst {
        e.weight = e.weight.sqrt();
    }
    mst
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mst_empty() {
        let data: Vec<Vec<f64>> = Vec::new();
        assert!(build_mst(&data, 1).is_empty());
    }

    #[test]
    fn test_mst_single_point() {
        let data = vec![vec![1.0, 2.0]];
        assert!(build_mst::<f64>(&data, 1).is_empty());
    }

    #[test]
    fn test_mst_two_points() {
        let data = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
        let mst: Vec<MstEdge<f64>> = build_mst(&data, 1);
        assert_eq!(mst.len(), 1);
        assert!((mst[0].weight - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_mst_simple_triangle() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.5, 0.866]];
        let mst = build_mst(&data, 1);
        assert_eq!(mst.len(), 2);
        let total: f64 = mst.iter().map(|e| e.weight).sum();
        assert!(total > 1.9 && total < 2.1);
    }

    #[test]
    fn test_mst_two_clusters() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
        ];
        let mst = build_mst(&data, 1);
        assert_eq!(mst.len(), 5);
        let long_edges: Vec<_> = mst.iter().filter(|e| e.weight > 5.0).collect();
        assert_eq!(long_edges.len(), 1);
    }

    #[test]
    fn test_mst_min_samples_increases_distances() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            vec![3.0, 0.0],
        ];
        let total_k1: f64 = build_mst(&data, 1).iter().map(|e| e.weight).sum();
        let total_k2: f64 = build_mst(&data, 2).iter().map(|e| e.weight).sum();
        assert!(total_k2 >= total_k1);
    }

    #[test]
    fn test_mst_spans_all_points() {
        let data = vec![vec![0.0], vec![1.0], vec![3.0], vec![6.0], vec![10.0]];
        let mst = build_mst(&data, 1);
        assert_eq!(mst.len(), 4);

        let mut seen = [false; 5];
        for e in &mst {
            seen[e.u] = true;
            seen[e.v] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn test_mst_collinear() {
        let data: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
        let mst = build_mst(&data, 1);
        assert_eq!(mst.len(), 4);
        let total: f64 = mst.iter().map(|e| e.weight).sum();
        assert!((total - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_mst_weights_are_euclidean() {
        let data = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
        let mst: Vec<MstEdge<f64>> = build_mst(&data, 1);
        assert!((mst[0].weight - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_mst_mutual_reachability_inflates() {
        let data = vec![vec![0.0, 0.0], vec![0.01, 0.0], vec![10.0, 0.0]];
        let mst = build_mst(&data, 2);
        assert_eq!(mst.len(), 2);
        for e in &mst {
            assert!(e.weight > 1.0, "Expected inflated weight, got {}", e.weight);
        }
    }

    #[test]
    fn test_mst_higher_dimensional() {
        let data = vec![vec![0.0; 8], vec![1.0; 8], vec![2.0; 8]];
        let mst: Vec<MstEdge<f64>> = build_mst(&data, 1);
        assert_eq!(mst.len(), 2);
        for e in &mst {
            assert!(e.weight > 0.0);
            assert!(e.weight.is_finite());
        }
    }
}
