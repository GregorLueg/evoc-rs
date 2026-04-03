//! Fuzzy graph generation for the UMAP-like embedding; however, there are
//! differences in this implementation due to how EVõC is designed.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::prelude::*;
use crate::utils::sparse::CoordinateList;

/////////////
// Helpers //
/////////////

/// Smooth k-NN distances via binary search to find sigma per point.
///
/// Uses log2(k) as the target entropy to match the EVoC/UMAP convention.
///
/// ### Params
///
/// * `dist` - kNN distance matrix where each row contains distances to k
///   nearest neighbours
/// * `k` - Number of nearest neighbours (used to compute target = log2(k))
/// * `local_connectivity` - Number of nearest neighbours to assume are at
///   distance zero (typically 1.0). Allows for local manifold structure.
/// * `bandwidth` - Convergence tolerance for binary search (typically 1e-5)
/// * `n_iter` - Maximum number of binary search iterations (typically 64)
///
/// ### Returns
///
/// * `sigmas` - Smoothing bandwidth for each point
/// * `rhos` - Distance to the `local_connectivity`-th nearest neighbour for
///   each point
pub fn smooth_knn_dist<T>(
    dists: &[Vec<T>],
    k: usize,
    local_connectivity: T,
    n_iter: usize,
) -> (Vec<T>, Vec<T>)
where
    T: EvocFloat,
{
    let target = (k as f64).log2();
    let tolerance = 1e-5;
    let two = T::one() + T::one();

    dists
        .par_iter()
        .map(|d| {
            let rho = if local_connectivity > T::zero() {
                let idx = (local_connectivity - T::one())
                    .max(T::zero())
                    .floor()
                    .to_usize()
                    .unwrap()
                    .min(d.len() - 1);
                let frac = (local_connectivity - T::one()).max(T::zero())
                    - (local_connectivity - T::one()).max(T::zero()).floor();
                if frac > T::zero() && idx + 1 < d.len() {
                    d[idx] * (T::one() - frac) + d[idx + 1] * frac
                } else {
                    d[idx]
                }
            } else {
                T::zero()
            };

            let mut lo = T::zero();
            let mut hi = T::max_value();
            let mut mid = T::one();

            for _ in 0..n_iter {
                let mut val = T::zero();
                for &dist in d.iter() {
                    let adjusted = (dist - rho).max(T::zero());
                    val += (-(adjusted / mid)).exp();
                }

                if (val.to_f64().unwrap() - target).abs() < tolerance {
                    break;
                }

                if val.to_f64().unwrap() > target {
                    hi = mid;
                    mid = (lo + hi) / two;
                } else {
                    lo = mid;
                    if hi == T::max_value() {
                        mid = mid * two;
                    } else {
                        mid = (lo + hi) / two;
                    }
                }
            }

            (mid, rho)
        })
        .unzip()
}

/// Convert k-NN graph to COO sparse format with Gaussian membership strengths
///
/// ### Params
///
/// * `knn_indices` - Indices of k nearest neighbours for each point
/// * `knn_dists` - Distances to k nearest neighbours for each point
/// * `sigmas` - Smoothing bandwidth for each point (from `smooth_knn_dist`)
/// * `rhos` - Local connectivity distance for each point (from
///   `smooth_knn_dist`)
///
/// ### Returns
///
/// Sparse graph in COO format where weights represent membership strengths
/// computed as exp(-(max(0, dist - rho) / sigma))
pub fn knn_to_coo<T>(
    knn_indices: &[Vec<usize>],
    knn_dists: &[Vec<T>],
    sigmas: &[T],
    rhos: &[T],
) -> CoordinateList<T>
where
    T: EvocFloat,
{
    let n = knn_indices.len();
    let capacity: usize = knn_indices.iter().map(|v| v.len()).sum();

    let mut rows = Vec::with_capacity(capacity);
    let mut cols = Vec::with_capacity(capacity);
    let mut vals = Vec::with_capacity(capacity);

    for (i, (neighbours, dists)) in knn_indices.iter().zip(knn_dists.iter()).enumerate() {
        let sigma = sigmas[i];
        let rho = rhos[i];

        for (&j, &dist) in neighbours.iter().zip(dists.iter()) {
            if i == j {
                continue;
            }
            let adjusted = (dist - rho).max(T::zero());
            let weight = if sigma > T::zero() {
                (-(adjusted / sigma)).exp()
            } else if adjusted > T::zero() {
                T::zero()
            } else {
                T::one()
            };

            rows.push(i);
            cols.push(j);
            vals.push(weight);
        }
    }

    CoordinateList {
        row_indices: rows,
        col_indices: cols,
        values: vals,
        n_samples: n,
    }
}

/// Symmetrise graph using probabilistic t-conorm (fuzzy set union)
///
/// Creates symmetric graph by combining directed edges using fuzzy union:
/// w_sym = w_ij + w_ji - w_ij * w_ji. In the case of this implementation the
/// mix weight is always 1.0.
///
/// ### Params
///
/// * `graph` - Input directed graph in COO format
///
/// ### Returns
///
/// Symmetrised graph in COO format
pub fn symmetrise_graph<T>(graph: &CoordinateList<T>) -> CoordinateList<T>
where
    T: EvocFloat,
{
    let n = graph.n_samples;

    let mut forward: Vec<FxHashMap<usize, T>> = vec![FxHashMap::default(); n];
    let mut backward: Vec<FxHashMap<usize, T>> = vec![FxHashMap::default(); n];

    for ((&i, &j), &w) in graph
        .row_indices
        .iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
    {
        forward[i].insert(j, w);
        backward[j].insert(i, w);
    }

    let edges: Vec<Vec<(usize, T)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut combined = FxHashMap::default();
            for &j in forward[i].keys().chain(backward[i].keys()) {
                let w_ij = forward[i].get(&j).copied().unwrap_or(T::zero());
                let w_ji = backward[i].get(&j).copied().unwrap_or(T::zero());
                let w_sym = w_ij + w_ji - w_ij * w_ji;
                if w_sym > T::zero() {
                    combined.insert(j, w_sym);
                }
            }
            combined.into_iter().collect()
        })
        .collect();

    let capacity: usize = edges.iter().map(|v| v.len()).sum();
    let mut rows = Vec::with_capacity(capacity);
    let mut cols = Vec::with_capacity(capacity);
    let mut vals = Vec::with_capacity(capacity);

    for (i, neighbours) in edges.into_iter().enumerate() {
        for (j, w) in neighbours {
            rows.push(i);
            cols.push(j);
            vals.push(w);
        }
    }

    CoordinateList {
        row_indices: rows,
        col_indices: cols,
        values: vals,
        n_samples: n,
    }
}

/// Convert COO sparse graph to adjacency list representation
///
/// More efficient for SGD optimisation where we need to iterate over neighbours
/// of each vertex.
///
/// ### Params
///
/// * `graph` - Sparse graph in COO format
///
/// ### Returns
///
/// Adjacency list where `result[i]` contains `(neighbour_index, edge_weight)`
pub fn coo_to_adjacency_list<T>(graph: &CoordinateList<T>) -> Vec<Vec<(usize, T)>>
where
    T: EvocFloat,
{
    let mut adj = vec![Vec::new(); graph.n_samples];
    for ((&i, &j), &w) in graph
        .row_indices
        .iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
    {
        adj[i].push((j, w));
    }
    adj
}

//////////
// Main //
//////////

/// Full fuzzy simplicial set from k-NN data
///
/// ### Params
///
/// * `knn_indices` - Indices of k nearest neighbours for each point
/// * `knn_dists` - Distances to k nearest neighbours for each point
/// * `effective_n_neighbors` - Number of nearest neighbours (used to compute
///   target = log2(k))
/// * `symmetrise` - Symmetrise the graph.
///
/// ### Returns
///
/// CoordinateList<T> of the fuzzy simplicial set
pub fn build_fuzzy_simplicial_set<T>(
    knn_indices: &[Vec<usize>],
    knn_dists: &[Vec<T>],
    effective_n_neighbors: T,
    symmetrise: bool,
) -> CoordinateList<T>
where
    T: EvocFloat,
{
    let k = effective_n_neighbors
        .to_usize()
        .unwrap_or(knn_dists[0].len());
    let (sigmas, rhos) = smooth_knn_dist(knn_dists, k, T::one(), 64);
    let graph = knn_to_coo(knn_indices, knn_dists, &sigmas, &rhos);

    if symmetrise {
        symmetrise_graph(&graph)
    } else {
        graph
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_data_gen {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_smooth_knn_dist_basic() {
        let dist = vec![vec![1.0, 2.0], vec![1.5, 3.0], vec![0.5, 1.5]];

        let (sigmas, rhos) = smooth_knn_dist(&dist, 2, 1.0, 64);

        assert_eq!(sigmas.len(), 3);
        assert_eq!(rhos.len(), 3);

        assert_relative_eq!(rhos[0], 1.0, epsilon = 1e-4);
        assert_relative_eq!(rhos[1], 1.5, epsilon = 1e-4);
        assert_relative_eq!(rhos[2], 0.5, epsilon = 1e-4);

        for sigma in sigmas.iter() {
            assert!(*sigma > 0.0);
        }
    }

    #[test]
    fn test_smooth_knn_dist_zero_local_connectivity() {
        let dist = vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]];

        let (sigmas, rhos) = smooth_knn_dist(&dist, 2, 0.0, 64);

        assert!(rhos.iter().all(|&r| r == 0.0));
        assert_eq!(sigmas.len(), 2);
    }

    #[test]
    fn test_knn_to_coo_basic() {
        let knn_indices = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let knn_dists = vec![vec![1.0, 2.0], vec![1.0, 1.5], vec![2.0, 1.5]];
        let sigmas = vec![1.0, 1.0, 1.0];
        let rhos = vec![0.0, 0.0, 0.0];

        let graph = knn_to_coo(&knn_indices, &knn_dists, &sigmas, &rhos);

        assert_eq!(graph.n_samples, 3);
        assert_eq!(graph.row_indices.len(), 6);
        assert_eq!(graph.col_indices.len(), 6);
        assert_eq!(graph.values.len(), 6);

        for &w in &graph.values {
            assert!((0.0..=1.0).contains(&w));
        }
    }

    #[test]
    fn test_knn_to_coo_self_loop_excluded() {
        let knn_indices = vec![vec![0, 1], vec![1, 0]];
        let knn_dists = vec![vec![0.0, 1.0], vec![0.0, 1.0]];
        let sigmas = vec![1.0, 1.0];
        let rhos = vec![0.0, 0.0];

        let graph = knn_to_coo(&knn_indices, &knn_dists, &sigmas, &rhos);

        assert_eq!(graph.values.len(), 2);
        assert!(
            graph
                .row_indices
                .iter()
                .zip(&graph.col_indices)
                .all(|(&i, &j)| i != j)
        );
    }

    #[test]
    fn test_symmetrise_graph_fuzzy_union() {
        // Directed: 0->1 with w=0.8, 1->0 with w=0.6
        let graph = CoordinateList {
            row_indices: vec![0, 1],
            col_indices: vec![1, 0],
            values: vec![0.8, 0.6],
            n_samples: 2,
        };

        let sym = symmetrise_graph(&graph);

        assert_eq!(sym.n_samples, 2);
        assert_eq!(sym.row_indices.len(), 2);

        // Full fuzzy union: w_sym = w_ij + w_ji - w_ij * w_ji
        // = 0.8 + 0.6 - 0.48 = 0.92 for both directions
        let expected = 0.8 + 0.6 - 0.8 * 0.6;

        for idx in 0..sym.row_indices.len() {
            assert_relative_eq!(sym.values[idx], expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_symmetrise_graph_one_direction() {
        // Only 0->1 exists, no reverse edge
        let graph = CoordinateList {
            row_indices: vec![0],
            col_indices: vec![1],
            values: vec![0.7],
            n_samples: 2,
        };

        let sym = symmetrise_graph(&graph);

        // Both 0->1 and 1->0 should appear
        // w_ij=0.7, w_ji=0.0 => w_sym = 0.7 + 0.0 - 0.0 = 0.7
        assert_eq!(sym.row_indices.len(), 2);

        for idx in 0..sym.row_indices.len() {
            assert_relative_eq!(sym.values[idx], 0.7, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_coo_to_adjacency_list() {
        let graph = CoordinateList {
            row_indices: vec![0, 0, 1, 2],
            col_indices: vec![1, 2, 2, 0],
            values: vec![0.5, 0.3, 0.8, 0.9],
            n_samples: 3,
        };

        let adj = coo_to_adjacency_list(&graph);

        assert_eq!(adj.len(), 3);
        assert_eq!(adj[0].len(), 2);
        assert_eq!(adj[1].len(), 1);
        assert_eq!(adj[2].len(), 1);

        assert!(adj[0].contains(&(1, 0.5)));
        assert!(adj[0].contains(&(2, 0.3)));
        assert!(adj[1].contains(&(2, 0.8)));
        assert!(adj[2].contains(&(0, 0.9)));
    }

    #[test]
    fn test_coo_to_adjacency_list_empty() {
        let graph: CoordinateList<f64> = CoordinateList {
            row_indices: vec![],
            col_indices: vec![],
            values: vec![],
            n_samples: 3,
        };

        let adj = coo_to_adjacency_list(&graph);

        assert_eq!(adj.len(), 3);
        assert!(adj[0].is_empty());
        assert!(adj[1].is_empty());
        assert!(adj[2].is_empty());
    }
}
