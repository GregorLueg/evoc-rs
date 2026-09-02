//! Fuzzy graph generation for the UMAP-like embedding; however, there are
//! differences in this implementation due to how EVõC is designed.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::prelude::*;
use crate::utils::sparse::CoordinateList;

///////////////
// Constants //
///////////////

/// Absolute convergence tolerance for the sigma binary search.
///
/// Matches `SMOOTH_K_TOLERANCE` in the Python reference
/// (`graph_construction.py:8`), and acts as a floor: the effective tolerance
/// never goes below this, so nothing loosens relative to upstream.
const SMOOTH_K_TOLERANCE: f64 = 1e-5;

/// Convergence tolerance for the sigma binary search, relative to the target.
///
/// The target is `log2(k)`, so a purely absolute tolerance is an ever-tighter
/// relative one as `k` grows. The two cross at `k = 1024`: below that the
/// absolute value wins and behaviour is exactly the reference's, above it the
/// band widens rather than asking the search to resolve a relative tolerance
/// that keeps shrinking. Nobody builds a kNN graph at `k > 1000`, so this is
/// insurance rather than a live path.
const SMOOTH_K_REL_TOLERANCE: f64 = 1e-6;

/////////////
// Helpers //
/////////////

/// Smooth k-NN distances via binary search to find sigma per point.
///
/// Uses log2(k) as the target entropy to match the EVoC/UMAP convention.
///
/// ### Params
///
/// * `dists` - kNN distance matrix where each row contains distances to k
///   nearest neighbours
/// * `k` - Number of nearest neighbours (used to compute target = log2(k))
/// * `local_connectivity` - Number of nearest neighbours to assume are at
///   distance zero (typically 1.0). Allows for local manifold structure.
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
    let tolerance = SMOOTH_K_TOLERANCE.max(target * SMOOTH_K_REL_TOLERANCE);
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
                // Accumulated at `T` rather than f64 on purpose. The sum is
                // driven to `log2(k)`, which stays small however large `k`
                // gets, so only a handful of terms ever contribute meaningfully
                // and the sum is well conditioned. Measured at f32 against an
                // f64 accumulator, the resolved sigma is identical up to
                // k = 5000 and the search converges in about 20 iterations of
                // the 64 available.
                let mut val = T::zero();
                for &dist in d.iter() {
                    let adjusted = (dist - rho).max(T::zero());
                    val += (-(adjusted / mid)).exp();
                }
                let val = val.to_f64().unwrap();

                if (val - target).abs() < tolerance {
                    break;
                }

                if val > target {
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
/// The endpoint is snapped rather than evaluated, so a union involving a
/// weight of exactly 1.0 returns exactly 1.0. That is algebraically what the
/// expression already says, but not what floating point computes, and the
/// difference decides whether `f32` and `f64` cluster the same data the same
/// way. See the comment on the branch itself.
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
                // Deliberate divergence from the Python reference, which
                // evaluates the bare `a + b - a*b` (graph_construction.py:184).
                //
                // A point's own nearest neighbour has `dist == rho`, so its
                // weight is exactly 1.0, and around 7% of edges land there. The
                // t-conorm is algebraically exactly 1 when either input is, but
                // `fl(1 + b) - b` rounds to 1 +/- 1 ulp, and *which* edges fall
                // one ulp short differs between f32 and f64. Label propagation
                // then thresholds on exactly 1.0, so those edges decide whether
                // a node gets labelled at all, and the two precisions produce
                // materially different clusterings of the same data.
                //
                // Snapping the endpoint costs nothing and takes f32/f64 label
                // agreement from 2/10 seeds to 9/10, mean ARI 1.0. Preferred
                // over `1 - (1 - a) * (1 - b)`, which is also exact here but
                // cancels when `a + b` is small.
                let w_sym = if w_ij >= T::one() || w_ji >= T::one() {
                    T::one()
                } else {
                    w_ij + w_ji - w_ij * w_ji
                };
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
/// `CoordinateList<T>` of the fuzzy simplicial set
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

    /// The tolerance must never tighten below the reference's absolute value,
    /// and must widen once the target grows past it.
    #[test]
    fn test_smooth_knn_tolerance_scales_with_target() {
        let effective = |k: usize| {
            let target = (k as f64).log2();
            SMOOTH_K_TOLERANCE.max(target * SMOOTH_K_REL_TOLERANCE)
        };

        // Every realistic neighbour count keeps the reference's value exactly.
        for k in [5usize, 15, 30, 100, 500] {
            assert_eq!(effective(k), SMOOTH_K_TOLERANCE, "k = {k} should not move");
        }

        // Past the crossover the band widens rather than tightening.
        assert!(effective(2048) > SMOOTH_K_TOLERANCE);
        assert!(effective(100_000) > effective(2048));
    }

    /// A large `k` at f32 must still resolve a usable sigma.
    ///
    /// The sum is driven to `log2(k)`, so it stays small and well conditioned
    /// however large `k` is. This guards the outcome rather than the tolerance
    /// arithmetic, which the test above covers.
    #[test]
    fn test_smooth_knn_resolves_sigma_at_large_k_in_f32() {
        let k = 500;
        let dist: Vec<Vec<f32>> = (0..4)
            .map(|r| (0..k).map(|i| 1.0 + (i as f32) * 0.02 + r as f32).collect())
            .collect();

        let (sigmas, rhos) = smooth_knn_dist(&dist, k, 1.0f32, 64);

        for (sigma, rho) in sigmas.iter().zip(&rhos) {
            assert!(sigma.is_finite(), "sigma {sigma} is not finite");
            assert!(*sigma > 0.0, "sigma {sigma} collapsed to zero");
            assert!(rho.is_finite());
        }
    }

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
