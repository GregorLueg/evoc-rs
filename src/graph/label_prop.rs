//! Label propagation initialisation for EVoC node embedding.
//!
//! Implements the recursive partition-and-embed strategy:
//!
//! 1. Partition the graph via seeded label propagation
//! 2. Build sparse reduction/expansion maps
//! 3. Coarsen the graph and (optionally) the data
//! 4. Recurse on the coarsened graph
//! 5. Run a short node embedding on the coarsened graph
//! 6. Expand back to the full graph via partition averaging

use rand::RngExt;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::graph::embedding::{EvocEmbeddingParams, evoc_embedding};
use crate::prelude::*;
use crate::utils::sparse::{CoordinateList, Csr, mat_to_vecs, vecs_to_mat};

///////////////////////
// Label propagation //
///////////////////////

/// Single parallel label propagation iteration (conservative variant).
///
/// Only unlabelled nodes (label < 0) participate. Each such node collects
/// weighted votes from its neighbours and adopts the label with the highest
/// total weight, provided that weight exceeds a threshold of 1.0. Ties are
/// broken by last-encountered, matching the Python Numba behaviour where an
/// unlabelled current node unconditionally accepts on ties.
///
/// Labelled nodes (label >= 0) are passed through unchanged.
///
/// ### Params
///
/// * `csr`    - Weighted adjacency matrix; rows are nodes, stored values are
///   edge weights
/// * `labels` - Current label assignment; negative values indicate unlabelled
///   nodes
///
/// ### Returns
///
/// A new label vector of length `csr.nrows` with propagated assignments
fn label_prop_iteration<T: EvocFloat>(csr: &Csr<T>, labels: &[i64]) -> Vec<i64> {
    (0..csr.nrows)
        .into_par_iter()
        .map(|i| {
            if labels[i] >= 0 {
                return labels[i];
            }
            // accumulate weighted votes from neighbours.
            // vec should be faster than hashmap for the size here.
            let mut votes: Vec<(i64, T)> = Vec::new();
            for idx in csr.indptr[i]..csr.indptr[i + 1] {
                let l = labels[csr.indices[idx]];
                let w = csr.data[idx];
                if let Some(entry) = votes.iter_mut().find(|e| e.0 == l) {
                    entry.1 += w;
                } else {
                    votes.push((l, w));
                }
            }
            // find winning label.
            let mut best = -1i64;
            let mut max_vote = T::one(); // threshold
            let mut _tie_count = 1usize;
            for &(l, v) in &votes {
                if l < 0 {
                    continue;
                }
                if v > max_vote {
                    max_vote = v;
                    best = l;
                    _tie_count = 1;
                } else if v == max_vote {
                    _tie_count += 1;
                    // Conservative variant: current_l is always -1 here, so the
                    // Python unconditionally accepts. Last-encountered wins.
                    best = l;
                }
            }
            best
        })
        .collect()
}

/// BFS mop-up for nodes that remain unlabelled after propagation.
///
/// For each unlabelled node, walks outward through neighbours up to 64 hops
/// looking for any labelled node. If one is found, the unlabelled node adopts
/// that label. If the BFS is exhausted without finding a label, a random
/// existing label is assigned instead.
///
/// Runs sequentially on the assumption that the number of outliers after
/// propagation is very small.
///
/// ### Params
///
/// * `csr`    - Weighted adjacency matrix defining the graph topology
/// * `labels` - Label assignments modified in place; negative values indicate
///   unlabelled nodes
/// * `seed`   - Base RNG seed; each unlabelled node derives its own seed via
///   `seed + i` to keep assignments deterministic and independent
fn label_outliers<T: EvocFloat>(csr: &Csr<T>, labels: &mut [i64], seed: u64) {
    let max_label = match labels.iter().copied().max() {
        Some(m) if m >= 0 => m,
        _ => return, // nothing labelled at all, nothing to propagate from
    };

    for i in 0..csr.nrows {
        if labels[i] >= 0 {
            continue;
        }
        let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
        let mut queue = vec![i];
        let mut found = false;
        for _ in 0..64 {
            if queue.is_empty() {
                break;
            }
            let node = queue.pop().unwrap();
            for idx in csr.indptr[node]..csr.indptr[node + 1] {
                let j = csr.indices[idx];
                if labels[j] >= 0 {
                    labels[i] = labels[j];
                    found = true;
                    break;
                }
                queue.push(j);
            }
            if found {
                break;
            }
        }
        if !found {
            labels[i] = rng.random_range(0..=max_label);
        }
    }
}

/// Remap labels to the contiguous range `0..n_unique`.
///
/// Positive labels are remapped in sorted order. Any remaining `-1` entries
/// (nodes that could not be labelled even after BFS mop-up) are each assigned
/// a distinct new id beyond the mapped range, so no node is left unlabelled.
///
/// ### Params
///
/// * `labels` - Label assignments modified in place
///
/// ### Returns
///
/// Total number of unique labels after remapping
fn remap_labels(labels: &mut [i64]) -> usize {
    let mut sorted: Vec<i64> = labels.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    sorted.retain(|&l| l >= 0);

    let mut map = std::collections::HashMap::with_capacity(sorted.len());
    for (new_id, &old_id) in sorted.iter().enumerate() {
        map.insert(old_id, new_id as i64);
    }

    let mut next = sorted.len() as i64;
    for l in labels.iter_mut() {
        if *l < 0 {
            *l = next;
            next += 1;
        } else {
            *l = map[l];
        }
    }

    next as usize
}

/// Full label propagation pipeline: seed, iterate, mop up outliers, remap.
///
/// Randomly seeds `approx_n_parts` nodes with distinct labels, then runs
/// `n_iter` rounds of `label_prop_iteration`. Nodes still unlabelled after
/// iteration are resolved by `label_outliers`, and finally all labels are
/// remapped to a contiguous `0..n_parts` range by `remap_labels`.
///
/// Seed collisions during initialisation mean the actual number of seed nodes
/// may be less than `approx_n_parts`; the true partition count is returned
/// alongside the assignments.
///
/// ### Params
///
/// * `csr` - Weighted adjacency matrix
/// * `approx_n_parts` - Target number of partitions (actual count may differ)
/// * `n_iter` - Number of propagation iterations to run
/// * `seed` - RNG seed for reproducibility
///
/// ### Returns
///
/// A tuple of:
/// * `Vec<usize>` of length `n` with partition assignments in `0..n_parts`
/// * `usize` with the actual number of partitions produced
fn label_prop_loop<T: EvocFloat>(
    csr: &Csr<T>,
    approx_n_parts: usize,
    n_iter: usize,
    seed: u64,
) -> (Vec<usize>, usize) {
    let n = csr.nrows;
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut labels = vec![-1i64; n];

    // seed random nodes. collisions (same index drawn twice) cause overwrites,
    // so the actual seed count may be less than approx_n_parts. This is fine.
    for i in 0..approx_n_parts {
        let idx = rng.random_range(0..n);
        labels[idx] = i as i64;
    }
    for _ in 0..n_iter {
        labels = label_prop_iteration(csr, &labels);
    }

    label_outliers(csr, &mut labels, seed.wrapping_add(2000));
    let n_parts = remap_labels(&mut labels);
    let partition: Vec<usize> = labels.iter().map(|&l| l as usize).collect();

    (partition, n_parts)
}

/////////
// PCA //
/////////

/// Centre each dimension and scale so the full observed range maps to
/// `[-1, 1]`.
///
/// Each column is shifted by its midpoint and divided by its half-range.
/// Columns with zero range (constant values) are left unchanged.
///
/// ### Params
///
/// * `data` - Row-major data modified in place; all rows must have the same
///   length
fn normalise_to_unit_range<T: EvocFloat>(data: &mut [Vec<T>]) {
    if data.is_empty() {
        return;
    }
    let d = data[0].len();
    for j in 0..d {
        let mut lo = T::infinity();
        let mut hi = T::neg_infinity();
        for row in data.iter() {
            if row[j] < lo {
                lo = row[j];
            }
            if row[j] > hi {
                hi = row[j];
            }
        }
        let range = hi - lo;
        if range > T::zero() {
            let mid = (hi + lo) / T::from(2.0).unwrap();
            let half_range = range / T::from(2.0).unwrap();
            for row in data.iter_mut() {
                row[j] = (row[j] - mid) / half_range;
            }
        }
    }
}

/// PCA initialisation for embedding small graphs.
///
/// Centres the data, computes a thin SVD, projects onto the first
/// `n_components` principal components, and normalises the result to `[-1, 1]`
/// per dimension. The centred matrix is built in `f64` regardless of `T` for
/// numerical stability.
///
/// The effective component count is clamped to `min(n_components, n, d)`.
///
/// ### Params
///
/// * `data` - Input data as `n` rows of length `d`
/// * `n_components` - Number of principal components to retain.
///
/// ### Returns
///
/// A `Vec` of `n` rows each of length `min(n_components, n, d)`, normalised
/// to `[-1, 1]` per dimension
fn pca_init<T: EvocFloat>(data: &[Vec<T>], n_components: usize) -> Vec<Vec<T>> {
    let n = data.len();
    let d = data[0].len();
    let k = n_components.min(n).min(d);

    // Column means
    let mut means = vec![0.0f64; d];
    for row in data {
        for (j, &v) in row.iter().enumerate() {
            means[j] += v.to_f64().unwrap();
        }
    }
    let inv_n = 1.0 / n as f64;
    for m in &mut means {
        *m *= inv_n;
    }

    // build centred matrix in f64 for numerical stability - important, can
    // blow up nastily... learnt that lesson
    let mat = faer::Mat::<f64>::from_fn(n, d, |i, j| data[i][j].to_f64().unwrap() - means[j]);

    let svd = mat.thin_svd().unwrap();
    let u = svd.U();
    let s = svd.S();

    let mut result: Vec<Vec<T>> = (0..n)
        .map(|i| (0..k).map(|j| T::from(u[(i, j)] * s[j]).unwrap()).collect())
        .collect();

    // Normalise to [-1, 1]
    normalise_to_unit_range(&mut result);

    result
}

/// Random initialisation for the base case of embedding.
///
/// Generates `n` points uniformly at random in `[-1, 1]^n_components`, then
/// L2-normalises each row so all points lie on the unit hypersphere. Zero
/// vectors (vanishingly unlikely) are left unchanged.
///
/// ### Params
///
/// * `n` - Number of points
/// * `n_components` - Embedding dimensionality
/// * `seed` - RNG seed for reproducibility
///
/// ### Returns
///
/// A `Vec` of `n` L2-normalised rows each of length `n_components`
fn random_init<T: EvocFloat>(n: usize, n_components: usize, seed: u64) -> Vec<Vec<T>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut result: Vec<Vec<T>> = (0..n)
        .map(|_| {
            (0..n_components)
                .map(|_| T::from(rng.random::<f64>() * 2.0 - 1.0).unwrap())
                .collect()
        })
        .collect();

    // L2-normalise each row
    for row in &mut result {
        let norm: f64 = row
            .iter()
            .map(|v| v.to_f64().unwrap().powi(2))
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            for v in row.iter_mut() {
                *v = T::from(v.to_f64().unwrap() / norm).unwrap();
            }
        }
    }
    result
}

///////////////////////
// Label propagation //
///////////////////////

#[allow(clippy::too_many_arguments)]
fn label_prop_init_inner<T: EvocFloat>(
    graph: &Csr<T>,
    n_components: usize,
    data: Option<&[Vec<T>]>,
    n_embedding_epochs: usize,
    approx_n_parts: usize,
    n_label_prop_iter: usize,
    base_init_threshold: usize,
    scaling: f64,
    seed: u64,
    verbose: bool,
) -> Vec<Vec<T>> {
    let n = graph.nrows;

    // Base case
    if n < base_init_threshold {
        return match data {
            Some(d) if d.len() == n && n_components <= d[0].len() => pca_init(d, n_components),
            _ => random_init(n, n_components, seed),
        };
    }

    // 1. Partition via label propagation
    let (partition, n_parts) = label_prop_loop(graph, approx_n_parts, n_label_prop_iter, seed);

    if verbose {
        println!("  Label prop: {} nodes -> {} partitions", n, n_parts);
    }

    // 2. Build reduction maps
    let base_reduction_map = Csr::<T>::from_partition(&partition, n_parts);
    let norm_reduction_map = base_reduction_map.normalise_cols_l2();
    let data_reducer = norm_reduction_map.transpose().normalise_rows_l1();

    // 3. Coarsen graph and data
    let nrt = norm_reduction_map.transpose();
    let mut reduced_graph = nrt.matmul(graph).matmul(&base_reduction_map);
    reduced_graph.clip_values(T::zero(), T::one());

    let reduced_data: Option<Vec<Vec<T>>> = data.map(|d| {
        let d_mat = vecs_to_mat(d);
        // NOTE: matmul_dense signature may need adjusting for MatRef vs &MatRef
        let rd_mat = data_reducer.matmul_dense(&d_mat.as_ref());
        mat_to_vecs(&rd_mat)
    });

    // 4. Recurse on the coarsened graph
    let reduced_init = label_prop_init_inner(
        &reduced_graph,
        n_components,
        reduced_data.as_deref(),
        n_embedding_epochs.min(255),
        approx_n_parts / 4,
        n_label_prop_iter,
        base_init_threshold,
        scaling,
        seed.wrapping_add(3000),
        verbose,
    );

    // 5. Short node embedding on the coarsened graph
    let adj_list = reduced_graph.to_adjacency_list();
    let emb_params = EvocEmbeddingParams {
        n_epochs: n_embedding_epochs.min(255),
        initial_alpha: T::from(0.001 * n_embedding_epochs as f64).unwrap(),
        ..Default::default()
    };
    let reduced_layout = evoc_embedding(
        &adj_list,
        n_components,
        &emb_params,
        Some(&reduced_init),
        seed.wrapping_add(4000),
        verbose,
    );

    // 6. Expand back to full graph (partition_expander strategy)
    let graph_t = graph.transpose();
    let sym = graph.elementwise_mul(&graph_t);
    let data_expander = sym.matmul(&norm_reduction_map).normalise_rows_l1();
    let nrm_l1 = norm_reduction_map.normalise_rows_l1();

    let layout_mat = vecs_to_mat(&reduced_layout);
    let expanded = data_expander.matmul_dense(&layout_mat.as_ref());
    let direct = nrm_l1.matmul_dense(&layout_mat.as_ref());

    // Average the two expansion terms
    let half = T::from(0.5).unwrap();
    let mut result = faer::Mat::from_fn(n, n_components, |i, j| {
        (expanded[(i, j)] + direct[(i, j)]) * half
    });

    // Centre each component (subtract column mean)
    let inv_n = T::from(1.0 / n as f64).unwrap();
    for j in 0..n_components {
        let mut sum = T::zero();
        for i in 0..n {
            sum += result[(i, j)];
        }
        let mean = sum * inv_n;
        for i in 0..n {
            result[(i, j)] = result[(i, j)] - mean;
        }
    }

    // Scale
    let scale = T::from(scaling).unwrap();
    let out: Vec<Vec<T>> = (0..n)
        .map(|i| (0..n_components).map(|j| scale * result[(i, j)]).collect())
        .collect();

    out
}

//////////
// Main //
//////////

/// Initialise a node embedding via recursive label-propagation partitioning.
///
/// This mirrors the Python `label_propagation_init`: large graphs are
/// recursively coarsened via label propagation, embedded at the coarsest level,
/// and expanded back up.
///
/// ### Params
///
/// * `graph` - The fuzzy simplicial set in COO format
/// * `n_components` - Embedding dimensionality (typically 4-16)
/// * `data` - Original data for PCA base case. If `None`, falls back to random.
/// * `seed` - RNG seed for reproducibility
/// * `verbose` - Print partition sizes at each recursion level
pub fn label_propagation_init<T: EvocFloat>(
    graph: &CoordinateList<T>,
    n_components: usize,
    data: Option<&[Vec<T>]>,
    seed: u64,
    verbose: bool,
) -> Vec<Vec<T>> {
    let csr = Csr::from_coo(graph);
    let n = csr.nrows;

    // Default partition count: scale with sqrt(n), clamped to [256, 16384]
    let approx_n_parts = ((8.0 * (n as f64).sqrt()) as usize)
        .clamp(256, 16384)
        .min(n);

    label_prop_init_inner(
        &csr,
        n_components,
        data,
        50, // n_embedding_epochs
        approx_n_parts,
        20,  // n_label_prop_iter
        64,  // base_init_threshold
        0.1, // scaling
        seed,
        verbose,
    )
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iteration_skips_labelled_nodes() {
        // 3 nodes: 0--1--2, all labelled
        let csr = Csr::new(vec![0, 1, 2, 3], vec![1, 0, 1], vec![1.0, 1.0, 1.0], 3, 3);
        let labels = vec![0i64, 1, 2];
        let result = label_prop_iteration(&csr, &labels);
        assert_eq!(result, labels, "Labelled nodes must not change");
    }

    #[test]
    fn iteration_propagates_strong_label() {
        // Node 0 unlabelled, connected to node 1 (label 5, weight 2.0) and
        // node 2 (label 7, weight 0.3). Only label 5 exceeds threshold of 1.0.
        let csr = Csr::new(vec![0, 2, 2, 2], vec![1, 2], vec![2.0, 0.3], 3, 3);
        let labels = vec![-1i64, 5, 7];
        let result = label_prop_iteration(&csr, &labels);
        assert_eq!(result[0], 5);
    }

    #[test]
    fn iteration_stays_unlabelled_below_threshold() {
        // Node 0 connected to node 1 (label 3, weight 0.5). Below threshold.
        let csr = Csr::new(vec![0, 1, 1], vec![1], vec![0.5], 2, 2);
        let labels = vec![-1i64, 3];
        let result = label_prop_iteration(&csr, &labels);
        assert_eq!(result[0], -1, "Weight below threshold should not propagate");
    }

    #[test]
    fn iteration_ignores_unlabelled_neighbours() {
        // Node 0 connected to node 1, both unlabelled. Vote for -1 is skipped.
        let csr = Csr::new(vec![0, 1, 1], vec![1], vec![5.0], 2, 2);
        let labels = vec![-1i64, -1];
        let result = label_prop_iteration(&csr, &labels);
        assert_eq!(result[0], -1);
    }

    // -- label_outliers -----------------------------------------------------

    #[test]
    fn outliers_finds_labelled_neighbour() {
        // 0 -- 1 -- 2, only node 2 labelled
        let csr = Csr::new(vec![0, 1, 3, 4], vec![1, 0, 2, 1], vec![1.0; 4], 3, 3);
        let mut labels = vec![-1i64, -1, 42];
        label_outliers(&csr, &mut labels, 99);
        // Node 1 finds node 2 directly, node 0 finds node 2 via node 1
        assert!(labels[0] >= 0, "Node 0 should be labelled");
        assert_eq!(labels[1], 42, "Node 1 adjacent to node 2");
    }

    #[test]
    fn outliers_random_fallback_for_isolated() {
        // Node 0 has no edges, node 1 is labelled
        let data: Vec<f64> = Vec::new();
        let csr = Csr::new(vec![0, 0, 0], Vec::new(), data, 2, 2);
        let mut labels = vec![-1i64, 5];
        label_outliers(&csr, &mut labels, 99);
        // Node 0 can't reach anything, gets random label in 0..=5
        assert!(labels[0] >= 0 && labels[0] <= 5);
    }

    // -- remap_labels -------------------------------------------------------

    #[test]
    fn remap_contiguous() {
        let mut labels = vec![10, 10, 20, 20, 10];
        let n = remap_labels(&mut labels);
        assert_eq!(n, 2); // two unique labels
        // 10 -> 0, 20 -> 1
        assert_eq!(labels, vec![0, 0, 1, 1, 0]);
    }

    #[test]
    fn remap_handles_negatives() {
        let mut labels = vec![5, -1, 5, -1];
        let n = remap_labels(&mut labels);
        // 5 -> 0, two -1s get 1 and 2
        assert_eq!(n, 3);
        assert_eq!(labels[0], 0);
        assert_eq!(labels[2], 0);
        assert!(labels[1] >= 1 && labels[1] <= 2);
        assert!(labels[3] >= 1 && labels[3] <= 2);
        assert_ne!(labels[1], labels[3]); // each -1 gets a unique id
    }

    // -- label_prop_loop ----------------------------------------------------

    #[test]
    fn loop_returns_valid_partition() {
        // Small ring graph: 0-1-2-3-4-0
        let n = 5;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            let j = (i + 1) % n;
            rows.push(i);
            cols.push(j);
            vals.push(1.0f64);
            rows.push(j);
            cols.push(i);
            vals.push(1.0);
        }
        let coo = CoordinateList {
            row_indices: rows,
            col_indices: cols,
            values: vals,
            n_samples: n,
        };
        let csr = Csr::from_coo(&coo);
        let (partition, n_parts) = label_prop_loop(&csr, 3, 10, 42);

        assert_eq!(partition.len(), n);
        assert!(n_parts > 0);
        for &p in &partition {
            assert!(p < n_parts);
        }
    }

    // -- normalise_to_unit_range --------------------------------------------

    #[test]
    fn unit_range_normalisation() {
        let mut data: Vec<Vec<f64>> = vec![vec![0.0, 10.0], vec![4.0, 20.0], vec![8.0, 30.0]];
        normalise_to_unit_range(&mut data);

        for row in &data {
            for &v in row {
                assert!(v >= -1.0 - 1e-12 && v <= 1.0 + 1e-12);
            }
        }
        // Min -> -1, max -> 1 for each column
        assert!((data[0][0] - (-1.0)).abs() < 1e-12);
        assert!((data[2][0] - 1.0).abs() < 1e-12);
        assert!((data[0][1] - (-1.0)).abs() < 1e-12);
        assert!((data[2][1] - 1.0).abs() < 1e-12);
    }

    // -- integration: full init on a small graph ----------------------------

    #[test]
    fn init_small_graph_random_base() {
        // Below threshold -> hits base case directly
        let n = 10;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                rows.push(i);
                cols.push(j);
                vals.push(0.5f64);
                rows.push(j);
                cols.push(i);
                vals.push(0.5);
            }
        }
        let coo = CoordinateList {
            row_indices: rows,
            col_indices: cols,
            values: vals,
            n_samples: n,
        };

        let result = label_propagation_init(&coo, 4, None, 42, false);
        assert_eq!(result.len(), n);
        for row in &result {
            assert_eq!(row.len(), 4);
            for &v in row {
                assert!(v.is_finite());
            }
        }
    }
}
