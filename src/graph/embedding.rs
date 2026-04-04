//! Node embedding for EVoC via parallelised Adam optimisation.
//!
//! This is *not* standard UMAP embedding. Key differences:
//!
//! - The attractive/repulsive gradient kernels are EVoC-specific (see below).
//! - Gamma (repulsive strength) is scheduled linearly from 0.5 to 1.5 over
//!   training rather than held constant.
//! - Adam optimiser (ported from manifolds-rs / uwot) replaces the standard
//!   SGD used in reference UMAP.
//! - Negative sampling schedule uses a 1.5x factor on the per-sample period.
//! - Edges are extracted as undirected (i < j) and processed via a CSR layout
//!   keyed per node, enabling parallel gradient accumulation without locks.

use rand::RngExt;
use rand::{SeedableRng, rngs::SmallRng};
use rayon::prelude::*;

use crate::prelude::*;

/// Run the EVoC node embedding via parallel Adam.
///
/// Takes a weighted adjacency list (typically from
/// `build_fuzzy_simplicial_set`) and produces a low-dimensional embedding for
/// subsequent density-based clustering.
///
/// ### Gradient kernels
///
/// These differ from standard UMAP's Cauchy-based kernels:
///
///   attractive: (-2 * noise_level * d - 2) / (2 * d^2 - 0.5 * d + 1)
///   repulsive:  gamma * 4 / ((1 + 0.25 * d^2) * d^2)
///
/// where `d` is Euclidean distance (not squared) for attractive, squared for
/// repulsive, and `gamma` is linearly scheduled from 0.5 to 1.5 over epochs.
///
/// ### Params
///
/// * `graph` - Weighted adjacency list; `graph[i]` contains `(j, weight)` pairs
/// * `n_components` - Embedding dimensionality (typically 4-16 for EVoC)
/// * `params` - Optimiser and kernel hyperparameters
/// * `initial_embedding` - Optional starting coordinates; random if `None`
/// * `seed` - RNG seed for reproducibility
/// * `verbose` - Print progress every 10 epochs
///
/// ### Returns
///
/// `Vec<Vec<T>>` of shape `[n_points, n_components]`
pub fn evoc_embedding<T>(
    graph: &[Vec<(usize, T)>],
    n_components: usize,
    params: &EvocEmbeddingParams<T>,
    initial_embedding: Option<&[Vec<T>]>,
    seed: u64,
    verbose: bool,
) -> Vec<Vec<T>>
where
    T: EvocFloat,
{
    let n = graph.len();
    if n == 0 {
        return Vec::new();
    }

    let dim = n_components;

    // initialise embedding
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut embd: Vec<T> = if let Some(init) = initial_embedding {
        init.iter().flat_map(|v| v.iter().copied()).collect()
    } else {
        (0..n * dim)
            .map(|_| T::from(rng.random_range(-0.25f64..0.25f64)).unwrap())
            .collect()
    };

    // extract undirected edges (i < j)
    let mut edges: Vec<(usize, usize, T)> = Vec::new();
    let mut degree = vec![0usize; n];

    for (i, neighbours) in graph.iter().enumerate() {
        for &(j, w) in neighbours {
            if i < j {
                edges.push((i, j, w));
                degree[i] += 1;
                degree[j] += 1;
            }
        }
    }

    if edges.is_empty() {
        return (0..n)
            .map(|i| embd[i * dim..(i + 1) * dim].to_vec())
            .collect();
    }

    // epoch sampling schedules
    let max_weight = edges.iter().map(|e| e.2).fold(T::zero(), |a, b| a.max(b));

    let epochs_per_sample: Vec<T> = edges
        .iter()
        .map(|(_, _, w)| {
            let norm = *w / max_weight;
            if norm > T::zero() {
                T::one() / norm
            } else {
                T::from(1e8).unwrap()
            }
        })
        .collect();

    let mut epoch_of_next_sample: Vec<T> = epochs_per_sample.clone();

    // 1.5x factor on negative schedule (matches Python reproducible path)
    let neg_factor = T::from(1.5).unwrap() / params.negative_sample_rate;
    let epochs_per_neg: Vec<T> = epochs_per_sample.iter().map(|e| *e * neg_factor).collect();
    let mut epoch_of_next_neg: Vec<T> = epochs_per_neg.clone();

    // CSR edge layout per node
    let mut offsets = vec![0usize; n + 1];
    for i in 0..n {
        offsets[i + 1] = offsets[i] + degree[i];
    }

    // (edge_idx, is_smaller_endpoint, other_node)
    let mut csr_edges = vec![(0usize, false, 0usize); edges.len() * 2];
    let mut cursor = offsets.clone();

    for (eidx, &(i, j, _)) in edges.iter().enumerate() {
        csr_edges[cursor[i]] = (eidx, true, j);
        cursor[i] += 1;
        csr_edges[cursor[j]] = (eidx, false, i);
        cursor[j] += 1;
    }

    // Adam state
    let mut m_buf = vec![T::zero(); n * dim];
    let mut v_buf = vec![T::zero(); n * dim];

    let one_m_b1 = T::one() - params.beta1;
    let one_m_b2 = T::one() - params.beta2;

    let bias_corrections: Vec<(T, T)> = (0..params.n_epochs)
        .map(|ep| {
            let t = T::from(ep + 1).unwrap();
            let b1t = params.beta1.powf(t);
            let b2t = params.beta2.powf(t);
            let sqrt_b2 = (T::one() - b2t).sqrt();
            (sqrt_b2 / (T::one() - b1t), sqrt_b2 * params.eps)
        })
        .collect();

    // gamma: 0.5 -> 1.5 linearly
    let denom = T::from(params.n_epochs.saturating_sub(1).max(1)).unwrap();
    let gamma_schedule: Vec<T> = (0..params.n_epochs)
        .map(|i| T::from(0.5).unwrap() + T::from(i).unwrap() / denom)
        .collect();

    let n_epochs_f = T::from(params.n_epochs).unwrap();

    // constants
    let two = T::from(2.0).unwrap();
    let half = T::from(0.5).unwrap();
    let four = T::from(4.0).unwrap();
    let quarter = T::from(0.25).unwrap();
    let clip = T::from(4.0).unwrap();
    let dsq_attr_min = T::from(1e-8).unwrap();
    let dsq_rep_min = T::from(0.01).unwrap();

    let mut grads = vec![T::zero(); n * dim];
    let mut active = vec![false; n];
    let mut node_rngs: Vec<SmallRng> = (0..n)
        .map(|i| SmallRng::seed_from_u64(seed + i as u64))
        .collect();

    // main loop
    for epoch in 0..params.n_epochs {
        let lr = params.initial_alpha * (T::one() - T::from(epoch).unwrap() / n_epochs_f);
        let epoch_t = T::from(epoch).unwrap();
        let (ad_scale, epsc) = bias_corrections[epoch];
        let gamma = gamma_schedule[epoch];
        let noise = params.noise_level;

        active.fill(false);

        // parallel gradient accumulation
        grads
            .par_chunks_exact_mut(dim)
            .zip(active.par_iter_mut())
            .zip(node_rngs.par_iter_mut())
            .enumerate()
            .for_each(|(ni, ((grad, is_active), rng))| {
                for g in grad.iter_mut() {
                    *g = T::zero();
                }

                let bi = ni * dim;
                let start = offsets[ni];
                let end = offsets[ni + 1];
                let mut touched = false;

                for &(eidx, is_smaller, other) in &csr_edges[start..end] {
                    if epoch_of_next_sample[eidx] > epoch_t {
                        continue;
                    }
                    touched = true;
                    let bo = other * dim;

                    let dsq = T::euclidean_simd(&embd[bi..bi + dim], &embd[bo..bo + dim]);

                    // attractive
                    if dsq >= dsq_attr_min {
                        let dist = dsq.sqrt();
                        let coeff =
                            (-(two * noise * dist) - two) / (two * dsq - half * dist + T::one());

                        for d in 0..dim {
                            grad[d] += coeff * (embd[bi + d] - embd[bo + d]);
                        }
                    }

                    // repulsive (only from smaller endpoint to avoid doubling)
                    if is_smaller {
                        let n_neg = ((epoch_t - epoch_of_next_neg[eidx]) / epochs_per_neg[eidx])
                            .floor()
                            .to_usize()
                            .unwrap_or(0);

                        for _ in 0..n_neg {
                            let k = rng.random_range(0..n);
                            if k == ni {
                                continue;
                            }
                            let bk = k * dim;

                            let dsq_k = T::euclidean_simd(&embd[bi..bi + dim], &embd[bk..bk + dim]);

                            if dsq_k > dsq_rep_min {
                                let gc = gamma * four / ((T::one() + quarter * dsq_k) * dsq_k);
                                for d in 0..dim {
                                    let diff = embd[bi + d] - embd[bk + d];
                                    grad[d] += (gc * diff).max(-clip).min(clip);
                                }
                            }
                        }
                    }
                }

                if touched {
                    *is_active = true;
                }
            });

        // parallel Adam moment update + embedding step
        grads
            .par_chunks_exact(dim)
            .zip(m_buf.par_chunks_exact_mut(dim))
            .zip(v_buf.par_chunks_exact_mut(dim))
            .zip(embd.par_chunks_exact_mut(dim))
            .zip(active.par_iter())
            .for_each(|((((g, m_n), v_n), e_n), &on)| {
                if !on {
                    return;
                }
                for d in 0..dim {
                    let gd = g[d];
                    m_n[d] = m_n[d] + one_m_b1 * (gd - m_n[d]);
                    v_n[d] = v_n[d] + one_m_b2 * (gd * gd - v_n[d]);
                    e_n[d] += lr * ad_scale * m_n[d] / (v_n[d].sqrt() + epsc);
                }
            });

        // edge schedule update
        epoch_of_next_sample
            .par_iter_mut()
            .zip(epoch_of_next_neg.par_iter_mut())
            .zip(epochs_per_sample.par_iter())
            .zip(epochs_per_neg.par_iter())
            .for_each(|(((ns, nn), &ps), &pn)| {
                if *ns <= epoch_t {
                    *ns += ps;
                    let neg_count = ((epoch_t - *nn) / pn).floor().to_usize().unwrap_or(0);
                    *nn += T::from(neg_count).unwrap() * pn;
                }
            });

        if verbose && ((epoch + 1) % 10 == 0 || epoch + 1 == params.n_epochs) {
            println!("  Embedding epoch {}/{}", epoch + 1, params.n_epochs);
        }
    }

    // reshape flat -> Vec<Vec<T>>
    (0..n)
        .map(|i| embd[i * dim..(i + 1) * dim].to_vec())
        .collect()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_embedding {
    use super::*;

    fn triangle_graph<T: EvocFloat>() -> Vec<Vec<(usize, T)>> {
        // Fully connected triangle with uniform weights
        vec![
            vec![(1, T::one()), (2, T::one())],
            vec![(0, T::one()), (2, T::one())],
            vec![(0, T::one()), (1, T::one())],
        ]
    }

    #[test]
    fn test_empty_graph() {
        let graph: Vec<Vec<(usize, f64)>> = Vec::new();
        let params = EvocEmbeddingParams::default();
        let result = evoc_embedding(&graph, 2, &params, None, 42, false);
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_node_no_edges() {
        let graph: Vec<Vec<(usize, f64)>> = vec![vec![]];
        let params = EvocEmbeddingParams::default();
        let result = evoc_embedding(&graph, 2, &params, None, 42, false);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 2);
    }

    #[test]
    fn test_output_shape() {
        let graph = triangle_graph::<f64>();
        let params = EvocEmbeddingParams::default();

        for dim in [2, 4, 8, 16] {
            let result = evoc_embedding(&graph, dim, &params, None, 42, false);
            assert_eq!(result.len(), 3);
            for row in &result {
                assert_eq!(row.len(), dim);
            }
        }
    }

    #[test]
    fn test_deterministic_with_same_seed() {
        let graph = triangle_graph::<f64>();
        let params = EvocEmbeddingParams::default();

        let a = evoc_embedding(&graph, 4, &params, None, 123, false);
        let b = evoc_embedding(&graph, 4, &params, None, 123, false);

        for (ra, rb) in a.iter().zip(b.iter()) {
            for (&va, &vb) in ra.iter().zip(rb.iter()) {
                assert_eq!(va.to_bits(), vb.to_bits());
            }
        }
    }

    #[test]
    fn test_different_seeds_differ() {
        let graph = triangle_graph::<f64>();
        let params = EvocEmbeddingParams::default();

        let a = evoc_embedding(&graph, 4, &params, None, 1, false);
        let b = evoc_embedding(&graph, 4, &params, None, 2, false);

        let any_differ = a
            .iter()
            .zip(b.iter())
            .any(|(ra, rb)| ra.iter().zip(rb.iter()).any(|(va, vb)| va != vb));
        assert!(any_differ);
    }

    #[test]
    fn test_initial_embedding_respected() {
        let graph: Vec<Vec<(usize, f64)>> = vec![vec![], vec![]];
        let params = EvocEmbeddingParams {
            n_epochs: 0,
            ..Default::default()
        };

        let init = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = evoc_embedding(&graph, 2, &params, Some(&init), 42, false);

        assert_eq!(result, init);
    }

    #[test]
    fn test_attractive_edges_contract() {
        // Two connected nodes should end up closer than they started
        let graph: Vec<Vec<(usize, f64)>> = vec![vec![(1, 1.0)], vec![(0, 1.0)]];

        let init = vec![vec![0.0, 0.0], vec![10.0, 0.0]];
        let params = EvocEmbeddingParams {
            n_epochs: 100,
            negative_sample_rate: 0.0, // disable repulsion
            ..Default::default()
        };

        let result = evoc_embedding(&graph, 2, &params, Some(&init), 42, false);

        let initial_dsq = 100.0; // 10^2
        let final_dsq: f64 = result[0]
            .iter()
            .zip(result[1].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        assert!(
            final_dsq < initial_dsq,
            "Expected contraction: initial dsq={}, final dsq={}",
            initial_dsq,
            final_dsq,
        );
    }

    #[test]
    fn test_no_nans_or_infs() {
        let graph = triangle_graph::<f64>();
        let params = EvocEmbeddingParams::default();
        let result = evoc_embedding(&graph, 8, &params, None, 42, false);

        for row in &result {
            for &v in row {
                assert!(v.is_finite(), "Got non-finite value: {}", v);
            }
        }
    }

    #[test]
    fn test_default_params_sane() {
        let p = EvocEmbeddingParams::<f64>::default();
        assert!(p.n_epochs > 0);
        assert!(p.initial_alpha > 0.0);
        assert!(p.beta1 > 0.0 && p.beta1 < 1.0);
        assert!(p.beta2 > 0.0 && p.beta2 < 1.0);
        assert!(p.eps > 0.0);
        assert!(p.noise_level > 0.0);
    }

    #[test]
    fn test_disconnected_components() {
        // Two pairs of nodes, no edges between pairs
        let graph: Vec<Vec<(usize, f64)>> = vec![
            vec![(1, 1.0)],
            vec![(0, 1.0)],
            vec![(3, 1.0)],
            vec![(2, 1.0)],
        ];
        let params = EvocEmbeddingParams::default();
        let result = evoc_embedding(&graph, 4, &params, None, 42, false);

        assert_eq!(result.len(), 4);
        for row in &result {
            assert_eq!(row.len(), 4);
            for &v in row {
                assert!(v.is_finite());
            }
        }
    }
}
