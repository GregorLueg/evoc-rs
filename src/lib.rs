//! EVoC - Embedding Vector Oriented Clustering
//!
//! Efficient clustering of high-dimensional embedding vectors (CLIP, sentence
//! transformers, etc.) by combining a UMAP-like node embedding with
//! HDBSCAN-style density-based clustering and multi-layer persistence
//! analysis. This is the Rust version/port which allows for different
//! approximate nearest neighbour search algorithms (for details, see
//! [ann-search-rs](https://crates.io/crates/ann-search-rs)).
//! This code is based on the original code from Leland McInnes, see the Python
//! implementation: [evoc](https://github.com/TutteInstitute/evoc)

#![allow(clippy::needless_range_loop)]
#![warn(missing_docs)]

pub mod clustering;
pub mod graph;
pub mod prelude;
pub mod utils;

use ann_search_rs::cpu::hnsw::{HnswIndex, HnswState};
use ann_search_rs::cpu::nndescent::{ApplySortedUpdates, NNDescent, NNDescentQuery};
use ann_search_rs::prelude::AnnSearchFloat;
use faer::MatRef;
use manifolds_rs::PreComputedKnn;
use manifolds_rs::data::nearest_neighbours::*;
use std::time::Instant;

use crate::clustering::condensed_tree::*;
use crate::clustering::linkage::mst_to_linkage_tree;
use crate::clustering::mst::build_mst;
use crate::clustering::persistence::build_cluster_layers;
use crate::graph::embedding::*;
use crate::graph::fuzzy_graph::*;
use crate::prelude::*;

////////////
// Params //
////////////

/// Parameters for EVoC clustering.
#[derive(Clone, Debug)]
pub struct EvocParams<T> {
    /// Number of nearest neighbours for graph construction.
    pub n_neighbours: usize,
    /// Noise level for the embedding gradient (0.0 = aggressive, 1.0 =
    /// conservative).
    pub noise_level: T,
    /// Number of embedding optimisation epochs.
    pub n_epochs: usize,
    /// Embedding dimensionality. If `None`, defaults to
    /// `min(max(n_neighbours / 4, 4), 16)`.
    pub embedding_dim: Option<usize>,
    /// Multiplier on effective neighbours for fuzzy graph construction.
    pub neighbour_scale: T,
    /// Whether to symmetrise the fuzzy graph.
    pub symmetrise: bool,
    /// Minimum samples for core distance in MST density estimation.
    pub min_samples: usize,
    /// Base minimum cluster size for the finest layer.
    pub base_min_cluster_size: usize,
    /// If set, binary-search for approximately this many clusters (single layer
    /// output).
    pub approx_n_clusters: Option<usize>,
    /// Jaccard similarity threshold for filtering redundant layers.
    pub min_similarity_threshold: f64,
    /// Maximum number of cluster layers to return.
    pub max_layers: usize,
}

/// Default implementation
impl<T: EvocFloat> Default for EvocParams<T> {
    fn default() -> Self {
        Self {
            n_neighbours: 15,
            noise_level: T::from(0.5).unwrap(),
            n_epochs: 50,
            embedding_dim: None,
            neighbour_scale: T::one(),
            symmetrise: true,
            min_samples: 5,
            base_min_cluster_size: 5,
            approx_n_clusters: None,
            min_similarity_threshold: 0.2,
            max_layers: 10,
        }
    }
}

/////////////
// Results //
/////////////

/// Result of EVoC clustering.
pub struct EvocResult<T> {
    /// Cluster labels per layer, sorted finest (most clusters) first.
    /// -1 indicates noise.
    pub cluster_layers: Vec<Vec<i64>>,
    /// Membership strengths per layer, in [0, 1].
    pub membership_strengths: Vec<Vec<T>>,
    /// Persistence score per layer (higher = more stable).
    pub persistence_scores: Vec<f64>,
    /// k-NN indices (excluding self).
    pub nn_indices: Vec<Vec<usize>>,
    /// k-NN distances (excluding self).
    pub nn_distances: Vec<Vec<T>>,
}

impl<T: EvocFloat> EvocResult<T> {
    /// Returns labels from the layer with the highest persistence score.
    ///
    /// Falls back to the only available layer when there is just one.
    pub fn best_labels(&self) -> &[i64] {
        if self.cluster_layers.len() <= 1 {
            &self.cluster_layers[0]
        } else {
            let best = self
                .persistence_scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            &self.cluster_layers[best]
        }
    }

    /// Returns membership strengths corresponding to
    /// [`EvocResult::best_labels`].
    pub fn best_strengths(&self) -> &[T] {
        if self.membership_strengths.len() <= 1 {
            &self.membership_strengths[0]
        } else {
            let best = self
                .persistence_scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            &self.membership_strengths[best]
        }
    }

    /// Number of clusters in the best layer, excluding noise points.
    pub fn n_clusters(&self) -> usize {
        let labels = self.best_labels();
        (labels.iter().copied().reduce(i64::max).unwrap_or(-1) + 1).max(0) as usize
    }
}

//////////
// Main //
//////////

/// Run EVoC clustering on high-dimensional embedding data.
///
/// 1. **kNN graph** — builds an approximate nearest-neighbour graph via the
///    selected ANN backend (`ann_type`), or uses `precomputed_knn` if
///    provided.
/// 2. **Fuzzy simplicial set** — smooths and optionally symmetrises the kNN
///    graph into a weighted undirected graph.
/// 3. **Node embedding** — optimises a low-dimensional layout using the EVoC
///    gradient (a modified UMAP repulsion term controlled by `noise_level`).
/// 4. **MST** — builds a minimum spanning tree over the embedding using
///    mutual reachability distances.
/// 5. **Cluster layers** — extracts a hierarchy of clusterings from the MST
///    via persistence analysis, or searches for a fixed number of clusters if
///    [`EvocParams::approx_n_clusters`] is set.
///
/// ### Params
///
/// * `data` — input matrix with shape `(n_points, n_features)`.
/// * `ann_type` — ANN backend identifier (e.g. `"nndescent"`, `"hnsw"`).
/// * `precomputed_knn` — pre-built `(indices, distances)` pair; pass `None`
///   to build the graph from `data`.
/// * `evoc_params` — clustering hyperparameters; see [`EvocParams`].
/// * `nn_params` — hyperparameters forwarded to the ANN backend.
/// * `seed` — random seed for reproducibility.
/// * `verbose` — print progress and timing to stdout.
///
/// ### Returns
///
/// An [`EvocResult`] containing cluster layers, membership strengths,
/// persistence scores, and the kNN graph.
pub fn evoc<T>(
    data: MatRef<T>,
    ann_type: String,
    precomputed_knn: PreComputedKnn<T>,
    evoc_params: &EvocParams<T>,
    nn_params: &NearestNeighbourParams<T>,
    seed: usize,
    verbose: bool,
) -> EvocResult<T>
where
    T: EvocFloat + AnnSearchFloat,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
    HnswIndex<T>: HnswState<T>,
{
    let start_all = Instant::now();

    // 1. kNN graph
    let (knn_indices, knn_dist) = match precomputed_knn {
        Some((indices, distances)) => {
            if verbose {
                println!("Using precomputed kNN graph...");
            }
            (indices, distances)
        }
        None => {
            if verbose {
                println!(
                    "Running approximate nearest neighbour search using {}...",
                    ann_type
                );
            }
            let start_knn = Instant::now();
            let result = run_ann_search(
                data,
                evoc_params.n_neighbours,
                ann_type,
                nn_params,
                seed,
                verbose,
            );
            if verbose {
                println!("kNN search done in {:.2?}.", start_knn.elapsed());
            }
            result
        }
    };

    // 2. Fuzzy simplicial set
    if verbose {
        println!("Constructing fuzzy simplicial set...");
    }
    let start_graph = Instant::now();
    let effective_k = evoc_params.neighbour_scale * T::from(evoc_params.n_neighbours).unwrap();
    let graph =
        build_fuzzy_simplicial_set(&knn_indices, &knn_dist, effective_k, evoc_params.symmetrise);
    let adj = coo_to_adjacency_list(&graph);
    if verbose {
        println!(
            "Fuzzy simplicial set done in {:.2?}.",
            start_graph.elapsed()
        );
    }

    // 3. Embedding dimensionality
    // original code did 4 to 15 -> went up one for SIMD
    let dim = evoc_params
        .embedding_dim
        .unwrap_or_else(|| (evoc_params.n_neighbours / 4).clamp(4, 16));

    // 4. Label propagation initialisation
    let start_init = Instant::now();
    let n = data.nrows();
    let d = data.ncols();
    let data_vecs: Vec<Vec<T>> = (0..n)
        .map(|i| (0..d).map(|j| data[(i, j)]).collect())
        .collect();

    if verbose {
        println!("Computing label propagation initialisation...");
    }
    let initial_embedding = crate::graph::label_prop::label_propagation_init(
        &graph,
        dim,
        Some(&data_vecs),
        seed as u64,
        verbose,
    );
    if verbose {
        println!("Label prop init done in {:.2?}.", start_init.elapsed());
    }

    // 5. Node embedding
    if verbose {
        println!(
            "Computing {}-d node embedding ({} epochs)...",
            dim, evoc_params.n_epochs
        );
    }
    let start_embed = Instant::now();
    let embed_params = EvocEmbeddingParams {
        n_epochs: evoc_params.n_epochs,
        noise_level: evoc_params.noise_level,
        initial_alpha: T::from(0.1).unwrap(),
        ..EvocEmbeddingParams::default()
    };

    let embedding = evoc_embedding(
        &adj,
        dim,
        &embed_params,
        Some(&initial_embedding),
        seed as u64,
        verbose,
    );
    if verbose {
        println!("Embedding done in {:.2?}.", start_embed.elapsed());
    }

    // 6. Clustering
    if verbose {
        println!("Running density-based clustering...");
    }
    let start_cluster = Instant::now();

    let (cluster_layers, membership_strengths, persistence_scores) =
        if let Some(target_k) = evoc_params.approx_n_clusters {
            let (labels, strengths) =
                search_for_n_clusters(&embedding, evoc_params.min_samples, target_k);
            (vec![labels], vec![strengths], vec![0.0])
        } else {
            build_cluster_layers(
                &embedding,
                evoc_params.min_samples,
                evoc_params.base_min_cluster_size,
                evoc_params.min_similarity_threshold,
                evoc_params.max_layers,
            )
        };

    if verbose {
        let n_layers = cluster_layers.len();
        println!(
            "Clustering done in {:.2?}: {} layer(s).",
            start_cluster.elapsed(),
            n_layers,
        );
        println!("EVoC total: {:.2?}.", start_all.elapsed());
    }

    EvocResult {
        cluster_layers,
        membership_strengths,
        persistence_scores,
        nn_indices: knn_indices,
        nn_distances: knn_dist,
    }
}

/// Binary-searches over `min_cluster_size` to find approximately `target_k`
/// clusters.
///
/// Builds the MST and linkage tree once from `embedding`, then re-condenses
/// the tree at different `min_cluster_size` thresholds until the number of
/// extracted leaves is close to `target_k`.
///
/// When the search converges, both boundary values (`lo` and `hi`) are
/// evaluated and the one whose cluster count is closer to `target_k` is
/// returned. Ties are broken in favour of whichever boundary assigns more
/// points to non-noise clusters.
///
/// ### Params
///
/// * `embedding` — low-dimensional node positions, one `Vec<T>` per point.
/// * `min_samples` — passed to [`build_mst`] for core-distance estimation.
/// * `target_k` — desired number of clusters.
///
/// ### Returns
///
/// A `(labels, membership_strengths)` pair for the selected clustering.
/// Labels follow the same convention as [`EvocResult::cluster_layers`]:
/// `-1` is noise, non-negative integers are cluster IDs.
pub fn search_for_n_clusters<T>(
    embedding: &[Vec<T>],
    min_samples: usize,
    target_k: usize,
) -> (Vec<i64>, Vec<T>)
where
    T: EvocFloat,
{
    let n = embedding.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut mst = build_mst(embedding, min_samples);
    let linkage = mst_to_linkage_tree(&mut mst, n);

    let mut lo = 2usize;
    let mut hi = n / 2;

    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if mid == lo || mid == hi {
            break;
        }

        let ct_mid = condense_tree(&linkage, n, mid);
        let leaves_mid = extract_leaves(&ct_mid);
        let mid_k = leaves_mid.len();

        if mid_k < target_k {
            // Need more clusters -> smaller min_cluster_size
            hi = mid;
        } else {
            // Have enough or too many -> larger min_cluster_size
            lo = mid;
        }
    }

    // Pick whichever bound is closer to target
    let ct_lo = condense_tree(&linkage, n, lo);
    let leaves_lo = extract_leaves(&ct_lo);
    let labels_lo = get_cluster_label_vector(&ct_lo, &leaves_lo, n);
    let lo_k = leaves_lo.len();

    let ct_hi = condense_tree(&linkage, n, hi);
    let leaves_hi = extract_leaves(&ct_hi);
    let labels_hi = get_cluster_label_vector(&ct_hi, &leaves_hi, n);
    let hi_k = leaves_hi.len();

    let lo_diff = (lo_k as isize - target_k as isize).unsigned_abs();
    let hi_diff = (hi_k as isize - target_k as isize).unsigned_abs();

    if lo_diff < hi_diff {
        let strengths = get_point_membership_strengths(&ct_lo, &leaves_lo, &labels_lo);
        (labels_lo, strengths)
    } else if hi_diff < lo_diff {
        let strengths = get_point_membership_strengths(&ct_hi, &leaves_hi, &labels_hi);
        (labels_hi, strengths)
    } else {
        // Tie: prefer whichever has more non-noise points (matches Python)
        let lo_assigned = labels_lo.iter().filter(|&&l| l >= 0).count();
        let hi_assigned = labels_hi.iter().filter(|&&l| l >= 0).count();
        if lo_assigned >= hi_assigned {
            let strengths = get_point_membership_strengths(&ct_lo, &leaves_lo, &labels_lo);
            (labels_lo, strengths)
        } else {
            let strengths = get_point_membership_strengths(&ct_hi, &leaves_hi, &labels_hi);
            (labels_hi, strengths)
        }
    }
}
