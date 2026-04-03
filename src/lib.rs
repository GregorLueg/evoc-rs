//! EVoC - Embedding Vector Oriented Clustering
//!
//! Efficient clustering of high-dimensional embedding vectors (CLIP, sentence
//! transformers, etc.) by combining a UMAP-like node embedding with
//! HDBSCAN-style density-based clustering and multi-layer persistence
//! analysis. This is the Rust version which allows for different approximate
//! nearest neighbour search algorithms.

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
    /// `min(max(n_neighbors / 4, 4), 15)`.
    pub embedding_dim: Option<usize>,
    /// Multiplier on effective neighbours for fuzzy graph construction.
    pub neighbor_scale: T,
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
    /// Distance metric for kNN. Typically `"cosine"`.
    pub metric: String,
}

/// Default implementation
impl<T: EvocFloat> Default for EvocParams<T> {
    fn default() -> Self {
        Self {
            n_neighbours: 15,
            noise_level: T::from(0.5).unwrap(),
            n_epochs: 50,
            embedding_dim: None,
            neighbor_scale: T::one(),
            symmetrise: true,
            min_samples: 5,
            base_min_cluster_size: 5,
            approx_n_clusters: None,
            min_similarity_threshold: 0.2,
            max_layers: 10,
            metric: "cosine".to_string(),
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
    /// Labels from the layer with the highest persistence score,
    /// or the base layer if only one exists.
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

    /// Membership strengths corresponding to `best_labels`.
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

    /// Number of clusters in the best layer (excluding noise).
    pub fn n_clusters(&self) -> usize {
        let labels = self.best_labels();
        (labels.iter().max().copied().unwrap_or(-1) + 1).max(0) as usize
    }
}

//////////
// Main //
//////////

/// Run EVoC clustering on high-dimensional embedding data.
///
/// Pipeline:
/// 1. Build k-NN graph via NNDescent
/// 2. Construct fuzzy simplicial set (smooth kNN + symmetrise)
/// 3. Compute low-dimensional node embedding (modified UMAP with EVoC gradient)
/// 4. Build MST with mutual reachability distance on the embedding
/// 5. Extract hierarchical cluster layers via persistence analysis
pub fn evoc<T>(
    data: MatRef<T>,
    ann_type: String,
    precomputed_knn: PreComputedKnn<T>,
    evoc_params: &EvocParams<T>,
    nn_params: &NearestNeighbourParams<T>,
    seed: usize,
    verbose: bool,
)
// -> EvocResult<T>
where
    T: EvocFloat + AnnSearchFloat,
    NNDescent<T>: ApplySortedUpdates<T> + NNDescentQuery<T>,
    HnswIndex<T>: HnswState<T>,
{
    let start_all = Instant::now();

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
                println!("kNN search done in: {:.2?}.", start_knn.elapsed());
            }
            result
        }
    };

    // 2. fuzzy simplicial set
    if verbose {
        println!("Constructing fuzzy simplicial set...");
    }
    let effective_k = evoc_params.neighbor_scale * T::from(evoc_params.n_neighbours).unwrap();
    let graph =
        build_fuzzy_simplicial_set(&knn_indices, &knn_dist, effective_k, evoc_params.symmetrise);
    let adj = coo_to_adjacency_list(&graph);

    if verbose {
        println!(
            " Construction of fuzzy simplicial set done in {:.2?}.",
            start_all.elapsed()
        );
    }

    let dim = evoc_params
        .embedding_dim
        .unwrap_or_else(|| (evoc_params.n_neighbours / 4).clamp(4, 16));

    if verbose {
        println!(
            "Computing {}-dimensional node embedding ({} epochs)...",
            dim, evoc_params.n_epochs
        );
    }

    let embed_params = EvocEmbeddingParams {
        n_epochs: evoc_params.n_epochs,
        noise_level: evoc_params.noise_level,
        initial_alpha: T::from(0.1).unwrap(),
        ..EvocEmbeddingParams::default()
    };

    let embd = evoc_embedding(&adj, dim, &embed_params, None, seed as u64, verbose);
}
