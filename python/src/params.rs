//! Rebuilding the crate's parameter structs from flat keyword arguments.
//!
//! pyo3 cannot hand a Rust struct straight across, and a `#[pyclass]` per
//! parameter struct would make the Python layer carry two objects the user has
//! to assemble before they can cluster anything. The estimator already presents
//! one flat constructor, so the flattening happens here instead: one call, one
//! long signature, structs rebuilt on this side.

use evoc_rs::prelude::*;
use evoc_rs::EvocParams;

/// A precomputed kNN graph as it arrives from and is handed to the core.
///
/// `(indices, distances)`, one row per sample, self excluded.
pub(crate) type Knn<T> = Option<(Vec<Vec<usize>>, Vec<Vec<T>>)>;

/// Narrow a precomputed kNN graph from the float64 Python hands over to f32.
///
/// ### Params
///
/// * `knn` - `(indices, distances)` pair, or `None`.
///
/// ### Returns
///
/// The same pair with the distances narrowed, or `None`.
pub(crate) fn narrow_knn(knn: Knn<f64>) -> Knn<f32> {
    knn.map(|(indices, distances)| {
        let distances = distances
            .into_iter()
            .map(|row| row.into_iter().map(|v| v as f32).collect())
            .collect();
        (indices, distances)
    })
}

/// Every knob of [`EvocParams`], as the scalars a `#[pyfunction]` can take.
///
/// `T` is the float type the clustering runs at, so the two dispatch arms build
/// the same struct at their own precision.
pub(crate) struct EvocArgs {
    /// Number of nearest neighbours for graph construction.
    pub n_neighbours: usize,
    /// Noise level for the embedding gradient.
    pub noise_level: f64,
    /// Number of embedding optimisation epochs.
    pub n_epochs: usize,
    /// Embedding dimensionality, `None` to derive it from `n_neighbours`.
    pub embedding_dim: Option<usize>,
    /// Multiplier on effective neighbours for fuzzy graph construction.
    pub neighbour_scale: f64,
    /// Whether to symmetrise the fuzzy graph.
    pub symmetrise: bool,
    /// Minimum samples for core distance in MST density estimation.
    pub min_samples: usize,
    /// Base minimum cluster size for the finest layer.
    pub base_min_cluster_size: usize,
    /// If set, binary-search for approximately this many clusters.
    pub approx_n_clusters: Option<usize>,
    /// Jaccard threshold for filtering redundant layers.
    pub min_similarity_threshold: f64,
    /// Maximum number of cluster layers to return.
    pub max_layers: usize,
}

impl EvocArgs {
    /// Rebuild the crate's parameter struct at the target precision.
    ///
    /// ### Returns
    ///
    /// An [`EvocParams`] with the float fields narrowed to `T`.
    pub(crate) fn build<T: EvocFloat>(&self) -> EvocParams<T> {
        EvocParams {
            n_neighbours: self.n_neighbours,
            noise_level: T::from(self.noise_level).unwrap(),
            n_epochs: self.n_epochs,
            embedding_dim: self.embedding_dim,
            neighbour_scale: T::from(self.neighbour_scale).unwrap(),
            symmetrise: self.symmetrise,
            min_samples: self.min_samples,
            base_min_cluster_size: self.base_min_cluster_size,
            approx_n_clusters: self.approx_n_clusters,
            min_similarity_threshold: self.min_similarity_threshold,
            max_layers: self.max_layers,
        }
    }
}

/// Every knob of [`NearestNeighbourParamsEvoc`], as plain scalars.
pub(crate) struct NnArgs {
    /// Distance metric, `"euclidean"` or `"cosine"`.
    pub dist_metric: String,
    /// Annoy: number of trees.
    pub n_tree: usize,
    /// Annoy: search budget per tree.
    pub search_budget: Option<usize>,
    /// HNSW: connections per layer.
    pub m: usize,
    /// HNSW: construction budget.
    pub ef_construction: usize,
    /// HNSW: search budget.
    pub ef_search: usize,
    /// NNDescent: diversification probability.
    pub diversify_prob: f64,
    /// NNDescent: convergence criterion.
    pub delta: f64,
    /// NNDescent: beam search budget for querying.
    pub ef_budget: Option<usize>,
    /// BallTree: proportion of N to search.
    pub bt_budget: f64,
    /// IVF: number of lists.
    pub n_list: Option<usize>,
    /// IVF: number of lists to probe.
    pub n_probes: Option<usize>,
}

impl NnArgs {
    /// Rebuild the crate's ANN parameter struct at the target precision.
    ///
    /// ### Returns
    ///
    /// A [`NearestNeighbourParamsEvoc`] with the float fields narrowed to `T`.
    pub(crate) fn build<T: EvocFloat>(&self) -> NearestNeighbourParamsEvoc<T> {
        NearestNeighbourParamsEvoc {
            dist_metric: self.dist_metric.clone(),
            n_tree: self.n_tree,
            search_budget: self.search_budget,
            m: self.m,
            ef_construction: self.ef_construction,
            ef_search: self.ef_search,
            diversify_prob: T::from(self.diversify_prob).unwrap(),
            delta: T::from(self.delta).unwrap(),
            ef_budget: self.ef_budget,
            bt_budget: T::from(self.bt_budget).unwrap(),
            n_list: self.n_list,
            n_probes: self.n_probes,
        }
    }
}
