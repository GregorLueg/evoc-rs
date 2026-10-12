//! MLX-backed (Apple Silicon) nearest neighbour generation via mlx-c.
//!
//! Compiled only when the `mlx` feature is on and mlx-c was found at build
//! time (`cfg(mlx_available)`, see `build.rs`). The backends in
//! `ann-search-rs` are `f32` only, so this module is too.

use ann_search_rs::{
    build_exhaustive_index_mlx, build_ivf_index_mlx, build_knn_graph_mlx,
    query_exhaustive_index_mlx_self, query_ivf_index_mlx_self,
};
use faer::MatRef;

use crate::nearest_neighbours::{rescale_distances, strip_self};
use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Which search algorithm to use for the MLX approximate nearest neighbour
/// search. Default is NN-Descent.
#[derive(Default)]
pub enum AnnSearchMlx {
    #[default]
    /// NN-Descent, returning the raw kNN graph
    NNDescentMlx,
    /// IVF
    IvfMlx,
    /// Exhaustive
    ExhaustiveMlx,
}

/// Parameters for the MLX nearest neighbour search
#[derive(Debug, Clone)]
pub struct NearestNeighbourParamsMlxEvoc {
    /// Distance metric, one of `"euclidean"` or `"cosine"`
    pub dist_metric: String,
    /// IVF-MLX: Number of lists. Defaults to `sqrt(n)` lists.
    pub n_list: Option<usize>,
    /// IVF-MLX: Number of lists to probe. Defaults to `sqrt(n_list)`.
    pub n_probes: Option<usize>,
    /// NNDescent-MLX: Internal working degree. Defaults to `1.5 * k` in
    /// ann-search-rs.
    pub k_build: Option<usize>,
    /// NNDescent-MLX: Maximum number of iterations. Defaults to 15 in
    /// ann-search-rs.
    pub max_iters: Option<usize>,
    /// NNDescent-MLX: Number of trees for the initialisation of the kNN graph.
    pub n_tree: Option<usize>,
    /// NNDescent-MLX: Termination criterium for the NNDescent iterations
    pub delta: f32,
    /// NNDescent-MLX: Sampling rate for the NNDescent iterations.
    pub rho: Option<f32>,
    /// NNDescent-MLX: Number of 2-hop refinement sweeps after the main loop.
    /// Defaults to 0 in ann-search-rs.
    pub refine_knn: Option<usize>,
}

impl NearestNeighbourParamsMlxEvoc {
    /// Generate a new instance
    ///
    /// ### Params
    ///
    /// * `dist_metric` - One of `"euclidean"` or `"cosine"`
    ///
    /// **IVF-MLX**
    ///
    /// * `n_list` - Number of clusters. Defaults to `sqrt(n)`.
    /// * `n_probes` - Number of clusters to probe during querying. Defaults
    ///   to `sqrt(n_list)`.
    ///
    /// **NNDescent-MLX**
    ///
    /// * `k_build` - Internal working degree. Defaults to `1.5 * k`.
    /// * `max_iters` - Maximum number of iterations. Defaults to 15.
    /// * `n_tree` - Number of trees for kNN graph initialisation.
    /// * `delta` - Convergence threshold for NNDescent iterations.
    /// * `rho` - Sampling rate for NNDescent iterations.
    /// * `refine_knn` - 2-hop refinement sweeps after the main loop.
    ///
    /// ### Returns
    ///
    /// Initialised self.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dist_metric: String,
        // ivf-mlx
        n_list: Option<usize>,
        n_probes: Option<usize>,
        // nndescent-mlx
        k_build: Option<usize>,
        max_iters: Option<usize>,
        n_tree: Option<usize>,
        delta: f32,
        rho: Option<f32>,
        refine_knn: Option<usize>,
    ) -> Self {
        Self {
            dist_metric,
            n_list,
            n_probes,
            k_build,
            max_iters,
            n_tree,
            delta,
            rho,
            refine_knn,
        }
    }
}

impl Default for NearestNeighbourParamsMlxEvoc {
    /// Returns sensible defaults for the MLX nearest neighbour search.
    ///
    /// ### Returns
    ///
    /// Initialised self with sensible default parameters.
    fn default() -> Self {
        Self {
            dist_metric: "euclidean".to_string(),
            n_list: None,
            n_probes: None,
            k_build: None,
            max_iters: None,
            n_tree: None,
            delta: 0.001,
            rho: None,
            refine_knn: None,
        }
    }
}

/// Parse the MLX ANN search variant to use
///
/// ### Params
///
/// * `s` - String identifying the MLX ANN search. One of `"exhaustive_mlx"`,
///   `"ivf_mlx"` or `"nndescent_mlx"`.
///
/// ### Returns
///
/// `Some(AnnSearchMlx)` if recognised, `None` otherwise.
pub fn parse_ann_search_mlx(s: &str) -> Option<AnnSearchMlx> {
    match s.to_lowercase().as_str() {
        "exhaustive_mlx" | "exhaustive" => Some(AnnSearchMlx::ExhaustiveMlx),
        "ivf_mlx" | "ivf" => Some(AnnSearchMlx::IvfMlx),
        "nndescent_mlx" | "nndescent" => Some(AnnSearchMlx::NNDescentMlx),
        _ => None,
    }
}

//////////
// Main //
//////////

/// Run the MLX approximate nearest neighbour search
///
/// Mirrors `run_ann_search_gpu`. NN-Descent hands back the raw kNN graph it
/// builds rather than CAGRA-pruning it and beam-searching it again: a
/// self-query of a kNN graph only redoes the build.
///
/// ### Params
///
/// * `data` - The data matrix with samples x features
/// * `k` - Number of neighbours to return (excluding self)
/// * `ann_type` - Which MLX search to use. One of `"exhaustive_mlx"`,
///   `"ivf_mlx"` or `"nndescent_mlx"`.
/// * `params_nn` - Parameters for the MLX nearest neighbour search.
/// * `seed` - Seed for reproducibility.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// `(knn_indices, knn_dist)` excluding self.
pub fn run_ann_search_mlx(
    data: MatRef<f32>,
    k: usize,
    ann_type: String,
    params_nn: &NearestNeighbourParamsMlxEvoc,
    seed: usize,
    verbose: usize,
) -> EvocKnnResults<f32> {
    let verbosity = parse_verbosity_level(verbose);

    let ann_search = parse_ann_search_mlx(&ann_type).unwrap_or_else(|| {
        println!(
            "Unrecognised MLX approximate nearest neighbour method provided: {:?}. Default to MLX NNDescent.",
            ann_type
        );
        AnnSearchMlx::default()
    });

    let (knn_indices_raw, knn_dist) = match ann_search {
        AnnSearchMlx::ExhaustiveMlx => {
            let index = build_exhaustive_index_mlx(data, &params_nn.dist_metric)?;

            query_exhaustive_index_mlx_self(&index, k + 1, true, verbosity.detailed_verbosity())?
        }
        AnnSearchMlx::IvfMlx => {
            let index = build_ivf_index_mlx(
                data,
                params_nn.n_list,
                None,
                &params_nn.dist_metric,
                seed,
                verbosity.detailed_verbosity(),
            )?;

            query_ivf_index_mlx_self(
                &index,
                k + 1,
                params_nn.n_probes,
                None,
                true,
                verbosity.detailed_verbosity(),
            )?
        }
        AnnSearchMlx::NNDescentMlx => {
            let graph = build_knn_graph_mlx(
                data,
                &params_nn.dist_metric,
                Some(k),
                params_nn.k_build,
                params_nn.max_iters,
                params_nn.n_tree,
                Some(params_nn.delta),
                params_nn.rho,
                params_nn.refine_knn,
                seed,
                verbosity.detailed_verbosity(),
            )?;

            // `include_self` so the row shape matches the query paths and the
            // shared self-filter below applies to all three.
            graph.extract_knn(Some(k + 1), true, true)
        }
    };

    let (knn_indices, knn_dist) = strip_self(
        knn_indices_raw,
        knn_dist.expect("distances requested from every backend"),
        k,
    );

    let knn_dist = rescale_distances(knn_dist, &params_nn.dist_metric);

    Ok((knn_indices, knn_dist))
}
