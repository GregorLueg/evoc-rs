//! Module containing GPU-accelerated (approximate) nearest neighbour generation
//! functions used in the different embedding methods.

use ann_search_rs::prelude::*;
use ann_search_rs::{
    build_exhaustive_index_gpu, build_ivf_index_gpu, build_knn_graph_nndescent_gpu,
    query_exhaustive_index_gpu_self, query_ivf_index_gpu_self,
};
use cubecl::Device;
use cubecl_utils_rs::CubeclFloat;
use faer::MatRef;

use crate::nearest_neighbours::{knn_graph_to_rows, rescale_distances, strip_self};
use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Which search algorithm to use for the GPU-accelerated approximate nearest
/// neighbour search. Default is set to NNDescentGpu GPU.
#[derive(Default)]
pub enum AnnSearchGpu {
    #[default]
    /// NNDescentGpu
    NNDescentGpu,
    /// IvfGpu
    IvfGpu,
    /// Exhaustive
    ExhaustiveGpu,
}

/// Parameters for the nearest neighbour search
#[derive(Debug, Clone)]
pub struct NearestNeighbourParamsGpuEvoc<T> {
    /// Distance metric, one of `"euclidean"` or `"cosine"`
    pub dist_metric: String,
    /// IVF-GPU: Number of lists, clusters to use. If not provided, will default
    /// to `sqrt(n)` lists.
    pub n_list: Option<usize>,
    /// IVF-GPU: Number of lists to probes. If not provided, will default to
    /// to `sqrt(n_list)` lists.
    pub n_probes: Option<usize>,
    /// NNDescent-GPU: Internal working degree of the descent. If `None`, the
    /// ann-search-rs crate defaults to `1.5 * k` with `k` the number of
    /// neighbours requested.
    pub build_k: Option<usize>,
    /// NNDescent-GPU: Maximum NNDescent iterations. If `None`, the
    /// ann-search-rs crate defaults to 15.
    pub max_iters: Option<usize>,
    /// NNDescent-GPU: Number of trees for the initialisation of the kNN graph
    /// prior to NNDescent. If `None`, sized from the number of samples.
    pub n_tree: Option<usize>,
    /// NNDescent-GPU: Termination criterium for the NNDescent iterations
    pub delta: T,
    /// NNDescent-GPU: Sampling rate for the NNDescent iterations.
    pub rho: Option<T>,
    /// NNDescent-GPU: 2-hop refinement sweeps after the main loop. If `None`,
    /// no refinement runs.
    pub refine_knn: Option<usize>,
}

impl<T> NearestNeighbourParamsGpuEvoc<T> {
    /// Generate a new instance
    ///
    /// ### Params
    ///
    /// General parameters
    ///
    /// * `dist_metric` - One of `"euclidean"` or `"cosine"`
    ///
    /// **IVF-GPU**
    ///
    /// * `n_list` - Number of clusters. Defaults to `sqrt(n)`.
    /// * `n_probes` - Number of clusters to probe during querying. Defaults
    ///   to `sqrt(n_list)`.
    ///
    /// **NNDescent-GPU**
    ///
    /// * `build_k` - Internal working degree. Defaults to `1.5 * k` in
    ///   ann-search-rs.
    /// * `max_iters` - Maximum NNDescent iterations. Defaults to `15`.
    /// * `n_tree` - Number of trees for kNN graph initialisation.
    /// * `delta` - Convergence threshold for NNDescent iterations.
    /// * `rho` - Sampling rate for NNDescent iterations.
    /// * `refine_knn` - 2-hop refinement sweeps after the main loop.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dist_metric: String,
        // ivf-gpu
        n_list: Option<usize>,
        n_probes: Option<usize>,
        // nndescent-gpu
        build_k: Option<usize>,
        max_iters: Option<usize>,
        n_tree: Option<usize>,
        delta: T,
        rho: Option<T>,
        refine_knn: Option<usize>,
    ) -> Self {
        Self {
            dist_metric,
            n_list,
            n_probes,
            build_k,
            max_iters,
            n_tree,
            delta,
            rho,
            refine_knn,
        }
    }
}

impl<T> Default for NearestNeighbourParamsGpuEvoc<T>
where
    T: AnnSearchFloat,
{
    /// Returns sensible defaults for the GPU-accelerated nearest neighbour
    /// search.
    ///
    /// ### Returns
    ///
    /// Initialised self with sensible default parameters.
    fn default() -> Self {
        Self {
            dist_metric: "euclidean".to_string(),
            // ivf-gpu
            n_list: None,
            n_probes: None,
            // nndescent-gpu
            build_k: None,
            max_iters: None,
            n_tree: None,
            delta: T::from(0.001).unwrap(),
            rho: None,
            refine_knn: None,
        }
    }
}

/// Parse the GPU ANN search variant to use
///
/// ### Params
///
/// * `s` - String identifying the GPU ANN search. One of `"exhaustive_gpu"`,
///   `"ivf_gpu"` or `"nndescent_gpu"`.
///
/// ### Returns
///
/// `Some(AnnSearchGpu)` if recognised, `None` otherwise.
pub fn parse_ann_search_gpu(s: &str) -> Option<AnnSearchGpu> {
    match s.to_lowercase().as_str() {
        "exhaustive_gpu" | "exhaustive" => Some(AnnSearchGpu::ExhaustiveGpu),
        "ivf_gpu" | "ivf" => Some(AnnSearchGpu::IvfGpu),
        "nndescent_gpu" | "nndescent" => Some(AnnSearchGpu::NNDescentGpu),
        _ => None,
    }
}

//////////
// Main //
//////////

/// Run the GPU-accelerated approximate nearest neighbour search
///
/// Mirrors `run_ann_search` but dispatches to GPU-backed indices. The returned
/// indices and distances exclude self, matching the CPU variant's contract.
///
/// ### Params
///
/// * `data` - The data matrix with samples x features
/// * `k` - Number of neighbours to return (excluding self)
/// * `ann_type` - Which GPU search to use. One of `"exhaustive_gpu"`,
///   `"ivf_gpu"` or `"nndescent_gpu"`.
/// * `params_nn` - Parameters for the GPU nearest neighbour search.
/// * `device` - The GPU device to use.
/// * `seed` - Seed for reproducibility.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// `(knn_indices, knn_dist)` excluding self.
#[cfg(feature = "gpu")]
pub fn run_ann_search_gpu<T>(
    data: MatRef<T>,
    k: usize,
    ann_type: String,
    params_nn: &NearestNeighbourParamsGpuEvoc<T>,
    device: Device,
    seed: usize,
    verbose: usize,
) -> EvocKnnResults<T>
where
    T: AnnSearchFloat + CubeclFloat,
{
    let verbosity = parse_verbosity_level(verbose);

    let ann_search = parse_ann_search_gpu(&ann_type).unwrap_or_else(|| {
        println!("Unrecognised GPU-accelerated approximate nearest neighbour method provided: {:?}. Default to GPU IVF.", ann_type);
        AnnSearchGpu::default()
    });

    let dist_metric = &params_nn.dist_metric;

    // The f32 self-queries return `k + 1` rows that usually, but not always,
    // lead with the point itself. The NNDescent builder returns the kNN graph
    // directly, which holds no self-edge.
    let (knn_indices, knn_dist) = match ann_search {
        AnnSearchGpu::ExhaustiveGpu => {
            let index = build_exhaustive_index_gpu::<T>(data, dist_metric, device, verbosity)?;

            let (idx, dist) = query_exhaustive_index_gpu_self(&index, k + 1, true, verbosity)?;
            strip_self(
                idx,
                dist.expect("self-query run with return_dist = true"),
                k,
            )
        }
        AnnSearchGpu::IvfGpu => {
            let index = build_ivf_index_gpu::<T>(
                data,
                params_nn.n_list,
                None,
                dist_metric,
                seed,
                verbosity,
                device,
            )?;

            let (idx, dist) =
                query_ivf_index_gpu_self(&index, k + 1, params_nn.n_probes, None, true, verbosity)?;
            strip_self(
                idx,
                dist.expect("self-query run with return_dist = true"),
                k,
            )
        }
        AnnSearchGpu::NNDescentGpu => {
            let nn_params = NNDescentGpuKnnParams::new(
                k,
                params_nn.build_k,
                params_nn.max_iters,
                params_nn.n_tree,
                params_nn.delta.to_f32(),
                params_nn.rho.map(|r| r.to_f32().unwrap()),
                params_nn.refine_knn,
                false,
                true,
            );
            let graph = build_knn_graph_nndescent_gpu::<T>(
                data,
                dist_metric,
                Some(nn_params),
                None,
                seed,
                verbosity,
                device,
            )?;

            knn_graph_to_rows(graph)
        }
    };

    let knn_dist = rescale_distances(knn_dist, dist_metric);

    Ok((knn_indices, knn_dist))
}
