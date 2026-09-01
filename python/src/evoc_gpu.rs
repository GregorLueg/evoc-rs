//! The GPU entry point.
//!
//! Only the kNN stage runs on the GPU; the fuzzy graph, embedding, MST and
//! persistence analysis stay on the CPU exactly as in [`crate::evoc`]. f32
//! only, because WGSL has no f64 and consumer GPUs cripple its throughput
//! anyway.
//!
//! The runtime is pinned rather than generic: a `R: Runtime` type parameter
//! cannot cross into Python, and wgpu is the only backend the wheel ships.

use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use evoc_rs::evoc_gpu;
use evoc_rs::prelude::*;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::convert;
use crate::error::EvocErr;
use crate::params::EvocArgs;
use crate::pool;

/// The runtime every GPU call here uses.
pub(crate) type Rt = WgpuRuntime;

/// The device every GPU call here uses.
///
/// ### Returns
///
/// wgpu's best available adapter.
pub(crate) fn default_device() -> WgpuDevice {
    WgpuDevice::default()
}

/// Every knob of `NearestNeighbourParamsGpuEvoc`, as plain scalars.
struct NnGpuArgs {
    /// Distance metric, `"euclidean"` or `"cosine"`.
    dist_metric: String,
    /// IVF-GPU: number of lists.
    n_list: Option<usize>,
    /// IVF-GPU: number of lists to probe.
    n_probes: Option<usize>,
    /// NNDescent-GPU: CAGRA graph degree after pruning.
    k: Option<usize>,
    /// NNDescent-GPU: build degree before pruning.
    k_build: Option<usize>,
    /// NNDescent-GPU: trees for the kNN graph initialisation.
    n_tree: Option<usize>,
    /// NNDescent-GPU: termination criterion.
    delta: f64,
    /// NNDescent-GPU: sampling rate.
    rho: Option<f64>,
    /// NNDescent-GPU: beam width when querying.
    beam_width: Option<usize>,
    /// NNDescent-GPU: beam iterations when querying.
    max_beam_iters: Option<usize>,
    /// NNDescent-GPU: entry points when querying.
    n_entry_points: Option<usize>,
}

impl NnGpuArgs {
    /// Rebuild the crate's GPU ANN parameter struct at f32.
    fn build(&self) -> NearestNeighbourParamsGpuEvoc<f32> {
        NearestNeighbourParamsGpuEvoc {
            dist_metric: self.dist_metric.clone(),
            n_list: self.n_list,
            n_probes: self.n_probes,
            k: self.k,
            k_build: self.k_build,
            n_tree: self.n_tree,
            delta: self.delta as f32,
            rho: self.rho.map(|v| v as f32),
            beam_width: self.beam_width,
            max_beam_iters: self.max_beam_iters,
            n_entry_points: self.n_entry_points,
        }
    }
}

/// Cluster a dense float32 matrix with GPU-accelerated kNN.
///
/// ### Params
///
/// * `x` - `(n_samples, n_features)` C-contiguous float32 array.
/// * `precomputed_knn` - `(indices, distances)` excluding self, or `None`.
/// * `ann_type` - One of `"exhaustive_gpu"`, `"ivf_gpu"`, `"nndescent_gpu"`.
///
/// ### Returns
///
/// The same five arrays [`crate::evoc::run_evoc`] returns.
#[pyfunction]
#[pyo3(signature = (
    x,
    *,
    precomputed_knn = None,
    ann_type,
    n_neighbours,
    noise_level,
    n_epochs,
    embedding_dim,
    neighbour_scale,
    symmetrise,
    min_samples,
    base_min_cluster_size,
    approx_n_clusters,
    min_similarity_threshold,
    max_layers,
    dist_metric,
    n_list,
    n_probes,
    k,
    k_build,
    n_tree,
    delta,
    rho,
    beam_width,
    max_beam_iters,
    n_entry_points,
    seed = 42,
    verbose = 0,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_evoc_gpu<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    precomputed_knn: crate::params::Knn<f64>,
    ann_type: &str,
    n_neighbours: usize,
    noise_level: f64,
    n_epochs: usize,
    embedding_dim: Option<usize>,
    neighbour_scale: f64,
    symmetrise: bool,
    min_samples: usize,
    base_min_cluster_size: usize,
    approx_n_clusters: Option<usize>,
    min_similarity_threshold: f64,
    max_layers: usize,
    dist_metric: String,
    n_list: Option<usize>,
    n_probes: Option<usize>,
    k: Option<usize>,
    k_build: Option<usize>,
    n_tree: Option<usize>,
    delta: f64,
    rho: Option<f64>,
    beam_width: Option<usize>,
    max_beam_iters: Option<usize>,
    n_entry_points: Option<usize>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let a = x
        .extract::<PyReadonlyArray2<'_, f32>>()
        .map_err(|_| PyTypeError::new_err("the GPU path is float32 only, since WGSL has no f64"))?;

    let evoc_args = EvocArgs {
        n_neighbours,
        noise_level,
        n_epochs,
        embedding_dim,
        neighbour_scale,
        symmetrise,
        min_samples,
        base_min_cluster_size,
        approx_n_clusters,
        min_similarity_threshold,
        max_layers,
    };
    let nn_args = NnGpuArgs {
        dist_metric,
        n_list,
        n_probes,
        k,
        k_build,
        n_tree,
        delta,
        rho,
        beam_width,
        max_beam_iters,
        n_entry_points,
    };

    let (data, n, dim) = convert::flat(&a)?;
    let params = evoc_args.build::<f32>();
    let nn_params = nn_args.build();
    let ann_type = ann_type.to_string();
    let precomputed = crate::params::narrow_knn(precomputed_knn);
    let device = default_device();

    let result = py
        .detach(|| {
            pool::run(|| {
                evoc_gpu::<f32, Rt>(
                    (data, n, dim),
                    ann_type,
                    precomputed,
                    &params,
                    &nn_params,
                    device,
                    seed,
                    verbose,
                )
            })
        })
        .map_err(EvocErr)?;

    let (knn_idx, knn_dist) = convert::pack_knn(py, &result.nn_indices, &result.nn_distances)?;
    let out = (
        convert::pack_layers(py, &result.cluster_layers)?,
        convert::pack_strengths(py, &result.membership_strengths)?,
        convert::pack_scores(py, result.persistence_scores),
        knn_idx,
        knn_dist,
    );
    Ok(out.into_pyobject(py)?.into_any())
}
