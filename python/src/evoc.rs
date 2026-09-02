//! The CPU entry point.
//!
//! One function rather than a handle class: EVoC is one-shot, there is no index
//! to keep and nothing to query afterwards, so the estimator state is the five
//! numpy arrays this returns and Python owns it from there.
//!
//! Nothing generic reaches `evoc_rs::evoc`. The dispatch macro expands to
//! concrete `f32` / `f64` code, so the core's trait bounds
//! (`NNDescent<T>: ApplySortedUpdates<T>` and friends) never have to be
//! restated here, and `ann-search-rs` stays out of this crate's manifest.

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

use crate::convert;
use crate::error::EvocErr;
use crate::params::{EvocArgs, NnArgs};
use crate::pool;

/// Expand one precision arm: borrow the array, run, pack the five outputs.
///
/// Note the nesting: `py.detach(|| pool::run(...))`, never the other way round.
/// `Python<'py>` is not `Send`, so a `pool::run` closure capturing it would not
/// satisfy `Ungil`, and the GIL should be dropped before the rayon fan-out
/// starts regardless.
macro_rules! evoc_arm {
    ($py:ident, $a:ident, $ty:ty, $knn:expr, $evoc:ident, $nn:ident, $ann:ident, $seed:ident, $verbose:ident) => {{
        let (data, n, dim) = convert::flat(&$a)?;
        let params = $evoc.build::<$ty>();
        let nn_params = $nn.build::<$ty>();
        let ann_type = $ann.to_string();
        let precomputed = $knn;

        let result = $py
            .detach(|| {
                pool::run(|| {
                    evoc_rs::evoc(
                        (data, n, dim),
                        ann_type,
                        precomputed,
                        &params,
                        &nn_params,
                        $seed,
                        $verbose,
                    )
                })
            })
            .map_err(EvocErr)?;

        let (knn_idx, knn_dist) = convert::pack_knn($py, &result.nn_indices, &result.nn_distances)?;
        let out = (
            convert::pack_layers($py, &result.cluster_layers)?,
            convert::pack_strengths($py, &result.membership_strengths)?,
            convert::pack_scores($py, result.persistence_scores),
            knn_idx,
            knn_dist,
        );
        Ok(out.into_pyobject($py)?.into_any())
    }};
}

/// Cluster a dense matrix with EVoC.
///
/// Every knob arrives flat. The Python estimator holds the defaults and the
/// validation; this only rebuilds the two parameter structs and dispatches on
/// dtype.
///
/// ### Params
///
/// * `x` - `(n_samples, n_features)` C-contiguous array of float32 or float64.
/// * `precomputed_knn` - `(indices, distances)` excluding self, or `None` to
///   build the graph from `x`. Distances always cross as float64, since Python
///   has one float type; a float32 run narrows them here rather than making the
///   caller match dtypes on a graph they may have built elsewhere.
/// * `ann_type` - ANN backend name.
///
/// ### Returns
///
/// `(cluster_layers, membership_strengths, persistence_scores, knn_indices,
/// knn_distances)`. The float arrays match the input dtype; the labels are
/// always `int64` and the scores always `float64`.
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
    n_tree,
    search_budget,
    m,
    ef_construction,
    ef_search,
    diversify_prob,
    delta,
    ef_budget,
    extract_knn,
    bt_budget,
    n_list,
    n_probes,
    seed = 42,
    verbose = 0,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_evoc<'py>(
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
    n_tree: usize,
    search_budget: Option<usize>,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    diversify_prob: f64,
    delta: f64,
    ef_budget: Option<usize>,
    extract_knn: bool,
    bt_budget: f64,
    n_list: Option<usize>,
    n_probes: Option<usize>,
    seed: usize,
    verbose: usize,
) -> PyResult<Bound<'py, PyAny>> {
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
    let nn_args = NnArgs {
        dist_metric,
        n_tree,
        search_budget,
        m,
        ef_construction,
        ef_search,
        diversify_prob,
        delta,
        ef_budget,
        extract_knn,
        bt_budget,
        n_list,
        n_probes,
    };

    if let Ok(a) = x.extract::<PyReadonlyArray2<'_, f32>>() {
        let knn = crate::params::narrow_knn(precomputed_knn);
        return evoc_arm!(py, a, f32, knn, evoc_args, nn_args, ann_type, seed, verbose);
    }

    if let Ok(a) = x.extract::<PyReadonlyArray2<'_, f64>>() {
        return evoc_arm!(
            py,
            a,
            f64,
            precomputed_knn,
            evoc_args,
            nn_args,
            ann_type,
            seed,
            verbose
        );
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "X must be a 2-D numpy array of dtype float32 or float64",
    ))
}
