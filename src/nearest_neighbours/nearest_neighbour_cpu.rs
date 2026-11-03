//! Module containing (approximate) nearest neighbour generation functions.

use ann_search_rs::prelude::*;

use ann_search_rs::*;
use faer::MatRef;
use num_traits::Float;
use rayon::prelude::*;
use std::default::Default;

use crate::nearest_neighbours::{knn_graph_to_rows, rescale_distances, strip_self};
use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Which search algorithm to use for the approximate nearest neighbour search
/// Default is set to Hnsw
#[derive(Default, Clone, Copy, Debug)]
pub enum AnnSearch {
    /// KmKnn
    #[default]
    KmKnn,
    /// HNSW
    Hnsw,
    /// NNDescent
    NNDescent,
    /// Annoy
    Annoy,
    /// IVF
    Ivf,
    /// BallTree
    BallTree,
    /// Exhaustive
    Exhaustive,
    /// Exhaustive on MLX (Apple Silicon, `mlx` feature)
    #[cfg(mlx_available)]
    ExhaustiveMlx,
    /// IVF on MLX (Apple Silicon, `mlx` feature)
    #[cfg(mlx_available)]
    IvfMlx,
    /// NNDescent on MLX (Apple Silicon, `mlx` feature)
    #[cfg(mlx_available)]
    NNDescentMlx,
}

/// Parameters for the nearest neighbour search
#[derive(Debug, Clone)]
pub struct NearestNeighbourParamsEvoc<T> {
    /// Distance metric, one of `"euclidean"` or `"cosine"`
    pub dist_metric: String,
    /// Annoy: Number of trees to use to build the index. Defaults to `50` like
    /// the `uwot` package.
    pub n_tree: usize,
    /// Annoy: Optional search budget per tree. If not provided, defaults to
    /// `k * n_tree * 20` candidates.
    pub search_budget: Option<usize>,
    /// HNSW: connections per given layer to use
    pub m: usize,
    /// HNSW: construction budget
    pub ef_construction: usize,
    /// HNSW: search budget
    pub ef_search: usize,
    /// NNDescent: convergence criterium. If less than these percentage of
    /// neighbours have been udpated, the algorithm counts as converged.
    pub delta: T,
    /// BallTree: Proportions of N to search in the BallTree
    pub bt_budget: T,
    /// IVF: Number of lists, clusters to use. If not provided, will default
    /// to `sqrt(n)` lists.
    pub n_list: Option<usize>,
    /// IVF: Number of lists to probes. If not provided, will default to
    /// to `sqrt(n_list)` lists.
    pub n_probes: Option<usize>,
}

impl<T> NearestNeighbourParamsEvoc<T> {
    /// Generate a new instance
    ///
    /// ### Params
    ///
    /// General parameters
    ///
    /// * `dist_metric` - One of `"euclidean"` or `"cosine"`
    ///
    /// **Annoy**
    ///
    /// * `dist_metric` - One of `"euclidean"` or `"cosine"`
    /// * `n_trees` - Number of trees to use to build the index. Defaults to `50`
    ///   like the `uwot` package.
    /// * `search_budget` - Optional search budget. The algorithm will set the
    ///   search budget to `10 * k * n_trees` if not provided.
    ///
    /// **HNSW**
    ///
    /// * `m` - Number of edges to generate per layer.
    /// * `ef_construction` - Budget during the construction of the index.
    /// * `ef_search` - Budget during the search of the index.
    ///
    /// **NN Descent**
    ///
    /// * `delta` - Early termination criterium.
    ///
    /// **BallTree**
    ///
    /// * `bt_budget` - Budget to use for BallTree
    ///
    /// **IVF**
    ///
    /// * `n_list` - Number of lists to use for IVF
    /// * `n_probes` - Number of lists/clusters to probe during querying
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dist_metric: String,
        // annoy
        n_tree: usize,
        search_budget: Option<usize>,
        // hnsw
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        // nndescent
        delta: T,
        // balltree
        bt_budget: T,
        // ivf / kmknn
        n_list: Option<usize>,
        n_probes: Option<usize>,
    ) -> Self {
        Self {
            dist_metric,
            n_tree,
            search_budget,
            m,
            ef_construction,
            ef_search,
            delta,
            bt_budget,
            n_list,
            n_probes,
        }
    }
}

impl<T> Default for NearestNeighbourParamsEvoc<T>
where
    T: Float,
{
    /// Returns sensible defaults for the approximate nearest neighbour search
    ///
    /// ### Returns
    ///
    /// Initialised self with sensible default parameters.
    fn default() -> Self {
        Self {
            dist_metric: "euclidean".to_string(),
            // annoy
            n_tree: 50,
            search_budget: None,
            // hnsw
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            // nndescent
            delta: T::from(0.001).unwrap(),
            // balltree
            bt_budget: T::from(0.1).unwrap(),
            // ivf
            n_list: None,
            n_probes: None,
        }
    }
}

/// Narrow a matrix to a flat row-major `f32` buffer for the neighbour searches.
///
/// Flat because `AnnMatrix` takes `FlattenData` without a copy, where a
/// `faer::Mat` would be flattened a second time.
///
/// ### Params
///
/// * `data` - The matrix to narrow, samples x features.
///
/// ### Returns
///
/// `(data, n_samples, n_features)` in `f32`.
fn cast_matrix_to_fp32<T>(data: MatRef<T>) -> FlattenData<f32>
where
    T: AnnSearchFloat,
{
    let n = data.nrows();
    let d = data.ncols();
    let mut flat = vec![0.0f32; n * d];

    flat.par_chunks_mut(d.max(1))
        .enumerate()
        .for_each(|(i, row)| {
            for (j, v) in row.iter_mut().enumerate() {
                *v = data[(i, j)].to_f32().unwrap();
            }
        });

    (flat, n, d)
}

/// Cast the `f32` distances of a neighbour search back up to `T`.
///
/// ### Params
///
/// * `knn` - `(knn_indices, optional distances)` from an `f32` search
///
/// ### Returns
///
/// The same rows with the distances in `T`.
fn cast_knn<T>(knn: KnnOptionResult<f32>) -> KnnOptionResult<T>
where
    T: AnnSearchFloat,
{
    let (indices, dist) = knn?;
    let dist = dist.map(|rows| {
        rows.into_par_iter()
            .map(|row| row.into_iter().map(|d| T::from(d).unwrap()).collect())
            .collect()
    });
    Ok((indices, dist))
}

/// Unwrap search results into rows that exclude self.
///
/// Goes by index rather than position: an exact duplicate can rank ahead of the
/// point itself, and points distinct in `f64` can be duplicates once narrowed.
///
/// ### Params
///
/// * `knn` - `(knn_indices, distances)` with distances requested
/// * `k` - Neighbours to keep per point
///
/// ### Returns
///
/// `(knn_indices, knn_dist)` excluding self.
fn finish_knn<T>(knn: KnnOptionResult<T>, k: usize) -> EvocKnnResults<T>
where
    T: Send,
{
    let (idx, dist) = knn?;
    let dist = dist.expect("search run with return_dist = true");

    Ok(strip_self(idx, dist, k))
}

/// Parse the AnnSearch to use
///
/// ### Params
///
/// * `s` - String defining the ANN search to use. `"exhaustive_mlx"`,
///   `"ivf_mlx"` and `"nndescent_mlx"` are only recognised when the MLX
///   backend is compiled in.
///
/// ### Return
///
/// Option of AnnSearch
pub fn parse_ann_search(s: &str) -> Option<AnnSearch> {
    match s.to_lowercase().as_str() {
        "annoy" => Some(AnnSearch::Annoy),
        "balltree" => Some(AnnSearch::BallTree),
        "exhaustive" => Some(AnnSearch::Exhaustive),
        "hnsw" => Some(AnnSearch::Hnsw),
        "ivf" => Some(AnnSearch::Ivf),
        "kmknn" => Some(AnnSearch::KmKnn),
        "nndescent" => Some(AnnSearch::NNDescent),
        #[cfg(mlx_available)]
        "exhaustive_mlx" => Some(AnnSearch::ExhaustiveMlx),
        #[cfg(mlx_available)]
        "ivf_mlx" => Some(AnnSearch::IvfMlx),
        #[cfg(mlx_available)]
        "nndescent_mlx" => Some(AnnSearch::NNDescentMlx),
        _ => None,
    }
}

//////////
// Main //
//////////

/// Run the approximate nearest neighbour search prior to UMAP
///
/// Every backend bar `"exhaustive"` searches in `f32` whatever `T` is, and the
/// distances are cast back up to `T`. `"exhaustive"` stays in `T`, so `f64`
/// ground truth remains available.
///
/// ### Params
///
/// * `data` - The data with samples x features
/// * `k` - Number of neighbours to return
/// * `ann_type` - Which approximate nearest neighbour search to use. One of
///   `"annoy"`, `"balltree"`, `"exhaustive"`, `"hnsw"`, `"ivf"`, `"kmknn"` or
///   `"nndescent"`. With the MLX backend compiled in also `"exhaustive_mlx"`,
///   `"ivf_mlx"` and `"nndescent_mlx"`.
/// * `params_nn` - The parameters for the approximate nearest neighbour search.
/// * `seed` - Seed for reproducibility.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// `(knn_indices, knn_dist)` excluding self.
pub fn run_ann_search<T>(
    data: MatRef<T>,
    k: usize,
    ann_type: String,
    params_nn: &NearestNeighbourParamsEvoc<T>,
    seed: usize,
    verbose: usize,
) -> EvocKnnResults<T>
where
    T: AnnSearchFloat,
{
    let verbosity = parse_verbosity_level(verbose);

    let ann_search = parse_ann_search(&ann_type).unwrap_or_else(|| {
        println!(
            "Unrecognised approximate nearest neighbour method provided: {:?}. Default to KmKnn.",
            ann_type
        );
        AnnSearch::default()
    });

    let dist_metric = &params_nn.dist_metric;

    // Self-queries return `k + 1` rows holding the point itself. The NNDescent
    // builders return the kNN graph directly, which holds no self-edge.
    let (knn_indices, knn_dist) = match ann_search {
        AnnSearch::Annoy => {
            let index = build_annoy_index::<f32>(
                cast_matrix_to_fp32(data),
                dist_metric,
                params_nn.n_tree,
                seed,
                verbosity,
            )?;

            finish_knn(
                cast_knn(query_annoy_self(
                    &index,
                    k + 1,
                    params_nn.search_budget,
                    true,
                    verbosity,
                )),
                k,
            )?
        }
        AnnSearch::Hnsw => {
            let index = build_hnsw_index::<f32>(
                cast_matrix_to_fp32(data),
                params_nn.m,
                params_nn.ef_construction,
                dist_metric,
                seed,
                verbosity,
            )?;

            finish_knn(
                cast_knn(query_hnsw_self::<f32>(
                    &index,
                    k + 1,
                    params_nn.ef_search,
                    true,
                    verbosity,
                )),
                k,
            )?
        }
        AnnSearch::NNDescent => {
            let nn_params = NNDescentKnnParams::new(
                k,
                None,
                None,
                None,
                params_nn.delta.to_f64().unwrap(),
                false,
                true,
            );
            let graph = build_knn_graph_nndescent::<f32>(
                cast_matrix_to_fp32(data),
                dist_metric,
                Some(nn_params),
                seed,
                verbosity,
            )?;
            let (idx, dist) = knn_graph_to_rows(graph);

            finish_knn(cast_knn(Ok((idx, Some(dist)))), k)?
        }
        AnnSearch::BallTree => {
            let index = build_balltree_index::<f32>(
                cast_matrix_to_fp32(data),
                dist_metric,
                seed,
                verbosity,
            )?;

            let budget = (data.nrows() as f32 * params_nn.bt_budget.to_f32().unwrap()) as usize;

            finish_knn(
                cast_knn(query_balltree_self(
                    &index,
                    k + 1,
                    Some(budget),
                    true,
                    verbosity,
                )),
                k,
            )?
        }
        AnnSearch::Exhaustive => {
            let index = build_exhaustive_index(data, dist_metric, verbosity);

            finish_knn(query_exhaustive_self(&index, k + 1, true, verbosity), k)?
        }
        AnnSearch::Ivf => {
            let index = build_ivf_index::<f32>(
                cast_matrix_to_fp32(data),
                params_nn.n_list,
                None,
                dist_metric,
                seed,
                verbosity,
            )?;

            finish_knn(
                cast_knn(query_ivf_self(
                    &index,
                    k + 1,
                    params_nn.n_probes,
                    true,
                    verbosity,
                )),
                k,
            )?
        }
        AnnSearch::KmKnn => {
            let index = build_kmknn_index::<f32>(
                cast_matrix_to_fp32(data),
                dist_metric,
                params_nn.n_list,
                None,
                seed,
                verbosity,
            )?;

            finish_knn(
                cast_knn(query_kmknn_self(&index, k + 1, true, verbosity)),
                k,
            )?
        }
        #[cfg(mlx_available)]
        AnnSearch::ExhaustiveMlx => {
            let index =
                build_exhaustive_index_mlx(cast_matrix_to_fp32(data), dist_metric, verbosity)?;

            finish_knn(
                cast_knn(query_exhaustive_index_mlx_self(
                    &index,
                    k + 1,
                    true,
                    verbosity,
                )),
                k,
            )?
        }
        #[cfg(mlx_available)]
        AnnSearch::IvfMlx => {
            let index = build_ivf_index_mlx(
                cast_matrix_to_fp32(data),
                params_nn.n_list,
                None,
                dist_metric,
                seed,
                verbosity,
            )?;

            finish_knn(
                cast_knn(query_ivf_index_mlx_self(
                    &index,
                    k + 1,
                    params_nn.n_probes,
                    None,
                    true,
                    verbosity,
                )),
                k,
            )?
        }
        #[cfg(mlx_available)]
        AnnSearch::NNDescentMlx => {
            let graph = build_knn_graph_mlx(
                cast_matrix_to_fp32(data),
                dist_metric,
                Some(k),
                None,
                None,
                None,
                params_nn.delta.to_f32(),
                None,
                None,
                seed,
                verbosity,
            )?;

            finish_knn(cast_knn(Ok(graph.extract_knn(None, false, true))), k)?
        }
    };

    let knn_dist = rescale_distances(knn_dist, dist_metric);

    Ok((knn_indices, knn_dist))
}
