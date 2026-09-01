"""The EVoC estimators.

`EVoC` runs the whole pipeline on the CPU. `EVoCGpu` moves the kNN stage onto
the GPU and leaves everything else where it was, which is the only part of EVoC
that is worth the transfer.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from beartype import beartype

from . import _evoc_rs as _core
from ._base import BaseEstimator
from ._validate import (
    CPU_BACKENDS,
    GPU_BACKENDS,
    METRICS,
    check_choice,
    check_matrix,
    check_n_neighbours,
    check_precomputed_knn,
)


class EVoC(BaseEstimator):
    """Cluster high-dimensional embeddings with EVoC.

    Six stages: an approximate kNN graph, a fuzzy simplicial set over it, a
    label-propagation initialisation, a UMAP-like node embedding, an MST over
    mutual reachability distances, then a hierarchy of clusterings pulled out of
    that tree by persistence.

    The hierarchy is the point. `cluster_layers_` holds one labelling per
    granularity, finest first, and `labels_` picks the layer with the highest
    persistence score. Set `approx_n_clusters` instead if you know how many
    clusters you want, and the finest layer is binary-searched for it.

    Args:
        n_neighbours: Neighbours per point in the kNN graph.
        noise_level: Repulsion strength in the embedding gradient. 0.0 is
            aggressive, 1.0 conservative.
        n_epochs: Embedding optimisation epochs.
        embedding_dim: Embedding dimensionality. None derives it from
            `n_neighbours` as `clamp(n_neighbours // 4, 4, 16)`.
        neighbour_scale: Multiplier on the effective neighbour count when
            building the fuzzy graph.
        symmetrise: Whether to symmetrise the fuzzy graph.
        min_samples: Points used for the core distance in the MST density
            estimate.
        base_min_cluster_size: Minimum cluster size at the finest layer.
        approx_n_clusters: Binary-search `base_min_cluster_size` for roughly
            this many clusters and return a single layer. None returns the full
            hierarchy.
        min_similarity_threshold: Jaccard similarity above which two layers
            count as redundant and the coarser one is dropped.
        max_layers: Cap on the layers returned.
        ann_algorithm: kNN backend. One of nndescent, hnsw, annoy, ivf, kmknn,
            balltree, exhaustive.
        metric: Distance metric, euclidean or cosine.
        n_tree: Annoy trees.
        search_budget: Annoy candidates per query. None uses the crate default.
        m: HNSW connections per layer.
        ef_construction: HNSW construction budget.
        ef_search: HNSW search budget.
        diversify_prob: NN-Descent diversification probability.
        delta: NN-Descent convergence threshold.
        ef_budget: NN-Descent beam budget when querying. None auto-picks.
        bt_budget: BallTree search budget, as a fraction of the sample count.
        n_list: IVF cells. None uses sqrt(n).
        n_probes: IVF cells probed per query. None uses sqrt(n_list).
        seed: Random seed.
        verbose: 0 silent, 1 normal, 2 detailed.

    Attributes:
        labels_: `(n_samples,)` int64 labels from the highest-persistence
            layer. -1 is noise.
        membership_strengths_: `(n_samples,)` strengths in [0, 1] for
            `labels_`.
        cluster_layers_: `(n_layers, n_samples)` int64, every layer, finest
            first.
        layer_strengths_: `(n_layers, n_samples)` strengths for every layer.
        persistence_scores_: `(n_layers,)` float64, higher is more stable.
        neighbour_graph_: `(distances, indices)` from the kNN stage, self
            excluded.
        n_clusters_: Non-noise clusters in `labels_`.
        n_features_in_: Columns seen during `fit`.

    Example:
        >>> import numpy as np
        >>> from evoc_rs import EVoC
        >>> rng = np.random.default_rng(0)
        >>> X = np.vstack([rng.normal(c * 20, 1, (200, 32)) for c in range(3)])
        >>> labels = EVoC(n_neighbours=15).fit_predict(X.astype(np.float32))
    """

    #: ANN backends this estimator will dispatch to.
    _BACKENDS: ClassVar[frozenset[str]] = CPU_BACKENDS
    #: Element type to force on `fit`, or None to keep the caller's.
    _FORCE_DTYPE: ClassVar[np.dtype | None] = None

    @beartype
    def __init__(
        self,
        n_neighbours: int = 15,
        *,
        noise_level: float = 0.5,
        n_epochs: int = 50,
        embedding_dim: int | None = None,
        neighbour_scale: float = 1.0,
        symmetrise: bool = True,
        min_samples: int = 5,
        base_min_cluster_size: int = 5,
        approx_n_clusters: int | None = None,
        min_similarity_threshold: float = 0.2,
        max_layers: int = 10,
        ann_algorithm: str = "nndescent",
        metric: str = "euclidean",
        n_tree: int = 50,
        search_budget: int | None = None,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        diversify_prob: float = 1.0,
        delta: float = 0.001,
        ef_budget: int | None = None,
        bt_budget: float = 0.05,
        n_list: int | None = None,
        n_probes: int | None = None,
        seed: int = 42,
        verbose: int = 0,
    ) -> None:
        self.n_neighbours = n_neighbours
        self.noise_level = noise_level
        self.n_epochs = n_epochs
        self.embedding_dim = embedding_dim
        self.neighbour_scale = neighbour_scale
        self.symmetrise = symmetrise
        self.min_samples = min_samples
        self.base_min_cluster_size = base_min_cluster_size
        self.approx_n_clusters = approx_n_clusters
        self.min_similarity_threshold = min_similarity_threshold
        self.max_layers = max_layers
        self.ann_algorithm = ann_algorithm
        self.metric = metric
        self.n_tree = n_tree
        self.search_budget = search_budget
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.diversify_prob = diversify_prob
        self.delta = delta
        self.ef_budget = ef_budget
        self.bt_budget = bt_budget
        self.n_list = n_list
        self.n_probes = n_probes
        self.seed = seed
        self.verbose = verbose

    # Unfitted state, as class attributes so `_check_fitted` has something to
    # read and pickling an unfitted estimator stays cheap.
    labels_: np.ndarray | None = None
    membership_strengths_: np.ndarray | None = None
    cluster_layers_: np.ndarray | None = None
    layer_strengths_: np.ndarray | None = None
    persistence_scores_: np.ndarray | None = None
    neighbour_graph_: tuple[np.ndarray, np.ndarray] | None = None
    n_features_in_: int | None = None

    def _run(self, X: np.ndarray, knn: Any) -> tuple[Any, ...]:
        """Call the compiled core. Overridden by the GPU subclass.

        Args:
            X: Validated design matrix.
            knn: Precomputed graph as nested lists, or None.

        Returns:
            The five arrays the core hands back.
        """
        return _core.run_evoc(
            X,
            precomputed_knn=knn,
            ann_type=self.ann_algorithm,
            n_neighbours=self.n_neighbours,
            noise_level=self.noise_level,
            n_epochs=self.n_epochs,
            embedding_dim=self.embedding_dim,
            neighbour_scale=self.neighbour_scale,
            symmetrise=self.symmetrise,
            min_samples=self.min_samples,
            base_min_cluster_size=self.base_min_cluster_size,
            approx_n_clusters=self.approx_n_clusters,
            min_similarity_threshold=self.min_similarity_threshold,
            max_layers=self.max_layers,
            dist_metric=self.metric,
            n_tree=self.n_tree,
            search_budget=self.search_budget,
            m=self.m,
            ef_construction=self.ef_construction,
            ef_search=self.ef_search,
            diversify_prob=self.diversify_prob,
            delta=self.delta,
            ef_budget=self.ef_budget,
            bt_budget=self.bt_budget,
            n_list=self.n_list,
            n_probes=self.n_probes,
            seed=self.seed,
            verbose=self.verbose,
        )

    @beartype
    def fit(
        self,
        X: np.ndarray,
        y: None = None,
        *,
        precomputed_knn: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> EVoC:
        """Cluster `X`.

        Args:
            X: `(n_samples, n_features)` design matrix.
            y: Ignored. Present so the estimator fits in a scikit-learn
                pipeline.
            precomputed_knn: `(indices, distances)`, both `(n_samples, k)` and
                excluding self. Skips the kNN stage entirely.

        Returns:
            The fitted estimator, so calls chain.
        """
        check_choice(self.ann_algorithm, self._BACKENDS, "ann_algorithm")
        check_choice(self.metric, METRICS, "metric")

        X = check_matrix(X, force_dtype=self._FORCE_DTYPE)
        check_n_neighbours(self.n_neighbours, X.shape[0])

        knn = (
            check_precomputed_knn(precomputed_knn, X.shape[0])
            if precomputed_knn is not None
            else None
        )

        layers, strengths, scores, knn_idx, knn_dist = self._run(X, knn)

        self.cluster_layers_ = layers
        self.layer_strengths_ = strengths
        self.persistence_scores_ = scores
        self.neighbour_graph_ = (knn_dist, knn_idx)
        self.n_features_in_ = X.shape[1]

        # Matches `EvocResult::best_labels`: the single layer when there is
        # one, otherwise the most persistent.
        best = 0 if len(scores) <= 1 else int(np.argmax(scores))
        self.labels_ = layers[best]
        self.membership_strengths_ = strengths[best]

        return self

    @beartype
    def fit_predict(
        self,
        X: np.ndarray,
        y: None = None,
        *,
        precomputed_knn: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Cluster `X` and return the labels.

        Args:
            X: `(n_samples, n_features)` design matrix.
            y: Ignored.
            precomputed_knn: As for `fit`.

        Returns:
            `(n_samples,)` int64 labels, -1 for noise.
        """
        return self.fit(X, precomputed_knn=precomputed_knn)._fitted_labels()

    @property
    def n_clusters_(self) -> int:
        """Non-noise clusters in `labels_`.

        Returns:
            The count, 0 when everything is noise.

        Raises:
            NotFittedError: If `fit` has not run.
        """
        return max(int(self._fitted_labels().max()) + 1, 0)


class EVoCGpu(EVoC):
    """`EVoC` with the kNN stage on the GPU.

    Everything downstream of the graph stays on the CPU, so this only pays off
    when the kNN search dominates: many points, high dimension, or an exhaustive
    search. float32 only, because WGSL has no f64.

    Args and attributes are those of `EVoC`, with these differences:

    Args:
        ann_algorithm: One of exhaustive_gpu, ivf_gpu, nndescent_gpu.
        k: CAGRA graph degree after pruning, for nndescent_gpu. None backfills
            from `n_neighbours`.
        k_build: Build degree before pruning. None backfills to
            `2 * n_neighbours`.
        n_tree: Trees initialising the kNN graph. None uses the crate default.
        rho: NN-Descent sampling rate. None auto-picks.
        beam_width: Beam width when querying. None auto-picks.
        max_beam_iters: Beam iterations when querying. None auto-picks.
        n_entry_points: Entry points when querying. None auto-picks.
    """

    _BACKENDS: ClassVar[frozenset[str]] = GPU_BACKENDS
    _FORCE_DTYPE: ClassVar[np.dtype | None] = np.dtype(np.float32)

    @beartype
    def __init__(
        self,
        n_neighbours: int = 15,
        *,
        noise_level: float = 0.5,
        n_epochs: int = 50,
        embedding_dim: int | None = None,
        neighbour_scale: float = 1.0,
        symmetrise: bool = True,
        min_samples: int = 5,
        base_min_cluster_size: int = 5,
        approx_n_clusters: int | None = None,
        min_similarity_threshold: float = 0.2,
        max_layers: int = 10,
        ann_algorithm: str = "ivf_gpu",
        metric: str = "euclidean",
        n_list: int | None = None,
        n_probes: int | None = None,
        k: int | None = None,
        k_build: int | None = None,
        n_tree: int | None = None,
        delta: float = 0.001,
        rho: float | None = None,
        beam_width: int | None = None,
        max_beam_iters: int | None = None,
        n_entry_points: int | None = None,
        seed: int = 42,
        verbose: int = 0,
    ) -> None:
        self.n_neighbours = n_neighbours
        self.noise_level = noise_level
        self.n_epochs = n_epochs
        self.embedding_dim = embedding_dim
        self.neighbour_scale = neighbour_scale
        self.symmetrise = symmetrise
        self.min_samples = min_samples
        self.base_min_cluster_size = base_min_cluster_size
        self.approx_n_clusters = approx_n_clusters
        self.min_similarity_threshold = min_similarity_threshold
        self.max_layers = max_layers
        self.ann_algorithm = ann_algorithm
        self.metric = metric
        self.n_list = n_list
        self.n_probes = n_probes
        self.k = k
        self.k_build = k_build
        self.n_tree = n_tree
        self.delta = delta
        self.rho = rho
        self.beam_width = beam_width
        self.max_beam_iters = max_beam_iters
        self.n_entry_points = n_entry_points
        self.seed = seed
        self.verbose = verbose

    def _run(self, X: np.ndarray, knn: Any) -> tuple[Any, ...]:
        """Call the compiled GPU core.

        Args:
            X: Validated float32 design matrix.
            knn: Precomputed graph as nested lists, or None.

        Returns:
            The five arrays the core hands back.

        Raises:
            ImportError: If this build has no GPU support.
        """
        if not hasattr(_core, "run_evoc_gpu"):
            raise ImportError(
                "this build has no GPU support; it was compiled with "
                "--no-default-features"
            )
        return _core.run_evoc_gpu(
            X,
            precomputed_knn=knn,
            ann_type=self.ann_algorithm,
            n_neighbours=self.n_neighbours,
            noise_level=self.noise_level,
            n_epochs=self.n_epochs,
            embedding_dim=self.embedding_dim,
            neighbour_scale=self.neighbour_scale,
            symmetrise=self.symmetrise,
            min_samples=self.min_samples,
            base_min_cluster_size=self.base_min_cluster_size,
            approx_n_clusters=self.approx_n_clusters,
            min_similarity_threshold=self.min_similarity_threshold,
            max_layers=self.max_layers,
            dist_metric=self.metric,
            n_list=self.n_list,
            n_probes=self.n_probes,
            k=self.k,
            k_build=self.k_build,
            n_tree=self.n_tree,
            delta=self.delta,
            rho=self.rho,
            beam_width=self.beam_width,
            max_beam_iters=self.max_beam_iters,
            n_entry_points=self.n_entry_points,
            seed=self.seed,
            verbose=self.verbose,
        )
