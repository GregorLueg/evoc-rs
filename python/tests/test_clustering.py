"""The estimator actually clusters."""

import numpy as np
import pytest
from conftest import N_CLUSTERS, agreement

import evoc_rs


def test_recovers_planted_clusters(blobs):
    X, y = blobs
    model = evoc_rs.EVoC(n_neighbours=15, seed=42).fit(X)

    assert model.labels_.shape == (len(y),)
    assert model.labels_.dtype == np.int64
    assert agreement(model.labels_, y) > 0.95


def test_layers_are_rectangular_and_ranked(blobs):
    X, _ = blobs
    model = evoc_rs.EVoC(n_neighbours=15, seed=42).fit(X)

    n_layers, n = model.cluster_layers_.shape
    assert n == X.shape[0]
    assert model.layer_strengths_.shape == (n_layers, n)
    assert model.persistence_scores_.shape == (n_layers,)

    # `labels_` is the most persistent layer, matching `best_labels` in Rust.
    best = 0 if n_layers == 1 else int(np.argmax(model.persistence_scores_))
    assert np.array_equal(model.labels_, model.cluster_layers_[best])


def test_membership_strengths_in_unit_interval(blobs):
    X, _ = blobs
    model = evoc_rs.EVoC(n_neighbours=15, seed=42).fit(X)

    assert model.membership_strengths_.min() >= 0.0
    assert model.membership_strengths_.max() <= 1.0


def test_approx_n_clusters_returns_one_layer(blobs):
    X, _ = blobs
    model = evoc_rs.EVoC(n_neighbours=15, approx_n_clusters=N_CLUSTERS, seed=42).fit(X)

    assert model.cluster_layers_.shape[0] == 1
    assert model.n_clusters_ == pytest.approx(N_CLUSTERS, abs=1)


def test_neighbour_graph_excludes_self(blobs):
    X, _ = blobs
    k = 15
    model = evoc_rs.EVoC(n_neighbours=k, seed=42).fit(X)

    distances, indices = model.neighbour_graph_
    assert indices.shape == (X.shape[0], k)
    assert distances.shape == (X.shape[0], k)
    assert not (indices == np.arange(X.shape[0])[:, None]).any()


def test_precomputed_knn_matches_the_built_graph(blobs):
    X, _ = blobs
    built = evoc_rs.EVoC(n_neighbours=15, ann_algorithm="exhaustive", seed=42).fit(X)
    distances, indices = built.neighbour_graph_

    reused = evoc_rs.EVoC(n_neighbours=15, seed=42).fit(
        X, precomputed_knn=(indices, distances)
    )

    assert np.array_equal(built.labels_, reused.labels_)


def test_float64_input_is_kept_at_float64(blobs):
    X, _ = blobs
    model = evoc_rs.EVoC(n_neighbours=15, seed=42).fit(X.astype(np.float64))

    assert model.membership_strengths_.dtype == np.float64


def test_reproducible_across_runs(blobs):
    X, _ = blobs
    a = evoc_rs.EVoC(n_neighbours=15, ann_algorithm="exhaustive", seed=7).fit(X)
    b = evoc_rs.EVoC(n_neighbours=15, ann_algorithm="exhaustive", seed=7).fit(X)

    assert np.array_equal(a.labels_, b.labels_)


@pytest.mark.parametrize(
    "backend", ["nndescent", "hnsw", "annoy", "ivf", "kmknn", "balltree", "exhaustive"]
)
def test_every_cpu_backend_dispatches(blobs, backend):
    X, y = blobs
    model = evoc_rs.EVoC(n_neighbours=15, ann_algorithm=backend, seed=42).fit(X)

    assert model.labels_.shape == (len(y),)
