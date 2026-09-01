"""The scikit-learn shaped surface."""

import pickle

import numpy as np
import pytest

import evoc_rs
from evoc_rs import NotFittedError


def test_get_params_round_trips():
    model = evoc_rs.EVoC(n_neighbours=20, metric="cosine")
    params = model.get_params()

    assert params["n_neighbours"] == 20
    assert params["metric"] == "cosine"
    assert evoc_rs.EVoC(**params).get_params() == params


def test_set_params_mutates_in_place():
    model = evoc_rs.EVoC()
    assert model.set_params(n_neighbours=30) is model
    assert model.n_neighbours == 30


def test_set_params_rejects_unknown_name():
    with pytest.raises(ValueError, match="is not a parameter"):
        evoc_rs.EVoC().set_params(nonsense=1)


def test_repr_shows_parameters():
    text = repr(evoc_rs.EVoC(n_neighbours=20))
    assert text.startswith("EVoC(")
    assert "n_neighbours=20" in text


def test_n_clusters_before_fit_raises():
    with pytest.raises(NotFittedError):
        _ = evoc_rs.EVoC().n_clusters_


def test_pickle_round_trip_keeps_labels(blobs):
    X, _ = blobs
    model = evoc_rs.EVoC(n_neighbours=15, seed=42).fit(X)

    restored = pickle.loads(pickle.dumps(model))

    assert np.array_equal(model.labels_, restored.labels_)
    assert restored.get_params() == model.get_params()


def test_fit_predict_matches_fit(blobs):
    X, _ = blobs
    labels = evoc_rs.EVoC(
        n_neighbours=15, ann_algorithm="exhaustive", seed=42
    ).fit_predict(X)
    model = evoc_rs.EVoC(n_neighbours=15, ann_algorithm="exhaustive", seed=42).fit(X)

    assert np.array_equal(labels, model.labels_)


def test_sklearn_clone_works(blobs):
    sklearn = pytest.importorskip("sklearn.base")
    model = evoc_rs.EVoC(n_neighbours=20)

    assert sklearn.clone(model).get_params() == model.get_params()


def test_version_strings_are_present():
    assert isinstance(evoc_rs.__version__, str)
    assert isinstance(evoc_rs.__core_version__, str)


def test_thread_override_round_trips():
    original = evoc_rs.num_threads()
    evoc_rs.set_num_threads(2)
    assert evoc_rs.num_threads() == 2
    evoc_rs.set_num_threads(0)
    assert evoc_rs.num_threads() == original
