"""Bad arguments are rejected here, not inside the core."""

import numpy as np
import pytest

import evoc_rs


def test_rejects_non_numeric_dtype():
    X = np.array([["a", "b"], ["c", "d"]])
    with pytest.raises(TypeError, match="must hold numbers"):
        evoc_rs.EVoC().fit(X)


def test_rejects_non_2d():
    with pytest.raises(ValueError, match="must be 2-D"):
        evoc_rs.EVoC().fit(np.zeros(10, dtype=np.float32))


def test_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        evoc_rs.EVoC().fit(np.zeros((0, 4), dtype=np.float32))


def test_rejects_non_finite():
    X = np.ones((50, 4), dtype=np.float32)
    X[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        evoc_rs.EVoC().fit(X)


def test_rejects_too_many_neighbours():
    X = np.ones((10, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="below the sample count"):
        evoc_rs.EVoC(n_neighbours=20).fit(X)


def test_rejects_unknown_backend(blobs):
    X, _ = blobs
    with pytest.raises(ValueError, match="ann_algorithm must be one of"):
        evoc_rs.EVoC(ann_algorithm="nonsense").fit(X)


def test_rejects_unknown_metric(blobs):
    X, _ = blobs
    with pytest.raises(ValueError, match="metric must be one of"):
        evoc_rs.EVoC(metric="mahalanobis").fit(X)


def test_rejects_gpu_backend_on_cpu_estimator(blobs):
    X, _ = blobs
    with pytest.raises(ValueError, match="ann_algorithm must be one of"):
        evoc_rs.EVoC(ann_algorithm="ivf_gpu").fit(X)


def test_integer_input_is_promoted_not_narrowed(blobs):
    X, _ = blobs
    model = evoc_rs.EVoC(n_neighbours=15, seed=42).fit(X.astype(np.int32))

    assert model.membership_strengths_.dtype == np.float64


def test_non_contiguous_input_is_accepted(blobs):
    X, _ = blobs
    # A strided view: every other column. The validator makes it contiguous.
    model = evoc_rs.EVoC(n_neighbours=15, seed=42).fit(X[:, ::2])

    assert model.n_features_in_ == X.shape[1] // 2


def test_precomputed_knn_shape_mismatch(blobs):
    X, _ = blobs
    indices = np.zeros((len(X), 5), dtype=np.int64)
    distances = np.zeros((len(X), 4), dtype=np.float64)

    with pytest.raises(ValueError, match="shapes disagree"):
        evoc_rs.EVoC().fit(X, precomputed_knn=(indices, distances))


def test_precomputed_knn_wrong_row_count(blobs):
    X, _ = blobs
    indices = np.zeros((7, 5), dtype=np.int64)
    distances = np.zeros((7, 5), dtype=np.float64)

    with pytest.raises(ValueError, match="rows, X has"):
        evoc_rs.EVoC().fit(X, precomputed_knn=(indices, distances))
