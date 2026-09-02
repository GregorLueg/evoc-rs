"""The GPU estimator.

Skipped twice over: once when the extension was built without the `gpu`
feature, once when there is no adapter on this machine. `gpu_available()`
answers both at once.
"""

import numpy as np
import pytest
from conftest import agreement

import evoc_rs

pytestmark = pytest.mark.skipif(
    not evoc_rs.gpu_available(),
    reason="no GPU support compiled in, or no adapter on this machine",
)


def test_recovers_planted_clusters(blobs):
    X, y = blobs
    model = evoc_rs.EVoCGpu(n_neighbours=15, seed=42).fit(X)

    assert agreement(model.labels_, y) > 0.95


@pytest.mark.parametrize("backend", ["exhaustive_gpu", "ivf_gpu", "nndescent_gpu"])
def test_every_gpu_backend_dispatches(blobs, backend):
    X, y = blobs
    model = evoc_rs.EVoCGpu(n_neighbours=15, ann_algorithm=backend, seed=42).fit(X)

    assert model.labels_.shape == (len(y),)


def test_float64_input_is_narrowed_to_float32(blobs):
    X, _ = blobs
    model = evoc_rs.EVoCGpu(n_neighbours=15, seed=42).fit(X.astype(np.float64))

    assert model.membership_strengths_.dtype == np.float32


def test_rejects_cpu_backend(blobs):
    X, _ = blobs
    with pytest.raises(ValueError, match="ann_algorithm must be one of"):
        evoc_rs.EVoCGpu(ann_algorithm="hnsw").fit(X)


def test_structurally_agrees_with_cpu(blobs):
    X, y = blobs
    cpu = evoc_rs.EVoC(n_neighbours=15, ann_algorithm="exhaustive", seed=42).fit(X)
    gpu = evoc_rs.EVoCGpu(n_neighbours=15, ann_algorithm="exhaustive_gpu", seed=42).fit(
        X
    )

    # Not label-identical: the two kNN paths break ties differently. Both must
    # still recover the planted structure.
    assert agreement(gpu.labels_, y) == pytest.approx(
        agreement(cpu.labels_, y), abs=0.05
    )
