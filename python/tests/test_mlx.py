"""The MLX kNN backends.

Only an `mlx` build on Apple Silicon has them, so the smoke tests skip
everywhere else. What every build can check is that an `*_mlx` name is refused
with a pointer to the build flag rather than the generic allowlist message.
"""

import numpy as np
import pytest
from conftest import agreement

import evoc_rs

needs_mlx = pytest.mark.skipif(not evoc_rs.mlx_available(), reason="not an mlx build")


@pytest.mark.skipif(evoc_rs.mlx_available(), reason="mlx build accepts the names")
def test_mlx_names_point_at_the_build_flag(blobs):
    X, _ = blobs
    with pytest.raises(ValueError, match="--features mlx"):
        evoc_rs.EVoC(ann_algorithm="nndescent_mlx").fit(X)


@needs_mlx
@pytest.mark.parametrize("backend", ["exhaustive_mlx", "ivf_mlx", "nndescent_mlx"])
def test_every_mlx_backend_recovers_clusters(blobs, backend):
    X, y = blobs
    model = evoc_rs.EVoC(n_neighbours=15, ann_algorithm=backend, seed=42).fit(X)

    assert agreement(model.labels_, y) > 0.95
    _, knn_idx = model.neighbour_graph_
    assert not (knn_idx == np.arange(X.shape[0])[:, None]).any()
