"""EVoC clustering for high-dimensional embeddings, in Rust.

EVoC clusters embedding vectors (CLIP, sentence transformers, single-cell
latent spaces) by embedding the kNN graph first and running density-based
clustering on that embedding, rather than on the original space. The result is
a *hierarchy* of clusterings ranked by persistence, not one labelling.

    >>> import numpy as np, evoc_rs
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal(c * 20, 1, (300, 32)) for c in range(4)])
    >>> model = evoc_rs.EVoC(n_neighbours=15).fit(X.astype(np.float32))
    >>> model.labels_            # the most persistent layer
    >>> model.cluster_layers_    # every layer, finest first
    >>> model.persistence_scores_

Already know how many clusters you want? Pass `approx_n_clusters` and the
finest layer is binary-searched for it, returning a single layer.

This is a port of TutteInstitute's `evoc`. Where behaviour diverges, the Python
original is the source of truth.
"""

from . import _evoc_rs
from ._base import BaseEstimator, NotFittedError
from ._evoc_rs import (
    EvocError,
    __core_version__,
    __version__,
    gpu_available,
    num_threads,
    set_num_threads,
)
from .estimator import EVoC

__all__ = [
    "BaseEstimator",
    "EVoC",
    "EvocError",
    "NotFittedError",
    "__core_version__",
    "__version__",
    "gpu_available",
    "num_threads",
    "set_num_threads",
]

# The GPU estimator exists only when the extension was built with it, which is
# fixed at wheel-build time. Re-exported when present so `evoc_rs.EVoCGpu`
# works; `evoc_rs.estimator` stays importable either way for anyone who wants
# the ImportError to say why.
if hasattr(_evoc_rs, "run_evoc_gpu"):  # pragma: no cover - build-dependent
    from .estimator import EVoCGpu

    __all__ += ["EVoCGpu"]
