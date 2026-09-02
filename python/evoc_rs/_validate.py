"""Argument checking at the FFI boundary.

The Rust core assumes what it is given is finite, 2-D and of a float type it
knows. It does not check, and a silently useless embedding is worse than an
exception, so everything is checked here instead.
"""

from __future__ import annotations

import numpy as np
from beartype import beartype

###########
# Globals #
###########

#: Element types the core compiles for. Anything else is promoted to float64.
_NATIVE: tuple[np.dtype, ...] = (np.dtype(np.float32), np.dtype(np.float64))

#: Metrics the kNN backends accept.
METRICS: frozenset[str] = frozenset({"euclidean", "cosine"})

#: CPU ANN backends `run_ann_search` dispatches on.
CPU_BACKENDS: frozenset[str] = frozenset(
    {"nndescent", "hnsw", "annoy", "ivf", "kmknn", "balltree", "exhaustive"}
)

#: GPU ANN backends `run_ann_search_gpu` dispatches on.
GPU_BACKENDS: frozenset[str] = frozenset({"exhaustive_gpu", "ivf_gpu", "nndescent_gpu"})


@beartype
def check_matrix(X: np.ndarray, *, force_dtype: np.dtype | None = None) -> np.ndarray:
    """Coerce a design matrix into something the core can take.

    Args:
        X: Candidate data, samples in rows and features in columns.
        force_dtype: Element type to cast to regardless of what came in. The
            GPU path pins float32; leave as None to keep the caller's width.

    Returns:
        A C-contiguous 2-D array of float32 or float64.

    Raises:
        TypeError: If the element type is not numeric.
        ValueError: If the array is not 2-D, is empty, or holds a non-finite
            value.
    """
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError(f"X must hold numbers, got dtype {X.dtype}")

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got {X.ndim} dimensions")

    if X.size == 0:
        raise ValueError("X is empty")

    # Never narrow on the caller's behalf: an unknown width goes up to float64,
    # not down to float32.
    target = force_dtype if force_dtype is not None else X.dtype
    if target not in _NATIVE:
        target = np.dtype(np.float64)

    arr = X.astype(target, copy=False)

    if not np.isfinite(arr).all():
        raise ValueError("X holds NaN or infinity; the core does not check for it")

    return np.ascontiguousarray(arr)


@beartype
def check_n_neighbours(n_neighbours: int, n_samples: int) -> None:
    """Reject a neighbour count the data cannot supply.

    The kNN stage queries `n_neighbours + 1` and drops self, so the ceiling is
    one below the sample count.

    Args:
        n_neighbours: Requested neighbours per point.
        n_samples: Rows in the design matrix.

    Raises:
        ValueError: If `n_neighbours` is below 1 or at least `n_samples`.
    """
    if n_neighbours < 1:
        raise ValueError(f"n_neighbours must be at least 1, got {n_neighbours}")

    if n_neighbours >= n_samples:
        raise ValueError(
            f"n_neighbours ({n_neighbours}) must be below the sample count "
            f"({n_samples})"
        )


@beartype
def check_choice(value: str, allowed: frozenset[str], name: str) -> str:
    """Check a string argument against the set the core dispatches on.

    The core falls back to a default on an unknown string and says so with a
    `println!`, which is invisible in a notebook. Nothing unvalidated reaches
    it.

    Args:
        value: The candidate.
        allowed: Names the core accepts.
        name: Argument name, for the message.

    Returns:
        The value unchanged.

    Raises:
        ValueError: If the value is not in `allowed`.
    """
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of {options}, got {value!r}")
    return value


@beartype
def check_precomputed_knn(
    knn: tuple[np.ndarray, np.ndarray], n_samples: int
) -> tuple[list[list[int]], list[list[float]]]:
    """Convert a precomputed kNN graph into the shape the core wants.

    Args:
        knn: `(indices, distances)`, both `(n_samples, k)`, self excluded.
        n_samples: Rows the design matrix has, for the shape check.

    Returns:
        `(indices, distances)` as nested lists, ready to cross the FFI.

    Raises:
        ValueError: If the two arrays disagree on shape, are not 2-D, or do not
            have one row per sample.
    """
    indices, distances = knn

    if indices.ndim != 2 or distances.ndim != 2:
        raise ValueError("precomputed_knn arrays must both be 2-D")

    if indices.shape != distances.shape:
        raise ValueError(
            f"precomputed_knn shapes disagree: {indices.shape} vs {distances.shape}"
        )

    if indices.shape[0] != n_samples:
        raise ValueError(
            f"precomputed_knn has {indices.shape[0]} rows, X has {n_samples}"
        )

    if (indices < 0).any():
        raise ValueError("precomputed_knn indices must be non-negative")

    return (
        indices.astype(np.int64, copy=False).tolist(),
        distances.astype(np.float64, copy=False).tolist(),
    )
