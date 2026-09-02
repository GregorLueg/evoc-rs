"""Shared fixtures.

One well-separated blob set, built once per session. Every clustering test runs
against it, so the recovery assertions all mean the same thing.
"""

import numpy as np
import pytest

###########
# Globals #
###########

#: Planted clusters in the fixture.
N_CLUSTERS = 4
#: Points per planted cluster.
N_PER_CLUSTER = 250
#: Feature count. Above the SIMD width, and typical of an embedding.
N_FEATURES = 32


@pytest.fixture(scope="session")
def blobs() -> tuple[np.ndarray, np.ndarray]:
    """Four well-separated gaussian blobs.

    Separation is 20 sigma, so any clustering that fails to recover these is
    broken rather than unlucky.

    Returns:
        `(X, y)` with `X` float32 `(n, 32)` and `y` the planted labels.
    """
    rng = np.random.default_rng(0)
    parts = []
    labels = []
    for c in range(N_CLUSTERS):
        centre = np.zeros(N_FEATURES)
        centre[c % N_FEATURES] = c * 20.0
        parts.append(rng.normal(centre, 1.0, (N_PER_CLUSTER, N_FEATURES)))
        labels.append(np.full(N_PER_CLUSTER, c))

    X = np.vstack(parts).astype(np.float32)
    y = np.concatenate(labels)
    return X, y


def agreement(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of points whose label agrees with the truth, up to permutation.

    Each predicted cluster is mapped to its majority true label; noise counts
    against the score.

    Args:
        predicted: Predicted labels, -1 for noise.
        truth: Planted labels.

    Returns:
        A value in [0, 1].
    """
    correct = 0
    for label in np.unique(predicted):
        if label < 0:
            continue
        members = truth[predicted == label]
        counts = np.bincount(members)
        correct += int(counts.max())
    return correct / len(truth)
