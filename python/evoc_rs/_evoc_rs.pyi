"""Type stub for the compiled extension.

`ty` reads this instead of importing the module, so the lint job never needs a
maturin build. The typed user-facing surface is `estimator.py`; the arguments
here are left loose on purpose, since they are a flat mirror of two Rust
parameter structs and repeating them buys nothing.
"""

import numpy as np

__version__: str
__core_version__: str

class EvocError(Exception): ...

def gpu_available() -> bool: ...
def num_threads() -> int: ...
def set_num_threads(n: int, /) -> None: ...
def run_evoc(
    x: np.ndarray,
    /,
    *,
    precomputed_knn: tuple[list[list[int]], list[list[float]]] | None = ...,
    ann_type: str,
    **kwargs: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...

# Present only in a build with the `gpu` feature.
def run_evoc_gpu(
    x: np.ndarray,
    /,
    *,
    precomputed_knn: tuple[list[list[int]], list[list[float]]] | None = ...,
    ann_type: str,
    **kwargs: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
