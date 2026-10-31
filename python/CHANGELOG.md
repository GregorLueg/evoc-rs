# News

Changes to the `evoc-rs` Python package. The Rust crate it wraps, `evoc-rs`, has
its own changelog at [`../CHANGELOG.md`](../CHANGELOG.md).

## 0.3.0

Requires `evoc-rs` 0.6.0.

**Breaking changes**

- `EVoC` drops `diversify_prob`, `ef_budget` and `extract_knn`. NN-Descent now
  hands back the kNN graph it builds, with no beam search.
- `EVoCGpu` drops `k`, `beam_width`, `max_beam_iters`, `n_entry_points` and
  `extract_knn`. `k_build` is now `build_k` and defaults to
  `1.5 * n_neighbours`. New `max_iters` and `refine_knn`.

**Features**

- Pull in `evoc-rs` 0.6.0 with `ann-search-rs` 0.12.
- `verbose` reaches the kNN stage: `1` prints build and query progress, `2`
  adds the timed build stages.

## 0.2.0

Requires `evoc-rs` 0.5.0.

**Features**

- Pull in `evoc-rs` 0.5.0 with `cubecl` 0.11 and `ann-search-rs` 0.11.
- New `mlx` build feature, off by default and not in the published wheel. On
  Apple Silicon with mlx-c installed, `maturin develop --release --features
  mlx` adds `exhaustive_mlx`, `ivf_mlx` and `nndescent_mlx` as
  `ann_algorithm` options on `EVoC`.
- New `mlx_available()`, fixed at build time.

## 0.1.6

Requires the `evoc-rs` 0.4.4.

**Features**

- Pull in the changes from `evoc-rs` with improved k-means clustering.

## 0.1.5

Requires the `evoc-rs` 0.4.3.

**Features**

- Pull in the changes from `evoc-rs` with the faster GPU-accelerated kNN
  searches.

## 0.1.4

Requires `evoc-rs` 0.4.2.

- Take in latest change from `ann-search-rs` version 0.9.3 with faster Annoy,
  BallTree and NNDescent.

## 0.1.3

Requires `evoc-rs` 0.4.1.

- Take in latest change from `ann-search-rs` version 0.9.1 with better k-means
  and Accelerate framework enabled for Mac users.


## 0.1.2

Requires `evoc-rs` 0.4.0.

- Take in latest change from `ann-search-rs` version 0.9.0.

## 0.1.1

Requires `evoc-rs` 0.3.1.

- AVX2 and AVX-512 distance kernels come through from the parent crate, so the
  kNN stage on x86_64 wheels is faster with no build flags.

## 0.1.0

Requires `evoc-rs` 0.3.0. First release on PyPI.

- Python bindings under `python/`, built with PyO3 and maturin. A scikit-learn
  shaped `EVoC` estimator over the CPU pipeline, plus `EVoCGpu` for the GPU kNN
  path.
