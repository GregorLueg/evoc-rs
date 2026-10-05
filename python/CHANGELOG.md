# News

Changes to the `evoc-rs` Python package. The Rust crate it wraps, `evoc-rs`, has
its own changelog at [`../CHANGELOG.md`](../CHANGELOG.md).

## 0.1.5

Requires the `evoc-rs` 0.4.2.

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
