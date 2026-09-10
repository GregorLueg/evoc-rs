# News

Changes to the `evoc-rs` Python package. The Rust crate it wraps, `evoc-rs`, has
its own changelog at [`../CHANGELOG.md`](../CHANGELOG.md).

## 0.1.1

Requires `evoc-rs` 0.3.1.

- AVX2 and AVX-512 distance kernels come through from the parent crate, so the
  kNN stage on x86_64 wheels is faster with no build flags.

## 0.1.0

Requires `evoc-rs` 0.3.0. First release on PyPI.

- Python bindings under `python/`, built with PyO3 and maturin. A scikit-learn
  shaped `EVoC` estimator over the CPU pipeline, plus `EVoCGpu` for the GPU kNN
  path.
