# News

## 0.3.0

**Features**

- NN-Descent can hand back the graph it already built instead of beam-searching
  it, via `extract_knn` on both `NearestNeighbourParamsEvoc` and
  `NearestNeighbourParamsGpuEvoc`. **On by default.** A self-kNN query
  re-searches a graph that is already a kNN graph, which is the work the descent
  just did. The graph degree is widened to cover `k` when it is set, since
  extraction cannot return more neighbours than the graph holds.
- `nndescent_gpu` replaces `ivf_gpu` as the default GPU kNN backend.
- More inputs accepted: `evoc` and `evoc_gpu` take a faer matrix, an ndarray
  2-D array (behind the new `ndarray` feature) or a row-major
  `(&[T], n_samples, n_features)` tuple, via the new `EvocMatrix` trait. Every
  layout bar a non-contiguous ndarray is zero-copy, and the pipeline below the
  entry points stays on `MatRef`.
- `evoc_rs::VERSION`, so a dependent can report which version of the numerics
  it was built against.
- Version update on `ann-search-rs` to 0.8.0, for substantially faster
  approximate nearest neighbour searches.

**Stability**

- `f32` and `f64` now return the same clustering for the same data. They did
  not before: `symmetrise_graph` evaluated the t-conorm as `a + b - a*b`, which
  is algebraically exactly 1 when either input is but rounds to 1 +/- 1 ulp in
  floating point. A point's own nearest neighbour has `dist == rho`, so around
  7% of graph edges carry exactly 1.0, and label propagation thresholds on
  exactly 1.0. Which edges fell one ulp short differed between precisions, so
  the two disagreed on which nodes to label and the clusterings diverged from
  there. Snapping the endpoint takes label agreement from 2 seeds in 10 to 9,
  mean ARI 1.0.

  **This is a deliberate divergence from the Python reference**, which
  evaluates the bare expression and carries the same fragility. Reproducibility
  across precisions was judged worth more than bit-parity, not least because
  the GPU path is `f32`-only and could not otherwise be checked against the CPU
  one. It does not make the clustering more accurate, only consistent.
  `integration_12_precisions_agree` guards the property.

- The persistence curve is computed in `f64` whatever `T` the pipeline runs at.
  `lambda_death` is `exp(-d)` for an MST distance `d`, which in `f32` goes
  subnormal past `d ~ 87` and flushes to zero past `d ~ 103`, where `f64` holds
  to `d ~ 745`. A zero contributes nothing to the curve, so an `f32` run on an
  embedding with a large spatial scale silently lost a whole cluster layer. The
  contributions are summed in `f64` for the same reason: they span many orders
  of magnitude and `find_peaks` compares the sums exactly.

**Python**

- Python bindings under `python/`, built with PyO3 and maturin. A
  scikit-learn shaped `EVoC` estimator over the CPU pipeline, plus `EVoCGpu`
  for the GPU kNN path. Documentation at
  <https://gregorlueg.github.io/evoc-rs/>.

**Breaking changes**

- `evoc` and `evoc_gpu` take `impl EvocMatrix<T>` rather than `MatRef<T>`.
  Existing call sites passing a `MatRef` are unaffected, since `MatRef`
  implements the trait.
- `ClusterBarcode::lambda_death` is `f64` rather than `T`,
  `compute_total_persistence` returns `(Vec<T>, Vec<f64>)`, and
  `select_diverse_peaks` takes `&[f64]` for the curve.
  `EvocResult::persistence_scores` was already `f64` and is unchanged, so this
  only affects callers reaching into the persistence module directly.
- `NearestNeighbourParamsEvoc::new` and `NearestNeighbourParamsGpuEvoc::new`
  take an `extract_knn` argument. Struct-literal and `..Default::default()`
  construction is unaffected.

## 0.2.7

**Feature:**

- Version update on `ann-search-rs` to profit from faster GPU code.

## 0.2.6

**Feature:**

- Version update on `ann-search-rs` to profit from more stable GPU code.

## 0.2.5

**Feature:**

- Version update on `ann-search-rs` to profit from faster GPU-acceleration.

## 0.2.4

**Fix:**

- Version bump to latest `ann-search-rs` that will ensure that the IVF indices
  do return the expected k neighbours and will query more lists.

## 0.2.3

**Features:**

- `CLAUDE.md` added.

**Fix:**

- Propagation of parameters to the GPU-accelerated NNDescent were not behaving.
  Fixed now.

## 0.2.2

**Fix:**

- Version update on `ann-search-rs` to avoid a nasty wgpu <> metal bug that
  affects the CAGRA-style ANN.

## 0.2.1

**Features:**

- Removed the coupling with `manifolds-rs` and sole dependency on
  `ann-search-rs`.

## 0.2.0

**Features:**

- Latest version bumps for packages (includes bug fixes on the GPU version on
  high dimensional data).
- Interface changed with verbose being a usize which gives more fine-grained
  control over the verbosity.

## 0.1.6

**Features:**

- Version bumps to latest `ann-search-rs` + improved error handling.
- Version bumps to latest `manifolds-rs` crate
- Bumping CubeCl to `0.10.0`

## 0.1.5

**Features:**

- Version bumps to latest `ann-search-rs` + improved error handling.

## 0.1.4

**Features:**

- Version bumps
- Updated CI/CD for release

## 0.1.3

**Features:**

- Version bump to latest version of `manifolds-rs` and `ann-search-rs`. In the
  former case, proper error handling is now happening.

## 0.1.2

**Features:**

- Version bump to latest version of `manifolds-rs`, unlocking GPU-accelerated
  kNN searches.
- GPU-accelerated version of EVoC that leverages the GPU-accelerated kNN
  searches under the hood.

## 0.1.1

**Features:**

- Version bump to latest version of `manifolds-rs`, unlocking IVF as an
  appproximate nearest neighbour searches.

## 0.1.0

**Features:**

- Release of the package.
