[![CI](https://github.com/GregorLueg/evoc-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/evoc-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/evoc-rs.svg)](https://crates.io/crates/evoc-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# evoc-rs

Rust implementation of **EVoC** (Embedding Vector Oriented Clustering): efficient
density-based clustering of high-dimensional embedding vectors (CLIP, sentence
transformers, and the like).

EVoC combines a UMAP-like node embedding with HDBSCAN-style density estimation
and multi-layer persistence analysis to produce stable, hierarchical clusterings
without requiring you to specify the number of clusters up front.

This crate is a port of the original Python implementation by Leland McInnes at
the [Tutte Institute](https://github.com/TutteInstitute/evoc), extended with
support for multiple approximate nearest neighbour backends via
[ann-search-rs](https://crates.io/crates/ann-search-rs), including optional
GPU-accelerated kNN via [cubecl](https://crates.io/crates/cubecl).

## How it works

1. **kNN graph** — approximate nearest-neighbour search via a selectable ANN
   backend (NNDescent, HNSW, or a precomputed graph). Optionally GPU-accelerated.
2. **Fuzzy simplicial set** — the kNN graph is smoothed and symmetrised into a
   weighted undirected graph.
3. **Node embedding** — a low-dimensional layout is optimised using the EVoC
   gradient, a modified UMAP repulsion term controlled by a `noise_level`
   parameter.
4. **MST** — a minimum spanning tree is built over the embedding using mutual
   reachability distances.
5. **Cluster layers** — a hierarchy of clusterings is extracted via persistence
   analysis, returning the most stable granularities. Alternatively, a target
   number of clusters can be specified and the algorithm will binary-search for
   the closest match.

## Installation

Just add the dependency to your `cargo.toml` file:

```toml
[dependencies]
evoc-rs = "*"
```

To enable GPU-accelerated kNN search:

```toml
[dependencies]
evoc-rs = { version = "*", features = ["gpu"] }
```

## Usage

Here is how you would use it.

```rust
use evoc_rs::{evoc, EvocParams};
use manifolds_rs::data::nearest_neighbours::NearestNeighbourParams;

// data is a faer MatRef<f32> with shape (n_points, n_features)
let params = EvocParams::default();
let nn_params = NearestNeighbourParams::default();
let result = evoc(
    data.as_ref(),
    "nndescent".to_string(),
    None,          // precomputed kNN (None = build from data)
    &params,
    &nn_params,
    42,            // seed
    true,          // verbose
);

// Best clustering by persistence score
let labels = result.best_labels();
println!("Found {} clusters.", result.n_clusters());

// Full layer hierarchy, finest first
for (i, layer) in result.cluster_layers.iter().enumerate() {
    println!("Layer {}: {} labels", i, layer.len());
}
```

If you already know roughly how many clusters you want:

```rust
let params = EvocParams {
    approx_n_clusters: Some(10),
    ..EvocParams::default()
};
```

### GPU-accelerated kNN (requires `gpu` feature)

For larger datasets, the nearest neighbour search is typically the bottleneck.
With the `gpu` feature, kNN runs on the GPU via `cubecl` (backends: Vulkan,
Metal, DX12 through wgpu, or CUDA). Graph construction, embedding, MST and
persistence analysis remain on the CPU.

```rust
use evoc_rs::{evoc_gpu, EvocParams};
use manifolds_rs::data::nearest_neighbours_gpu::NearestNeighbourParamsGpu;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

// data is a faer MatRef<f32> (f32 only — see note below)
let params = EvocParams::<f32>::default();
let nn_params = NearestNeighbourParamsGpu::<f32>::default();
let device = WgpuDevice::default();

let result = evoc_gpu::<f32, WgpuRuntime>(
    data.as_ref(),
    "ivf_gpu".to_string(),   // "exhaustive_gpu", "ivf_gpu", or "nndescent_gpu"
    None,
    &params,
    &nn_params,
    device,
    42,
    true,
);

let labels = result.best_labels();
```

A note on precision: GPU computation runs in `f32`. WGSL (the wgpu shader
language) has no `f64`, and `f64` throughput on consumer GPUs is typically
1/32 to 1/64 of `f32` regardless. If you need double precision, stick to the
CPU path. GPU results are also not bit-reproducible across runs due to
non-deterministic parallel reduction order; structural quality is consistent.

To use CUDA instead of wgpu, swap `WgpuRuntime`/`WgpuDevice` for
`CudaRuntime`/`CudaDevice` from `cubecl::cuda`.

## Credits

Based on the original EVoC algorithm and Python implementation by
[Leland McInnes](https://github.com/lmcinnes) at the
[Tutte Institute for Mathematics and Computing](https://github.com/TutteInstitute/evoc).

## Licence

This project is licensed under the MIT Licence for original contributions –
see [LICENSE](LICENSE) for details.

Portions of this work are derived from the
[EVoC](https://github.com/TutteInstitute/evoc) project by the Tutte Institute
for Mathematics and Computing, licensed under the BSD 2-Clause Licence; see
[LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY) for details.
