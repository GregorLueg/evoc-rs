# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# CPU-only build/test (matches CI test-cpu)
cargo test --release --no-default-features

# GPU-enabled build/test (matches CI test-gpu, needs Vulkan/Metal/CUDA)
cargo test --release --all-features -- --no-capture

# Single test
cargo test --release --no-default-features integration_01_mst_two_clusters

# Feature-gated code
cargo build --features gpu
cargo doc --no-deps --all-features --open
```

Linux GPU CI installs Vulkan (`libvulkan1`, `mesa-vulkan-drivers`) and sets `WGPU_BACKEND=vulkan`.

## Architecture

Library crate (`src/lib.rs`) exposing two top-level entry points that share the same 6-stage pipeline:

- `evoc(...)` — CPU pipeline
- `evoc_gpu::<T, R>(...)` — GPU kNN, CPU everything else (only under `gpu` feature)

Both take a `faer::MatRef<T>` of shape `(n_points, n_features)` and return `Result<EvocResult<T>, EvocErrors>`.

**Pipeline stages** (in order, matched between CPU and GPU paths):

1. kNN graph — `nearest_neighbours::{cpu, gpu}` dispatches to backends in `ann-search-rs` (CPU: `nndescent`, `hnsw`, `annoy`, `ivf`, `kmknn`, `balltree`, `exhaustive`; GPU: `exhaustive_gpu`, `ivf_gpu`, `nndescent_gpu`). Can be short-circuited with `precomputed_knn`.
2. Fuzzy simplicial set — `graph::fuzzy_graph` smooths kNN into a weighted symmetric graph (COO), then `coo_to_adjacency_list`.
3. Label propagation init — `graph::label_prop::label_propagation_init` gives the embedding a starting layout.
4. Node embedding — `graph::embedding::evoc_embedding` runs the EVoC gradient (modified UMAP repulsion tuned by `noise_level`) with Adam.
5. MST + linkage — `clustering::mst::build_mst` uses mutual reachability distances via `clustering::kd_tree`; `clustering::linkage::mst_to_linkage_tree` produces the hierarchy.
6. Cluster extraction — either `clustering::persistence::build_cluster_layers` (returns multi-layer hierarchy, ranked by persistence) OR `search_for_n_clusters` (binary-searches `min_cluster_size` when `EvocParams::approx_n_clusters` is set).

### Traits and generics

Everything is generic over `T: EvocFloat` (`src/utils/traits.rs`) which composes `Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + Default + SimdDistance + AddAssign + ComplexField + RealField + 'static`. This is a blanket impl, not something to implement manually — it works for `f32` and `f64` on the CPU path. **GPU path is `f32` only** (WGSL has no `f64`, and consumer GPUs cripple `f64` throughput).

### Feature gate

The `gpu` feature enables `ann-search-rs/gpu` + the `cubecl` dependency. Everything GPU-related is behind `#[cfg(feature = "gpu")]`:

- `src/nearest_neighbours/nearest_neighbour_gpu.rs`
- `evoc_gpu` in `src/lib.rs`
- `NearestNeighbourParamsGpuEvoc` re-export in `src/prelude.rs`
- `tests/integration_tests_gpu.rs` (guarded with `#![cfg(feature = "gpu")]`)

When adding GPU-touching code, mirror this pattern.

### Prelude

`src/prelude.rs` re-exports the trait `EvocFloat`, `EvocErrors`, `EvocEmbeddingParams`, `NearestNeighbourParamsEvoc`, `CoordinateList`, the `Verbosity` enum + `parse_verbosity_level`, and the type aliases `EvocKnnResults<T>` / `PreComputedKnn<T>`. Consumers use `use evoc_rs::prelude::*;` — keep new public types in scope by adding them here.

### Verbosity

Numeric API: `0 = Quiet`, `1 = Normal`, `2 = Detailed`. `parse_verbosity_level` maps to the `Verbosity` enum; both `evoc` and `evoc_gpu` accept `usize` for FFI ergonomics.

## Conventions

- Rust 2024 edition, MSRV whatever `dtolnay/rust-toolchain@stable` resolves to in CI.
- `#![warn(missing_docs)]` at the crate root — public items need doc comments.
- `#![allow(clippy::needless_range_loop)]` is applied at both the crate root and in tests — indexed loops are used deliberately for readability in numerical kernels; keep them when hot.
- Errors flow through `EvocErrors` (`thiserror`), which currently just wraps `ann_search_rs::errors::AnnSearchErrors`.
- Version bumps happen in `Cargo.toml`, README install snippets, and `docs/news.md` together.

## Upstream reference

Port of the Python [evoc](https://github.com/TutteInstitute/evoc) by Leland McInnes. When behaviour diverges from the reference, the Python implementation is the source of truth (see the tie-breaking comment in `search_for_n_clusters` — "matches Python").
