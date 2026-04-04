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
[ann-search-rs](https://crates.io/crates/ann-search-rs).

## How it works

1. **kNN graph** — approximate nearest-neighbour search via a selectable ANN
   backend (NNDescent, HNSW, or a precomputed graph).
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
