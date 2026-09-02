# evoc-rs

Python bindings for [`evoc-rs`](https://github.com/GregorLueg/evoc-rs): EVoC
clustering for high-dimensional embeddings. The Rust crate does the work. This
is a thin scikit-learn shaped layer over it.

**Documentation: <https://gregorlueg.github.io/evoc-rs/>**

## Install

```bash
uv pip install evoc-rs
```

## Use

```python
import numpy as np
import evoc_rs

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(c * 20, 1, (500, 32)) for c in range(4)]).astype(np.float32)

model = evoc_rs.EVoC(n_neighbours=15).fit(X)

model.labels_  # the most persistent layer, -1 is noise
model.cluster_layers_  # (n_layers, n_samples), finest first
model.persistence_scores_  # (n_layers,), higher is more stable
```

Parameters go in the constructor, data goes into `fit`, results come off the
fitted estimator. `get_params`, `set_params`, `fit` and `fit_predict` are all
there, so it drops into scikit-learn pipelines and `GridSearchCV` without
scikit-learn being an install requirement.

## The hierarchy is the point

Most clustering libraries hand you one labelling. EVoC hands you several, one
per granularity, ranked by how stable each is. `labels_` picks the most
persistent for you, but the interesting bit is often a layer or two either side:

```python
for layer, score in zip(model.cluster_layers_, model.persistence_scores_):
    print(f"{layer.max() + 1:3d} clusters, persistence {score:.3f}")
```

Already know how many clusters you want? Pass `approx_n_clusters` and the finest
layer is binary-searched for it, returning a single layer.

```python
model = evoc_rs.EVoC(approx_n_clusters=8).fit(X)
```

## How it works

Six stages, all in Rust:

1. Approximate kNN graph, via [`ann-search-rs`](https://github.com/GregorLueg/ann-search-rs).
2. Fuzzy simplicial set over that graph.
3. Label-propagation initialisation.
4. UMAP-like node embedding with a repulsion term tuned by `noise_level`.
5. MST over mutual reachability distances in the embedding.
6. Cluster layers pulled out of the tree by persistence.

Clustering the embedding rather than the original space is what makes it fast
and what makes the hierarchy meaningful.

## kNN backends

`ann_algorithm` picks the backend for stage 1. `nndescent` is the default and
the right first choice for a self-kNN graph. `exhaustive` is exact, and worth it
below roughly 10k points. `hnsw`, `annoy`, `ivf`, `kmknn` and `balltree` are all
there if you have a reason.

Bring your own graph instead, if you already have one:

```python
model = evoc_rs.EVoC().fit(X, precomputed_knn=(indices, distances))
```

Both arrays are `(n_samples, k)` and exclude self.

## GPU

`EVoCGpu` moves the kNN stage onto the GPU via wgpu and leaves everything
downstream on the CPU. That only pays off when the search dominates: many
points, high dimension, or an exhaustive search. float32 only, because WGSL has
no f64.

```python
if evoc_rs.gpu_available():
    model = evoc_rs.EVoCGpu(n_neighbours=15).fit(X)
```

`gpu_available()` answers both "was this wheel built with GPU support" and "is
there an adapter here" at once. The published wheel carries it.

## Threads

```python
evoc_rs.set_num_threads(8)  # cap it
evoc_rs.set_num_threads(0)  # back to one per core
```

## Credit

Port of [evoc](https://github.com/TutteInstitute/evoc) by Leland McInnes. Where
behaviour diverges, the Python original is the source of truth.
