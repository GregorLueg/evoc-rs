# Quickstart

## Fit

```python
import numpy as np
import evoc_rs

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(c * 20, 1, (500, 32)) for c in range(4)]).astype(np.float32)

model = evoc_rs.EVoC(n_neighbours=15).fit(X)
```

Parameters go in the constructor, data goes into `fit`, results come off the
fitted estimator. `fit_predict` is there if you only want the labels.

## What comes back

| Attribute | Shape | What it is |
| --- | --- | --- |
| `labels_` | `(n,)` int64 | The most persistent layer. `-1` is noise. |
| `membership_strengths_` | `(n,)` | Strength in `[0, 1]` for `labels_`. |
| `cluster_layers_` | `(n_layers, n)` int64 | Every layer, finest first. |
| `layer_strengths_` | `(n_layers, n)` | Strengths for every layer. |
| `persistence_scores_` | `(n_layers,)` float64 | Higher is more stable. |
| `neighbour_graph_` | `(distances, indices)` | The kNN graph, self excluded. |
| `n_clusters_` | int | Non-noise clusters in `labels_`. |

Distances come first in `neighbour_graph_`, matching scikit-learn and FAISS.

## Fixing the cluster count

Pass `approx_n_clusters` and the finest layer is binary-searched for it. You get
one layer back rather than the hierarchy.

```python
model = evoc_rs.EVoC(approx_n_clusters=8).fit(X)
model.n_clusters_
```

It is approximate on purpose: the search moves `base_min_cluster_size` until the
extracted count lands near the target, and some data has no setting that hits it
exactly.

## dtype

float32 and float64 both run natively and the output width follows the input.
Anything else is promoted to float64, never narrowed. If memory is tight, hand
it float32 yourself.

```python
model.membership_strengths_.dtype  # float32 in, float32 out
```

## Bring your own graph

Already built a kNN graph? The whole search stage is skipped.

```python
distances, indices = model.neighbour_graph_
reused = evoc_rs.EVoC(n_neighbours=15).fit(X, precomputed_knn=(indices, distances))
```

Both arrays are `(n_samples, k)` and exclude self. This is the fast path when
you are sweeping clustering parameters over fixed data: build the graph once,
refit as often as you like.

## Threads

```python
evoc_rs.set_num_threads(8)  # cap it
evoc_rs.set_num_threads(0)  # back to one per core
evoc_rs.num_threads()
```

Not a global rayon pool, so calling it twice in a notebook is fine.

## Reproducibility

Same `seed` and same `ann_algorithm` gives the same labels. The approximate
backends are seeded too, so the only thing that moves between runs is nothing.

```python
a = evoc_rs.EVoC(seed=7).fit_predict(X)
b = evoc_rs.EVoC(seed=7).fit_predict(X)
assert (a == b).all()
```
