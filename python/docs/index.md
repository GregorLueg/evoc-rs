# evoc-rs

EVoC clustering for high-dimensional embeddings. The
[Rust crate](https://github.com/GregorLueg/evoc-rs) does the work; this is a
thin scikit-learn shaped layer over it.

```bash
uv pip install evoc-rs
```

```python
import numpy as np
import evoc_rs

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(c * 20, 1, (500, 32)) for c in range(4)]).astype(np.float32)

model = evoc_rs.EVoC(n_neighbours=15).fit(X)
model.labels_
```

## What EVoC does differently

Most density-based clustering runs on your data as it stands. EVoC embeds the
kNN graph first, with a UMAP-like optimiser, then builds an MST over *that* and
pulls clusters out by persistence. Two consequences follow.

It is fast on wide data. The expensive stages see a 4-to-16 dimensional
embedding, not your 768-dimensional one, and the only stage that touches the
original space is the kNN search.

And you get a hierarchy, not a labelling. `cluster_layers_` holds one labelling
per granularity, `persistence_scores_` says how stable each is, and `labels_`
picks the most stable for you. That last part is a convenience, not the answer:
the layers either side are often the interesting ones. See
[the cluster hierarchy](hierarchy.md).

## Where to go next

[Quickstart](quickstart.md) is the five-minute version.
[kNN backends](backends.md) covers the seven graph builders and when the default
is wrong. [GPU](gpu.md) covers `EVoCGpu` and when it pays off.
[Changelog](https://github.com/GregorLueg/evoc-rs/blob/main/python/CHANGELOG.md)
covers what changed in this package. It versions separately from the Rust crate,
which keeps its own.

## Credit

Port of [evoc](https://github.com/TutteInstitute/evoc) by Leland McInnes. Where
behaviour diverges, the Python original is the source of truth.
