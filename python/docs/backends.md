# kNN backends

Stage one builds a k-nearest-neighbour graph, and on wide data it is the
expensive stage. `ann_algorithm` picks how.

| Backend | Notes |
| --- | --- |
| `nndescent` | Default. Fastest route to a full self-kNN graph. |
| `exhaustive` | Exact. Blocked GEMM, not a naive scan. Ground truth. |
| `hnsw` | Hierarchical small-world graph. The usual first choice elsewhere. |
| `annoy` | Random projection forest. |
| `ivf` | Inverted file over k-means cells. |
| `kmknn` | Exact, k-means pruned. |
| `balltree` | Metric tree of nested hyperspheres. |

All seven come from [`ann-search-rs`](https://github.com/GregorLueg/ann-search-rs),
where they are documented properly and benchmarked. An `mlx` build on Apple
Silicon adds three more, see [MLX on Apple Silicon](#mlx-on-apple-silicon).

## Which to use

`nndescent` for anything above roughly 10k points. EVoC wants a self-kNN graph
for every point, which is exactly what NN-Descent is built for; the graph-based
indices have to be constructed and then queried n times, and that second pass is
wasted work here.

`exhaustive` below 10k. It is exact, and at that size the approximation buys you
very little wall clock. Also what you want when you are comparing runs and need
the graph out of the equation.

Everything else is there for a reason you already have. If you do not have one,
the two above cover it.

## Per-backend knobs

The constructor carries all of them, prefixed by nothing, so they sit flat
alongside the clustering parameters. Only the ones matching your
`ann_algorithm` are read.

| Backend | Knobs |
| --- | --- |
| `annoy` | `n_tree`, `search_budget` |
| `hnsw` | `m`, `ef_construction`, `ef_search` |
| `nndescent` | `diversify_prob`, `delta`, `ef_budget`, `extract_knn` |
| `balltree` | `bt_budget` |
| `ivf`, `ivf_mlx` | `n_list`, `n_probes` |
| `nndescent_mlx` | `delta`, `extract_knn` |

### The NN-Descent fast track

`extract_knn` is on by default and worth leaving on. NN-Descent's whole job is
to build a kNN graph, so querying it for a self-kNN graph afterwards re-does
work that has already been done. Extraction hands back the graph directly and
skips the beam search, along with every beam parameter (`ef_budget` on the CPU,
`beam_width` and friends on the GPU).

The graph degree is widened to cover `k` when it is set, since extraction
cannot return more neighbours than the graph holds. `None` on any optional knob
means the crate picks. `n_list` defaults to `sqrt(n)`, `n_probes` to
`sqrt(n_list)`, and the NN-Descent query budget is derived from `n_neighbours`.

## MLX on Apple Silicon

Experimental. A wheel built with the `mlx` feature on a Mac with mlx-c
installed takes three more `ann_algorithm` values on the plain `EVoC`:

| Backend | Notes |
| --- | --- |
| `exhaustive_mlx` | Exact, on the Apple GPU. |
| `ivf_mlx` | Inverted file over k-means cells, on the Apple GPU. |
| `nndescent_mlx` | NN-Descent on the Apple GPU. With `extract_knn` it hands back the raw graph; without, it builds a CAGRA index and beam-searches it. |

```bash
brew install mlx-c
maturin develop --release --features mlx
```

```python
evoc_rs.mlx_available()  # True only in an mlx build that found mlx-c
model = evoc_rs.EVoC(ann_algorithm="nndescent_mlx").fit(X)
```

The search runs in float32 whatever you pass in, so float64 input loses
precision in the kNN stage only; the rest of the pipeline keeps your dtype. No
published wheel carries this, since it links mlx-c dynamically and would only
import on a Mac that has it. In a build without it, an `*_mlx` name raises with
a pointer to the build flag rather than the generic "unknown backend" message.
How it stacks up against `EVoCGpu` on the same Mac is not benchmarked yet.

## Metric

`euclidean` or `cosine`. Nothing else, and an unknown string raises rather than
silently falling back.

Cosine is usually right for embeddings out of a transformer, since the vectors
carry meaning in direction rather than magnitude. Euclidean is right for
anything already scaled.

## Skipping the stage

Any graph you can produce as `(indices, distances)` is accepted, so you are not
limited to these seven:

```python
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=16).fit(X)
distances, indices = nn.kneighbors(X)

# Drop the self column that scikit-learn includes.
model = evoc_rs.EVoC(n_neighbours=15).fit(
    X, precomputed_knn=(indices[:, 1:], distances[:, 1:])
)
```

Both arrays are `(n_samples, k)` and must exclude self.
