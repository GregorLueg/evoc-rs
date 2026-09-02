# GPU

`EVoCGpu` moves the kNN stage onto the GPU and leaves everything downstream
where it was.

```python
import evoc_rs

if evoc_rs.gpu_available():
    model = evoc_rs.EVoCGpu(n_neighbours=15).fit(X)
else:
    model = evoc_rs.EVoC(n_neighbours=15).fit(X)
```

## Is it available

```python
evoc_rs.gpu_available()
```

Two questions, one answer: was this wheel built with GPU support, and is there
an adapter here. The published wheel is built with it, so a `False` on a normal
machine means no adapter. Backend is wgpu, so Metal, Vulkan and DX12 all work
and there is no CUDA runtime to install.

`EVoCGpu` is only exported when the extension carries it, so
`hasattr(evoc_rs, "EVoCGpu")` is the import-time version of the same question.

## When it pays off

Only the kNN search moves. The fuzzy graph, the embedding, the MST and the
persistence analysis all stay on the CPU, so the ceiling is however much of your
runtime stage one was.

That makes it worth it on many points in high dimension, and especially with
`exhaustive_gpu`, where the GPU does what it is good at. On 5k points in 32
dimensions the setup cost eats the win and the CPU path is as quick. On Apple
Silicon in particular the GPU rarely blows the CPU out of the water; measure
before you commit to it.

## Backends

| Backend | Notes |
| --- | --- |
| `ivf_gpu` | Default. Inverted file over k-means cells. |
| `exhaustive_gpu` | Exact. Where the GPU wins by the largest margin. |
| `nndescent_gpu` | CAGRA-style graph, built and queried on device. |

Knobs are `n_list` and `n_probes` for `ivf_gpu`, and `k`, `k_build`, `n_tree`,
`delta`, `rho`, `beam_width`, `max_beam_iters`, `n_entry_points` and
`extract_knn` for `nndescent_gpu`.

`extract_knn` is on by default and hands back the CAGRA graph the build already
produced rather than beam-searching it, which skips every beam parameter above.
If you want to make sure your graph is as good as possible, use
`extract_knn = False`.

The two CAGRA degrees are worth knowing about. `k` is the graph degree after
pruning and `k_build` the degree before it, and both are independent of the
`n_neighbours` you are querying for. Left at the crate defaults they sit below
`n_neighbours` whenever you ask for more than 30 neighbours, and the beam search
then walks a graph too small to answer the query well. `None` on either
backfills from `n_neighbours`, so leave them alone unless you have a reason.

## float32 only

WGSL has no `f64`, and consumer GPUs cripple its throughput anyway. `EVoCGpu`
casts float64 input down rather than refusing it, which is the one place in this
library that narrows on your behalf.

```python
model = evoc_rs.EVoCGpu().fit(X.astype(np.float64))
model.membership_strengths_.dtype  # float32
```

## CPU-only wheels

Building from source with `--no-default-features` gives a wheel without any of
this. `gpu_available()` returns `False`, `EVoCGpu` is not exported, and
constructing one anyway raises an `ImportError` naming the flag.
