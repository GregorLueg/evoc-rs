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
| `nndescent_gpu` | NN-Descent on device, handing back the kNN graph directly. |

Knobs are `n_list` and `n_probes` for `ivf_gpu`, and `build_k`, `max_iters`,
`n_tree`, `delta`, `rho` and `refine_knn` for `nndescent_gpu`.

`nndescent_gpu` builds the graph with `n_neighbours` per point and returns it,
with no CAGRA pruning and no beam search on top. `build_k` is the working degree
during the descent and defaults to `1.5 * n_neighbours`. Datasets too big for a
single device buffer are built in balanced batches automatically.

## float32 only

WGSL has no `f64`, and consumer GPUs cripple its throughput anyway. `EVoCGpu`
casts float64 input down rather than refusing it, which is the one place in this
library that narrows on your behalf.

```python
model = evoc_rs.EVoCGpu().fit(X.astype(np.float64))
model.membership_strengths_.dtype  # float32
```

## MLX instead

On Apple Silicon there is a second route to the GPU: the experimental MLX
backends, which ride on the plain `EVoC` rather than `EVoCGpu`. See
[MLX on Apple Silicon](backends.md#mlx-on-apple-silicon).

## CPU-only wheels

Building from source with `--no-default-features` gives a wheel without any of
this. `gpu_available()` returns `False`, `EVoCGpu` is not exported, and
constructing one anyway raises an `ImportError` naming the flag.
