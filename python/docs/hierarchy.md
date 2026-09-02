# The cluster hierarchy

This is the part that differs from HDBSCAN, and the part worth understanding
before you take `labels_` at face value.

## What the layers are

EVoC builds an MST over mutual reachability distances in the embedding, then
condenses it repeatedly at increasing `min_cluster_size`. Each pass gives one
labelling. Together they are `cluster_layers_`, sorted finest first: layer 0 has
the most clusters, the last has the fewest.

```python
for layer, score in zip(model.cluster_layers_, model.persistence_scores_):
    n = layer.max() + 1
    noise = (layer < 0).mean()
    print(f"{n:3d} clusters  persistence {score:.3f}  noise {noise:.1%}")
```

Layers too similar to the one below are dropped before you see them: if the
Jaccard similarity between two consecutive layers is above
`min_similarity_threshold`, the coarser of the two carries no new information.
`max_layers` caps what survives.

## What persistence measures

Persistence is how long a cluster survives as the density threshold sweeps. A
cluster that appears at one threshold and dissolves at the next is an artefact
of where you happened to cut. One that holds across a wide band is structure.

The score on each layer aggregates that over its clusters, so a high-scoring
layer is one whose clusters would still be there had you cut somewhere else.
That is why `labels_` defaults to `argmax(persistence_scores_)`.

It is a heuristic, not an oracle. On data with real structure at two scales, two
layers score similarly and picking between them is a question about your
problem, not about the data.

## Using a layer other than the default

```python
# The finest layer, whatever its persistence.
fine = model.cluster_layers_[0]

# The coarsest.
coarse = model.cluster_layers_[-1]

# The layer closest to a count you have in mind.
counts = model.cluster_layers_.max(axis=1) + 1
chosen = model.cluster_layers_[np.abs(counts - 12).argmin()]
```

If you know the count up front, `approx_n_clusters` is the better tool: it
searches for that count directly rather than picking the nearest layer out of
whatever the hierarchy happened to produce.

## Noise

`-1` is noise, the HDBSCAN convention. Points land there when they sit in a
low-density region the condensed tree never assigned to a cluster. Noise
fraction climbs with `base_min_cluster_size` and with `min_samples`, so a
labelling that is 40% noise usually means one of those is too high for the
data rather than that the data has no structure.

`membership_strengths_` is `0.0` on noise points and in `[0, 1]` elsewhere. Use
it to threshold soft assignments:

```python
confident = model.labels_.copy()
confident[model.membership_strengths_ < 0.5] = -1
```

## Tuning, roughly in order of effect

`n_neighbours` sets the scale everything else sees. Larger means a more
connected graph, smoother embedding, fewer and broader clusters. It is the first
thing to move.

`base_min_cluster_size` sets the floor at the finest layer, and therefore how
many layers there are at all.

`min_samples` controls the density estimate. Higher makes the mutual
reachability distances more conservative and pushes more points into noise.

`noise_level` tunes the repulsion in the embedding. Lower separates harder,
which can manufacture structure that is not there; higher is conservative.
