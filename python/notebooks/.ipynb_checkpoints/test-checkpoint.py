# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import evoc_rs

# %%
rng = np.random.default_rng(0)
X = np.vstack([rng.normal(c * 20, 1, (500, 32)) for c in range(4)]).astype(np.float32)

# %%
evoc_rs.mlx_available()

# %%
model = evoc_rs.EVoC(n_neighbours=15, ann_algorithm="nndescent_mlx",verbose=2).fit(X)

# %%
