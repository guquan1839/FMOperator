# FMO — Feature Interaction Modeling for Neural Operators

Reference implementation for
**[[2607.28762] Feature Interaction Modeling for Neural Operators](https://arxiv.org/abs/2607.28762)**.

> **Status: work in progress.** The paper is still being revised — the experiments
> are being completed and corrected, so the variants, the exact configurations and
> the reported numbers may still change. This repository tracks the code that is
> actually used to produce the results and is published early so that the models
> can be inspected and reused while the paper is finalised. Please treat every
> number as preliminary.

---

## Contents

```
v1/
├── FMO_BSI.py          low-rank bilinear interaction, uniform field-pair weights (baseline)
├── FMO_ABSI.py         FMO_BSI + input-dependent (attentive) field-pair weights
├── FMO_MBSI.py         FMO_BSI with per-head MLP interaction factors
├── FMO_MABSI.py        MLP interaction factors + per-head attentive field-pair weights
└── README.md
```

Every model file is **fully self-contained**: it imports PyTorch and the standard
library only, and it never imports another file of this repository. Importing one
model cannot influence another, and any file can be copied into a training script
as-is.

## Model family

All four models share one skeleton. A history window plus a query `(x, t)` is
split into four *fields*

| field | content |
| --- | --- |
| `history` | the `n_frames × n_sensors` input window, encoded by a convolutional encoder |
| `x` | first query coordinate |
| `t` | second query coordinate |
| `stats` | `n_stats` scalar statistics of the history |

and the prediction is

$$
y = \mathrm{decoder}\Big(\mathrm{pool}\big(\{\phi_h(i,j)\}\big)\Big) + w^\top [\text{history}, x, t, \text{stats}]
$$

The interaction representation of one head $h$ and one unordered field pair
$i<j$ is the **symmetric low-rank bilinear form**

$$
\phi_h(i,j) = L_{h,i}(e_i) \odot R_{h,j}(e_j) + L_{h,j}(e_j) \odot R_{h,i}(e_i) \in \mathbb{R}^{r}
$$

where $e_i$ are the field embeddings, $\odot$ is the Hadamard product,
$L_{h,i}, R_{h,i}:\mathbb{R}^{d}\to\mathbb{R}^{r}$ are learned projections, $H$ is
the number of heads and $r$ the per-head rank. The four variants differ only in
how the projections are parameterised and how the six pairs are aggregated:

| file | interaction factors $L, R$ | field-pair pooling | parameters* |
| --- | --- | --- | --- |
| `FMO_BSI.py` | bias-free linear | uniform sum $\sum_{i<j}\phi_h(i,j)$ | 632,711 |
| `FMO_ABSI.py` | bias-free linear | attention $\alpha_h=\mathrm{softmax}_{i<j}\langle w_h,\phi_h(i,j)\rangle$ | 632,839 |
| `FMO_MBSI.py` | per-head 2-layer MLP (SiLU, identity output) | uniform sum | 1,162,119 |
| `FMO_MABSI.py` | per-head 2-layer MLP (SiLU, identity output) | per-head attention | 1,162,247 |

\* default configuration: `history = 20×64`, `embed_dim = 128`, `heads = 4`,
`rank = 32`, `decoder_hidden = 128`, `decoder_depth = 2`.

Naming: **B**ilinear **S**um **I**nteraction; **A** = input-dependent
(attentive) field-pair weights; **M** = MLP-parameterised interaction factors.
The attentive variants add only `heads × rank` parameters (the gate) on top of
their non-attentive counterpart.

## Design notes

* **Heads are a re-parameterisation, not extra capacity.** With bias-free linear
  factors and a linear read-out, $H$ heads of rank $r$ are *exactly* equivalent
  to a single head of rank $Hr$ — concatenating the per-head projection weights
  reproduces the same function with the same parameter count. The capacity knob
  is therefore the total rank $R = Hr$; prefer to report $R$, or use the
  per-head non-linearity of `FMO_MBSI`/`FMO_MABSI`, which does break the
  equivalence.
* **Attention starts at the baseline.** The gate $w_h$ is zero-initialised, so
  $\alpha_h$ is uniform at step 0 and the attentive models are numerically
  identical to their non-attentive counterparts at initialisation.
* **Identity output activation on the factors.** `FMO_MBSI`/`FMO_MABSI` apply
  the non-linearity in the hidden layer of the factor MLP and leave the factors
  themselves linear (`proj_out_act="identity"`). Squashing the factors right
  before the Hadamard product (`proj_out_act="silu"`) was consistently worse on
  the paper's datasets.
* **`pair_pool="sum"`** is accepted by the attentive files as an ablation key
  and reduces them exactly to their non-attentive counterpart.

## Quick start

```python
import torch
from FMO_ABSI import FMO_ABSI

model = FMO_ABSI(n_frames=20, n_sensors=64, n_stats=3, heads=4, rank=32)

sensors = torch.randn(8, 20, 64)   # history window; (8, 1280) is accepted as well
coords  = torch.rand(8, 2)         # (x, t)
stats   = torch.randn(8, 3)        # scalar history statistics
y = model(sensors, coords, stats)  # -> (8,)

# inspect the field-pair weights (B, heads, pairs), pair order is
# (h,x) (h,t) (h,stats) (x,t) (x,stats) (t,stats)
weights = model.pair_weights(sensors, coords, stats)
```

Every file can also be executed directly for a parameter-count self-check:

```bash
python FMO_MBSI.py
```

## Experiments

Released runs are under `Experiments/`: `KS` at three training-set sizes, and
one run per equation for the others.

Protocol, identical for every run below: 50,000 optimizer steps, seed 38, query
batch of 1024 sampled `(trajectory, x, t)` points, stride 2 (64 of the 128 grid
points), 20 input frames -> 20 output frames, per-trajectory history
normalisation, and the same convolutional history encoder, decoder and wide
linear skip. The three models differ **only** in how the four fields
`(history, x, t, stats)` are fused. Reported errors are always on the held-out
**test** split.

The two columns per model are the two checkpoint-selection rules, both measured
on that same test split: `train` selects the checkpoint with the lowest
training-batch loss, `val` the one with the lowest loss on a fixed validation
minibatch. They are selection rules, not train/validation errors.

* **RMSE** = `sqrt(mean over test points of (prediction - target)^2)`, in the
  physical units of the field.
* **Relative L2** = mean over test trajectories of
  `||prediction - target||_2 / ||target||_2` (dimensionless, hence the one to
  compare across equations).

The **bold** entry in each row is the smallest error of that row (markdown
does not support colour).

### RMSE (absolute, physical units)

| equation (n_train) | FM-BSI train | FM-BSI val | FM-ABSI train | FM-ABSI val | FM-NFM train | FM-NFM val |
|---|---|---|---|---|---|---|
| kuramoto_sivashinsky1d (384) | 0.009922 | **0.009793** | 0.01197 | 0.01251 | 0.1293 | 0.1266 |
| kuramoto_sivashinsky1d (1000) | 0.002844 | 0.00278 | 0.002703 | **0.002689** | 0.1098 | 0.0972 |
| kuramoto_sivashinsky1d (10000) | 0.001526 | 0.001531 | 0.001517 | 0.001558 | **0.001455** | 0.001491 |
| square_advection1d (1000) | 0.0884 | 0.08746 | 0.09022 | **0.08375** | 0.09134 | 0.09111 |
| lwr1d (1000) | 0.03565 | **0.03435** | 0.03455 | 0.03444 | 0.03464 | 0.03472 |
| buckley_leverett1d (1000) | 0.03142 | 0.03146 | **0.03085** | 0.03113 | 0.04021 | 0.03896 |

### Relative L2 error

| equation (n_train) | FM-BSI train | FM-BSI val | FM-ABSI train | FM-ABSI val | FM-NFM train | FM-NFM val |
|---|---|---|---|---|---|---|
| kuramoto_sivashinsky1d (384) | 0.02905 | **0.02873** | 0.03382 | 0.03664 | 0.3434 | 0.3401 |
| kuramoto_sivashinsky1d (1000) | 0.008164 | 0.008239 | **0.007692** | 0.007719 | 0.1912 | 0.201 |
| kuramoto_sivashinsky1d (10000) | 0.004632 | 0.004602 | 0.004597 | 0.004754 | **0.00447** | 0.004575 |
| square_advection1d (1000) | 0.178 | 0.1735 | 0.1822 | **0.1711** | 0.1824 | 0.1811 |
| lwr1d (1000) | 0.05757 | 0.05703 | **0.05448** | 0.05567 | 0.05815 | 0.05597 |
| buckley_leverett1d (1000) | 0.05841 | 0.05888 | **0.05745** | 0.0581 | 0.07228 | 0.07083 |

## Requirements

* Python ≥ 3.9
* PyTorch ≥ 2.0 (CPU is enough; the scripts use CUDA automatically when available)

## Citation

```bibtex
@misc{gu2026featureinteractionmodelingneural,
  title         = {Feature Interaction Modeling for Neural Operators},
  author        = {Quan Gu and Xiaoduo Li and Hongxia Liu},
  year          = {2026},
  eprint        = {2607.28762},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2607.28762}
}
```
