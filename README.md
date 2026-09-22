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

Released runs are under `Experiments/<protocol>/<equation>/<n_train>/`, where the
top level is the sampling protocol:

* `seed38/` — the runs reported in the tables below for
  `kuramoto_sivashinsky1d (1000)` and all five equations. The process hash seed
  is pinned (`PYTHONHASHSEED=38`), so every model of a comparison draws its
  query batches from the **same** sampling stream (strictly paired runs). These
  folders also carry the four baseline models used in the paper
  (`DeepONet`, `POD_DeepONet`, `Shift_DeepONet`, `NOMAD`, `result.json` only for
  the equations).
* `randomseed/` — the earlier releases, produced without pinning the
  per-process sampling stream. `KS` at 384 and 10000 training trajectories only
  have this variant; they are kept as an archive and are not part of any
  comparison below (all comparisons use `seed38/`).

Protocol, identical for every run: 50,000 optimizer steps, seed 38, query
batch of 1024 sampled `(trajectory, x, t)` points, stride 2 (64 of the 128 grid
points), 20 input frames -> 20 output frames, per-trajectory history
normalisation, and the same convolutional history encoder, decoder and wide
linear skip. The three FMO models differ **only** in how the four fields
`(history, x, t, stats)` are fused; the four baselines (`DeepONet`,
`POD-DeepONet`, `Shift-DeepONet`, `NOMAD`) are trained on the same data with the
same protocol and a latent dimension of 100. Reported errors are always on the
held-out **test** split.

Each model column reports `train / val`, the two checkpoint-selection rules,
both measured on that same test split: `train` selects the checkpoint with the
lowest training-batch loss, `val` the one with the lowest loss on a fixed
validation minibatch. They are selection rules, not train/validation errors.

* **RMSE** = `sqrt(mean over test points of (prediction - target)^2)`, in the
  physical units of the field.
* **Relative L2** = mean over test trajectories of
  `||prediction - target||_2 / ||target||_2` (dimensionless, hence the one to
  compare across equations).

The **bold** entry is the smallest error of each row (markdown does not support
colour).

### RMSE (absolute, physical units)

| equation (n_train) | FM-BSI | FM-ABSI | FM-NFM | DeepONet | POD-DeepONet | Shift-DeepONet | NOMAD |
|---|---|---|---|---|---|---|---|
| kuramoto_sivashinsky1d (1000) | 0.002454 / **0.002411** | 0.002619 / 0.002660 | 0.02614 / 0.02604 | 0.03484 / 0.02831 | 0.05948 / 0.05200 | 0.1020 / 0.07381 | 0.1161 / 0.1171 |
| square_advection1d (1000) | 0.09054 / **0.08046** | 0.08794 / 0.08249 | 0.09475 / 0.08815 | 0.09328 / 0.09192 | 0.08731 / 0.08583 | 0.09552 / 0.09588 | 0.09151 / 0.09478 |
| lwr1d (1000) | 0.03681 / 0.03606 | 0.03518 / 0.03458 | 0.03685 / 0.03505 | 0.03891 / 0.03782 | 0.04055 / 0.04383 | 0.03141 / 0.03324 | **0.02974** / 0.02979 |
| buckley_leverett1d (1000) | 0.03118 / 0.03118 | 0.03106 / **0.03033** | 0.03712 / 0.03677 | 0.04843 / 0.04810 | 0.05527 / 0.05081 | 0.05677 / 0.05690 | 0.05598 / 0.05499 |
| cubic_conservation1d (1000) | 0.01419 / 0.01419 | **0.01391** / 0.01404 | 0.07949 / 0.07648 | 0.04858 / 0.05014 | 0.06951 / 0.06024 | 0.06753 / 0.06835 | 0.09279 / 0.09421 |
| burgers1d (1000) | 0.02615 / 0.02745 | **0.02558** / 0.03248 | 0.03443 / 0.03520 | 0.07588 / 0.07815 | 0.09000 / 0.08514 | 0.08954 / 0.08944 | 0.07907 / 0.08198 |

### Relative L2 error

| equation (n_train) | FM-BSI | FM-ABSI | FM-NFM | DeepONet | POD-DeepONet | Shift-DeepONet | NOMAD |
|---|---|---|---|---|---|---|---|
| kuramoto_sivashinsky1d (1000) | 0.007304 / **0.007114** | 0.007560 / 0.007781 | 0.06397 / 0.06397 | 0.1078 / 0.08660 | 0.1815 / 0.1586 | 0.2190 / 0.1484 | 0.3237 / 0.3270 |
| square_advection1d (1000) | 0.1833 / **0.1630** | 0.1727 / 0.1670 | 0.1882 / 0.1717 | 0.1962 / 0.1967 | 0.1787 / 0.1765 | 0.1962 / 0.1962 | 0.1812 / 0.1883 |
| lwr1d (1000) | 0.05931 / 0.05769 | 0.05548 / 0.05575 | 0.06099 / 0.05764 | 0.07109 / 0.06956 | 0.07271 / 0.07796 | 0.05183 / 0.05375 | **0.04929** / 0.04933 |
| buckley_leverett1d (1000) | 0.05807 / 0.05807 | 0.05791 / **0.05671** | 0.06662 / 0.06630 | 0.09135 / 0.09100 | 0.1044 / 0.09627 | 0.1055 / 0.1041 | 0.1027 / 0.1009 |
| cubic_conservation1d (1000) | 0.03833 / 0.03822 | **0.03806** / 0.03832 | 0.1740 / 0.1729 | 0.1356 / 0.1387 | 0.1966 / 0.1690 | 0.1821 / 0.1838 | 0.2596 / 0.2617 |
| burgers1d (1000) | 0.09704 / 0.1002 | **0.09342** / 0.1185 | 0.1288 / 0.1312 | 0.2905 / 0.2982 | 0.3457 / 0.3248 | 0.3214 / 0.3276 | 0.2961 / 0.3052 |

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
