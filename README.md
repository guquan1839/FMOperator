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
├── FMO_ABSI.py         FMO_BSI + attentive field-pair weights
├── FMO_MBSI.py         FMO_BSI with per-head MLP interaction factors
├── FMO_MABSI.py        MLP interaction factors + attentive field-pair weights
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

## Results

The complete source tables are stored in [`data`](data). The six equation-level
figures below show four clusters per equation: best train RMSE, best val RMSE,
best train relative L2, and best val relative L2. Every cluster contains all
seven models.

### Kuramoto–Sivashinsky 1D

![Kuramoto–Sivashinsky 1D bar chart](figures/kuramoto_sivashinsky1d_bar_chart.png)

### Square Advection 1D

![Square Advection 1D bar chart](figures/square_advection1d_bar_chart.png)

### LWR 1D

![LWR 1D bar chart](figures/lwr1d_bar_chart.png)

### Buckley–Leverett 1D

![Buckley–Leverett 1D bar chart](figures/buckley_leverett1d_bar_chart.png)

### Cubic Conservation 1D

![Cubic Conservation 1D bar chart](figures/cubic_conservation1d_bar_chart.png)

### Burgers 1D

![Burgers 1D bar chart](figures/burgers1d_bar_chart.png)

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
