"""FMO-BSI: Bilinear Sum Interaction.

Self-contained reference implementation of the FMO baseline used in
"Feature Interaction Modeling for Neural Operators".

Paper: https://arxiv.org/abs/2607.28762

This file is deliberately standalone: it imports PyTorch (and the standard
library) only, it never imports any other file of this repository, and it can
therefore be vendored into an arbitrary training script as-is.

Fields and notation
-------------------
The model consumes four fields.

    0  history : the (n_frames x n_sensors) input window
    1  x       : the first query coordinate
    2  t       : the second query coordinate
    3  stats   : n_stats scalar statistics of the history

Every field is embedded into R^d (d = ``embed_dim``): the history by a
convolutional encoder, the remaining fields by per-field linear maps.  For an
unordered field pair (i, j) and a head h in {1, ..., H}, the interaction
representation is the symmetric low-rank bilinear form

    phi_h(i, j) = L_hi(e_i) * R_hj(e_j) + R_hi(e_i) * L_hj(e_j)   in R^r

where ``*`` is the Hadamard product, L_hi, R_hi : R^d -> R^r are bias-free
learned projections, H is the number of heads and r is the per-head rank.
BSI aggregates the six field pairs with *uniform* weights,

    S_h = sum_{i<j} phi_h(i, j)   in R^r,

concatenates the heads and reads them out with

    pooled = SiLU(W_out [S_1 ; ... ; S_H]),   W_out : R^{H*r} -> R^d,

after which an MLP decoder predicts the target and a wide linear skip over the
raw concatenated fields is added:

    y = decoder(pooled) + w^T [history ; x ; t ; stats].

Design note
-----------
With bias-free linear projections and a linear read-out, H heads of rank r are
*exactly* equivalent to a single head of rank R = H * r: concatenating the
per-head projection weights along the output dimension reproduces the same
function with the same parameter count.  The head split is therefore a
re-parameterisation and the capacity knob is the total rank R = H * r.  Use
FMO_MBSI/FMO_MABSI for per-head parameterisations that are *not* equivalent to
a single larger head.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Tuple

import torch
from torch import nn


__all__ = ["HistoryEncoder", "FMO_BSI", "FIELD_NAMES", "FIELD_PAIRS"]

FIELD_NAMES: Tuple[str, ...] = ("history", "x", "t", "stats")
FIELD_PAIRS: Tuple[Tuple[int, int], ...] = tuple(combinations(range(len(FIELD_NAMES)), 2))


def make_mlp(in_dim: int, out_dim: int, hidden: int, depth: int) -> nn.Sequential:
    """``depth`` SiLU hidden layers followed by a linear output layer."""
    layers: List[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
    for _ in range(int(depth) - 1):
        layers += [nn.Linear(hidden, hidden), nn.SiLU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class HistoryEncoder(nn.Module):
    """Encode a (n_frames x n_sensors) history window into one embedding.

    ``mode="cnn"`` (default) treats the window as a 2D image and encodes it
    with three stride-2 convolutions plus a linear head, which preserves
    spatial structure (e.g. the location of a shock).  ``mode="mlp"`` encodes
    the flattened window with a two-layer MLP.
    """

    def __init__(
        self,
        n_frames: int,
        n_sensors: int,
        out_dim: int = 128,
        mode: str = "cnn",
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.n_frames = int(n_frames)
        self.n_sensors = int(n_sensors)
        self.mode = str(mode)
        if self.mode not in {"cnn", "mlp"}:
            raise ValueError(f"unknown history encoder mode: {mode!r}")
        if self.mode == "cnn":
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            )
            height, width = self.n_frames, self.n_sensors
            for _ in range(3):
                height, width = (height + 1) // 2, (width + 1) // 2
            flat_dim = 64 * height * width
        else:
            self.encoder = nn.Flatten(1)
            flat_dim = self.n_frames * self.n_sensors
        self.proj = nn.Sequential(
            nn.Linear(flat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, sensors: torch.Tensor) -> torch.Tensor:
        if self.mode == "cnn":
            image = sensors.reshape(sensors.shape[0], 1, self.n_frames, self.n_sensors)
            return self.proj(self.encoder(image).flatten(1))
        return self.proj(self.encoder(sensors))


class FMO_BSI(nn.Module):
    """FMO with symmetric low-rank bilinear pooling and uniform pair weights.

    Parameters
    ----------
    n_frames, n_sensors:
        Shape of the history window (a flat ``n_frames * n_sensors`` history is
        also accepted at call time).
    n_stats:
        Number of history statistics supplied as the fourth field.
    embed_dim:
        Width d of the per-field embeddings.
    heads, rank:
        H heads of rank r; the total interaction rank is H * r.
    decoder_hidden, decoder_depth:
        Shape of the MLP decoder that maps the pooled representation to a
        scalar.
    history_encoder, history_hidden:
        History encoder flavour ("cnn" or "mlp") and its hidden width.
    wide_skip:
        Whether to add the wide linear skip over the raw concatenated fields.
    """

    def __init__(
        self,
        n_frames: int = 20,
        n_sensors: int = 64,
        n_stats: int = 3,
        embed_dim: int = 128,
        heads: int = 4,
        rank: int = 32,
        decoder_hidden: int = 128,
        decoder_depth: int = 2,
        history_encoder: str = "cnn",
        history_hidden: int = 256,
        wide_skip: bool = True,
    ) -> None:
        super().__init__()
        self.n_frames = int(n_frames)
        self.n_sensors = int(n_sensors)
        self.n_stats = int(n_stats)
        self.embed_dim = int(embed_dim)
        self.heads = int(heads)
        self.rank = int(rank)
        self.total_rank = self.heads * self.rank
        self.field_count = len(FIELD_NAMES)
        self.pairs_count = len(FIELD_PAIRS)

        self.history_encoder = HistoryEncoder(
            self.n_frames, self.n_sensors, self.embed_dim, history_encoder, history_hidden
        )
        self.embedders = nn.ModuleList(
            [
                nn.Linear(1, self.embed_dim),
                nn.Linear(1, self.embed_dim),
                nn.Linear(self.n_stats, self.embed_dim),
            ]
        )
        self.left = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Linear(self.embed_dim, self.rank, bias=False) for _ in FIELD_NAMES]
                )
                for _ in range(self.heads)
            ]
        )
        self.right = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Linear(self.embed_dim, self.rank, bias=False) for _ in FIELD_NAMES]
                )
                for _ in range(self.heads)
            ]
        )
        self.output = nn.Sequential(nn.Linear(self.total_rank, self.embed_dim), nn.SiLU())
        self.decoder = make_mlp(self.embed_dim, 1, decoder_hidden, decoder_depth)
        self.linear = (
            nn.Linear(self.n_frames * self.n_sensors + 2 + self.n_stats, 1)
            if wide_skip
            else None
        )

    # ------------------------------------------------------------------ utils
    def _check_sensors(self, sensors: torch.Tensor) -> torch.Tensor:
        if sensors.dim() == 2:
            sensors = sensors.reshape(sensors.shape[0], self.n_frames, self.n_sensors)
        if sensors.dim() != 3 or sensors.shape[1:] != (self.n_frames, self.n_sensors):
            raise ValueError(
                f"expected history of shape (B, {self.n_frames}, {self.n_sensors}), "
                f"got {tuple(sensors.shape)}"
            )
        return sensors

    def _embed(
        self, sensors: torch.Tensor, coords: torch.Tensor, stats: torch.Tensor
    ) -> List[torch.Tensor]:
        if coords.shape[1] != 2:
            raise ValueError(f"expected coords of shape (B, 2), got {tuple(coords.shape)}")
        x, t = coords.split(1, dim=1)
        return [
            self.history_encoder(sensors),
            self.embedders[0](x),
            self.embedders[1](t),
            self.embedders[2](stats),
        ]

    def _wide_input(
        self, sensors: torch.Tensor, coords: torch.Tensor, stats: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([sensors.flatten(1), coords, stats], dim=1)

    @staticmethod
    def _pair_features(
        left: List[nn.Module], right: List[nn.Module], embeddings: List[torch.Tensor]
    ) -> torch.Tensor:
        """Symmetric pair features of shape (B, pairs, rank)."""
        left_projection = [project(value) for project, value in zip(left, embeddings)]
        right_projection = [project(value) for project, value in zip(right, embeddings)]
        pairs = [
            left_projection[i] * right_projection[j] + left_projection[j] * right_projection[i]
            for i in range(len(embeddings))
            for j in range(i + 1, len(embeddings))
        ]
        return torch.stack(pairs, dim=1)

    def _pool(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        heads = [
            self._pair_features(left, right, embeddings).sum(dim=1)
            for left, right in zip(self.left, self.right)
        ]
        return self.output(torch.cat(heads, dim=1))

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        sensors: torch.Tensor,
        coords: torch.Tensor,
        stats: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict one scalar query per sample.

        ``meta`` is accepted for pipeline compatibility and is not used.
        """
        sensors = self._check_sensors(sensors)
        embeddings = self._embed(sensors, coords, stats)
        pooled = self._pool(embeddings)
        prediction = self.decoder(pooled).squeeze(-1)
        if self.linear is not None:
            prediction = prediction + self.linear(self._wide_input(sensors, coords, stats)).squeeze(-1)
        return prediction


if __name__ == "__main__":  # tiny self-check, no data required
    torch.manual_seed(0)
    model = FMO_BSI()
    output = model(torch.randn(4, 20, 64), torch.rand(4, 2), torch.randn(4, 3))
    print(
        f"{type(model).__name__}: params={sum(p.numel() for p in model.parameters()):,} "
        f"output={tuple(output.shape)} mean={output.mean().item():+.4f}"
    )
