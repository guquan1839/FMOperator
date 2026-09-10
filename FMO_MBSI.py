"""FMO-MBSI: MLP-parameterised Bilinear Sum Interaction.

FMO-BSI with each interaction *factor* produced by a small per-head MLP
instead of a single bias-free linear map.  The pair aggregation is the plain
uniform sum, so the only difference with respect to FMO_BSI is the
parameterisation of the left/right projections.

Paper: https://arxiv.org/abs/2607.28762

This file is deliberately standalone: it imports PyTorch (and the standard
library) only, it never imports any other file of this repository, and it can
therefore be vendored into an arbitrary training script as-is.

Interaction factors
-------------------
For head h, side s in {left, right} and field i, FMO_BSI uses

    e_i  ->  W_{h,s,i} e_i                      (linear, bias-free)

whereas FMO-MBSI uses

    e_i  ->  G_{h,s,i}^T act( A_{h,s,i} e_i ),  A : R^d -> R^{p},  G : R^{p} -> R^r

with ``proj_hidden = p`` (default 128), ``act = SiLU`` on the hidden layer and
an *identity* activation on the output factors (``proj_out_act="identity"``,
the paper's default).  The identity output activation keeps the factors linear
at the point where they are multiplied, which is important for small data:
applying a squashing non-linearity to the factors themselves
(``proj_out_act="silu"``) was consistently worse in our experiments.

Heads that are not equivalent to a wider head
---------------------------------------------
Because each head has its own non-linearity, H heads of rank r are no longer
equivalent to a single head of rank H * r (this is the reason FMO-MBSI uses a
per-head parameterisation at all).  ``proj_hidden=0`` recovers the bias-free
linear factors of FMO_BSI.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


__all__ = ["HistoryEncoder", "make_head_projection", "FMO_MBSI", "FIELD_NAMES", "FIELD_PAIRS"]

FIELD_NAMES: Tuple[str, ...] = ("history", "x", "t", "stats")
FIELD_PAIRS: Tuple[Tuple[int, int], ...] = tuple(combinations(range(len(FIELD_NAMES)), 2))

_ACTIVATIONS = {"silu": nn.SiLU, "tanh": nn.Tanh, "identity": None}


def make_mlp(in_dim: int, out_dim: int, hidden: int, depth: int) -> nn.Sequential:
    """``depth`` SiLU hidden layers followed by a linear output layer."""
    layers: List[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
    for _ in range(int(depth) - 1):
        layers += [nn.Linear(hidden, hidden), nn.SiLU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


def make_head_projection(
    embed_dim: int,
    proj_hidden: int,
    rank: int,
    mid_act: str = "silu",
    out_act: str = "identity",
) -> nn.Module:
    """Projection used for one interaction factor of one head and one field.

    ``proj_hidden <= 0`` returns the bias-free linear projection of FMO_BSI;
    a positive value returns a two-layer MLP whose hidden activation is
    ``mid_act`` and whose output factors are optionally passed through
    ``out_act`` (``"identity"``, the default, leaves them untouched).
    """
    if mid_act not in _ACTIVATIONS or out_act not in _ACTIVATIONS:
        raise ValueError(f"unknown activation: {mid_act!r} / {out_act!r}")
    if int(proj_hidden) <= 0:
        return nn.Linear(embed_dim, rank, bias=False)
    layers: List[nn.Module] = [
        nn.Linear(embed_dim, int(proj_hidden)), _ACTIVATIONS[mid_act]()
    ]
    layers.append(nn.Linear(int(proj_hidden), rank))
    if _ACTIVATIONS[out_act] is not None:
        layers.append(_ACTIVATIONS[out_act]())
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


class FMO_MBSI(nn.Module):
    """FMO with per-head MLP interaction factors and uniform pair weights.

    Parameters
    ----------
    proj_hidden:
        Hidden width of the per-head factor MLPs; ``0`` reverts to the
        bias-free linear factors of FMO_BSI.
    proj_mid_act, proj_out_act:
        Hidden / output activations of those MLPs, see
        :func:`make_head_projection`.
    Other parameters are identical to FMO_BSI.
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
        proj_hidden: int = 128,
        proj_mid_act: str = "silu",
        proj_out_act: str = "identity",
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
        self.proj_hidden = int(proj_hidden)
        self.proj_mid_act = str(proj_mid_act)
        self.proj_out_act = str(proj_out_act)

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
                    [
                        make_head_projection(
                            self.embed_dim, self.proj_hidden, self.rank,
                            self.proj_mid_act, self.proj_out_act,
                        )
                        for _ in FIELD_NAMES
                    ]
                )
                for _ in range(self.heads)
            ]
        )
        self.right = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        make_head_projection(
                            self.embed_dim, self.proj_hidden, self.rank,
                            self.proj_mid_act, self.proj_out_act,
                        )
                        for _ in FIELD_NAMES
                    ]
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

    def factor_parameter_counts(self) -> Dict[str, int]:
        """Parameter count of the interaction factors, split by head."""

        def count(module: nn.Module) -> int:
            return sum(parameter.numel() for parameter in module.parameters())

        return {
            f"head{head}": count(self.left[head]) + count(self.right[head])
            for head in range(self.heads)
        }

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
    model = FMO_MBSI()
    output = model(torch.randn(4, 20, 64), torch.rand(4, 2), torch.randn(4, 3))
    print(
        f"{type(model).__name__}: params={sum(p.numel() for p in model.parameters()):,} "
        f"output={tuple(output.shape)} mean={output.mean().item():+.4f} "
        f"proj_hidden={model.proj_hidden} out_act={model.proj_out_act}"
    )
