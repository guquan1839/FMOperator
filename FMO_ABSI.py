"""FMO-ABSI: Attentive Bilinear Sum Interaction.

FMO-BSI plus an *input-dependent* weighting of the six field pairs (an
attention / field-selection layer, in the spirit of attentive factorisation
machines).  Everything else (embeddings, low-rank symmetric bilinear factors,
read-out, decoder, wide skip) is identical to FMO_BSI, so the two models differ
by exactly ``heads * rank`` parameters.

Paper: https://arxiv.org/abs/2607.28762

This file is deliberately standalone: it imports PyTorch (and the standard
library) only, it never imports any other file of this repository, and it can
therefore be vendored into an arbitrary training script as-is.

Field-pair attention
--------------------
Given the pair features phi_h(i, j) in R^r of FMO_BSI, a learned vector
w_h in R^r scores every pair,

    logit_h(i, j) = <w_h, phi_h(i, j)>,

and the pair aggregation becomes a convex combination

    alpha_h = softmax_{pairs} (logit_h),
    S_h     = P * sum_{i<j} alpha_h(i, j) * phi_h(i, j)      (P = 6 pairs).

``pair_pool="shared"`` (default) sums the logits over the heads before the
softmax, so all heads share one field-pair selection; ``pair_pool="per_head"``
gives every head its own softmax.  The factor P keeps the scale of S_h
comparable to the uniform sum of FMO_BSI.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Tuple

import torch
from torch import nn


__all__ = ["HistoryEncoder", "FMO_ABSI", "FIELD_NAMES", "FIELD_PAIRS"]

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


class FMO_ABSI(nn.Module):
    """FMO with symmetric low-rank bilinear pooling and attentive pair weights.

    Parameters
    ----------
    pair_pool:
        ``"shared"`` (default) -> one field-pair softmax shared by all heads;
        ``"per_head"`` -> an independent softmax per head;
        ``"sum"`` -> uniform weights, i.e. exactly FMO_BSI (kept for ablation).
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
        pair_pool: str = "shared",
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
        self.pair_pool = str(pair_pool)
        if self.pair_pool not in {"sum", "shared", "per_head"}:
            raise ValueError(f"unknown pair_pool: {pair_pool!r}")

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
        # Zero-initialised: the attention starts from the uniform sum of
        # FMO_BSI, so the two models are exactly equal at initialisation.
        self.gate = (
            nn.Parameter(torch.zeros(self.heads, self.rank))
            if self.pair_pool != "sum"
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

    def _stacked_pair_features(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """All heads' pair features of shape (B, heads, pairs, rank)."""
        return torch.stack(
            [
                self._pair_features(left, right, embeddings)
                for left, right in zip(self.left, self.right)
            ],
            dim=1,
        )

    def _attention_weights(self, pair_features: torch.Tensor) -> torch.Tensor:
        """Softmax field-pair weights of shape (B, heads, pairs)."""
        if self.gate is None:
            shape = (pair_features.shape[0], self.heads, self.pairs_count)
            return pair_features.new_full(shape, 1.0 / self.pairs_count)
        logits = (pair_features * self.gate[None, :, None, :]).sum(dim=-1)
        if self.pair_pool == "shared":
            logits = logits.sum(dim=1, keepdim=True).expand(-1, self.heads, -1)
        return torch.softmax(logits, dim=-1)

    def _pool(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        if self.pair_pool == "sum":
            heads = [
                self._pair_features(left, right, embeddings).sum(dim=1)
                for left, right in zip(self.left, self.right)
            ]
        else:
            pair_features = self._stacked_pair_features(embeddings)
            weights = self._attention_weights(pair_features)
            pair_features = pair_features * (self.pairs_count * weights).unsqueeze(-1)
            heads = [pair_features[:, head].sum(dim=1) for head in range(self.heads)]
        return self.output(torch.cat(heads, dim=1))

    @torch.no_grad()
    def pair_weights(
        self, sensors: torch.Tensor, coords: torch.Tensor, stats: torch.Tensor
    ) -> torch.Tensor:
        """Field-pair attention weights used for the given inputs, (B, H, P)."""
        sensors = self._check_sensors(sensors)
        embeddings = self._embed(sensors, coords, stats)
        return self._attention_weights(self._stacked_pair_features(embeddings))

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
    model = FMO_ABSI()
    output = model(torch.randn(4, 20, 64), torch.rand(4, 2), torch.randn(4, 3))
    print(
        f"{type(model).__name__}: params={sum(p.numel() for p in model.parameters()):,} "
        f"output={tuple(output.shape)} mean={output.mean().item():+.4f} "
        f"pool={model.pair_pool}"
    )
