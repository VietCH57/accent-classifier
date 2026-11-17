"""Trainable head on top of the frozen encoder:
    hidden states [B, L, T, D] -> softmax-weighted layer sum -> Linear(D, proj) -> pooling over time
    (attentive statistics or mean+std) -> MLP -> 2 logits (0 = non-Vietnamese, 1 = Vietnamese)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerSum(nn.Module):
    def __init__(self, num_layers: int, layer: int | None = None):
        super().__init__()
        self.layer = layer
        self.logits = nn.Parameter(torch.zeros(num_layers))

    def weights(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=0)

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        if self.layer is not None:
            return hs[:, self.layer]
        return torch.einsum("l,bltd->btd", self.weights().to(hs.dtype), hs)


class MeanStdPool(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(x.dtype)
        n = m.sum(1).clamp(min=1)
        mean = (x * m).sum(1) / n
        var = ((x - mean.unsqueeze(1)) ** 2 * m).sum(1) / n
        return torch.cat([mean, var.clamp(min=1e-6).sqrt()], dim=-1)


class AttentiveStatsPool(nn.Module):
    """Attentive statistics pooling (Okabe et al. 2018) with a masked softmax over time."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a = self.att(x).squeeze(-1).masked_fill(~mask, float("-inf"))
        a = F.softmax(a, dim=1).unsqueeze(-1)
        mean = (a * x).sum(1)
        var = (a * (x - mean.unsqueeze(1)) ** 2).sum(1)
        return torch.cat([mean, var.clamp(min=1e-6).sqrt()], dim=-1)


class AccentHead(nn.Module):
    def __init__(self, num_layers: int, dim: int, proj: int = 256, hidden: int = 256, pool: str = "asp",
                 dropout: float = 0.2, layer: int | None = None, num_classes: int = 2):
        super().__init__()
        self.layers = LayerSum(num_layers, layer)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, proj)
        self.pool = AttentiveStatsPool(proj) if pool == "asp" else MeanStdPool()
        self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * proj, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def embed(self, hs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.norm(self.layers(hs).float()))
        return self.pool(x, mask)

    def forward(self, hs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embed(hs, mask))
