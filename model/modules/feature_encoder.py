from __future__ import annotations

import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    """
    Per-node, per-timestep feature encoder: x[t,i] -> h[t,i] in R^{d_model}.
    """

    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N, F)
        h = self.proj(x)
        h = self.norm(h)
        h = self.act(h)
        return self.drop(h)

