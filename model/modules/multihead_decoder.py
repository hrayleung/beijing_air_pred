from __future__ import annotations

import torch
import torch.nn as nn


class _SmallHead(nn.Module):
    def __init__(self, d_in: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadHorizonDecoder(nn.Module):
    """
    Multi-head decoder: one small head per pollutant.

    Input representation is station-wise (B, N, C). A horizon embedding is concatenated
    so each head can produce distinct outputs for each horizon.

    Output: (B, H, N, D)
    """

    def __init__(
        self,
        *,
        d_in: int,
        horizon: int,
        d_h: int,
        num_targets: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.num_targets = int(num_targets)

        self.h_emb = nn.Embedding(self.horizon, int(d_h))

        d_cat = int(d_in) + int(d_h)
        self.heads = nn.ModuleList(
            [_SmallHead(d_cat, int(hidden_dim), float(dropout)) for _ in range(self.num_targets)]
        )

    def forward(self, rep: torch.Tensor) -> torch.Tensor:
        """
        rep: (B, N, C)
        returns: (B, H, N, D)
        """
        if rep.ndim != 3:
            raise ValueError(f"Expected rep as (B,N,C); got {tuple(rep.shape)}")
        B, N, C = rep.shape

        h_idx = torch.arange(self.horizon, device=rep.device)
        h = self.h_emb(h_idx)  # (H, d_h)
        h = h.view(1, self.horizon, 1, -1).expand(B, self.horizon, N, -1)  # (B,H,N,d_h)

        r = rep.unsqueeze(1).expand(B, self.horizon, N, C)  # (B,H,N,C)
        x = torch.cat([r, h], dim=-1)  # (B,H,N,C+d_h)

        outs = []
        for head in self.heads:
            y = head(x)  # (B,H,N,1)
            outs.append(y)
        return torch.cat(outs, dim=-1)  # (B,H,N,D)

