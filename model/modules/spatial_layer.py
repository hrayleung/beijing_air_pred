from __future__ import annotations

import torch
import torch.nn as nn


class SpatialMessagePassing(nn.Module):
    """
    Directed message passing per timestep:
      z[t] = (A_t @ h[t]) @ W_s
    """

    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_model, out_dim, bias=False)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # h: (B, L, N, d), A: (B, L, N, N)
        msg = torch.einsum("blij,bljd->blid", A, h)  # (B, L, N, d)
        return self.proj(msg)

