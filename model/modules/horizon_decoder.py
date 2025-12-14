from __future__ import annotations

import torch
import torch.nn as nn


class HorizonDecoder(nn.Module):
    """
    Horizon-conditioned decoder using learnable horizon embeddings.
    """

    def __init__(
        self,
        d_in: int,
        horizon: int,
        d_h: int,
        out_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.h_emb = nn.Embedding(self.horizon, d_h)

        self.mlp = nn.Sequential(
            nn.Linear(d_in + d_h, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
          r: (B, N, d_in)
        Returns:
          yhat: (B, H, N, out_dim)
        """
        B, N, d_in = r.shape
        h_idx = torch.arange(self.horizon, device=r.device)
        h = self.h_emb(h_idx).view(1, self.horizon, 1, -1)  # (1,H,1,d_h)
        r_exp = r.unsqueeze(1).expand(B, self.horizon, N, d_in)
        x = torch.cat([r_exp, h.expand(B, -1, N, -1)], dim=-1)
        yhat = self.mlp(x)
        return yhat

