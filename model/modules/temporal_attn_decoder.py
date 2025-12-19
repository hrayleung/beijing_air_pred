from __future__ import annotations

import torch
import torch.nn as nn


class TemporalAttentionHorizonDecoder(nn.Module):
    """
    Horizon-conditioned decoder that attends over the full lookback sequence.

    Input: rep_seq (B, N, C, L)
    Output: yhat (B, H, N, D)
    """

    def __init__(
        self,
        *,
        d_in: int,
        horizon: int,
        d_h: int,
        out_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.d_h = int(d_h)

        self.h_emb = nn.Embedding(self.horizon, self.d_h)
        self.key = nn.Linear(int(d_in), self.d_h, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(int(d_in) + self.d_h, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, rep_seq: torch.Tensor) -> torch.Tensor:
        if rep_seq.ndim != 4:
            raise ValueError(f"Expected rep_seq as (B,N,C,L); got {tuple(rep_seq.shape)}")
        B, N, C, L = rep_seq.shape

        # (B,N,C,L) -> (B,N,L,C)
        rep = rep_seq.permute(0, 1, 3, 2)

        # Keys: (B,N,L,d_h)
        k = self.key(rep)

        # Horizon queries: (H,d_h)
        h_idx = torch.arange(self.horizon, device=rep_seq.device)
        q = self.h_emb(h_idx)

        # Scores: (B,N,H,L)
        scores = torch.einsum("bnld,hd->bnhl", k, q) / (self.d_h**0.5)
        attn = torch.softmax(scores, dim=-1)

        # Context: (B,N,H,C)
        ctx = torch.einsum("bnhl,bnlc->bnhc", attn, rep)

        # (B,N,H,C) -> (B,H,N,C)
        ctx = ctx.permute(0, 2, 1, 3).contiguous()

        # Concatenate horizon embedding and predict.
        q_exp = q.view(1, self.horizon, 1, self.d_h).expand(B, self.horizon, N, self.d_h)
        x = torch.cat([ctx, q_exp], dim=-1)
        return self.mlp(x)

