from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.graphs.adjacency import add_self_loops, row_normalize


class WindGatedDynamicGraphBuilder(nn.Module):
    """
    Builds a directed adjacency A_t per timestep:
      A_t = RowNormalize( a*A_static + b*A_learn + c*A_dyn(t) + I )
    where:
      A_learn is derived from trainable node embeddings,
      A_dyn(t) is attention-based and wind-gated.
    """

    def __init__(
        self,
        num_nodes: int,
        d_model: int,
        d_qk: int,
        d_node_emb: int,
        *,
        wind_gate_hidden: int = 32,
        lambda_gate: float = 0.5,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
        gamma_init: float = 1.0,
        add_loops: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.d_qk = d_qk
        self.lambda_gate = float(lambda_gate)
        self.add_loops = bool(add_loops)

        self.Wq = nn.Linear(d_model, d_qk, bias=False)
        self.Wk = nn.Linear(d_model, d_qk, bias=False)

        self.node_emb = nn.Parameter(torch.randn(num_nodes, d_node_emb) * 0.02)

        self.gate_mlp = nn.Sequential(
            nn.Linear(3, wind_gate_hidden),
            nn.GELU(),
            nn.Linear(wind_gate_hidden, 1),
        )

        self._alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self._beta = nn.Parameter(torch.tensor(float(beta_init)))
        self._gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def _positive(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    def build_A_learn(self) -> torch.Tensor:
        # (N, d_e) @ (d_e, N) -> (N, N)
        scores = self.node_emb @ self.node_emb.t()
        return F.softmax(scores, dim=-1)

    def build_wind_gate(self, wind_uvs: torch.Tensor) -> torch.Tensor:
        # wind_uvs: (B, L, N, 3) = [u, v, wspm]
        g = torch.sigmoid(self.gate_mlp(wind_uvs))  # (B, L, N, 1)
        return g

    def build_A_dyn(self, h: torch.Tensor, gate: torch.Tensor, A_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # h: (B, L, N, d_model), gate: (B, L, N, 1)
        q = self.Wq(h)  # (B, L, N, d_qk)
        k = self.Wk(h)  # (B, L, N, d_qk)

        # logits: (B, L, N, N)
        logits = torch.einsum("blid,bljd->blij", q, k) / (self.d_qk ** 0.5)
        # Wind gating: scale attention "temperature" per source node (i).
        # NOTE: adding a row-wise constant would cancel out under softmax; scaling does not.
        logits = logits * (1.0 + (self.lambda_gate * gate))
        return F.softmax(logits, dim=-1)

    def forward(self, h: torch.Tensor, A_static: torch.Tensor, wind_uvs: torch.Tensor) -> torch.Tensor:
        """
        Args:
          h: (B, L, N, d_model)
          A_static: (N, N) on device
          wind_uvs: (B, L, N, 3) = [u, v, wspm]
        Returns:
          A: (B, L, N, N) row-normalized
        """
        A_learn = self.build_A_learn()  # (N, N)
        gate = self.build_wind_gate(wind_uvs)  # (B, L, N, 1)
        A_dyn = self.build_A_dyn(h, gate)  # (B, L, N, N)

        a = self._positive(self._alpha)
        b = self._positive(self._beta)
        c = self._positive(self._gamma)

        fused = a * A_static + b * A_learn  # (N, N)
        fused = fused.unsqueeze(0).unsqueeze(0) + (c * A_dyn)  # (B, L, N, N)

        if self.add_loops:
            fused = add_self_loops(fused, weight=1.0)

        return row_normalize(fused)
