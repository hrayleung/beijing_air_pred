from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from model.modules.feature_encoder import FeatureEncoder
from model.modules.dynamic_graph import WindGatedDynamicGraphBuilder
from model.modules.spatial_layer import SpatialMessagePassing
from model.modules.tcn import TemporalConvNet
from model.modules.horizon_decoder import HorizonDecoder
from model.modules.multihead_decoder import MultiHeadHorizonDecoder
from model.modules.residual_baseline import compute_persistence_baseline


@dataclass(frozen=True)
class WGDTMConfig:
    num_nodes: int
    in_features: int
    horizon: int
    num_targets: int
    d_model: int
    d_qk: int
    d_node_emb: int
    dropout: float
    wind_gate_hidden: int
    lambda_gate: float
    alpha_init: float
    beta_init: float
    gamma_init: float
    add_self_loops: bool
    spatial_out_dim: int
    tcn_channels: int
    tcn_layers: int
    tcn_kernel: int
    tcn_dropout: float
    dec_h_emb_dim: int
    dec_hidden_dim: int
    dec_dropout: float
    decoder_type: str = "shared"  # "shared" | "multihead"
    use_residual_forecasting: bool = False
    assert_shapes: bool = False


class WGDGTM(nn.Module):
    """
    Wind-Gated Dynamic Graph + TCN Temporal Model.
    """

    def __init__(
        self,
        cfg: WGDTMConfig,
        A_static: torch.Tensor,
        wind_feature_indices: Dict[str, int],
        *,
        target_feature_indices: Sequence[int],
        input_center: torch.Tensor,
        input_scale: torch.Tensor,
    ):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("A_static", A_static)
        self.wind_idx = wind_feature_indices
        self.target_feature_indices = list(target_feature_indices)
        self.register_buffer("input_center", input_center)
        self.register_buffer("input_scale", input_scale)

        self.encoder = FeatureEncoder(cfg.in_features, cfg.d_model, dropout=cfg.dropout)
        self.graph = WindGatedDynamicGraphBuilder(
            num_nodes=cfg.num_nodes,
            d_model=cfg.d_model,
            d_qk=cfg.d_qk,
            d_node_emb=cfg.d_node_emb,
            wind_gate_hidden=cfg.wind_gate_hidden,
            lambda_gate=cfg.lambda_gate,
            alpha_init=cfg.alpha_init,
            beta_init=cfg.beta_init,
            gamma_init=cfg.gamma_init,
            add_loops=cfg.add_self_loops,
        )
        self.spatial = SpatialMessagePassing(cfg.d_model, cfg.spatial_out_dim)

        if cfg.spatial_out_dim != cfg.tcn_channels:
            self.tcn_in = nn.Linear(cfg.spatial_out_dim, cfg.tcn_channels)
        else:
            self.tcn_in = nn.Identity()

        self.tcn = TemporalConvNet(
            channels=cfg.tcn_channels,
            num_layers=cfg.tcn_layers,
            kernel_size=cfg.tcn_kernel,
            dropout=cfg.tcn_dropout,
        )

        if cfg.decoder_type == "multihead":
            self.decoder = MultiHeadHorizonDecoder(
                d_in=cfg.tcn_channels,
                horizon=cfg.horizon,
                d_h=cfg.dec_h_emb_dim,
                num_targets=cfg.num_targets,
                hidden_dim=cfg.dec_hidden_dim,
                dropout=cfg.dec_dropout,
            )
        elif cfg.decoder_type == "shared":
            self.decoder = HorizonDecoder(
                d_in=cfg.tcn_channels,
                horizon=cfg.horizon,
                d_h=cfg.dec_h_emb_dim,
                out_dim=cfg.num_targets,
                hidden_dim=cfg.dec_hidden_dim,
                dropout=cfg.dec_dropout,
            )
        else:
            raise ValueError(f"Unknown decoder_type: {cfg.decoder_type}")

    def _wind_uvs(self, X: torch.Tensor) -> torch.Tensor:
        # X is input-scaled; unscale only the wind-related channels for physical gating.
        idx_sin = self.wind_idx["wd_sin"]
        idx_cos = self.wind_idx["wd_cos"]
        idx_wspm = self.wind_idx["WSPM"]

        wd_sin = X[:, :, :, idx_sin] * self.input_scale[idx_sin] + self.input_center[idx_sin]
        wd_cos = X[:, :, :, idx_cos] * self.input_scale[idx_cos] + self.input_center[idx_cos]
        wspm = X[:, :, :, idx_wspm] * self.input_scale[idx_wspm] + self.input_center[idx_wspm]
        u = wspm * wd_cos
        v = wspm * wd_sin
        return torch.stack([u, v, wspm], dim=-1)  # (B, L, N, 3)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, L, N, F) scaled features
        returns Yhat: (B, H, N, D) in raw target units
        """
        B, L, N, F = X.shape
        h = self.encoder(X)  # (B, L, N, d_model)

        wind_uvs = self._wind_uvs(X)
        A = self.graph(h, self.A_static, wind_uvs)  # (B, L, N, N)

        z = self.spatial(h, A)  # (B, L, N, spatial_out)
        z = self.tcn_in(z)  # (B, L, N, C)

        # TCN expects (B*N, C, L)
        z = z.permute(0, 2, 3, 1).reshape(B * N, self.cfg.tcn_channels, L)
        y = self.tcn(z)  # (B*N, C, L)
        r_last = y[:, :, -1].reshape(B, N, self.cfg.tcn_channels)  # (B, N, C)

        delta_hat = self.decoder(r_last)  # (B, H, N, D)

        if self.cfg.use_residual_forecasting:
            y_base = compute_persistence_baseline(
                X,
                horizon=self.cfg.horizon,
                target_feature_indices=self.target_feature_indices,
                input_center=self.input_center,
                input_scale=self.input_scale,
                assert_checks=self.cfg.assert_shapes,
            )
            yhat = y_base + delta_hat
        else:
            yhat = delta_hat

        if self.cfg.assert_shapes:
            assert yhat.shape == (B, self.cfg.horizon, self.cfg.num_nodes, self.cfg.num_targets)
        return yhat
