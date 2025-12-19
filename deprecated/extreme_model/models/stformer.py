from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass(frozen=True)
class STFormerConfig:
    num_nodes: int
    in_features: int
    lookback: int
    horizon: int
    num_targets: int
    d_model: int
    n_heads: int
    enc_layers: int
    dec_layers: int
    ff_dim: int
    dropout: float
    use_future_time_features: bool = True
    baseline_mode: str = "none"  # "none" | "persistence" | "seasonal" | "mix"
    assert_shapes: bool = False


class STFormer(nn.Module):
    """
    Spatio-Temporal Transformer (encoder-decoder).

    Encoder tokens: (t, node) over the full lookback window.
    Decoder queries: (horizon, node), optionally enriched with known future time features
    derived from the last observed hour/month sin/cos in X.

    X input is expected to be *scaled* (RobustScaler per-feature).
    Output is in raw target units (same as Y in P1_deep NPZ).
    """

    def __init__(
        self,
        cfg: STFormerConfig,
        *,
        time_feature_indices: Dict[str, int],
        input_center: torch.Tensor,
        input_scale: torch.Tensor,
    ):
        super().__init__()
        self.cfg = cfg

        needed = {"hour_sin", "hour_cos", "month_sin", "month_cos"}
        missing = needed.difference(time_feature_indices.keys())
        if missing:
            raise ValueError(f"Missing time_feature_indices keys: {sorted(missing)}")
        self.time_idx = {k: int(v) for k, v in time_feature_indices.items()}

        self.register_buffer("input_center", input_center)
        self.register_buffer("input_scale", input_scale)

        self.in_proj = nn.Linear(int(cfg.in_features), int(cfg.d_model))
        self.time_pos = nn.Embedding(int(cfg.lookback), int(cfg.d_model))
        self.node_emb = nn.Embedding(int(cfg.num_nodes), int(cfg.d_model))
        self.horizon_emb = nn.Embedding(int(cfg.horizon), int(cfg.d_model))

        self.time_feat_proj = nn.Linear(4, int(cfg.d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(cfg.d_model),
            nhead=int(cfg.n_heads),
            dim_feedforward=int(cfg.ff_dim),
            dropout=float(cfg.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(cfg.enc_layers))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=int(cfg.d_model),
            nhead=int(cfg.n_heads),
            dim_feedforward=int(cfg.ff_dim),
            dropout=float(cfg.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(cfg.dec_layers))

        self.out_proj = nn.Sequential(
            nn.Linear(int(cfg.d_model), int(cfg.d_model)),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.d_model), int(cfg.num_targets)),
        )

        if str(cfg.baseline_mode).lower() == "mix":
            # Per-horizon per-target mixing between persistence and seasonal baselines.
            # Initialize to prefer persistence for short horizons and seasonal for long horizons.
            init = torch.linspace(-2.0, 2.0, steps=int(cfg.horizon)).view(int(cfg.horizon), 1).repeat(1, int(cfg.num_targets))
            self.baseline_gate_logits = nn.Parameter(init)
        else:
            self.baseline_gate_logits = None

        # Precompute hour rotation deltas for horizons h=1..H.
        h = torch.arange(1, int(cfg.horizon) + 1, dtype=torch.float32)
        delta = (2.0 * math.pi / 24.0) * h
        self.register_buffer("_hour_cos_delta", torch.cos(delta))
        self.register_buffer("_hour_sin_delta", torch.sin(delta))

    def _unscale_feature(self, X_scaled: torch.Tensor, feature_idx: int) -> torch.Tensor:
        # X_scaled: (B, L, N, F)
        c = self.input_center[int(feature_idx)]
        s = self.input_scale[int(feature_idx)]
        return X_scaled[..., int(feature_idx)] * s + c

    def _baseline_forecast(self, X_scaled: torch.Tensor) -> torch.Tensor:
        """
        Build a strong deterministic baseline forecast from past target features in X.

        Returns: (B, H, N, D) in raw target units.
        """
        mode = str(self.cfg.baseline_mode).lower()
        if mode == "none":
            raise RuntimeError("_baseline_forecast called with baseline_mode=none")

        B, L, N, _ = X_scaled.shape
        H = int(self.cfg.horizon)
        D = int(self.cfg.num_targets)

        if D > int(self.cfg.in_features):
            raise ValueError(f"num_targets D={D} must be <= in_features F={int(self.cfg.in_features)}")

        center = self.input_center[:D].view(1, 1, D)
        scale = self.input_scale[:D].view(1, 1, D)

        # Persistence baseline: repeat last observed targets for all horizons.
        last_scaled = X_scaled[:, L - 1, :, :D]  # (B,N,D)
        last_raw = last_scaled * scale + center  # (B,N,D)
        persistence = last_raw.unsqueeze(1).expand(B, H, N, D)  # (B,H,N,D)

        if mode in {"persistence"}:
            return persistence

        # Seasonal baseline: use the last 24 hours trajectory as next-24 forecast (h=1..H <= 24).
        if int(self.cfg.lookback) < 24 or H > 24:
            raise ValueError(f"baseline_mode={mode} requires lookback>=24 and horizon<=24; got lookback={L}, horizon={H}")
        seasonal_scaled = X_scaled[:, L - 24 : L, :, :D]  # (B,24,N,D)
        seasonal_raw = seasonal_scaled * self.input_scale[:D].view(1, 1, 1, D) + self.input_center[:D].view(1, 1, 1, D)
        seasonal = seasonal_raw[:, :H, :, :]  # (B,H,N,D)

        if mode == "seasonal":
            return seasonal

        if mode == "mix":
            if self.baseline_gate_logits is None:
                raise RuntimeError("baseline_gate_logits is None but baseline_mode=mix")
            g = torch.sigmoid(self.baseline_gate_logits).view(1, H, 1, D)  # (1,H,1,D)
            return g * seasonal + (1.0 - g) * persistence

        raise ValueError(f"Unknown baseline_mode={self.cfg.baseline_mode!r}")

    def _future_time_features(self, X_scaled: torch.Tensor) -> torch.Tensor:
        """
        Build known future time features for each horizon using the last observed
        (hour_sin, hour_cos) rotated by +h hours, and month_sin/cos copied.

        Returns: (B, H, 4) = [hour_sin, hour_cos, month_sin, month_cos]
        """
        B, L, N, F = X_scaled.shape

        # Use station 0; these time features are global and identical across stations.
        x_last = X_scaled[:, L - 1, 0, :]  # (B, F)

        hour_sin = x_last[:, self.time_idx["hour_sin"]]
        hour_cos = x_last[:, self.time_idx["hour_cos"]]
        month_sin = x_last[:, self.time_idx["month_sin"]]
        month_cos = x_last[:, self.time_idx["month_cos"]]

        # Unscale to true sin/cos values.
        hour_sin = hour_sin * self.input_scale[self.time_idx["hour_sin"]] + self.input_center[self.time_idx["hour_sin"]]
        hour_cos = hour_cos * self.input_scale[self.time_idx["hour_cos"]] + self.input_center[self.time_idx["hour_cos"]]
        month_sin = month_sin * self.input_scale[self.time_idx["month_sin"]] + self.input_center[self.time_idx["month_sin"]]
        month_cos = month_cos * self.input_scale[self.time_idx["month_cos"]] + self.input_center[self.time_idx["month_cos"]]

        cos_d = self._hour_cos_delta.view(1, -1)  # (1,H)
        sin_d = self._hour_sin_delta.view(1, -1)  # (1,H)

        hour_sin_f = hour_sin.view(B, 1) * cos_d + hour_cos.view(B, 1) * sin_d
        hour_cos_f = hour_cos.view(B, 1) * cos_d - hour_sin.view(B, 1) * sin_d

        month_sin_f = month_sin.view(B, 1).expand(B, self.cfg.horizon)
        month_cos_f = month_cos.view(B, 1).expand(B, self.cfg.horizon)

        return torch.stack([hour_sin_f, hour_cos_f, month_sin_f, month_cos_f], dim=-1)  # (B,H,4)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (B, L, N, F) scaled
        returns: (B, H, N, D) raw units
        """
        if X.ndim != 4:
            raise ValueError(f"Expected X as (B,L,N,F); got {tuple(X.shape)}")
        B, L, N, F = X.shape
        if L != int(self.cfg.lookback):
            raise ValueError(f"Expected lookback L={int(self.cfg.lookback)}; got {L}")
        if N != int(self.cfg.num_nodes):
            raise ValueError(f"Expected num_nodes N={int(self.cfg.num_nodes)}; got {N}")
        if F != int(self.cfg.in_features):
            raise ValueError(f"Expected in_features F={int(self.cfg.in_features)}; got {F}")

        # Encoder tokens
        x = self.in_proj(X)  # (B,L,N,C)
        t = torch.arange(L, device=X.device)
        n = torch.arange(N, device=X.device)
        x = x + self.time_pos(t).view(1, L, 1, -1) + self.node_emb(n).view(1, 1, N, -1)
        x = x.reshape(B, L * N, self.cfg.d_model)  # (B, LN, C)

        mem = self.encoder(x)  # (B, LN, C)

        # Decoder queries
        h = torch.arange(int(self.cfg.horizon), device=X.device)
        h_emb = self.horizon_emb(h).view(1, int(self.cfg.horizon), 1, -1)
        n_emb = self.node_emb(n).view(1, 1, N, -1)

        if self.cfg.use_future_time_features:
            tf = self._future_time_features(X)  # (B,H,4)
            tf_emb = self.time_feat_proj(tf).view(B, int(self.cfg.horizon), 1, -1)
        else:
            tf_emb = 0.0

        q = h_emb + n_emb + tf_emb  # (B,H,N,C)
        q = q.reshape(B, int(self.cfg.horizon) * N, self.cfg.d_model)  # (B, HN, C)

        out = self.decoder(tgt=q, memory=mem)  # (B, HN, C)
        yhat = self.out_proj(out).reshape(B, int(self.cfg.horizon), N, int(self.cfg.num_targets))

        # Optional residual learning around a strong baseline forecast.
        if str(self.cfg.baseline_mode).lower() != "none":
            baseline = self._baseline_forecast(X)
            yhat = yhat + baseline

        if self.cfg.assert_shapes:
            assert yhat.shape == (B, int(self.cfg.horizon), N, int(self.cfg.num_targets))
        return yhat
