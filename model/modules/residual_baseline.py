from __future__ import annotations

from typing import Iterable, Sequence

import torch


def compute_persistence_baseline(
    X_scaled: torch.Tensor,
    *,
    horizon: int,
    target_feature_indices: Sequence[int],
    input_center: torch.Tensor,
    input_scale: torch.Tensor,
    assert_checks: bool = False,
) -> torch.Tensor:
    """
    Persistence baseline in RAW target units:
      y_base(t+h) = y(t) for all horizons h=1..H.

    Args:
      X_scaled: (B, L, N, F) scaled inputs (RobustScaler per-feature).
      horizon: H (e.g., 24)
      target_feature_indices: indices of the 6 pollutant features inside X's feature dim.
      input_center/input_scale: (F,) tensors matching preprocessing scaler params.
      assert_checks: when True, validate baseline invariants.

    Returns:
      y_base_raw: (B, H, N, D) in raw units, constant over horizon.
    """
    if X_scaled.ndim != 4:
        raise ValueError(f"Expected X_scaled as (B,L,N,F); got {tuple(X_scaled.shape)}")
    B, L, N, F = X_scaled.shape
    idx = torch.as_tensor(list(target_feature_indices), device=X_scaled.device, dtype=torch.long)
    if idx.numel() == 0:
        raise ValueError("target_feature_indices is empty")
    if idx.min().item() < 0 or idx.max().item() >= F:
        raise ValueError(f"target_feature_indices out of range for F={F}: {list(target_feature_indices)}")
    if input_center.ndim != 1 or input_scale.ndim != 1 or input_center.shape != input_scale.shape:
        raise ValueError("input_center/input_scale must be 1D and have the same shape")
    if input_center.shape[0] != F:
        raise ValueError(f"input_center/input_scale length mismatch: got {input_center.shape[0]} expected {F}")

    # Take last lookback step t (index L-1), inverse-transform only pollutant channels.
    x_t_scaled = X_scaled[:, L - 1, :, :].index_select(dim=-1, index=idx)  # (B, N, D)
    center = input_center.index_select(dim=0, index=idx).view(1, 1, -1)  # (1,1,D)
    scale = input_scale.index_select(dim=0, index=idx).view(1, 1, -1)  # (1,1,D)
    y_t_raw = x_t_scaled * scale + center  # (B, N, D)

    y_base = y_t_raw.unsqueeze(1).expand(B, int(horizon), N, idx.numel()).contiguous()  # (B,H,N,D)

    if assert_checks:
        if y_base.shape != (B, int(horizon), N, idx.numel()):
            raise AssertionError(f"Bad baseline shape: {tuple(y_base.shape)}")
        # Baseline is constant across horizon.
        if not torch.allclose(y_base[:, 0], y_base[:, -1], atol=0.0, rtol=0.0):
            raise AssertionError("Persistence baseline must be constant across horizons")
        # Baseline equals y(t) at all horizons.
        if not torch.allclose(y_base[:, 0], y_t_raw, atol=0.0, rtol=0.0):
            raise AssertionError("Persistence baseline does not match raw y(t)")

    return y_base

