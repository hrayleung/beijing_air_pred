from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def weighted_masked_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Std-weighted masked MAE (per pollutant).

    pred/target/mask: (B, H, N, D)
    weights: (D,)
    """
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)} mask={tuple(mask.shape)}")
    if weights.ndim != 1 or weights.shape[0] != pred.shape[-1]:
        raise ValueError(f"weights must be (D,), got {tuple(weights.shape)} for D={pred.shape[-1]}")
    w = weights.view(1, 1, 1, -1)
    abs_error = torch.abs(pred - target) * mask * w
    denom = mask.sum().clamp_min(eps)
    return abs_error.sum() / denom


def compute_target_std_weights_from_npz(train_npz_path: str, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes per-pollutant std on TRAIN observed targets only.
    Returns (std, weights) arrays of shape (D,).
    """
    data = np.load(train_npz_path, allow_pickle=True)
    Y = data["Y"].astype(np.float32)
    Y_mask = data["Y_mask"].astype(np.float32)

    D = Y.shape[-1]
    std = np.zeros((D,), dtype=np.float32)
    weights = np.zeros((D,), dtype=np.float32)
    for d in range(D):
        vals = Y[:, :, :, d][Y_mask[:, :, :, d] == 1]
        std[d] = np.float32(vals.std() if vals.size else 0.0)
        weights[d] = np.float32(1.0 / (float(std[d]) + float(eps))) if std[d] > 0 else np.float32(1.0)
    return std, weights

