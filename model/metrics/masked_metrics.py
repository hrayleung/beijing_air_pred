from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    denom = float(mask.sum())
    if denom <= 0:
        return float("nan")
    return float((values * mask).sum() / denom)


def masked_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    return _masked_mean(np.abs(pred - target), mask)


def masked_rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    mse = _masked_mean((pred - target) ** 2, mask)
    return float(np.sqrt(mse))


def masked_smape(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> float:
    num = np.abs(pred - target)
    denom = np.abs(pred) + np.abs(target) + eps
    return 100.0 * _masked_mean(num / denom, mask)


def per_pollutant_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    pollutant_names: List[str],
    horizons: Tuple[int, ...] = (1, 6, 12, 24),
) -> Dict[str, Dict[str, float]]:
    """
    pred/target/mask: (S, H, N, D)
    """
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}, mask={mask.shape}")
    H = pred.shape[1]
    D = pred.shape[3]

    out: Dict[str, Dict[str, float]] = {}
    for d in range(D):
        name = pollutant_names[d]
        m = mask[:, :, :, d]
        p = pred[:, :, :, d]
        y = target[:, :, :, d]
        row = {
            "MAE": masked_mae(p, y, m),
            "RMSE": masked_rmse(p, y, m),
            "sMAPE": masked_smape(p, y, m),
        }
        for h in horizons:
            if h < 1 or h > H:
                raise ValueError(f"Invalid horizon {h} for H={H}")
            row[f"MAE_h{h}"] = masked_mae(p[:, h - 1, :], y[:, h - 1, :], m[:, h - 1, :])
        out[name] = row
    return out


def macro_average(per_pollutant: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    keys = ["MAE", "RMSE", "sMAPE"]
    out: Dict[str, float] = {}
    for k in keys:
        out[f"macro_{k}"] = float(np.nanmean([v[k] for v in per_pollutant.values()]))
    return out


def horizon_variation_check(pred: np.ndarray, eps: float = 1e-3) -> float:
    """
    Returns mean(|h1 - h24|) averaged over all dims.
    """
    if pred.shape[1] < 24:
        raise ValueError("Expected H>=24")
    delta = float(np.mean(np.abs(pred[:, 0] - pred[:, 23])))
    if delta <= eps:
        raise ValueError(f"Horizon variation too small: {delta} <= {eps}")
    return delta

