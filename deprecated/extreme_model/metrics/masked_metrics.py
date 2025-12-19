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


def compute_all_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": masked_mae(pred, target, mask),
        "RMSE": masked_rmse(pred, target, mask),
        "sMAPE": masked_smape(pred, target, mask),
    }


def compute_per_horizon_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """
    pred/target/mask: (S, H, N, D)
    returns: dict of horizon (1-indexed) -> {MAE, RMSE, sMAPE} over all pollutants
    """
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}, mask={mask.shape}")
    if pred.ndim != 4:
        raise ValueError(f"Expected 4D arrays (S,H,N,D); got {pred.shape}")
    H = int(pred.shape[1])
    out: Dict[int, Dict[str, float]] = {}
    for h in range(H):
        out[h + 1] = compute_all_metrics(pred[:, h, :, :], target[:, h, :, :], mask[:, h, :, :])
    return out


def compute_per_pollutant_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    pollutant_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    pred/target/mask: (S, H, N, D)
    returns: dict pollutant -> {MAE, RMSE, sMAPE} aggregated over all horizons/stations/samples
    """
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}, mask={mask.shape}")
    if pred.ndim != 4:
        raise ValueError(f"Expected 4D arrays (S,H,N,D); got {pred.shape}")
    D = int(pred.shape[3])
    if len(pollutant_names) != D:
        raise ValueError(f"Expected pollutant_names length {D}; got {len(pollutant_names)}")

    out: Dict[str, Dict[str, float]] = {}
    for d in range(D):
        name = pollutant_names[d]
        out[name] = compute_all_metrics(pred[:, :, :, d], target[:, :, :, d], mask[:, :, :, d])
    return out


def compute_per_pollutant_report(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    pollutant_names: List[str],
    horizons: Tuple[int, ...] = (1, 6, 12, 24),
) -> Dict[str, Dict[str, float]]:
    """
    pred/target/mask: (S, H, N, D)
    returns: pollutant -> {MAE, RMSE, sMAPE, MAE_h1, MAE_h6, ...}
    """
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}, mask={mask.shape}")
    if pred.ndim != 4:
        raise ValueError(f"Expected 4D arrays (S,H,N,D); got {pred.shape}")
    H = int(pred.shape[1])
    D = int(pred.shape[3])
    if len(pollutant_names) != D:
        raise ValueError(f"Expected pollutant_names length {D}; got {len(pollutant_names)}")

    out: Dict[str, Dict[str, float]] = {}
    for d in range(D):
        name = pollutant_names[d]
        p = pred[:, :, :, d]
        y = target[:, :, :, d]
        m = mask[:, :, :, d]
        row = compute_all_metrics(p, y, m)
        for h in horizons:
            if h < 1 or h > H:
                raise ValueError(f"Invalid horizon {h} for H={H}")
            row[f"MAE_h{h}"] = masked_mae(p[:, h - 1, :], y[:, h - 1, :], m[:, h - 1, :])
        out[name] = row
    return out


def macro_average_per_pollutant(per_pollutant: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    keys = ["MAE", "RMSE", "sMAPE"]
    out: Dict[str, float] = {}
    for k in keys:
        out[f"macro_{k}"] = float(np.nanmean([float(v.get(k, np.nan)) for v in per_pollutant.values()]))
    return out


# Backwards-compatible aliases (older code expects these names).
def per_pollutant_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    pollutant_names: List[str],
    horizons: Tuple[int, ...] = (1, 6, 12, 24),
) -> Dict[str, Dict[str, float]]:
    return compute_per_pollutant_report(pred, target, mask, pollutant_names, horizons=horizons)


def macro_average(per_pollutant: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return macro_average_per_pollutant(per_pollutant)


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
