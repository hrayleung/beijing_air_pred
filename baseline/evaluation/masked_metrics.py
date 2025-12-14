"""
Masked metrics for evaluation with missing values.
All metrics operate on numpy arrays and respect Y_mask.
"""
import numpy as np
from typing import Dict, Optional, Tuple


def masked_mae(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray
) -> float:
    """
    Compute Masked Mean Absolute Error.
    
    Args:
        pred: Predictions (any shape)
        target: Ground truth (same shape as pred)
        mask: Binary mask, 1=valid, 0=missing (same shape)
        
    Returns:
        MAE computed only over valid positions
    """
    valid_count = mask.sum()
    if valid_count == 0:
        return np.nan
    
    abs_error = np.abs(pred - target) * mask
    return abs_error.sum() / valid_count


def masked_rmse(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray
) -> float:
    """
    Compute Masked Root Mean Squared Error.
    """
    valid_count = mask.sum()
    if valid_count == 0:
        return np.nan
    
    sq_error = ((pred - target) ** 2) * mask
    mse = sq_error.sum() / valid_count
    return np.sqrt(mse)


def masked_smape(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    epsilon: float = 1e-8
) -> float:
    """
    Compute Masked Symmetric Mean Absolute Percentage Error.
    
    sMAPE = 100 * mean(|pred - target| / (|pred| + |target| + eps))
    """
    valid_count = mask.sum()
    if valid_count == 0:
        return np.nan
    
    numerator = np.abs(pred - target)
    denominator = np.abs(pred) + np.abs(target) + epsilon
    smape_values = (numerator / denominator) * mask
    
    return 100.0 * smape_values.sum() / valid_count


def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray
) -> Dict[str, float]:
    """Compute all metrics at once."""
    return {
        'MAE': masked_mae(pred, target, mask),
        'RMSE': masked_rmse(pred, target, mask),
        'sMAPE': masked_smape(pred, target, mask)
    }


def compute_per_horizon_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray
) -> Dict[int, Dict[str, float]]:
    """
    Compute metrics per horizon.
    
    Args:
        pred: (samples, H, N, D)
        target: (samples, H, N, D)
        mask: (samples, H, N, D)
        
    Returns:
        Dict mapping horizon (1-indexed) to metrics dict
    """
    H = pred.shape[1]
    results = {}
    
    for h in range(H):
        results[h + 1] = compute_all_metrics(
            pred[:, h, :, :],
            target[:, h, :, :],
            mask[:, h, :, :]
        )
    
    return results


def compute_per_pollutant_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    pollutant_names: list = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics per pollutant.
    
    Args:
        pred: (samples, H, N, D)
        target: (samples, H, N, D)
        mask: (samples, H, N, D)
        pollutant_names: List of pollutant names
        
    Returns:
        Dict mapping pollutant name to metrics dict
    """
    if pollutant_names is None:
        pollutant_names = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    
    D = pred.shape[3]
    results = {}
    
    for d in range(D):
        name = pollutant_names[d] if d < len(pollutant_names) else f"target_{d}"
        results[name] = compute_all_metrics(
            pred[:, :, :, d],
            target[:, :, :, d],
            mask[:, :, :, d]
        )
    
    return results


def compute_per_pollutant_report(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    pollutant_names: list = None,
    horizons: Tuple[int, ...] = (1, 6, 12, 24),
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-pollutant metrics aggregated over all horizons, plus MAE at selected horizons.

    Args:
        pred/target/mask: (samples, H, N, D)
        pollutant_names: list of D names
        horizons: 1-indexed horizons to report MAE at

    Returns:
        Dict pollutant -> {MAE, RMSE, sMAPE, MAE_h1, MAE_h6, ...}
    """
    if pollutant_names is None:
        pollutant_names = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}, mask={mask.shape}")
    if pred.ndim != 4:
        raise ValueError(f"Expected 4D arrays (S,H,N,D); got {pred.shape}")

    H = pred.shape[1]
    D = pred.shape[3]
    results: Dict[str, Dict[str, float]] = {}

    for d in range(D):
        name = pollutant_names[d] if d < len(pollutant_names) else f"target_{d}"
        out = compute_all_metrics(pred[:, :, :, d], target[:, :, :, d], mask[:, :, :, d])

        for h in horizons:
            if h < 1 or h > H:
                raise ValueError(f"Invalid horizon {h}; H={H}")
            out[f"MAE_h{h}"] = masked_mae(
                pred[:, h - 1, :, d],
                target[:, h - 1, :, d],
                mask[:, h - 1, :, d],
            )

        results[name] = out

    return results


def macro_average_per_pollutant(per_pollutant: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Macro-average metrics across pollutants (equal weight per pollutant).
    """
    keys = ["MAE", "RMSE", "sMAPE"]
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(v.get(k, np.nan)) for v in per_pollutant.values()]
        out[f"macro_{k}"] = float(np.nanmean(vals))
    return out


def compute_detailed_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    pollutant_names: list = None
) -> Tuple[Dict, Dict, Dict]:
    """
    Compute overall, per-horizon, and per-pollutant metrics.
    
    Returns:
        overall_metrics, per_horizon_metrics, per_pollutant_metrics
    """
    overall = compute_all_metrics(pred, target, mask)
    per_horizon = compute_per_horizon_metrics(pred, target, mask)
    per_pollutant = compute_per_pollutant_metrics(pred, target, mask, pollutant_names)
    
    return overall, per_horizon, per_pollutant
