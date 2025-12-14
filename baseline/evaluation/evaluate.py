"""
Unified evaluation entrypoint.
"""
import os
import json
from typing import Dict, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .masked_metrics import compute_detailed_metrics, compute_per_pollutant_report
from .per_pollutant_report import write_metrics_overall, write_metrics_per_pollutant


def _json_default(obj):
    """
    JSON serializer for objects not serializable by default `json` code.
    Handles numpy scalars/arrays (and torch tensors when available).
    """
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _validate_pred_target_mask_shapes(pred: np.ndarray, target: np.ndarray, mask: np.ndarray):
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError(
            "pred/target/mask shape mismatch: "
            f"pred={pred.shape}, target={target.shape}, mask={mask.shape}"
        )
    if pred.ndim != 4:
        raise ValueError(f"Expected 4D arrays (S,H,N,D); got pred.ndim={pred.ndim} with shape={pred.shape}")
    _, H, N, D = pred.shape
    if (H, N, D) != (24, 12, 6):
        raise ValueError(f"Expected (H,N,D)=(24,12,6); got {(H, N, D)} with shape={pred.shape}")


def _inverse_transform_targets(
    arr: np.ndarray,
    target_scaler,
) -> np.ndarray:
    """
    Inverse the per-pollutant RobustScaler used in preprocessing v2.1 when `scale_targets=true`.
    Expected `target_scaler.center_` and `target_scaler.scale_` shaped (D,).
    """
    if target_scaler is None:
        raise ValueError("scale_targets is True but target_scaler is None")
    center = np.asarray(getattr(target_scaler, "center_", None))
    scale = np.asarray(getattr(target_scaler, "scale_", None))
    if center.ndim != 1 or scale.ndim != 1 or center.shape != scale.shape:
        raise ValueError(f"Invalid target_scaler params: center={center.shape}, scale={scale.shape}")
    if arr.shape[-1] != center.shape[0]:
        raise ValueError(f"Target dim mismatch: arr.D={arr.shape[-1]} vs scaler.D={center.shape[0]}")
    return (arr * scale.reshape(1, 1, 1, -1)) + center.reshape(1, 1, 1, -1)


def _check_horizon_variation(pred: np.ndarray, epsilon: float = 1e-3) -> Dict[str, float]:
    """
    Sanity check: ensure predictions vary across horizon.
    Returns per-pollutant deltas between h=1 and h=24 (mean abs diff across samples/stations).
    """
    deltas = {}
    for d in range(pred.shape[3]):
        deltas[str(d)] = float(np.mean(np.abs(pred[:, 0, :, d] - pred[:, -1, :, d])))
    if max(deltas.values()) <= float(epsilon):
        raise ValueError(
            "Predictions appear constant across horizons "
            f"(max mean|h1-h24|={max(deltas.values()):.6g} <= {epsilon}). "
            "This usually indicates an output reshape/broadcast bug."
        )
    return deltas


def evaluate_predictions(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    model_name: str,
    split: str = "test",
    pollutant_names: list = None,
    results_dir: str = "baseline/results",
    *,
    scale_targets: bool = False,
    target_scaler=None,
    pred_is_scaled: bool = False,
    expected_shape: Optional[tuple] = None,
    require_horizon_variation: bool = False,
    horizon_variation_epsilon: float = 1e-3,
) -> Dict:
    """
    Evaluate predictions and save results.
    
    Args:
        pred: (samples, H, N, D) predictions in RAW units
        target: (samples, H, N, D) ground truth in RAW units
        mask: (samples, H, N, D) validity mask
        model_name: Name of the model
        split: Data split name
        pollutant_names: List of pollutant names
        results_dir: Directory to save results
        
    Returns:
        Dict with all metrics
    """
    if pollutant_names is None:
        pollutant_names = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

    _validate_pred_target_mask_shapes(pred, target, mask)
    if expected_shape is not None and tuple(pred.shape) != tuple(expected_shape):
        raise ValueError(f"Unexpected pred shape {pred.shape}; expected {expected_shape}")

    # Ensure all metrics are computed in RAW target units.
    if scale_targets:
        target_raw = _inverse_transform_targets(target, target_scaler)
        pred_raw = _inverse_transform_targets(pred, target_scaler) if pred_is_scaled else pred
    else:
        target_raw = target
        pred_raw = pred

    # Sanity check for multi-horizon outputs (fails fast on broadcast/reshape bugs).
    if require_horizon_variation:
        _check_horizon_variation(pred_raw, epsilon=horizon_variation_epsilon)
    
    # Compute metrics
    overall, per_horizon, per_pollutant = compute_detailed_metrics(
        pred_raw, target_raw, mask, pollutant_names
    )

    # Replace per-pollutant metrics with report-ready version (adds MAE_h1/h6/h12/h24).
    per_pollutant = compute_per_pollutant_report(pred_raw, target_raw, mask, pollutant_names)
    
    # Build results dict
    results = {
        'model': model_name,
        'split': split,
        'overall': overall,
        'per_horizon': per_horizon,
        'per_pollutant': per_pollutant
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {model_name} on {split}")
    print(f"{'='*60}")
    print(f"Overall MAE:  {overall['MAE']:.4f}")
    print(f"Overall RMSE: {overall['RMSE']:.4f}")
    print(f"Overall sMAPE: {overall['sMAPE']:.2f}%")
    
    print(f"\nPer-Horizon MAE (h=1,6,12,24):")
    for h in [1, 6, 12, 24]:
        print(f"  h={h:2d}: {per_horizon[h]['MAE']:.4f}")
    
    print(f"\nPer-Pollutant MAE:")
    for name, metrics in per_pollutant.items():
        print(f"  {name:8s}: {metrics['MAE']:.4f}")
    
    # Save to CSV
    save_metrics_to_csv(results, results_dir)

    # Save per-pollutant + macro-average CSVs
    write_metrics_per_pollutant(results, results_dir)
    write_metrics_overall(results, results_dir)
    
    # Save detailed JSON
    save_detailed_json(results, results_dir)
    
    # Save per-model plots
    save_model_plots(results, pred_raw, target_raw, mask, pollutant_names, results_dir)
    
    return results


def save_detailed_json(results: Dict, results_dir: str):
    """Save detailed metrics to JSON file."""
    json_dir = os.path.join(results_dir, 'logs')
    os.makedirs(json_dir, exist_ok=True)
    
    model_name = results['model']
    split = results['split']
    
    # Convert per_horizon keys to strings for JSON
    json_results = {
        'model': model_name,
        'split': split,
        'overall': results['overall'],
        'per_horizon': {str(k): v for k, v in results['per_horizon'].items()},
        'per_pollutant': results['per_pollutant']
    }
    
    json_path = os.path.join(json_dir, f'{model_name}_{split}_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=_json_default)
    print(f"Detailed metrics saved to {json_path}")


def save_model_plots(results: Dict, pred: np.ndarray, target: np.ndarray, 
                     mask: np.ndarray, pollutant_names: list, results_dir: str):
    """Save evaluation plots for a single model."""
    plot_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    model_name = results['model']
    
    # 1. MAE vs Horizon plot
    _plot_mae_vs_horizon(results, model_name, plot_dir)
    
    # 2. Per-pollutant bar chart
    _plot_per_pollutant_mae(results, model_name, pollutant_names, plot_dir)
    
    # 3. Sample prediction plots (3 random samples for PM2.5)
    _plot_sample_predictions(pred, target, mask, model_name, pollutant_names, plot_dir)


def _plot_mae_vs_horizon(results: Dict, model_name: str, plot_dir: str):
    """Plot MAE vs forecast horizon."""
    horizons = sorted(results['per_horizon'].keys())
    mae_values = [results['per_horizon'][h]['MAE'] for h in horizons]
    
    plt.figure(figsize=(10, 6))
    plt.plot(horizons, mae_values, 'b-o', linewidth=2, markersize=4)
    plt.xlabel('Forecast Horizon (hours)', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title(f'{model_name.upper()} - MAE vs Forecast Horizon', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(plot_dir, f'{model_name}_mae_vs_horizon.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"MAE vs horizon plot saved to {save_path}")


def _plot_per_pollutant_mae(results: Dict, model_name: str, pollutant_names: list, plot_dir: str):
    """Plot MAE per pollutant."""
    mae_values = [results['per_pollutant'][p]['MAE'] for p in pollutant_names]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(pollutant_names)))
    bars = plt.bar(pollutant_names, mae_values, color=colors, edgecolor='black')
    
    # Add value labels on bars
    for bar, val in zip(bars, mae_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Pollutant', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title(f'{model_name.upper()} - MAE by Pollutant', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    save_path = os.path.join(plot_dir, f'{model_name}_mae_by_pollutant.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"MAE by pollutant plot saved to {save_path}")


def _plot_sample_predictions(pred: np.ndarray, target: np.ndarray, mask: np.ndarray,
                             model_name: str, pollutant_names: list, plot_dir: str):
    """Plot sample predictions vs ground truth."""
    np.random.seed(42)
    n_samples = min(3, pred.shape[0])
    sample_indices = np.random.choice(pred.shape[0], n_samples, replace=False)
    
    H = pred.shape[1]
    horizons = np.arange(1, H + 1)
    
    # Plot for PM2.5 (index 0)
    pollutant_idx = 0
    pollutant_name = pollutant_names[pollutant_idx]
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i, (ax, sample_idx) in enumerate(zip(axes, sample_indices)):
        # Average across stations for this sample
        pred_avg = pred[sample_idx, :, :, pollutant_idx].mean(axis=1)
        target_avg = target[sample_idx, :, :, pollutant_idx].mean(axis=1)
        mask_avg = mask[sample_idx, :, :, pollutant_idx].mean(axis=1)
        
        ax.plot(horizons, pred_avg, 'b-o', label='Prediction', linewidth=2, markersize=4)
        ax.plot(horizons, target_avg, 'r-s', label='Ground Truth', linewidth=2, markersize=4)
        
        ax.set_xlabel('Forecast Horizon (hours)', fontsize=11)
        ax.set_ylabel(f'{pollutant_name} (μg/m³)', fontsize=11)
        ax.set_title(f'Sample {sample_idx} - Station Average', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name.upper()} - {pollutant_name} Predictions', fontsize=14, y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(plot_dir, f'{model_name}_sample_predictions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sample predictions plot saved to {save_path}")


def save_metrics_to_csv(results: Dict, results_dir: str):
    """Save metrics to CSV file."""
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "metrics_summary.csv")
    
    rows = []
    model = results['model']
    split = results['split']
    
    # Overall row
    rows.append({
        'model': model,
        'split': split,
        'pollutant': 'ALL',
        'horizon': 'ALL',
        'MAE': results['overall']['MAE'],
        'RMSE': results['overall']['RMSE'],
        'sMAPE': results['overall']['sMAPE']
    })
    
    # Per-horizon rows
    for h, metrics in results['per_horizon'].items():
        rows.append({
            'model': model,
            'split': split,
            'pollutant': 'ALL',
            'horizon': h,
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'sMAPE': metrics['sMAPE']
        })
    
    # Per-pollutant rows
    for pollutant, metrics in results['per_pollutant'].items():
        rows.append({
            'model': model,
            'split': split,
            'pollutant': pollutant,
            'horizon': 'ALL',
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'sMAPE': metrics['sMAPE']
        })
    
    df = pd.DataFrame(rows)
    
    # Append to existing or create new
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        # Remove existing entries for this model/split
        existing = existing[~((existing['model'] == model) & (existing['split'] == split))]
        df = pd.concat([existing, df], ignore_index=True)
    
    df.to_csv(csv_path, index=False)
    print(f"\nMetrics saved to {csv_path}")
