"""
Plotting utilities for evaluation.
"""
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .masked_metrics import compute_per_horizon_metrics


def plot_error_vs_horizon(
    results_dict: Dict[str, Dict],
    metric: str = 'MAE',
    save_path: str = None,
    title: str = None
):
    """
    Plot error vs horizon for multiple models.
    
    Args:
        results_dict: Dict mapping model_name to evaluation results
        metric: 'MAE', 'RMSE', or 'sMAPE'
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    for model_name, results in results_dict.items():
        horizons = sorted(results['per_horizon'].keys())
        values = [results['per_horizon'][h][metric] for h in horizons]
        plt.plot(horizons, values, marker='o', label=model_name, linewidth=2)
    
    plt.xlabel('Forecast Horizon (hours)', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.title(title or f'{metric} vs Forecast Horizon', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()


def plot_prediction_example(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    sample_idx: int = 0,
    station_idx: int = 0,
    pollutant_idx: int = 0,
    pollutant_name: str = "PM2.5",
    save_path: str = None
):
    """
    Plot prediction vs ground truth for a single example.
    
    Args:
        pred: (samples, H, N, D)
        target: (samples, H, N, D)
        mask: (samples, H, N, D)
        sample_idx: Sample index
        station_idx: Station index
        pollutant_idx: Pollutant index
        pollutant_name: Name for title
        save_path: Path to save figure
    """
    H = pred.shape[1]
    horizons = np.arange(1, H + 1)
    
    pred_vals = pred[sample_idx, :, station_idx, pollutant_idx]
    target_vals = target[sample_idx, :, station_idx, pollutant_idx]
    mask_vals = mask[sample_idx, :, station_idx, pollutant_idx]
    
    plt.figure(figsize=(10, 5))
    
    # Plot prediction
    plt.plot(horizons, pred_vals, 'b-o', label='Prediction', linewidth=2)
    
    # Plot target (only valid points)
    valid_h = horizons[mask_vals == 1]
    valid_t = target_vals[mask_vals == 1]
    plt.plot(valid_h, valid_t, 'r-s', label='Ground Truth', linewidth=2)
    
    # Mark missing points
    missing_h = horizons[mask_vals == 0]
    if len(missing_h) > 0:
        plt.scatter(missing_h, [0] * len(missing_h), c='gray', marker='x', 
                   s=100, label='Missing', zorder=5)
    
    plt.xlabel('Forecast Horizon (hours)', fontsize=12)
    plt.ylabel(f'{pollutant_name} (μg/m³)', fontsize=12)
    plt.title(f'{pollutant_name} Prediction vs Truth (Sample {sample_idx}, Station {station_idx})', 
              fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()


def plot_per_pollutant_comparison(
    results_dict: Dict[str, Dict],
    metric: str = 'MAE',
    save_path: str = None
):
    """
    Bar plot comparing models across pollutants.
    """
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    models = list(results_dict.keys())
    
    x = np.arange(len(pollutants))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, model in enumerate(models):
        values = [results_dict[model]['per_pollutant'].get(p, {}).get(metric, 0) 
                  for p in pollutants]
        ax.bar(x + i * width, values, width, label=model)
    
    ax.set_xlabel('Pollutant', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{metric} by Pollutant', fontsize=14)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(pollutants)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
