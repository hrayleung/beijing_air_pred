from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def plot_error_vs_horizon(mae_by_h: List[float], out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    horizons = np.arange(1, len(mae_by_h) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(horizons, mae_by_h, "b-o", linewidth=2, markersize=3)
    plt.xlabel("Horizon (hours)")
    plt.ylabel("Macro MAE")
    plt.title("Macro MAE vs Horizon")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_prediction_example(pred: np.ndarray, y: np.ndarray, mask: np.ndarray, out_path: str, *, station_idx: int = 0, pollutant_idx: int = 0, pollutant_name: str = "PM2.5") -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rng = np.random.default_rng(42)
    sample = int(rng.integers(0, pred.shape[0]))
    horizons = np.arange(1, pred.shape[1] + 1)
    p = pred[sample, :, station_idx, pollutant_idx]
    t = y[sample, :, station_idx, pollutant_idx]
    m = mask[sample, :, station_idx, pollutant_idx].astype(bool)

    plt.figure(figsize=(10, 5))
    plt.plot(horizons[m], t[m], "k-o", label="truth", linewidth=2, markersize=3)
    plt.plot(horizons, p, "r-", label="pred", linewidth=2)
    plt.title(f"Example forecast — sample={sample}, station={station_idx}, pollutant={pollutant_name}")
    plt.xlabel("Horizon (hours)")
    plt.ylabel("Raw units")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path

