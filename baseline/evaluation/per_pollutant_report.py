"""
CSV reporting utilities for PRSA baselines.

Outputs:
  - baseline/results/metrics_per_pollutant.csv
  - baseline/results/metrics_overall.csv
"""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

from .masked_metrics import macro_average_per_pollutant


POLLUTANT_COLUMNS = [
    "model",
    "pollutant",
    "MAE",
    "RMSE",
    "sMAPE",
    "MAE_h1",
    "MAE_h6",
    "MAE_h12",
    "MAE_h24",
]


def _upsert_csv(path: str, rows: List[Dict], key_cols: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_df = pd.DataFrame(rows)
    if new_df.empty:
        return

    if os.path.exists(path):
        old_df = pd.read_csv(path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        # Keep last occurrence per key
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
    else:
        combined = new_df

    combined.to_csv(path, index=False)


def write_metrics_per_pollutant(results: Dict, results_dir: str) -> str:
    """
    Write/Upsert per-pollutant metrics for a single model evaluation.

    Expects:
      results['model']
      results['per_pollutant'] with keys MAE/RMSE/sMAPE and MAE_h{1,6,12,24}
    """
    model = results["model"]
    per_pollutant = results["per_pollutant"]

    rows: List[Dict] = []
    for pollutant, metrics in per_pollutant.items():
        row = {"model": model, "pollutant": pollutant}
        for col in POLLUTANT_COLUMNS:
            if col in ("model", "pollutant"):
                continue
            row[col] = float(metrics.get(col, float("nan")))
        rows.append(row)

    out_path = os.path.join(results_dir, "metrics_per_pollutant.csv")
    _upsert_csv(out_path, rows, key_cols=["model", "pollutant"])
    return out_path


def write_metrics_overall(results: Dict, results_dir: str) -> str:
    """
    Write/Upsert overall + macro-average metrics for a single model evaluation.
    """
    model = results["model"]
    overall = results["overall"]
    per_pollutant = results["per_pollutant"]
    macro = macro_average_per_pollutant(per_pollutant)

    row = {
        "model": model,
        "MAE": float(overall.get("MAE")),
        "RMSE": float(overall.get("RMSE")),
        "sMAPE": float(overall.get("sMAPE")),
        **macro,
    }

    # Also include overall per-horizon MAE at key horizons when present.
    per_h = results.get("per_horizon", {})
    for h in (1, 6, 12, 24):
        if h in per_h:
            row[f"MAE_h{h}"] = float(per_h[h].get("MAE"))

    out_path = os.path.join(results_dir, "metrics_overall.csv")
    _upsert_csv(out_path, [row], key_cols=["model"])
    return out_path


def plot_seasonal_naive_sanity(
    pred,
    target,
    mask,
    results_dir: str,
    *,
    pollutant_idx: int = 0,
    pollutant_name: str = "PM2.5",
    station_idx: int = 0,
    sample_indices: List[int] | None = None,
) -> str:
    """
    Save a small sanity plot: true vs seasonal naive for PM2.5 at one station across 24 horizons.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if sample_indices is None:
        rng = np.random.default_rng(42)
        k = min(3, pred.shape[0])
        sample_indices = [int(i) for i in rng.choice(pred.shape[0], size=k, replace=False)]

    horizons = np.arange(1, pred.shape[1] + 1)
    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(10, 3.2 * len(sample_indices)), sharex=True)
    if len(sample_indices) == 1:
        axes = [axes]

    for ax, i in zip(axes, sample_indices):
        y = target[i, :, station_idx, pollutant_idx]
        m = mask[i, :, station_idx, pollutant_idx].astype(bool)
        p = pred[i, :, station_idx, pollutant_idx]

        ax.plot(horizons[m], y[m], "k-o", label="true", linewidth=2, markersize=3)
        ax.plot(horizons, p, "b-", label="seasonal_naive", linewidth=2)
        ax.set_title(f"sample={i}, station={station_idx}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("horizon (hours)")
    axes[0].set_ylabel(f"{pollutant_name} (raw)")
    plt.tight_layout()

    out_path = os.path.join(results_dir, "plots", "seasonal_naive_sanity.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
