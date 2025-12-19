from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_train_history(history: List[dict], out_path: str) -> str:
    if not history:
        return out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    epochs = [int(r["epoch"]) for r in history]
    train_loss = [float(r["train_loss"]) for r in history]
    val_mae = [float(r.get("val_macro_MAE", float("nan"))) for r in history]
    val_rmse = [float(r.get("val_macro_RMSE", float("nan"))) for r in history]
    val_smape = [float(r.get("val_macro_sMAPE", float("nan"))) for r in history]

    best_idx = int(np.nanargmin(val_mae))
    best_epoch = epochs[best_idx]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, "b-", linewidth=2)
    ax.set_title("Train Loss (weighted masked MAE)")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, val_mae, "r-", linewidth=2)
    ax.axvline(best_epoch, color="k", linestyle="--", linewidth=1)
    ax.set_title(f"Val Macro MAE (best@{best_epoch})")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, val_rmse, "g-", linewidth=2)
    ax.axvline(best_epoch, color="k", linestyle="--", linewidth=1)
    ax.set_title("Val Macro RMSE")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, val_smape, "m-", linewidth=2)
    ax.axvline(best_epoch, color="k", linestyle="--", linewidth=1)
    ax.set_title("Val Macro sMAPE")
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_loss_curve(history: List[dict], out_path: str, *, model_display_name: str) -> str:
    """
    Baseline-style training curve: train_loss vs val_loss.
    """
    if not history:
        return out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    epochs = [int(r["epoch"]) for r in history]
    train_loss = [float(r["train_loss"]) for r in history]
    val_loss = [float(r.get("val_loss", float("nan"))) for r in history]
    best_val = float(np.nanmin(val_loss)) if np.any(np.isfinite(val_loss)) else float("nan")

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, "b-", label="Train Loss", linewidth=2)
    plt.plot(epochs, val_loss, "r-", label="Val Loss", linewidth=2)
    if np.isfinite(best_val):
        plt.axhline(y=best_val, color="g", linestyle="--", label=f"Best Val: {best_val:.4f}")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (Masked MAE)", fontsize=12)
    plt.title(f"{model_display_name} Training Curve", fontsize=14)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_mae_vs_horizon(per_horizon: Dict[int, Dict[str, float]], out_path: str, *, model_display_name: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    horizons = sorted(per_horizon.keys())
    mae_by_h = [float(per_horizon[h]["MAE"]) for h in horizons]
    plt.figure(figsize=(10, 6))
    plt.plot(horizons, mae_by_h, "b-o", linewidth=2, markersize=4)
    plt.xlabel("Forecast Horizon (hours)", fontsize=12)
    plt.ylabel("MAE", fontsize=12)
    plt.title(f"{model_display_name} - MAE vs Forecast Horizon", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_sample_predictions(
    pred: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    out_path: str,
    *,
    pollutant_idx: int = 0,
    pollutant_name: str = "PM2.5",
    model_display_name: str,
    n_samples: int = 3,
) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    horizons = np.arange(1, pred.shape[1] + 1)

    rng = np.random.default_rng(42)
    k = min(int(n_samples), int(pred.shape[0]))
    sample_indices = [int(i) for i in rng.choice(pred.shape[0], size=k, replace=False)]

    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(12, 4 * len(sample_indices)))
    if len(sample_indices) == 1:
        axes = [axes]

    for ax, sample_idx in zip(axes, sample_indices):
        p = pred[sample_idx, :, :, pollutant_idx]  # (H,N)
        t = y[sample_idx, :, :, pollutant_idx]  # (H,N)
        m = mask[sample_idx, :, :, pollutant_idx]  # (H,N)

        denom = m.sum(axis=1)  # (H,)
        denom = np.where(denom > 0, denom, np.nan)
        pred_avg = (p * m).sum(axis=1) / denom
        true_avg = (t * m).sum(axis=1) / denom

        ax.plot(horizons, pred_avg, "b-o", label="Prediction", linewidth=2, markersize=4)
        ax.plot(horizons, true_avg, "r-s", label="Ground Truth", linewidth=2, markersize=4)
        ax.set_xlabel("Forecast Horizon (hours)", fontsize=11)
        ax.set_ylabel(f"{pollutant_name} (raw)", fontsize=11)
        ax.set_title(f"Sample {sample_idx} - Station-Masked Average", fontsize=12)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{model_display_name} - {pollutant_name} Predictions", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_mae_by_pollutant(per_pollutant: Dict[str, Dict[str, float]], out_path: str, *, model_display_name: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    names = list(per_pollutant.keys())
    maes = [float(per_pollutant[n]["MAE"]) for n in names]

    plt.figure(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = plt.bar(names, maes, color=colors, edgecolor="black")

    y_pad = max(maes) * 0.01 if maes else 0.0
    for bar, val in zip(bars, maes):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            float(bar.get_height()) + float(y_pad),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xlabel("Pollutant", fontsize=12)
    plt.ylabel("MAE", fontsize=12)
    plt.title(f"{model_display_name} - MAE by Pollutant", fontsize=14)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_mae_by_pollutant_horizons(per_pollutant: Dict[str, Dict[str, float]], out_path: str, *, horizons=(1, 6, 12, 24)) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    names = list(per_pollutant.keys())
    H = list(horizons)
    vals = np.array([[float(per_pollutant[n][f"MAE_h{h}"]) for h in H] for n in names], dtype=np.float32)

    x = np.arange(len(names))
    width = 0.18
    plt.figure(figsize=(12, 5))
    for i, h in enumerate(H):
        plt.bar(x + (i - (len(H) - 1) / 2) * width, vals[:, i], width=width, label=f"h{h}")
    plt.xticks(x, names)
    plt.ylabel("MAE (raw units)")
    plt.title("MAE by Pollutant @ Selected Horizons")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_all_models_vs_horizon(
    baseline_metrics_summary_csv: str,
    this_per_horizon: Dict[int, Dict[str, float]],
    out_path: str,
    *,
    metric: str = "MAE",
    model_label: str = "ST-Former",
    split: str = "test",
    models_keep: Optional[List[str]] = None,
) -> str:
    """
    Baseline-style all-models curve plot by reading baseline metrics_summary.csv
    and overlaying the current model curve.
    """
    if not os.path.exists(baseline_metrics_summary_csv):
        return out_path
    try:
        import pandas as pd
    except Exception:
        return out_path

    df = pd.read_csv(baseline_metrics_summary_csv)
    df = df[(df["split"] == split) & (df["pollutant"] == "ALL") & (df["horizon"] != "ALL")].copy()
    if df.empty:
        return out_path
    df["horizon"] = df["horizon"].astype(int)

    models = sorted(df["model"].unique().tolist())
    if models_keep is not None:
        models = [m for m in models_keep if m in models]

    plt.figure(figsize=(10, 6))
    for m in models:
        sub = df[df["model"] == m].sort_values("horizon")
        if len(sub) == 0:
            continue
        plt.plot(sub["horizon"].values, sub[metric].values, marker="o", linewidth=2, label=m)

    horizons = sorted(this_per_horizon.keys())
    vals = [float(this_per_horizon[h][metric]) for h in horizons]
    plt.plot(horizons, vals, marker="o", linewidth=2, label=model_label)

    plt.xlabel("Forecast Horizon (hours)", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.title(f"{metric} vs Forecast Horizon", fontsize=14)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# Backwards-compatible wrappers for older filenames (kept for convenience).
def plot_error_vs_horizon(mae_by_h: List[float], out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    horizons = np.arange(1, len(mae_by_h) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(horizons, mae_by_h, "b-o", linewidth=2, markersize=4)
    plt.xlabel("Forecast Horizon (hours)", fontsize=12)
    plt.ylabel("MAE", fontsize=12)
    plt.title("MAE vs Forecast Horizon", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_prediction_example(
    pred: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    out_path: str,
    *,
    station_idx: int = 0,
    pollutant_idx: int = 0,
    pollutant_name: str = "PM2.5",
) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rng = np.random.default_rng(42)
    sample = int(rng.integers(0, pred.shape[0]))
    horizons = np.arange(1, pred.shape[1] + 1)
    p = pred[sample, :, station_idx, pollutant_idx]
    t = y[sample, :, station_idx, pollutant_idx]
    m = mask[sample, :, station_idx, pollutant_idx].astype(bool)

    plt.figure(figsize=(10, 5))
    plt.plot(horizons[m], t[m], "k-o", label="Ground Truth", linewidth=2, markersize=3)
    plt.plot(horizons, p, "b-", label="Prediction", linewidth=2)
    plt.title(f"Example Forecast (sample={sample}, station={station_idx}, pollutant={pollutant_name})", fontsize=14)
    plt.xlabel("Forecast Horizon (hours)", fontsize=12)
    plt.ylabel("Raw units", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
