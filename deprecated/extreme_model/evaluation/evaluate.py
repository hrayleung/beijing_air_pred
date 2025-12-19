from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from extreme_model.metrics.masked_metrics import (
    compute_all_metrics,
    compute_per_horizon_metrics,
    compute_per_pollutant_metrics,
    compute_per_pollutant_report,
    horizon_variation_check,
    macro_average_per_pollutant,
)
from extreme_model.evaluation.plots import (
    plot_all_models_vs_horizon,
    plot_mae_by_pollutant,
    plot_mae_vs_horizon,
    plot_sample_predictions,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _upsert_csv(path: str, df: pd.DataFrame, *, key_cols: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if df.empty:
        return
    if os.path.exists(path):
        old = pd.read_csv(path)
        combined = pd.concat([old, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
    else:
        combined = df
    combined.to_csv(path, index=False)


def evaluate_model(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    pollutant_names: List[str],
    results_dir: str,
    model_name: str = "stformer",
    model_display_name: Optional[str] = None,
    split: str = "test",
    baseline_results_dir: str = "baseline/results",
) -> Dict:
    model.eval()
    preds = []
    ys = []
    masks = []
    with torch.no_grad():
        it = loader
        if tqdm is not None:
            it = tqdm(loader, desc="[eval] iter", total=len(loader), leave=False, dynamic_ncols=True)
        for batch in it:
            X = batch["X"].to(device)
            pred = model(X).detach().cpu().numpy().astype(np.float32)
            preds.append(pred)
            ys.append(batch["Y"].numpy().astype(np.float32))
            masks.append(batch["Y_mask"].numpy().astype(np.float32))

    pred = np.concatenate(preds, axis=0)
    y = np.concatenate(ys, axis=0)
    m = np.concatenate(masks, axis=0)

    hv = horizon_variation_check(pred, eps=1e-3)

    overall = compute_all_metrics(pred, y, m)
    per_horizon = compute_per_horizon_metrics(pred, y, m)
    per_pollutant = compute_per_pollutant_report(pred, y, m, pollutant_names)
    per_pollutant_simple = compute_per_pollutant_metrics(pred, y, m, pollutant_names)
    macro = macro_average_per_pollutant(per_pollutant_simple)

    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    display_name = model_display_name or model_name

    # Baseline-compatible CSV outputs (same schema as baseline/results).
    # metrics_summary.csv: (model, split, pollutant, horizon, MAE, RMSE, sMAPE)
    summary_rows = [
        {
            "model": model_name,
            "split": split,
            "pollutant": "ALL",
            "horizon": "ALL",
            **overall,
        }
    ]
    for h in sorted(per_horizon.keys()):
        summary_rows.append(
            {
                "model": model_name,
                "split": split,
                "pollutant": "ALL",
                "horizon": int(h),
                **per_horizon[h],
            }
        )
    for pollutant in pollutant_names:
        summary_rows.append(
            {
                "model": model_name,
                "split": split,
                "pollutant": pollutant,
                "horizon": "ALL",
                "MAE": float(per_pollutant[pollutant]["MAE"]),
                "RMSE": float(per_pollutant[pollutant]["RMSE"]),
                "sMAPE": float(per_pollutant[pollutant]["sMAPE"]),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    _upsert_csv(os.path.join(results_dir, "metrics_summary.csv"), summary_df, key_cols=["model", "split", "pollutant", "horizon"])

    # metrics_per_pollutant.csv: (model, pollutant, MAE, RMSE, sMAPE, MAE_h1/h6/h12/h24)
    per_pollutant_rows = []
    for pollutant in pollutant_names:
        per_pollutant_rows.append({"model": model_name, "pollutant": pollutant, **per_pollutant[pollutant]})
    per_pollutant_df = pd.DataFrame(per_pollutant_rows)
    _upsert_csv(os.path.join(results_dir, "metrics_per_pollutant.csv"), per_pollutant_df, key_cols=["model", "pollutant"])

    # metrics_overall.csv: (model, MAE, RMSE, sMAPE, macro_MAE, macro_RMSE, macro_sMAPE, MAE_h1/h6/h12/h24)
    overall_row = {"model": model_name, **overall, **macro}
    for h in (1, 6, 12, 24):
        if h in per_horizon:
            overall_row[f"MAE_h{h}"] = float(per_horizon[h]["MAE"])
    overall_df = pd.DataFrame([overall_row])
    _upsert_csv(os.path.join(results_dir, "metrics_overall.csv"), overall_df, key_cols=["model"])

    # Optional: write a combined model_comparison.csv against baseline results (if present).
    comparison_path = os.path.join(results_dir, "model_comparison.csv")
    base_comparison = os.path.join(baseline_results_dir, "model_comparison.csv")
    this_row = {
        "Model": model_name,
        "MAE": float(overall["MAE"]),
        "RMSE": float(overall["RMSE"]),
        "sMAPE": float(overall["sMAPE"]),
        "MAE_h1": float(per_horizon[1]["MAE"]),
        "MAE_h6": float(per_horizon[6]["MAE"]),
        "MAE_h12": float(per_horizon[12]["MAE"]),
        "MAE_h24": float(per_horizon[24]["MAE"]),
    }
    if os.path.exists(base_comparison):
        base_df = pd.read_csv(base_comparison)
        base_df = base_df[base_df["Model"] != model_name]
        cmp_df = pd.concat([base_df, pd.DataFrame([this_row])], ignore_index=True)
    else:
        cmp_df = pd.DataFrame([this_row])
    cmp_df = cmp_df.sort_values("MAE")
    cmp_df.to_csv(comparison_path, index=False)

    # Plots (baseline-style names under plots/).
    plot_mae_vs_horizon(
        per_horizon,
        os.path.join(plots_dir, f"{model_name}_mae_vs_horizon.png"),
        model_display_name=display_name,
    )
    plot_mae_by_pollutant(
        per_pollutant,
        os.path.join(plots_dir, f"{model_name}_mae_by_pollutant.png"),
        model_display_name=display_name,
    )
    plot_sample_predictions(
        pred,
        y,
        m,
        os.path.join(plots_dir, f"{model_name}_sample_predictions.png"),
        model_display_name=display_name,
        pollutant_idx=0,
        pollutant_name=pollutant_names[0],
    )

    # Optional: comparison plots to baseline curves (if baseline metrics_summary.csv exists).
    baseline_summary = os.path.join(baseline_results_dir, "metrics_summary.csv")
    plot_all_models_vs_horizon(
        baseline_summary,
        per_horizon,
        os.path.join(plots_dir, "all_models_mae_vs_horizon.png"),
        metric="MAE",
        model_label=display_name,
    )
    plot_all_models_vs_horizon(
        baseline_summary,
        per_horizon,
        os.path.join(plots_dir, "all_models_rmse_vs_horizon.png"),
        metric="RMSE",
        model_label=display_name,
    )

    # Extra diagnostics for reproducibility/debug.
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    pd.DataFrame([{"model": model_name, **macro, "horizon_variation": hv}]).to_csv(
        os.path.join(metrics_dir, "macro_avg_metrics.csv"), index=False
    )

    return {
        "model": model_name,
        "split": split,
        "overall": overall,
        "per_horizon": per_horizon,
        "per_pollutant": per_pollutant,
        "macro": macro,
        "horizon_variation": hv,
    }
