from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from model.metrics.masked_metrics import horizon_variation_check, macro_average, per_pollutant_metrics
from model.evaluation.plots import plot_error_vs_horizon, plot_prediction_example

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def evaluate_model(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    pollutant_names: List[str],
    results_dir: str,
    model_name: str = "wgdgtm",
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

    # Horizon variation check: fail fast on collapsed constant outputs.
    hv = horizon_variation_check(pred, eps=1e-3)

    per_p = per_pollutant_metrics(pred, y, m, pollutant_names)
    macro = macro_average(per_p)

    # Per-horizon macro MAE curve
    mae_by_h = []
    for h in range(pred.shape[1]):
        per_p_h = per_pollutant_metrics(
            pred[:, h : h + 1],
            y[:, h : h + 1],
            m[:, h : h + 1],
            pollutant_names,
            horizons=(1,),
        )
        macro_h = macro_average(per_p_h)
        mae_by_h.append(macro_h["macro_MAE"])

    metrics_dir = os.path.join(results_dir, "metrics")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    rows = []
    for pollutant, vals in per_p.items():
        row = {"model": model_name, "pollutant": pollutant, **vals}
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(metrics_dir, "metrics_per_pollutant.csv"), index=False)
    pd.DataFrame([{"model": model_name, **macro, "horizon_variation": hv}]).to_csv(
        os.path.join(metrics_dir, "macro_avg_metrics.csv"), index=False
    )

    plot_error_vs_horizon(mae_by_h, os.path.join(plots_dir, "error_vs_horizon.png"))
    plot_prediction_example(
        pred,
        y,
        m,
        os.path.join(plots_dir, "prediction_vs_truth.png"),
        station_idx=0,
        pollutant_idx=0,
        pollutant_name=pollutant_names[0],
    )

    return {"per_pollutant": per_p, "macro": macro, "horizon_variation": hv}
