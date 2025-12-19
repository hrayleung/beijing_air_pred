#!/usr/bin/env python3
"""
Special-case evaluation for Beijing air quality forecasting.

Adds two analyses that are commonly requested in experiment sections:
  1) Holiday / weekend vs normal days performance
  2) Robustness under increased input missingness (simulated)

Outputs (default):
  - baseline/results/special_cases/holiday_weekend_metrics.csv
  - baseline/results/special_cases/missing_rate_metrics.csv
  - baseline/results/special_cases/plots/*.png

Run (recommended env has numpy/torch/lightgbm):
  /disk_n/conda/envs/dl/bin/python scripts/special_case_analysis.py
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Ensure the project root is on sys.path so `baseline/` and `model/` are importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from baseline.data.loader_lgbm import align_lgbm_predictions_to_npz, load_lgbm_data
from baseline.data.loader_npz import (
    load_feature_list,
    load_metadata,
    load_npz_data,
    load_scaler,
    load_target_list,
)
from baseline.evaluation.masked_metrics import compute_per_pollutant_report, macro_average_per_pollutant
from baseline.models import LightGBMMultiHorizon, NaivePersistence
from baseline.models.lstm_seq2seq import LSTMDirect
from baseline.models.tcn import TCN
from model.data.prsa_npz_dataset import load_feature_and_target_lists
from model.data.scalers import input_scaler_center_scale, load_input_scaler
from model.graphs.adjacency import add_self_loops, load_static_adjacency, row_normalize, to_torch_adjacency
from model.models.wgdgtm import WGDGTM, WGDTMConfig


# A lightweight holiday calendar for China (major public holidays) within 2013-03..2017-02.
# Note: This is intentionally simple; if you need the *official* adjusted-workday calendar,
# replace these ranges with your course/organization's calendar.
HOLIDAY_RANGES: List[Tuple[str, str, str]] = [
    # 2013 (dataset starts 2013-03-01)
    ("2013-04-04", "2013-04-06", "Qingming"),
    ("2013-04-29", "2013-05-01", "Labor Day"),
    ("2013-10-01", "2013-10-07", "National Day"),
    # 2014
    ("2014-01-01", "2014-01-03", "New Year"),
    ("2014-01-31", "2014-02-06", "Spring Festival"),
    ("2014-04-05", "2014-04-07", "Qingming"),
    ("2014-05-01", "2014-05-03", "Labor Day"),
    ("2014-10-01", "2014-10-07", "National Day"),
    # 2015
    ("2015-01-01", "2015-01-03", "New Year"),
    ("2015-02-18", "2015-02-24", "Spring Festival"),
    ("2015-04-04", "2015-04-06", "Qingming"),
    ("2015-05-01", "2015-05-03", "Labor Day"),
    ("2015-10-01", "2015-10-07", "National Day"),
    # 2016
    ("2016-01-01", "2016-01-03", "New Year"),
    ("2016-02-07", "2016-02-13", "Spring Festival"),
    ("2016-04-02", "2016-04-04", "Qingming"),
    ("2016-04-30", "2016-05-02", "Labor Day"),
    ("2016-10-01", "2016-10-07", "National Day"),
    # 2017 (dataset ends 2017-02-28)
    ("2017-01-01", "2017-01-03", "New Year"),
    ("2017-01-27", "2017-02-02", "Spring Festival"),
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _build_holiday_dates(ranges: Sequence[Tuple[str, str, str]]) -> np.ndarray:
    days: List[np.datetime64] = []
    for start_s, end_s, _name in ranges:
        start = np.datetime64(start_s, "D")
        end = np.datetime64(end_s, "D")
        if end < start:
            raise ValueError(f"Invalid holiday range: {start_s}..{end_s}")
        cur = start
        while cur <= end:
            days.append(cur)
            cur = cur + np.timedelta64(1, "D")
    return np.array(sorted(set(days)), dtype="datetime64[D]")


def _forecast_datetimes(datetime_origins: np.ndarray, horizon: int) -> np.ndarray:
    if datetime_origins.ndim != 1:
        raise ValueError(f"Expected datetime_origins shape (S,), got {datetime_origins.shape}")
    offsets = (np.arange(1, horizon + 1, dtype=np.int64) * np.timedelta64(1, "h")).reshape(1, -1)
    return datetime_origins.reshape(-1, 1) + offsets  # (S, H)


def _weekend_mask(forecast_dt: np.ndarray) -> np.ndarray:
    flat = pd.to_datetime(forecast_dt.reshape(-1))
    # Monday=0 ... Sunday=6
    dow = flat.dayofweek.to_numpy()
    is_weekend = (dow >= 5).reshape(forecast_dt.shape)
    return is_weekend


def _macro_metrics_from_per_pollutant(per_pollutant: Dict[str, Dict[str, float]], *, horizons: Sequence[int]) -> Dict[str, float]:
    macro = macro_average_per_pollutant(per_pollutant)
    out = dict(macro)
    for h in horizons:
        vals = [float(per_pollutant[p].get(f"MAE_h{h}", np.nan)) for p in per_pollutant.keys()]
        out[f"macro_MAE_h{h}"] = float(np.nanmean(vals))
    return out


def _compute_subset_macro_metrics(
    pred: np.ndarray,
    y: np.ndarray,
    y_mask: np.ndarray,
    *,
    pollutant_names: List[str],
    time_mask: np.ndarray,
    sample_mask: Optional[np.ndarray] = None,
    horizons: Sequence[int] = (1, 6, 12, 24),
) -> Dict[str, float]:
    if pred.shape != y.shape or pred.shape != y_mask.shape:
        raise ValueError(f"pred/y/y_mask shape mismatch: {pred.shape} vs {y.shape} vs {y_mask.shape}")
    if time_mask.shape != pred.shape[:2]:
        raise ValueError(f"time_mask shape mismatch: time_mask={time_mask.shape} vs pred[:2]={pred.shape[:2]}")

    mask = y_mask.astype(np.float32, copy=False)
    if sample_mask is not None:
        if sample_mask.shape != (pred.shape[0],):
            raise ValueError(f"sample_mask shape mismatch: {sample_mask.shape} vs {(pred.shape[0],)}")
        mask = mask * sample_mask.reshape(-1, 1, 1, 1).astype(np.float32)
    mask = mask * time_mask.reshape(pred.shape[0], pred.shape[1], 1, 1).astype(np.float32)

    per_pollutant = compute_per_pollutant_report(pred, y, mask, pollutant_names=pollutant_names, horizons=tuple(horizons))
    out = _macro_metrics_from_per_pollutant(per_pollutant, horizons=horizons)
    out["n_eval"] = float(mask.sum())
    return out


@torch.no_grad()
def _predict_torch_flat_model(
    model: torch.nn.Module,
    X: np.ndarray,
    *,
    batch_size: int,
    num_stations: int,
    num_targets: int,
    drop_mask: Optional[np.ndarray] = None,
    drop_feature_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Predict for models that take flattened X: (B, L, N*F) and output (B, H, N*D).
    We accept X as (S, L, N, F) and flatten inside for consistency with missingness injection.
    """
    model.eval()
    device = torch.device("cpu")
    model = model.to(device)

    S, L, N, F = X.shape
    if N != num_stations:
        raise ValueError(f"Unexpected N: {N} != {num_stations}")

    preds: List[np.ndarray] = []
    for start in range(0, S, batch_size):
        xb = X[start : start + batch_size].copy()
        if drop_mask is not None and drop_feature_indices is not None:
            dm = drop_mask[start : start + batch_size]
            sub = xb[:, :, :, drop_feature_indices]
            sub[dm] = 0.0
            xb[:, :, :, drop_feature_indices] = sub

        xb_flat = xb.reshape(xb.shape[0], L, N * F)
        x_t = torch.from_numpy(xb_flat).to(device=device, dtype=torch.float32)
        out = model(x_t).cpu().numpy().astype(np.float32)  # (B, H, N*D)
        out = out.reshape(out.shape[0], out.shape[1], num_stations, num_targets)
        preds.append(out)
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def _predict_wgdgtm(
    model: torch.nn.Module,
    X: np.ndarray,
    *,
    batch_size: int,
    drop_mask: Optional[np.ndarray] = None,
    drop_feature_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    model.eval()
    device = torch.device("cpu")
    model = model.to(device)

    S = X.shape[0]
    preds: List[np.ndarray] = []
    for start in range(0, S, batch_size):
        xb = X[start : start + batch_size].copy()
        if drop_mask is not None and drop_feature_indices is not None:
            dm = drop_mask[start : start + batch_size]
            sub = xb[:, :, :, drop_feature_indices]
            sub[dm] = 0.0
            xb[:, :, :, drop_feature_indices] = sub

        x_t = torch.from_numpy(xb).to(device=device, dtype=torch.float32)
        out = model(x_t).cpu().numpy().astype(np.float32)  # (B, H, N, D)
        preds.append(out)
    return np.concatenate(preds, axis=0)


def _predict_naive(
    model: NaivePersistence,
    X: np.ndarray,
    *,
    batch_size: int,
    drop_mask: Optional[np.ndarray] = None,
    drop_feature_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    S = X.shape[0]
    preds: List[np.ndarray] = []
    for start in range(0, S, batch_size):
        xb = X[start : start + batch_size].copy()
        if drop_mask is not None and drop_feature_indices is not None:
            dm = drop_mask[start : start + batch_size]
            sub = xb[:, :, :, drop_feature_indices]
            sub[dm] = 0.0
            xb[:, :, :, drop_feature_indices] = sub
        preds.append(model.predict(xb))
    return np.concatenate(preds, axis=0)


def _wind_feature_indices(feature_list: list) -> Dict[str, int]:
    needed = ["wd_sin", "wd_cos", "WSPM"]
    idx = {}
    for k in needed:
        if k not in feature_list:
            raise ValueError(f"Missing required wind feature {k} in feature_list")
        idx[k] = feature_list.index(k)
    return idx


def _load_wgdgtm(cfg_path: str, ckpt_path: str) -> Tuple[WGDGTM, List[str], List[str]]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    feature_list, target_list = load_feature_and_target_lists(cfg["data"]["processed_dir"])
    input_scaler = load_input_scaler(cfg["data"]["p1_deep_dir"])
    input_center_np, input_scale_np = input_scaler_center_scale(input_scaler)
    target_feature_indices = [feature_list.index(t) for t in target_list]

    device = torch.device("cpu")
    A_np, _ = load_static_adjacency(cfg["data"]["graphs_dir"])
    A = to_torch_adjacency(A_np, device=device)
    A = row_normalize(add_self_loops(A, weight=1.0))

    mcfg = cfg["model"]
    tcfg = cfg["task"]
    cfg_obj = WGDTMConfig(
        num_nodes=int(tcfg["num_stations"]),
        in_features=len(feature_list),
        horizon=int(tcfg["horizon"]),
        num_targets=int(tcfg["num_targets"]),
        d_model=int(mcfg["d_model"]),
        d_qk=int(mcfg["d_qk"]),
        d_node_emb=int(mcfg["d_node_emb"]),
        dropout=float(mcfg["dropout"]),
        wind_gate_hidden=int(mcfg["wind_gate"]["hidden_dim"]),
        lambda_gate=float(mcfg["wind_gate"]["lambda_gate"]),
        alpha_init=float(mcfg["graph_fusion"]["alpha_init"]),
        beta_init=float(mcfg["graph_fusion"]["beta_init"]),
        gamma_init=float(mcfg["graph_fusion"]["gamma_init"]),
        add_self_loops=bool(mcfg["graph_fusion"]["add_self_loops"]),
        spatial_out_dim=int(mcfg["spatial"]["out_dim"]),
        tcn_channels=int(mcfg["tcn"]["channels"]),
        tcn_layers=int(mcfg["tcn"]["num_layers"]),
        tcn_kernel=int(mcfg["tcn"]["kernel_size"]),
        tcn_dropout=float(mcfg["tcn"]["dropout"]),
        dec_h_emb_dim=int(mcfg["decoder"]["horizon_emb_dim"]),
        dec_hidden_dim=int(mcfg["decoder"]["hidden_dim"]),
        dec_dropout=float(mcfg["decoder"]["dropout"]),
        decoder_type=str(mcfg.get("decoder", {}).get("type", "shared")),
        use_residual_forecasting=bool(mcfg.get("use_residual_forecasting", False)),
        assert_shapes=bool(cfg.get("debug", {}).get("assert_shapes", False)),
    )

    model = WGDGTM(
        cfg_obj,
        A_static=A,
        wind_feature_indices=_wind_feature_indices(feature_list),
        target_feature_indices=target_feature_indices,
        input_center=torch.tensor(input_center_np, device=device),
        input_scale=torch.tensor(input_scale_np, device=device),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module.") :]: v for k, v in state.items()}
    model.load_state_dict(state)
    return model, feature_list, target_list


def _plot_holiday_mae_curves(
    out_path: str,
    curves: Dict[str, Dict[str, List[float]]],
    *,
    title: str,
) -> None:
    _ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(10, 5))
    horizons = np.arange(1, 25)
    for model_name, by_subset in curves.items():
        for subset_name, ys in by_subset.items():
            plt.plot(horizons, ys, marker="o", linewidth=2, markersize=3, label=f"{model_name}-{subset_name}")
    plt.xlabel("Horizon (hours)")
    plt.ylabel("macro_MAE")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_missing_rate_curve(out_path: str, df: pd.DataFrame, *, metric: str = "macro_MAE") -> None:
    _ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(8, 5))
    for model_name, g in df.groupby("model"):
        g = g.sort_values("missing_rate")
        plt.plot(g["missing_rate"] * 100.0, g[metric], marker="o", linewidth=2, label=model_name)
    plt.xlabel("Additional input missing rate (%)")
    plt.ylabel(metric)
    plt.title(f"{metric} vs missing rate (pollutant channels)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="baseline/results/special_cases")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing_rates", default="0,0.1,0.3,0.5")
    parser.add_argument("--p1_deep_dir", default="processed/P1_deep")
    parser.add_argument("--processed_dir", default="processed")
    parser.add_argument("--tabular_dir", default="processed/tabular_lgbm")
    parser.add_argument("--baseline_ckpt_dir", default="baseline/results/checkpoints")
    parser.add_argument("--wgdgtm_config", default="model/configs/wgdgtm.yaml")
    parser.add_argument("--wgdgtm_ckpt", default="model/results/checkpoints/best.pt")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    out_dir = args.out_dir
    plots_dir = os.path.join(out_dir, "plots")
    _ensure_dir(out_dir)
    _ensure_dir(plots_dir)

    feature_list = load_feature_list(args.processed_dir)
    target_list = load_target_list(args.processed_dir)
    meta = load_metadata(args.processed_dir)
    station_list = meta["station_list"]
    pollutant_feature_indices = [feature_list.index(t) for t in target_list]

    npz = load_npz_data(args.p1_deep_dir, splits=("test",))
    test = npz["test"]
    X = test["X"].astype(np.float32, copy=False)  # (S,168,12,17) scaled
    X_mask = test["X_mask"].astype(np.float32, copy=False)
    Y = test["Y"].astype(np.float32, copy=False)
    Y_mask = test["Y_mask"].astype(np.float32, copy=False)
    datetime_origins = test["datetime_origins"]

    S, H, N, D = Y.shape
    if H != 24 or N != 12 or D != 6:
        raise ValueError(f"Unexpected test Y shape: {Y.shape}; expected (S,24,12,6)")

    forecast_dt = _forecast_datetimes(datetime_origins, horizon=H)
    forecast_dates = forecast_dt.astype("datetime64[D]")
    holiday_dates = _build_holiday_dates(HOLIDAY_RANGES)
    is_holiday = np.isin(forecast_dates, holiday_dates)
    is_non_holiday = ~is_holiday
    is_weekend = _weekend_mask(forecast_dt)
    is_weekday = ~is_weekend

    # LightGBM coverage mask (some origins may be absent due to valid_start filtering).
    lgbm_df = load_lgbm_data(args.tabular_dir, splits=("test",))["test"]
    lgbm = LightGBMMultiHorizon(n_jobs=1)
    lgbm.load(os.path.join(args.baseline_ckpt_dir, "lgbm"))
    raw_pred = lgbm.predict(lgbm_df)  # (num_rows,24,6)
    lgbm_pred, present_mask = align_lgbm_predictions_to_npz(raw_pred, lgbm_df, test, station_list)
    sample_mask_common = present_mask.astype(bool)

    # Load scalers for naive baselines
    scalers = load_scaler(args.p1_deep_dir)

    # Load torch baselines (LSTM/TCN) from checkpoints
    input_dim = N * X.shape[3]  # N*F
    output_dim = N * D
    with open("baseline/configs/lstm.yaml", "r", encoding="utf-8") as f:
        lstm_cfg = yaml.safe_load(f)
    with open("baseline/configs/tcn.yaml", "r", encoding="utf-8") as f:
        tcn_cfg = yaml.safe_load(f)

    lstm = LSTMDirect(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=int(lstm_cfg.get("lstm", {}).get("hidden_dim", 256)),
        num_layers=int(lstm_cfg.get("lstm", {}).get("num_layers", 2)),
        dropout=float(lstm_cfg.get("lstm", {}).get("dropout", 0.2)),
        horizon=H,
    )
    ckpt_lstm = torch.load(os.path.join(args.baseline_ckpt_dir, "lstm_best.pt"), map_location="cpu")
    lstm.load_state_dict(ckpt_lstm["model_state_dict"])

    tcn = TCN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_channels=int(tcn_cfg.get("tcn", {}).get("hidden_channels", 64)),
        num_layers=int(tcn_cfg.get("tcn", {}).get("num_layers", 6)),
        kernel_size=int(tcn_cfg.get("tcn", {}).get("kernel_size", 3)),
        dropout=float(tcn_cfg.get("tcn", {}).get("dropout", 0.2)),
        horizon=H,
    )
    ckpt_tcn = torch.load(os.path.join(args.baseline_ckpt_dir, "tcn_best.pt"), map_location="cpu")
    tcn.load_state_dict(ckpt_tcn["model_state_dict"])

    # Load WG-DGTM
    wgdgtm, _w_feat, _w_targets = _load_wgdgtm(args.wgdgtm_config, args.wgdgtm_ckpt)

    # ---------- 1) Holiday / weekend split metrics ----------
    subset_defs = {
        "holiday": is_holiday,
        "non_holiday": is_non_holiday,
        "weekend": is_weekend,
        "weekday": is_weekday,
    }

    holiday_rows: List[Dict[str, float]] = []
    curves: Dict[str, Dict[str, List[float]]] = {}

    def add_model_holiday_rows(model_name: str, pred: np.ndarray):
        for subset_name, tmask in subset_defs.items():
            met = _compute_subset_macro_metrics(
                pred,
                Y,
                Y_mask,
                pollutant_names=target_list,
                time_mask=tmask,
                sample_mask=sample_mask_common,
                horizons=(1, 6, 12, 24),
            )
            holiday_rows.append({"model": model_name, "subset": subset_name, **met})

    # Naive persistence
    naive = NaivePersistence(
        input_scaler=scalers["input_scaler"],
        feature_list=feature_list,
        target_list=target_list,
    )
    pred_naive = _predict_naive(naive, X, batch_size=int(args.batch_size))
    add_model_holiday_rows("naive_persistence", pred_naive)

    # LSTM / TCN
    pred_lstm = _predict_torch_flat_model(
        lstm, X, batch_size=int(args.batch_size), num_stations=N, num_targets=D
    )
    add_model_holiday_rows("lstm", pred_lstm)

    pred_tcn = _predict_torch_flat_model(
        tcn, X, batch_size=int(args.batch_size), num_stations=N, num_targets=D
    )
    add_model_holiday_rows("tcn", pred_tcn)

    # LightGBM (already aligned)
    add_model_holiday_rows("lightgbm", lgbm_pred.astype(np.float32, copy=False))

    # WG-DGTM
    pred_wg = _predict_wgdgtm(wgdgtm, X, batch_size=int(args.batch_size))
    add_model_holiday_rows("wgdgtm", pred_wg)

    # Per-horizon curves for a small comparison figure (LightGBM vs WG-DGTM)
    for model_name, pred in [("lightgbm", lgbm_pred), ("wgdgtm", pred_wg)]:
        curves[model_name] = {}
        for subset_name in ("holiday", "non_holiday"):
            ys: List[float] = []
            for h in range(1, H + 1):
                tmask = subset_defs[subset_name][:, h - 1 : h]
                met_h = _compute_subset_macro_metrics(
                    pred[:, h - 1 : h],
                    Y[:, h - 1 : h],
                    Y_mask[:, h - 1 : h],
                    pollutant_names=target_list,
                    time_mask=tmask,
                    sample_mask=sample_mask_common,
                    horizons=(1,),
                )
                ys.append(float(met_h["macro_MAE"]))
            curves[model_name][subset_name] = ys

    holiday_df = pd.DataFrame(holiday_rows)
    holiday_csv = os.path.join(out_dir, "holiday_weekend_metrics.csv")
    holiday_df.to_csv(holiday_csv, index=False)
    _plot_holiday_mae_curves(
        os.path.join(plots_dir, "holiday_macro_mae_vs_horizon.png"),
        curves,
        title="Holiday vs Non-holiday: macro_MAE vs Horizon",
    )

    # ---------- 2) Missingness robustness (simulate additional missing in X) ----------
    missing_rates = [float(x.strip()) for x in str(args.missing_rates).split(",") if x.strip() != ""]
    for r in missing_rates:
        if r < 0 or r >= 1:
            raise ValueError(f"Invalid missing rate {r}; expected in [0,1)")

    rng_base = int(args.seed)
    missing_rows: List[Dict[str, float]] = []
    # Only compare models that share the same sequential sensor inputs (P1_deep X).
    missing_models = [
        ("naive_persistence", "naive"),
        ("lstm", "lstm"),
        ("tcn", "tcn"),
        ("wgdgtm", "wgdgtm"),
    ]

    observed = (X_mask[:, :, :, pollutant_feature_indices] == 1.0)
    for r in missing_rates:
        rng = np.random.default_rng(rng_base + int(round(r * 10_000)))
        drop = (rng.random(size=observed.shape) < r) & observed  # (S,168,12,6)

        # Predict & evaluate each model under the same missingness mask.
        pred = _predict_naive(
            naive,
            X,
            batch_size=int(args.batch_size),
            drop_mask=drop,
            drop_feature_indices=pollutant_feature_indices,
        )
        per_p = compute_per_pollutant_report(pred, Y, Y_mask, pollutant_names=target_list)
        met = _macro_metrics_from_per_pollutant(per_p, horizons=(1, 6, 12, 24))
        missing_rows.append({"model": "naive_persistence", "missing_rate": r, **met})

        pred = _predict_torch_flat_model(
            lstm,
            X,
            batch_size=int(args.batch_size),
            num_stations=N,
            num_targets=D,
            drop_mask=drop,
            drop_feature_indices=pollutant_feature_indices,
        )
        per_p = compute_per_pollutant_report(pred, Y, Y_mask, pollutant_names=target_list)
        met = _macro_metrics_from_per_pollutant(per_p, horizons=(1, 6, 12, 24))
        missing_rows.append({"model": "lstm", "missing_rate": r, **met})

        pred = _predict_torch_flat_model(
            tcn,
            X,
            batch_size=int(args.batch_size),
            num_stations=N,
            num_targets=D,
            drop_mask=drop,
            drop_feature_indices=pollutant_feature_indices,
        )
        per_p = compute_per_pollutant_report(pred, Y, Y_mask, pollutant_names=target_list)
        met = _macro_metrics_from_per_pollutant(per_p, horizons=(1, 6, 12, 24))
        missing_rows.append({"model": "tcn", "missing_rate": r, **met})

        pred = _predict_wgdgtm(
            wgdgtm,
            X,
            batch_size=int(args.batch_size),
            drop_mask=drop,
            drop_feature_indices=pollutant_feature_indices,
        )
        per_p = compute_per_pollutant_report(pred, Y, Y_mask, pollutant_names=target_list)
        met = _macro_metrics_from_per_pollutant(per_p, horizons=(1, 6, 12, 24))
        missing_rows.append({"model": "wgdgtm", "missing_rate": r, **met})

    missing_df = pd.DataFrame(missing_rows)
    missing_csv = os.path.join(out_dir, "missing_rate_metrics.csv")
    missing_df.to_csv(missing_csv, index=False)
    _plot_missing_rate_curve(os.path.join(plots_dir, "missing_rate_macro_mae.png"), missing_df, metric="macro_MAE")

    print(f"[done] wrote: {holiday_csv}")
    print(f"[done] wrote: {missing_csv}")
    print(f"[done] plots: {plots_dir}")


if __name__ == "__main__":
    main()
