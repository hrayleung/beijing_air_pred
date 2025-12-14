"""
Debug checks for baseline evaluation correctness (A–E).

Writes:
  - baseline/results/logs/scale_check.txt
  - baseline/results/logs/horizon_variation_check.txt
  - baseline/results/plots/debug_pm25_curves.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


def _load_yaml(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _masked_mean_std(values: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    obs = mask.astype(bool)
    if obs.sum() == 0:
        return float("nan"), float("nan")
    v = values[obs]
    return float(v.mean()), float(v.std())


def _inverse_targets(arr: np.ndarray, target_scaler) -> np.ndarray:
    center = np.asarray(getattr(target_scaler, "center_", None))
    scale = np.asarray(getattr(target_scaler, "scale_", None))
    if center.ndim != 1 or scale.ndim != 1 or center.shape != scale.shape:
        raise ValueError(f"Invalid target_scaler params: center={center.shape}, scale={scale.shape}")
    if arr.shape[-1] != center.shape[0]:
        raise ValueError(f"Target dim mismatch: arr.D={arr.shape[-1]} vs scaler.D={center.shape[0]}")
    return (arr * scale.reshape(1, 1, 1, -1)) + center.reshape(1, 1, 1, -1)


def _write_scale_report(
    out_path: str,
    truth_by_model: Dict[str, np.ndarray],
    mask_by_model: Dict[str, np.ndarray],
    preds_by_model: Dict[str, np.ndarray],
    pollutant_names: List[str],
):
    lines: List[str] = []
    lines.append("Scale alignment check (masked by Y_mask==1)")
    lines.append("")
    for model_name, pred_raw in preds_by_model.items():
        y_true_raw = truth_by_model[model_name]
        y_mask = mask_by_model[model_name]
        if pred_raw.shape != y_true_raw.shape:
            lines.append(f"[{model_name}] ERROR: pred shape {pred_raw.shape} != y_true shape {y_true_raw.shape}")
            lines.append("")
            continue
        lines.append(f"[{model_name}]")
        for d, p in enumerate(pollutant_names):
            true_mean, true_std = _masked_mean_std(y_true_raw[:, :, :, d], y_mask[:, :, :, d])
            pred_mean, pred_std = _masked_mean_std(pred_raw[:, :, :, d], y_mask[:, :, :, d])
            lines.append(
                f"  {p:5s}  y_true mean/std: {true_mean:10.4f} / {true_std:10.4f}  |  "
                f"y_pred mean/std: {pred_mean:10.4f} / {pred_std:10.4f}"
            )
        lines.append("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _write_horizon_variation_report(
    out_path: str,
    preds_by_model: Dict[str, np.ndarray],
    pollutant_names: List[str],
):
    lines: List[str] = []
    lines.append("Horizon-variation sanity check: mean(|pred[h1]-pred[h24]|) per pollutant")
    lines.append("")

    for model_name, pred in preds_by_model.items():
        if pred.ndim != 4 or pred.shape[1] != 24:
            lines.append(f"[{model_name}] ERROR: expected (S,24,N,D), got {pred.shape}")
            lines.append("")
            continue
        lines.append(f"[{model_name}]")
        for d, p in enumerate(pollutant_names):
            delta = float(np.mean(np.abs(pred[:, 0, :, d] - pred[:, 23, :, d])))
            lines.append(f"  {p:5s}  mean|h1-h24|: {delta:.6f}")
        lines.append("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _write_per_horizon_metric_index_report(
    out_path: str,
    truth_by_model: Dict[str, np.ndarray],
    mask_by_model: Dict[str, np.ndarray],
    preds_by_model: Dict[str, np.ndarray],
):
    from baseline.evaluation.masked_metrics import compute_per_horizon_metrics, masked_mae

    lines: List[str] = []
    lines.append("Per-horizon metric indexing check (MAE_h1 uses horizon index 0; MAE_h24 uses index 23)")
    lines.append("")

    for model_name, pred in preds_by_model.items():
        y = truth_by_model[model_name]
        m = mask_by_model[model_name]
        if pred.shape != y.shape:
            continue
        per_h = compute_per_horizon_metrics(pred, y, m)
        mae_h1_manual = float(masked_mae(pred[:, 0], y[:, 0], m[:, 0]))
        mae_h24_manual = float(masked_mae(pred[:, 23], y[:, 23], m[:, 23]))
        lines.append(f"[{model_name}]")
        lines.append(f"  h1  per_horizon: {per_h[1]['MAE']:.6f}  manual(slice): {mae_h1_manual:.6f}")
        lines.append(f"  h24 per_horizon: {per_h[24]['MAE']:.6f}  manual(slice): {mae_h24_manual:.6f}")
        lines.append("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _plot_pm25_example(
    out_path: str,
    y_true_raw: np.ndarray,
    y_mask: np.ndarray,
    preds_by_name: Dict[str, np.ndarray],
    sample_idx: int,
    station_idx: int,
    pollutant_idx: int = 0,
    pollutant_name: str = "PM2.5",
):
    import matplotlib.pyplot as plt

    horizons = np.arange(1, 25)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)

    y = y_true_raw[sample_idx, :, station_idx, pollutant_idx]
    m = y_mask[sample_idx, :, station_idx, pollutant_idx].astype(bool)
    ax.plot(horizons[m], y[m], "k-o", label="true", linewidth=2, markersize=3)

    for name, pred in preds_by_name.items():
        p = pred[sample_idx, :, station_idx, pollutant_idx]
        ax.plot(horizons, p, label=name, linewidth=2)

    ax.set_title(f"{pollutant_name} @ station_idx={station_idx}, sample_idx={sample_idx}")
    ax.set_xlabel("horizon (hours)")
    ax.set_ylabel("raw units")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _seasonal_naive_index_sanity(
    out_path: str,
    X_scaled: np.ndarray,
    input_scaler,
    feature_list: List[str],
    target_list: List[str],
    seasonal_pred_raw: np.ndarray,
    sample_idx: int,
    station_idx: int,
    pollutant_name: str = "PM2.5",
):
    """
    For one sample, verify seasonal prediction comes from the correct lookback index.
    """
    feature_pollutant_idx = feature_list.index(pollutant_name)
    target_pollutant_idx = target_list.index(pollutant_name)

    # Inverse transform the entire lookback for a single sample/station for the pollutant channel.
    L, N, F = X_scaled.shape[1:]
    x_sample = X_scaled[sample_idx]  # (L, N, F)
    x_flat = x_sample.reshape(-1, F)
    x_raw = input_scaler.inverse_transform(x_flat).reshape(L, N, F)

    lines: List[str] = []
    lines.append("SeasonalNaive24 indexing sanity check")
    lines.append(f"sample_idx={sample_idx}, station_idx={station_idx}, pollutant={pollutant_name}")
    lines.append("")
    lines.append("h  lookback_idx  X_raw(lookback_idx)  seasonal_pred[h]")

    for h in range(1, 25):
        lookback_idx = L + h - 25
        x_val = float(x_raw[lookback_idx, station_idx, feature_pollutant_idx])
        p_val = float(seasonal_pred_raw[sample_idx, h - 1, station_idx, target_pollutant_idx])
        lines.append(f"{h:2d} {lookback_idx:11d} {x_val:17.6f} {p_val:17.6f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _predict_torch_from_checkpoint(
    model_name: str,
    checkpoint_path: str,
    data: Dict[str, Dict[str, np.ndarray]],
    config: dict,
    target_list: List[str],
    adj: Optional[np.ndarray],
    device: str,
) -> np.ndarray:
    import torch

    _, _, N, F = data["test"]["X"].shape
    H = data["test"]["Y"].shape[1]
    D = len(target_list)

    model_cfg = config.get(model_name, {})

    if model_name == "lstm":
        from baseline.models.lstm_seq2seq import LSTMDirect

        model = LSTMDirect(
            input_dim=N * F,
            output_dim=N * D,
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.2),
            horizon=H,
        )
        flatten_x = True
    elif model_name == "tcn":
        from baseline.models import TCN

        model = TCN(
            input_dim=N * F,
            output_dim=N * D,
            hidden_channels=model_cfg.get("hidden_channels", 64),
            num_layers=model_cfg.get("num_layers", 6),
            kernel_size=model_cfg.get("kernel_size", 3),
            dropout=model_cfg.get("dropout", 0.2),
            horizon=H,
        )
        flatten_x = True
    elif model_name == "stgcn":
        from baseline.models import STGCN

        if adj is None:
            raise ValueError("adj is required for stgcn")
        model = STGCN(
            num_nodes=N,
            in_channels=F,
            out_channels=D,
            hidden_channels=model_cfg.get("hidden_channels", 64),
            num_layers=model_cfg.get("num_layers", 2),
            kernel_size=model_cfg.get("kernel_size", 3),
            K=model_cfg.get("K", 3),
            horizon=H,
            dropout=model_cfg.get("dropout", 0.2),
            time_pool=model_cfg.get("time_pool", "mean"),
        )
        model.set_adjacency(adj)
        flatten_x = False
    elif model_name == "gwnet":
        from baseline.models import GraphWaveNet

        if adj is None:
            raise ValueError("adj is required for gwnet")
        model = GraphWaveNet(
            num_nodes=N,
            in_channels=F,
            out_channels=D,
            hidden_channels=model_cfg.get("hidden_channels", 32),
            skip_channels=model_cfg.get("skip_channels", 64),
            num_layers=model_cfg.get("num_layers", 4),
            horizon=H,
            dropout=model_cfg.get("dropout", 0.2),
            use_adaptive_adj=model_cfg.get("use_adaptive_adj", True),
            time_pool=model_cfg.get("time_pool", "mean"),
        )
        model.set_adjacency(adj)
        flatten_x = False
    else:
        raise ValueError(f"Unsupported torch model: {model_name}")

    from baseline.data.loader_npz import create_dataloaders
    from baseline.training.checkpointing import load_checkpoint

    loaders = create_dataloaders({"test": data["test"]}, batch_size=256, flatten_x=flatten_x, num_workers=0, pin_memory=False)

    model = model.to(device)
    try:
        load_checkpoint(checkpoint_path, model, optimizer=None, device=device)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load checkpoint {checkpoint_path} into a freshly-constructed {model_name} model. "
            "This usually means the checkpoint was trained with different model hyperparameters "
            "(e.g., hidden size / num layers). Re-run training with the desired config, or run "
            "`baseline.scripts.debug_checks` with a config that matches the checkpoint."
        ) from e
    model.eval()

    preds: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loaders["test"]:
            X = batch["X"].to(device)
            out = model(X)
            preds.append(out.detach().cpu().numpy())

    pred = np.concatenate(preds, axis=0)
    if pred.ndim == 3 and pred.shape[2] == N * D:
        pred = pred.reshape(pred.shape[0], H, N, D)

    return pred.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Baseline debug checks (A–E)")
    parser.add_argument("--config", default="baseline/configs/default.yaml")
    parser.add_argument("--results-dir", default=None, help="Override results_dir from config")
    parser.add_argument("--checkpoints-dir", default=None, help="Directory containing model checkpoints (defaults to results_dir/checkpoints)")
    parser.add_argument("--out-dir", default=None, help="Directory to write logs/plots (defaults to results_dir)")
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda:0")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--station-idx", type=int, default=0)
    parser.add_argument(
        "--models",
        default="naive,seasonal,lgbm,lstm,tcn,stgcn,gwnet",
        help="Comma-separated list",
    )
    args = parser.parse_args()

    # Avoid bytecode writes in environments where the repo is not writable by the current user.
    sys.dont_write_bytecode = True

    cfg = _load_yaml(args.config)
    results_dir = args.results_dir or cfg["output"]["results_dir"]
    checkpoints_dir = args.checkpoints_dir or os.path.join(results_dir, "checkpoints")
    out_dir = args.out_dir or results_dir

    from baseline.data.loader_npz import (
        load_npz_data,
        load_scaler,
        load_metadata,
        load_feature_list,
        load_target_list,
    )
    from baseline.data.loader_lgbm import load_lgbm_data, align_lgbm_predictions_to_npz
    from baseline.data.graph import load_adjacency
    from baseline.models import NaivePersistence, SeasonalNaive24
    from baseline.models.lgbm_multioutput import LightGBMMultiHorizon

    npz = load_npz_data(cfg["data"]["p1_deep_dir"], splits=("test",))
    scaler_dict = load_scaler(cfg["data"]["p1_deep_dir"])
    metadata = load_metadata(cfg["data"]["processed_dir"])
    feature_list = load_feature_list(cfg["data"]["processed_dir"])
    target_list = load_target_list(cfg["data"]["processed_dir"])

    scale_targets = bool(metadata.get("scale_targets", False))
    target_scaler = scaler_dict.get("target_scaler")
    input_scaler = scaler_dict["input_scaler"]

    X_test = npz["test"]["X"]
    Y_test = npz["test"]["Y"]
    Y_mask = npz["test"]["Y_mask"]

    y_true_raw = _inverse_targets(Y_test, target_scaler) if scale_targets else Y_test

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    preds_raw: Dict[str, np.ndarray] = {}
    truth_by_model: Dict[str, np.ndarray] = {}
    mask_by_model: Dict[str, np.ndarray] = {}

    # Default truth/mask (full NPZ test set)
    for name in model_names:
        truth_by_model[name] = y_true_raw
        mask_by_model[name] = Y_mask

    if "naive" in model_names:
        preds_raw["naive"] = NaivePersistence(input_scaler, feature_list, target_list).predict(X_test)

    if "seasonal" in model_names:
        seasonal_pred = SeasonalNaive24(input_scaler, feature_list, target_list).predict(X_test)
        preds_raw["seasonal"] = seasonal_pred

        # Indexing sanity output (writes to logs alongside scale check).
        seasonal_log = os.path.join(out_dir, "logs", "seasonal_index_check.txt")
        _seasonal_naive_index_sanity(
            seasonal_log,
            X_test,
            input_scaler,
            feature_list,
            target_list,
            seasonal_pred,
            sample_idx=args.sample_idx,
            station_idx=args.station_idx,
            pollutant_name="PM2.5",
        )

    if "lgbm" in model_names:
        lgbm_ckpt_dir = os.path.join(checkpoints_dir, "lgbm")
        lgbm_model = LightGBMMultiHorizon()
        lgbm_model.load(lgbm_ckpt_dir)
        lgbm_data = load_lgbm_data(cfg["data"]["tabular_dir"], splits=("test",))
        adj, station_list = load_adjacency(cfg["data"]["graphs_dir"])
        raw_pred = lgbm_model.predict(lgbm_data["test"])
        aligned, present = align_lgbm_predictions_to_npz(raw_pred, lgbm_data["test"], npz["test"], station_list)
        preds_raw["lgbm"] = aligned[present]
        truth_by_model["lgbm"] = y_true_raw[present]
        mask_by_model["lgbm"] = Y_mask[present]

    # Torch models: load from checkpoints and run inference.
    device = args.device
    if any(m in model_names for m in ("lstm", "tcn", "stgcn", "gwnet")):
        import torch

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

    adj = None
    if any(m in model_names for m in ("stgcn", "gwnet")):
        adj, _ = load_adjacency(cfg["data"]["graphs_dir"])

    for torch_model in ("lstm", "tcn", "stgcn", "gwnet"):
        if torch_model not in model_names:
            continue
        ckpt_path = os.path.join(checkpoints_dir, f"{torch_model}_best.pt")
        pred = _predict_torch_from_checkpoint(
            torch_model,
            ckpt_path,
            npz,
            cfg,
            target_list,
            adj,
            device,
        )
        pred_raw = _inverse_targets(pred, target_scaler) if scale_targets else pred
        preds_raw[torch_model] = pred_raw

    # A) scale alignment report
    scale_path = os.path.join(out_dir, "logs", "scale_check.txt")
    _write_scale_report(scale_path, truth_by_model, mask_by_model, preds_raw, target_list)

    # B) horizon variation report (deep models only)
    hv_models = {k: v for k, v in preds_raw.items() if k in {"lstm", "tcn", "stgcn", "gwnet"}}
    hv_path = os.path.join(out_dir, "logs", "horizon_variation_check.txt")
    _write_horizon_variation_report(hv_path, hv_models, target_list)

    # E) per-horizon metric indexing check
    idx_path = os.path.join(out_dir, "logs", "per_horizon_metric_check.txt")
    _write_per_horizon_metric_index_report(idx_path, truth_by_model, mask_by_model, preds_raw)

    # C) one example plot: true vs persistence vs seasonal for PM2.5
    plot_path = os.path.join(out_dir, "plots", "debug_pm25_curves.png")
    plot_preds = {k: preds_raw[k] for k in ("naive", "seasonal") if k in preds_raw}
    _plot_pm25_example(
        plot_path,
        truth_by_model.get("naive", y_true_raw),
        mask_by_model.get("naive", Y_mask),
        plot_preds,
        sample_idx=args.sample_idx,
        station_idx=args.station_idx,
        pollutant_idx=0,
        pollutant_name="PM2.5",
    )

    print(f"Wrote: {scale_path}")
    print(f"Wrote: {hv_path}")
    print(f"Wrote: {idx_path}")
    print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()
