from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from extreme_model.data.prsa_npz_dataset import PRSANPZDataset, load_feature_and_target_lists
from extreme_model.losses.masked_losses import compute_target_std_weights_from_npz
from extreme_model.models.stformer import STFormer, STFormerConfig
from extreme_model.training.callbacks import EarlyStopping
from extreme_model.training.trainer import Trainer

from model.data.scalers import input_scaler_center_scale, load_input_scaler


def _time_feature_indices(feature_list: list) -> Dict[str, int]:
    needed = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
    idx = {}
    for k in needed:
        if k not in feature_list:
            raise ValueError(f"Missing required time feature {k} in feature_list")
        idx[k] = feature_list.index(k)
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    results_cfg = cfg.get("results", {}) or {}
    model_name = str(results_cfg.get("model_id") or results_cfg.get("model_name") or "stformer")
    model_display_name = str(results_cfg.get("display_name") or "ST-Former")

    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    results_dir = os.path.join(cfg["results"]["dir"], cfg["results"].get("experiment_name", "default"))
    os.makedirs(results_dir, exist_ok=True)
    # Snapshot config for reproducibility.
    with open(os.path.join(results_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Run metadata.
    os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)
    meta = {
        "phase": "train",
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "config_path": args.config,
        "seed": int(cfg.get("seed", 42)),
        "model_name": model_name,
        "model_display_name": model_display_name,
        "device": str(device),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    }
    try:
        import subprocess

        meta["git_commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__), stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        meta["git_commit"] = None
    with open(os.path.join(results_dir, "logs", "run_metadata_train.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        print(f"[setup] device={device} visible_gpus={gpu_count} gpu_names={gpu_names} results_dir={results_dir}", flush=True)
    else:
        print(f"[setup] device={device} results_dir={results_dir}", flush=True)

    feature_list, target_list = load_feature_and_target_lists(cfg["data"]["processed_dir"])
    input_scaler = load_input_scaler(cfg["data"]["p1_deep_dir"])
    input_center_np, input_scale_np = input_scaler_center_scale(input_scaler)

    # Data
    train_ds = PRSANPZDataset(os.path.join(cfg["data"]["p1_deep_dir"], "train.npz"))
    val_ds = PRSANPZDataset(os.path.join(cfg["data"]["p1_deep_dir"], "val.npz"))
    train_loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True, num_workers=int(cfg["training"]["num_workers"]))
    val_loader = DataLoader(val_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, num_workers=int(cfg["training"]["num_workers"]))
    print(f"[data] train={len(train_ds)} val={len(val_ds)} batch_size={cfg['training']['batch_size']}", flush=True)

    # Loss weights from TRAIN
    std, weights = compute_target_std_weights_from_npz(os.path.join(cfg["data"]["p1_deep_dir"], "train.npz"), eps=float(cfg["training"]["loss_eps"]))
    loss_weighting = str(cfg.get("training", {}).get("loss_weighting", "std")).lower()
    if loss_weighting in {"std", "inv_std", "inverse_std"}:
        used_weights = weights
    elif loss_weighting in {"none", "uniform"}:
        used_weights = np.ones_like(weights, dtype=np.float32)
    else:
        raise ValueError(f"Unknown training.loss_weighting={loss_weighting!r} (expected 'std' or 'none')")
    os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
    with open(os.path.join(results_dir, "metrics", "target_std_weights.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "std": std.tolist(),
                "weights": weights.tolist(),
                "used_weights": used_weights.tolist(),
                "loss_weighting": loss_weighting,
                "targets": target_list,
            },
            f,
            indent=2,
        )
    weights_t = torch.tensor(used_weights, dtype=torch.float32)
    print(f"[loss] loss_weighting={loss_weighting} saved {os.path.join(results_dir, 'metrics', 'target_std_weights.json')}", flush=True)

    # Model
    mcfg = cfg["model"]
    tcfg = cfg["task"]
    cfg_obj = STFormerConfig(
        num_nodes=int(tcfg["num_stations"]),
        in_features=len(feature_list),
        lookback=int(tcfg["lookback"]),
        horizon=int(tcfg["horizon"]),
        num_targets=int(tcfg["num_targets"]),
        d_model=int(mcfg["d_model"]),
        n_heads=int(mcfg["n_heads"]),
        enc_layers=int(mcfg["enc_layers"]),
        dec_layers=int(mcfg["dec_layers"]),
        ff_dim=int(mcfg["ff_dim"]),
        dropout=float(mcfg["dropout"]),
        use_future_time_features=bool(mcfg.get("use_future_time_features", True)),
        baseline_mode=str(mcfg.get("baseline_mode", "none")),
        assert_shapes=bool(cfg.get("debug", {}).get("assert_shapes", False)),
    )

    model = STFormer(
        cfg_obj,
        time_feature_indices=_time_feature_indices(feature_list),
        input_center=torch.tensor(input_center_np, device=device),
        input_scale=torch.tensor(input_scale_np, device=device),
    ).to(device)

    # Optional multi-GPU (DataParallel).
    use_dp = bool(cfg.get("training", {}).get("use_data_parallel", False))
    gpu_ids = cfg.get("training", {}).get("gpus", None)
    if use_dp and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if gpu_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        elif isinstance(gpu_ids, int):
            device_ids = list(range(int(gpu_ids)))
        else:
            device_ids = [int(x) for x in gpu_ids]
        if len(device_ids) >= 2:
            print(f"[setup] using DataParallel device_ids={device_ids}", flush=True)
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        else:
            print("[setup] DataParallel requested but <2 GPUs selected; using single GPU", flush=True)
    elif use_dp and torch.cuda.is_available():
        print("[setup] DataParallel requested but only 1 GPU visible; using single GPU", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))

    trainer = Trainer(
        model=model,
        optimizer=optim,
        device=device,
        weights=weights_t,
        grad_clip=float(cfg["training"]["grad_clip"]),
        loss_eps=float(cfg["training"]["loss_eps"]),
        results_dir=results_dir,
        early_stopping=EarlyStopping(patience=int(cfg["training"]["patience"]), min_delta=float(cfg["training"].get("min_delta", 0.0))),
        model_name=model_name,
        model_display_name=model_display_name,
        log_interval=int(cfg["training"].get("log_interval", 50)),
    )

    state = trainer.fit(train_loader, val_loader, epochs=int(cfg["training"]["epochs"]), pollutant_names=target_list)
    print(f"Best checkpoint: {state.best_path} (val_macro_MAE={state.best_val:.6f})")
    print(f"[logs] {os.path.join(results_dir, 'logs', 'train_history.json')}", flush=True)


if __name__ == "__main__":
    main()
