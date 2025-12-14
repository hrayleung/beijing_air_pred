from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader

from model.data.prsa_npz_dataset import PRSANPZDataset, load_feature_and_target_lists
from model.data.scalers import input_scaler_center_scale, load_input_scaler
from model.graphs.adjacency import load_static_adjacency, row_normalize, add_self_loops, to_torch_adjacency
from model.losses.masked_losses import compute_target_std_weights_from_npz
from model.models.wgdgtm import WGDGTM, WGDTMConfig
from model.training.callbacks import EarlyStopping
from model.training.trainer import Trainer


def _wind_feature_indices(feature_list: list) -> Dict[str, int]:
    needed = ["wd_sin", "wd_cos", "WSPM"]
    idx = {}
    for k in needed:
        if k not in feature_list:
            raise ValueError(f"Missing required wind feature {k} in feature_list")
        idx[k] = feature_list.index(k)
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(int(cfg.get("seed", 42)))
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    results_dir = cfg["results"]["dir"]
    os.makedirs(results_dir, exist_ok=True)
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

    # Static graph
    A_np, station_list = load_static_adjacency(cfg["data"]["graphs_dir"])
    A = to_torch_adjacency(A_np, device=device)
    A = row_normalize(add_self_loops(A, weight=1.0))
    print(f"[graph] loaded A_static shape={tuple(A.shape)} stations={len(station_list)}", flush=True)

    # Loss weights from TRAIN
    std, weights = compute_target_std_weights_from_npz(os.path.join(cfg["data"]["p1_deep_dir"], "train.npz"), eps=float(cfg["training"]["loss_eps"]))
    os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
    with open(os.path.join(results_dir, "metrics", "target_std_weights.json"), "w", encoding="utf-8") as f:
        json.dump({"std": std.tolist(), "weights": weights.tolist(), "targets": target_list}, f, indent=2)
    weights_t = torch.tensor(weights, dtype=torch.float32)
    print(f"[loss] target_std_weights saved to {os.path.join(results_dir, 'metrics', 'target_std_weights.json')}", flush=True)

    # Model
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
        assert_shapes=bool(cfg.get("debug", {}).get("assert_shapes", False)),
    )

    model = WGDGTM(
        cfg_obj,
        A_static=A,
        wind_feature_indices=_wind_feature_indices(feature_list),
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
        log_interval=int(cfg["training"].get("log_interval", 50)),
    )

    state = trainer.fit(train_loader, val_loader, epochs=int(cfg["training"]["epochs"]), pollutant_names=target_list)
    print(f"Best checkpoint: {state.best_path} (val_macro_MAE={state.best_val:.6f})")
    print(f"[logs] {os.path.join(results_dir, 'logs', 'train_history.json')}", flush=True)


if __name__ == "__main__":
    main()
