from __future__ import annotations

import argparse
import os
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader

from model.data.prsa_npz_dataset import PRSANPZDataset, load_feature_and_target_lists
from model.data.scalers import input_scaler_center_scale, load_input_scaler
from model.graphs.adjacency import load_static_adjacency, row_normalize, add_self_loops, to_torch_adjacency
from model.evaluation.evaluate import evaluate_model
from model.models.wgdgtm import WGDGTM, WGDTMConfig


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
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

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
    test_ds = PRSANPZDataset(os.path.join(cfg["data"]["p1_deep_dir"], "test.npz"))
    test_loader = DataLoader(test_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, num_workers=int(cfg["training"]["num_workers"]))

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
        assert_shapes=bool(cfg.get("debug", {}).get("assert_shapes", False)),
    )

    model = WGDGTM(
        cfg_obj,
        A_static=A,
        wind_feature_indices=_wind_feature_indices(feature_list),
        input_center=torch.tensor(input_center_np, device=device),
        input_scale=torch.tensor(input_scale_np, device=device),
    ).to(device)

    # Optional multi-GPU (DataParallel) for faster eval.
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

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model_state"]
    # Accept checkpoints saved from DataParallel or single-GPU.
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module.") :]: v for k, v in state.items()}
    target_model = model.module if hasattr(model, "module") else model
    target_model.load_state_dict(state)

    out = evaluate_model(model, test_loader, device=device, pollutant_names=target_list, results_dir=results_dir, model_name="wgdgtm")
    print(f"Saved metrics/plots to {results_dir}")


if __name__ == "__main__":
    main()
