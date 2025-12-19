from __future__ import annotations

import argparse
import os
import sys
import json
from datetime import datetime, timezone
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader

from extreme_model.data.prsa_npz_dataset import PRSANPZDataset, load_feature_and_target_lists
from extreme_model.evaluation.evaluate import evaluate_model
from extreme_model.models.stformer import STFormer, STFormerConfig

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
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    results_cfg = cfg.get("results", {}) or {}
    model_name = str(results_cfg.get("model_id") or results_cfg.get("model_name") or "stformer")
    model_display_name = str(results_cfg.get("display_name") or "ST-Former")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    results_dir = os.path.join(cfg["results"]["dir"], cfg["results"].get("experiment_name", "default"))
    os.makedirs(results_dir, exist_ok=True)
    # Snapshot config for reproducibility (eval-time).
    with open(os.path.join(results_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)
    meta = {
        "phase": "eval",
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "config_path": args.config,
        "ckpt": args.ckpt,
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
    with open(os.path.join(results_dir, "logs", "run_metadata_eval.json"), "w", encoding="utf-8") as f:
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

    test_ds = PRSANPZDataset(os.path.join(cfg["data"]["p1_deep_dir"], "test.npz"))
    test_loader = DataLoader(test_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, num_workers=int(cfg["training"]["num_workers"]))

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
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module.") :]: v for k, v in state.items()}
    target_model = model.module if hasattr(model, "module") else model
    ckpt_root = os.path.abspath(os.path.dirname(os.path.dirname(args.ckpt)))
    results_root = os.path.abspath(results_dir)
    if ckpt_root != results_root:
        print(
            f"[warn] checkpoint is under {ckpt_root} but results_dir is {results_root}; "
            "this usually means the eval config does not match the checkpoint experiment.",
            flush=True,
        )
    try:
        target_model.load_state_dict(state)
    except RuntimeError as e:
        has_gate = isinstance(state, dict) and ("baseline_gate_logits" in state)
        cfg_mode = str(getattr(cfg_obj, "baseline_mode", "none"))
        if has_gate and cfg_mode.lower() != "mix":
            raise RuntimeError(
                f"{e}\n\n"
                "Checkpoint contains 'baseline_gate_logits' (trained with model.baseline_mode: 'mix'), "
                f"but your config has model.baseline_mode: {cfg_mode!r}. "
                "Please re-run eval with `--config extreme_model/configs/stformer_residual.yaml` "
                "(or any config with baseline_mode: 'mix')."
            ) from e
        raise

    evaluate_model(
        model,
        test_loader,
        device=device,
        pollutant_names=target_list,
        results_dir=results_dir,
        model_name=model_name,
        model_display_name=model_display_name,
        split="test",
        baseline_results_dir="baseline/results",
    )
    print(f"Saved metrics/plots to {results_dir}")


if __name__ == "__main__":
    main()
