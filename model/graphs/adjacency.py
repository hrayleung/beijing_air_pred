from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch


def load_static_adjacency(graphs_dir: str) -> Tuple[np.ndarray, list]:
    """
    Loads `adjacency_corr_topk.npy` and `station_list.json`.
    """
    import json

    adj_path = os.path.join(graphs_dir, "adjacency_corr_topk.npy")
    stations_path = os.path.join(graphs_dir, "station_list.json")

    A = np.load(adj_path).astype(np.float32)
    with open(stations_path, "r", encoding="utf-8") as f:
        station_list = json.load(f)
    return A, station_list


def add_self_loops(A: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    n = A.shape[-1]
    I = torch.eye(n, device=A.device, dtype=A.dtype) * float(weight)
    return A + I


def row_normalize(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    denom = A.sum(dim=-1, keepdim=True).clamp_min(eps)
    return A / denom


def to_torch_adjacency(A_np: np.ndarray, device: torch.device) -> torch.Tensor:
    A = torch.tensor(A_np, dtype=torch.float32, device=device)
    return A

