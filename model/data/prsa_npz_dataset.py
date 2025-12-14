from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PRSANPZPaths:
    p1_deep_dir: str

    def split_path(self, split: str) -> str:
        return os.path.join(self.p1_deep_dir, f"{split}.npz")


class PRSANPZDataset(Dataset):
    """
    Loads preprocessed PRSA P1_deep NPZ windows.

    X: (S, L, N, F)  scaled
    Y: (S, H, N, D)  raw units (zeros at missing positions)
    Y_mask: same shape as Y, 1=observed, 0=missing
    """

    def __init__(
        self,
        npz_path: str,
        *,
        mmap_mode: str = "r",
    ):
        self.npz_path = npz_path
        self._npz = np.load(npz_path, allow_pickle=True, mmap_mode=mmap_mode)

        self.X = self._npz["X"].astype(np.float32)
        self.Y = self._npz["Y"].astype(np.float32)
        self.Y_mask = self._npz["Y_mask"].astype(np.float32)

        if self.X.ndim != 4 or self.Y.ndim != 4 or self.Y_mask.ndim != 4:
            raise ValueError("Unexpected NPZ shapes")
        if self.Y.shape != self.Y_mask.shape:
            raise ValueError(f"Y/Y_mask mismatch: {self.Y.shape} vs {self.Y_mask.shape}")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.Y[idx])
        y_mask = torch.from_numpy(self.Y_mask[idx])
        return {"X": x, "Y": y, "Y_mask": y_mask}


def load_feature_and_target_lists(processed_dir: str) -> Tuple[list, list]:
    import json

    with open(os.path.join(processed_dir, "feature_list.json"), "r", encoding="utf-8") as f:
        feature_list = json.load(f)
    with open(os.path.join(processed_dir, "target_list.json"), "r", encoding="utf-8") as f:
        target_list = json.load(f)
    return feature_list, target_list


def load_metadata(processed_dir: str) -> Dict:
    import json

    with open(os.path.join(processed_dir, "metadata.json"), "r", encoding="utf-8") as f:
        return json.load(f)

