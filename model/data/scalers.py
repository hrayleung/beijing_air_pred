from __future__ import annotations

import os
import pickle
from typing import Tuple

import numpy as np


def load_input_scaler(p1_deep_dir: str):
    path = os.path.join(p1_deep_dir, "scaler.pkl")
    with open(path, "rb") as f:
        scalers = pickle.load(f)
    if isinstance(scalers, dict):
        return scalers["input_scaler"]
    return scalers


def input_scaler_center_scale(input_scaler) -> Tuple[np.ndarray, np.ndarray]:
    center = np.asarray(getattr(input_scaler, "center_", None), dtype=np.float32)
    scale = np.asarray(getattr(input_scaler, "scale_", None), dtype=np.float32)
    if center.ndim != 1 or scale.ndim != 1 or center.shape != scale.shape:
        raise ValueError("input_scaler must provide 1D center_ and scale_ arrays")
    return center, scale

