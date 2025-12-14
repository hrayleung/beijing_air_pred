"""
LightGBM Tabular Data Loader.
"""
import os
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np


def load_lgbm_data(
    data_dir: str = "processed/tabular_lgbm",
    splits: Tuple[str, ...] = ("train", "val", "test")
) -> Dict[str, pd.DataFrame]:
    """Load LightGBM tabular data for specified splits."""
    data = {}
    for split in splits:
        path = os.path.join(data_dir, f"lgbm_{split}.csv")
        data[split] = pd.read_csv(path)
    return data


def get_feature_target_columns(
    df: pd.DataFrame,
    targets: List[str] = None
) -> Tuple[List[str], List[str]]:
    """
    Separate feature and target columns.
    
    Features: all columns except datetime, station, and target columns (*_h*)
    Targets: columns matching {target}_h{horizon} pattern
    """
    if targets is None:
        targets = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    
    # Target columns: {pollutant}_h{1-24}
    target_cols = [c for c in df.columns 
                   if any(f'{t}_h' in c for t in targets)]
    
    # Meta columns to exclude
    meta_cols = ['datetime', 'station']
    
    # Feature columns: everything else
    feature_cols = [c for c in df.columns 
                    if c not in target_cols and c not in meta_cols]
    
    return feature_cols, target_cols


def get_horizon_targets(
    target_cols: List[str],
    horizon: int,
    targets: List[str] = None
) -> List[str]:
    """Get target columns for a specific horizon."""
    if targets is None:
        targets = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    return [f'{t}_h{horizon}' for t in targets]


def align_lgbm_predictions_to_npz(
    predictions: np.ndarray,
    lgbm_df: pd.DataFrame,
    npz_data: Dict[str, np.ndarray],
    station_list: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align LightGBM predictions to NPZ test ordering.
    
    LightGBM predictions are per (datetime, station) row.
    NPZ test has shape (samples, 24, 12, 6).
    
    Args:
        predictions: (num_rows, 6) for single horizon or (num_rows, 24, 6) for all
        lgbm_df: DataFrame with datetime and station columns
        npz_data: NPZ test data with datetime_origins
        station_list: Ordered list of stations
        
    Returns:
        Aligned predictions (samples, 24, 12, 6)
    """
    # NPZ sample origins (length = num_samples)
    npz_origins = pd.to_datetime(npz_data['datetime_origins'])
    num_samples = len(npz_origins)
    H, N, D = 24, 12, 6
    
    # Create station index mapping
    station_to_idx = {s: i for i, s in enumerate(station_list)}
    
    # Initialize output
    aligned = np.zeros((num_samples, H, N, D), dtype=np.float32)
    
    # Normalize lgbm_df so row indices align with `predictions` positional axis.
    lgbm_df = lgbm_df.copy().reset_index(drop=True)
    lgbm_df['datetime'] = pd.to_datetime(lgbm_df['datetime'])

    if predictions.ndim != 3 or predictions.shape[1:] != (H, D):
        raise ValueError(f"Expected predictions with shape (num_rows, {H}, {D}); got {predictions.shape}")

    # Map datetime -> list of row positions for fast lookup.
    dt_to_rows: Dict[pd.Timestamp, List[int]] = {}
    for row_pos, (dt, station) in enumerate(zip(lgbm_df['datetime'].values, lgbm_df['station'].values)):
        dt_to_rows.setdefault(pd.Timestamp(dt), []).append(row_pos)

    present_mask = np.zeros((num_samples,), dtype=bool)
    for sample_idx, origin in enumerate(npz_origins):
        rows = dt_to_rows.get(pd.Timestamp(origin))
        if not rows:
            continue
        present_mask[sample_idx] = True
        for row_pos in rows:
            station = lgbm_df.at[row_pos, 'station']
            station_idx = station_to_idx.get(station)
            if station_idx is not None:
                aligned[sample_idx, :, station_idx, :] = predictions[row_pos]

    return aligned, present_mask
