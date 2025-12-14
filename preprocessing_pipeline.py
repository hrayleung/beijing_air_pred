#!/usr/bin/env python3
"""
Reproducible Preprocessing Pipeline for UCI Beijing Multi-Site Air Quality (PRSA) Dataset
Supports: Persistence, Seasonal Naive, LightGBM, LSTM, TCN, STGCN, Graph WaveNet/MTGNN

Author: ML Pipeline
Date: 2024
"""

import os
import sys
import json
import pickle
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Pipeline configuration - all parameters in one place for reproducibility."""
    
    # Random seed
    SEED = 42
    
    # Data paths
    DATA_DIR = "PRSA_Data_20130301-20170228"
    OUTPUT_DIR = "processed"
    
    # Task parameters (FIXED)
    LOOKBACK = 168  # L = 168 hours (7 days)
    HORIZON = 24    # H = 24 hours
    NUM_STATIONS = 12
    
    # Target pollutants (fixed order)
    TARGETS = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    
    # Meteorology features
    METEO_FEATURES = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    
    # Wind direction encoding
    WD_ENCODING = 'sincos'  # 'sincos' or 'onehot'
    
    # Time features
    USE_TIME_FEATURES = True
    
    # Split boundaries (default)
    TRAIN_START = "2013-03-01 00:00:00"
    TRAIN_END = "2016-02-29 23:00:00"
    VAL_START = "2016-03-01 00:00:00"
    VAL_END = "2016-10-31 23:00:00"
    TEST_START = "2016-11-01 00:00:00"
    TEST_END = "2017-02-28 23:00:00"
    
    # Cap value handling
    CAP_VALUE_MODE = 'A'  # 'A' = convert to NaN, 'B' = keep as-is
    CAP_VALUES = {
        'PM2.5': 999,
        'PM10': 999,
        'CO': 10000
    }
    OUTLIER_PERCENTILE = 99.9  # Report outliers above this percentile
    
    # Imputation settings
    MAX_INTERP_GAP = 6  # Maximum gap (hours) for linear interpolation
    
    # Scaling
    SCALE_TARGETS = False  # Keep targets in original scale by default
    
    # Graph construction
    GRAPH_TOP_K = 4  # Top-k neighbors for adjacency matrix
    GRAPH_CORR_FEATURE = 'PM2.5'  # Feature to use for correlation
    
    # LightGBM lag features
    LGBM_LAGS = [1, 2, 3, 4, 5, 6, 12, 18, 24, 48, 72, 168]
    LGBM_ROLLING_WINDOWS = [24, 72, 168]


# Wind direction mapping (16 compass points + calm)
WD_ANGLE_MAP = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
    'cv': np.nan  # calm/variable - treat as missing
}


def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging to file and console."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'reports', 'preprocessing_log.txt')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logger = logging.getLogger('preprocessing')
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# =============================================================================
# STEP 1: Load and Clean Data
# =============================================================================

def load_station_data(config: Config, logger: logging.Logger) -> Tuple[pd.DataFrame, List[str]]:
    """Load all station CSVs, construct datetime, verify coverage."""
    
    logger.info("=" * 60)
    logger.info("STEP 1: Loading and cleaning data")
    logger.info("=" * 60)
    
    import glob
    import re
    
    # Find all station files
    pattern = os.path.join(config.DATA_DIR, "PRSA_Data_*_20130301-20170228.csv")
    csv_files = sorted(glob.glob(pattern))
    
    if len(csv_files) != 12:
        logger.warning(f"Expected 12 station files, found {len(csv_files)}")
    
    all_dfs = []
    station_list = []
    
    for filepath in csv_files:
        # Parse station name from filename
        basename = os.path.basename(filepath)
        match = re.search(r'PRSA_Data_(.+)_20130301-20170228\.csv', basename)
        station_name = match.group(1) if match else None
        
        if station_name is None:
            logger.error(f"Could not parse station name from {basename}")
            continue
        
        # Load CSV
        df = pd.read_csv(filepath)
        
        # Construct datetime
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        
        # Drop 'No' column
        if 'No' in df.columns:
            df = df.drop(columns=['No'])
        
        # Ensure station column exists
        if 'station' not in df.columns:
            df['station'] = station_name
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        all_dfs.append(df)
        station_list.append(station_name)
        
        logger.info(f"  {station_name}: {len(df)} rows, "
                   f"{df['datetime'].min()} to {df['datetime'].max()}")
    
    # Sort station list alphabetically for consistent ordering
    station_list = sorted(station_list)
    logger.info(f"\nStation order (alphabetical): {station_list}")
    
    # Combine all data
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Verify coverage
    logger.info("\nVerifying hourly coverage:")
    expected_start = pd.Timestamp("2013-03-01 00:00:00")
    expected_end = pd.Timestamp("2017-02-28 23:00:00")
    expected_hours = pd.date_range(start=expected_start, end=expected_end, freq='h')
    expected_count = len(expected_hours)
    logger.info(f"  Expected hourly timestamps: {expected_count}")
    
    for station in station_list:
        station_df = combined_df[combined_df['station'] == station]
        actual_count = len(station_df)
        if actual_count != expected_count:
            logger.warning(f"  {station}: {actual_count} rows (expected {expected_count})")
        else:
            logger.info(f"  {station}: {actual_count} rows ✓")
    
    return combined_df, station_list


# =============================================================================
# STEP 2: Define Features
# =============================================================================

def encode_wind_direction(df: pd.DataFrame, method: str = 'sincos') -> pd.DataFrame:
    """Encode wind direction as sin/cos or one-hot."""
    
    df = df.copy()
    
    if method == 'sincos':
        # Map to angles
        df['wd_angle'] = df['wd'].map(WD_ANGLE_MAP)
        # Convert to radians and compute sin/cos
        df['wd_sin'] = np.sin(np.radians(df['wd_angle']))
        df['wd_cos'] = np.cos(np.radians(df['wd_angle']))
        # Drop intermediate column
        df = df.drop(columns=['wd_angle'])
        wd_features = ['wd_sin', 'wd_cos']
    else:
        # One-hot encoding
        wd_dummies = pd.get_dummies(df['wd'], prefix='wd')
        df = pd.concat([df, wd_dummies], axis=1)
        wd_features = [c for c in df.columns if c.startswith('wd_')]
    
    return df, wd_features


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features (hour-of-day, month)."""
    
    df = df.copy()
    
    # Hour of day (0-23) -> sin/cos
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
    
    # Month (1-12) -> sin/cos
    df['month_sin'] = np.sin(2 * np.pi * df['datetime'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['datetime'].dt.month / 12)
    
    return df, ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']


def define_features(df: pd.DataFrame, config: Config, logger: logging.Logger) -> Tuple[pd.DataFrame, List[str]]:
    """Define and create all input features."""
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Defining features")
    logger.info("=" * 60)
    
    # Start with pollutants and meteorology
    feature_list = config.TARGETS.copy() + config.METEO_FEATURES.copy()
    logger.info(f"  Base features: {feature_list}")
    
    # Encode wind direction
    df, wd_features = encode_wind_direction(df, config.WD_ENCODING)
    feature_list.extend(wd_features)
    logger.info(f"  Wind direction ({config.WD_ENCODING}): {wd_features}")
    
    # Add time features if configured
    if config.USE_TIME_FEATURES:
        df, time_features = add_time_features(df)
        feature_list.extend(time_features)
        logger.info(f"  Time features: {time_features}")
    
    logger.info(f"\n  Total input features: {len(feature_list)}")
    logger.info(f"  Feature order: {feature_list}")
    logger.info(f"  Target order: {config.TARGETS}")
    
    return df, feature_list


# =============================================================================
# STEP 3: Cap Value Handling
# =============================================================================

def handle_cap_values(df: pd.DataFrame, config: Config, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Handle sensor cap values and detect outliers."""
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Cap value handling")
    logger.info("=" * 60)
    logger.info(f"  Mode: {'A (convert to NaN)' if config.CAP_VALUE_MODE == 'A' else 'B (keep as-is)'}")
    
    df = df.copy()
    cap_report = []
    
    for var, cap_val in config.CAP_VALUES.items():
        if var not in df.columns:
            continue
        
        # Count cap values by station
        for station in df['station'].unique():
            mask = (df['station'] == station) & (df[var] == cap_val)
            count = mask.sum()
            
            if count > 0:
                cap_report.append({
                    'station': station,
                    'variable': var,
                    'cap_value': cap_val,
                    'count': count,
                    'action': 'converted_to_NaN' if config.CAP_VALUE_MODE == 'A' else 'kept'
                })
                
                if config.CAP_VALUE_MODE == 'A':
                    df.loc[mask, var] = np.nan
    
    # Detect extreme outliers (report only, don't remove)
    logger.info(f"\n  Outlier detection (>{config.OUTLIER_PERCENTILE}th percentile):")
    for var in config.TARGETS + config.METEO_FEATURES:
        if var not in df.columns:
            continue
        
        threshold = df[var].quantile(config.OUTLIER_PERCENTILE / 100)
        outlier_count = (df[var] > threshold).sum()
        
        if outlier_count > 0:
            max_val = df[var].max()
            logger.info(f"    {var}: {outlier_count} values > {threshold:.2f} (max: {max_val:.2f})")
            cap_report.append({
                'station': 'ALL',
                'variable': var,
                'cap_value': f'>{threshold:.2f}',
                'count': outlier_count,
                'action': 'reported_only'
            })
    
    cap_report_df = pd.DataFrame(cap_report)
    
    if len(cap_report_df) > 0:
        logger.info(f"\n  Cap values found and processed:")
        for _, row in cap_report_df[cap_report_df['action'] != 'reported_only'].iterrows():
            logger.info(f"    {row['station']}/{row['variable']}: {row['count']} values = {row['cap_value']} -> {row['action']}")
    
    return df, cap_report_df


# =============================================================================
# STEP 4: Build Spatiotemporal Tensor
# =============================================================================

def build_spatiotemporal_tensor(
    df: pd.DataFrame, 
    station_list: List[str], 
    feature_list: List[str],
    config: Config,
    logger: logging.Logger
) -> Tuple[np.ndarray, pd.DatetimeIndex, pd.DataFrame]:
    """Build aligned tensor [T, N_stations, F_features]."""
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Building spatiotemporal tensor")
    logger.info("=" * 60)
    
    # Create global datetime index
    global_index = pd.date_range(
        start=config.TRAIN_START,
        end=config.TEST_END,
        freq='h'
    )
    T = len(global_index)
    N = len(station_list)
    F = len(feature_list)
    
    logger.info(f"  Global datetime index: {global_index[0]} to {global_index[-1]}")
    logger.info(f"  Tensor shape: ({T}, {N}, {F}) = (time, stations, features)")
    
    # Initialize tensor with NaN
    data_tensor = np.full((T, N, F), np.nan, dtype=np.float32)
    
    # Fill tensor station by station
    for s_idx, station in enumerate(station_list):
        station_df = df[df['station'] == station].copy()
        station_df = station_df.set_index('datetime')
        
        # Reindex to global index
        station_df = station_df.reindex(global_index)
        
        # Extract features
        for f_idx, feat in enumerate(feature_list):
            if feat in station_df.columns:
                data_tensor[:, s_idx, f_idx] = station_df[feat].values
    
    # Create long-format dataframe for debugging
    debug_records = []
    for t_idx, dt in enumerate(global_index):
        for s_idx, station in enumerate(station_list):
            record = {'datetime': dt, 'station': station}
            for f_idx, feat in enumerate(feature_list):
                record[feat] = data_tensor[t_idx, s_idx, f_idx]
            debug_records.append(record)
    
    debug_df = pd.DataFrame(debug_records)
    
    # Report missingness
    total_cells = T * N * F
    missing_cells = np.isnan(data_tensor).sum()
    logger.info(f"  Total cells: {total_cells:,}")
    logger.info(f"  Missing cells: {missing_cells:,} ({100*missing_cells/total_cells:.2f}%)")
    
    return data_tensor, global_index, debug_df


# =============================================================================
# STEP 5: Time-Based Split
# =============================================================================

def time_based_split(
    data_tensor: np.ndarray,
    datetime_index: pd.DatetimeIndex,
    config: Config,
    logger: logging.Logger
) -> Dict[str, Tuple[np.ndarray, pd.DatetimeIndex]]:
    """Split data chronologically into train/val/test."""
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Time-based split")
    logger.info("=" * 60)
    
    splits = {}
    
    boundaries = {
        'train': (pd.Timestamp(config.TRAIN_START), pd.Timestamp(config.TRAIN_END)),
        'val': (pd.Timestamp(config.VAL_START), pd.Timestamp(config.VAL_END)),
        'test': (pd.Timestamp(config.TEST_START), pd.Timestamp(config.TEST_END))
    }
    
    for split_name, (start, end) in boundaries.items():
        mask = (datetime_index >= start) & (datetime_index <= end)
        split_data = data_tensor[mask]
        split_index = datetime_index[mask]
        
        splits[split_name] = (split_data, split_index)
        
        logger.info(f"  {split_name.upper()}: {start} to {end}")
        logger.info(f"    Shape: {split_data.shape}, Hours: {len(split_index)}")
    
    return splits, boundaries


# =============================================================================
# STEP 6: Missing Value Handling
# =============================================================================

def create_missing_mask(data: np.ndarray) -> np.ndarray:
    """Create binary mask: 1 = observed, 0 = missing."""
    return (~np.isnan(data)).astype(np.float32)


def impute_within_split(
    data: np.ndarray,
    method: str = 'ffill_bfill',
    max_gap: int = 6
) -> np.ndarray:
    """Impute missing values within a single split (no leakage)."""
    
    T, N, F = data.shape
    imputed = data.copy()
    
    for s in range(N):
        for f in range(F):
            series = pd.Series(imputed[:, s, f])
            
            if method == 'ffill_bfill':
                # Forward fill then backward fill
                series = series.ffill().bfill()
            
            elif method == 'linear_ffill_bfill':
                # Linear interpolation for short gaps, then ffill/bfill edges
                series = series.interpolate(method='linear', limit=max_gap)
                series = series.ffill().bfill()
            
            imputed[:, s, f] = series.values
    
    return imputed


def process_pipeline_p1(
    splits: Dict[str, Tuple[np.ndarray, pd.DatetimeIndex]],
    feature_list: List[str],
    config: Config,
    logger: logging.Logger
) -> Dict[str, Dict[str, np.ndarray]]:
    """Pipeline P1: Deep learning rigorous with masks."""
    
    logger.info("\n  Pipeline P1 (Deep Learning - with masks):")
    
    processed = {}
    
    for split_name, (data, dt_index) in splits.items():
        # Create masks BEFORE imputation
        X_mask = create_missing_mask(data)
        
        # Impute X values (conservative: ffill/bfill only)
        X_imputed = impute_within_split(data, method='ffill_bfill')
        
        # For targets, keep original (with NaN) for Y_mask creation
        target_indices = [feature_list.index(t) for t in config.TARGETS]
        Y_raw = data[:, :, target_indices].copy()
        Y_mask = create_missing_mask(Y_raw)
        
        processed[split_name] = {
            'data': X_imputed,
            'X_mask': X_mask,
            'Y_mask': Y_mask,
            'datetime_index': dt_index
        }
        
        missing_before = np.isnan(data).sum()
        missing_after = np.isnan(X_imputed).sum()
        logger.info(f"    {split_name}: Missing before={missing_before:,}, after={missing_after:,}")
    
    return processed


def process_pipeline_p2(
    splits: Dict[str, Tuple[np.ndarray, pd.DatetimeIndex]],
    config: Config,
    logger: logging.Logger
) -> Dict[str, Dict[str, np.ndarray]]:
    """Pipeline P2: Simple baseline-friendly (no masks)."""
    
    logger.info("\n  Pipeline P2 (Simple - no masks):")
    
    processed = {}
    
    for split_name, (data, dt_index) in splits.items():
        # Impute with linear interpolation + ffill/bfill
        X_imputed = impute_within_split(
            data, 
            method='linear_ffill_bfill',
            max_gap=config.MAX_INTERP_GAP
        )
        
        processed[split_name] = {
            'data': X_imputed,
            'datetime_index': dt_index
        }
        
        missing_before = np.isnan(data).sum()
        missing_after = np.isnan(X_imputed).sum()
        logger.info(f"    {split_name}: Missing before={missing_before:,}, after={missing_after:,}")
    
    return processed


# =============================================================================
# STEP 7: Scaling
# =============================================================================

def fit_scalers(
    train_data: np.ndarray,
    feature_list: List[str],
    config: Config,
    logger: logging.Logger
) -> Tuple[RobustScaler, Optional[RobustScaler], Dict]:
    """Fit scalers on TRAIN data only."""
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: Fitting scalers (TRAIN only)")
    logger.info("=" * 60)
    
    T, N, F = train_data.shape
    
    # Reshape to 2D for sklearn: (T*N, F)
    train_flat = train_data.reshape(-1, F)
    
    # Fit RobustScaler for inputs
    input_scaler = RobustScaler()
    input_scaler.fit(train_flat)
    
    logger.info(f"  Input scaler: RobustScaler fitted on {train_flat.shape}")
    
    # Target scaler (optional)
    target_scaler = None
    target_indices = [feature_list.index(t) for t in config.TARGETS]
    
    if config.SCALE_TARGETS:
        target_data = train_flat[:, target_indices]
        target_scaler = RobustScaler()
        target_scaler.fit(target_data)
        logger.info(f"  Target scaler: RobustScaler fitted on targets")
    else:
        logger.info(f"  Target scaler: None (targets kept in original scale)")
    
    # Save scaler params as JSON-serializable dict
    scaler_params = {
        'input_center': input_scaler.center_.tolist(),
        'input_scale': input_scaler.scale_.tolist(),
        'target_indices': target_indices,
        'scale_targets': config.SCALE_TARGETS
    }
    
    if target_scaler is not None:
        scaler_params['target_center'] = target_scaler.center_.tolist()
        scaler_params['target_scale'] = target_scaler.scale_.tolist()
    
    return input_scaler, target_scaler, scaler_params


def apply_scaling(
    data: np.ndarray,
    input_scaler: RobustScaler,
    target_scaler: Optional[RobustScaler],
    feature_list: List[str],
    config: Config
) -> np.ndarray:
    """Apply fitted scalers to data."""
    
    T, N, F = data.shape
    data_flat = data.reshape(-1, F)
    
    # Scale all features
    scaled_flat = input_scaler.transform(data_flat)
    
    # If not scaling targets, restore original target values
    if not config.SCALE_TARGETS:
        target_indices = [feature_list.index(t) for t in config.TARGETS]
        scaled_flat[:, target_indices] = data_flat[:, target_indices]
    
    return scaled_flat.reshape(T, N, F)


# =============================================================================
# STEP 8: Supervised Window Generation
# =============================================================================

def generate_supervised_windows(
    data: np.ndarray,
    datetime_index: pd.DatetimeIndex,
    feature_list: List[str],
    config: Config,
    X_mask: Optional[np.ndarray] = None,
    Y_mask_full: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """Generate supervised learning windows with fixed shapes.
    
    X: (num_samples, L=168, N=12, F_in)
    Y: (num_samples, H=24, N=12, 6)
    """
    
    T, N, F = data.shape
    L = config.LOOKBACK
    H = config.HORIZON
    
    target_indices = [feature_list.index(t) for t in config.TARGETS]
    
    # Calculate valid forecast origins
    # Origin t: X uses [t-L+1, t], Y uses [t+1, t+H]
    # Valid range: t >= L-1 and t+H <= T-1
    # So: t in [L-1, T-H-1]
    
    num_samples = T - L - H + 1
    
    if num_samples <= 0:
        raise ValueError(f"Not enough data for windows: T={T}, L={L}, H={H}")
    
    # Initialize arrays
    X = np.zeros((num_samples, L, N, F), dtype=np.float32)
    Y = np.zeros((num_samples, H, N, len(config.TARGETS)), dtype=np.float32)
    
    if X_mask is not None:
        X_mask_out = np.zeros((num_samples, L, N, F), dtype=np.float32)
        Y_mask_out = np.zeros((num_samples, H, N, len(config.TARGETS)), dtype=np.float32)
    
    # Collect datetime indices for each sample
    sample_datetimes = []
    
    for i in range(num_samples):
        # Forecast origin is at index (L-1 + i)
        origin_idx = L - 1 + i
        
        # X window: [origin_idx - L + 1, origin_idx] inclusive
        x_start = origin_idx - L + 1
        x_end = origin_idx + 1  # exclusive
        
        # Y window: [origin_idx + 1, origin_idx + H] inclusive
        y_start = origin_idx + 1
        y_end = origin_idx + H + 1  # exclusive
        
        X[i] = data[x_start:x_end]
        Y[i] = data[y_start:y_end, :, target_indices]
        
        if X_mask is not None:
            X_mask_out[i] = X_mask[x_start:x_end]
            Y_mask_out[i] = Y_mask_full[y_start:y_end]
        
        sample_datetimes.append(datetime_index[origin_idx])
    
    result = {
        'X': X,
        'Y': Y,
        'datetime_origins': np.array(sample_datetimes, dtype='datetime64[ns]')
    }
    
    if X_mask is not None:
        result['X_mask'] = X_mask_out
        result['Y_mask'] = Y_mask_out
    
    # Also create flattened versions for time-only models
    result['X_flat'] = X.reshape(num_samples, L, N * F)
    result['Y_flat'] = Y.reshape(num_samples, H, N * len(config.TARGETS))
    
    return result


# =============================================================================
# STEP 9: Tabular Dataset for LightGBM (Vectorized)
# =============================================================================

def create_lgbm_features(
    data: np.ndarray,
    datetime_index: pd.DatetimeIndex,
    station_list: List[str],
    feature_list: List[str],
    config: Config,
    logger: logging.Logger
) -> pd.DataFrame:
    """Create tabular features for LightGBM with lag features (VECTORIZED).
    
    Option A: Global model with station_id as categorical.
    Each row = (forecast_origin_time, station)
    """
    
    T, N, F = data.shape
    L = config.LOOKBACK
    H = config.HORIZON
    
    # Features to create lags for (pollutants + meteorology)
    lag_features = config.TARGETS + config.METEO_FEATURES
    
    # Process each station separately, then concatenate
    all_station_dfs = []
    
    for s_idx, station in enumerate(station_list):
        # Extract station data: (T, F) -> DataFrame
        station_data = pd.DataFrame(
            data[:, s_idx, :],
            index=datetime_index,
            columns=feature_list
        )
        
        # Start building features DataFrame
        df = pd.DataFrame(index=datetime_index)
        df['datetime'] = datetime_index
        df['station'] = station
        df['station_id'] = s_idx
        
        # Time features (vectorized)
        df['hour_sin'] = np.sin(2 * np.pi * datetime_index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * datetime_index.hour / 24)
        df['month_sin'] = np.sin(2 * np.pi * datetime_index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * datetime_index.month / 12)
        df['dayofweek'] = datetime_index.dayofweek
        
        # Lag features (vectorized using shift)
        for f_name in lag_features:
            if f_name in feature_list:
                series = station_data[f_name]
                for lag in config.LGBM_LAGS:
                    df[f'{f_name}_lag{lag}'] = series.shift(lag - 1).values
        
        # Rolling statistics (vectorized)
        for f_name in lag_features:
            if f_name in feature_list:
                series = station_data[f_name]
                for window in config.LGBM_ROLLING_WINDOWS:
                    rolling = series.rolling(window=window, min_periods=1)
                    df[f'{f_name}_roll{window}_mean'] = rolling.mean().values
                    df[f'{f_name}_roll{window}_std'] = rolling.std().values
        
        # Target columns for all horizons (vectorized using shift)
        for target_name in config.TARGETS:
            if target_name in feature_list:
                series = station_data[target_name]
                for h in range(1, H + 1):
                    df[f'{target_name}_h{h}'] = series.shift(-h).values
        
        all_station_dfs.append(df)
    
    # Concatenate all stations
    combined_df = pd.concat(all_station_dfs, ignore_index=True)
    
    # Filter to valid forecast origins only: t in [L-1, T-H-1]
    # This means datetime should be in range [datetime_index[L-1], datetime_index[T-H-1]]
    valid_start = datetime_index[L - 1]
    valid_end = datetime_index[T - H - 1]
    
    combined_df = combined_df[
        (combined_df['datetime'] >= valid_start) & 
        (combined_df['datetime'] <= valid_end)
    ].reset_index(drop=True)
    
    # Convert station to categorical
    combined_df['station'] = combined_df['station'].astype('category')
    
    return combined_df


# =============================================================================
# STEP 10: Graph Construction
# =============================================================================

def build_adjacency_matrix(
    train_data: np.ndarray,
    station_list: List[str],
    feature_list: List[str],
    config: Config,
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray]:
    """Build adjacency matrix from station correlations (TRAIN only)."""
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 10: Building adjacency matrix")
    logger.info("=" * 60)
    
    T, N, F = train_data.shape
    
    # Get correlation feature index
    corr_feat_idx = feature_list.index(config.GRAPH_CORR_FEATURE)
    
    # Extract feature data for all stations: (T, N)
    corr_data = train_data[:, :, corr_feat_idx]
    
    # Handle NaN values - use pandas for NaN-aware correlation
    corr_df = pd.DataFrame(corr_data, columns=station_list)
    corr_matrix = corr_df.corr(method='pearson').values  # (N, N)
    
    # Fill any remaining NaN with 0 (shouldn't happen after imputation)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    logger.info(f"  Correlation feature: {config.GRAPH_CORR_FEATURE}")
    logger.info(f"  Full correlation matrix shape: {corr_matrix.shape}")
    
    # Build top-k adjacency
    k = config.GRAPH_TOP_K
    adj_topk = np.zeros((N, N), dtype=np.float32)
    
    for i in range(N):
        # Get correlations for node i (excluding self)
        corrs = corr_matrix[i].copy()
        corrs[i] = -np.inf  # Exclude self
        
        # Find top-k neighbors
        top_k_indices = np.argsort(corrs)[-k:]
        
        for j in top_k_indices:
            adj_topk[i, j] = corr_matrix[i, j]
            adj_topk[j, i] = corr_matrix[j, i]  # Symmetric
    
    # Set diagonal to 1
    np.fill_diagonal(adj_topk, 1.0)
    
    # Ensure non-negative (some correlations might be negative)
    adj_topk = np.maximum(adj_topk, 0)
    
    logger.info(f"  Top-k adjacency (k={k}): {(adj_topk > 0).sum()} non-zero entries")
    logger.info(f"  Adjacency matrix:\n{adj_topk.round(3)}")
    
    return adj_topk, corr_matrix.astype(np.float32)


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(
    config: Config,
    station_list: List[str],
    feature_list: List[str],
    boundaries: Dict,
    p1_processed: Dict,
    p2_processed: Dict,
    input_scaler: RobustScaler,
    target_scaler: Optional[RobustScaler],
    scaler_params: Dict,
    lgbm_dfs: Dict[str, pd.DataFrame],
    adj_topk: np.ndarray,
    adj_full: np.ndarray,
    cap_report_df: pd.DataFrame,
    missingness_report: pd.DataFrame,
    logger: logging.Logger
):
    """Save all outputs to processed/ directory."""
    
    logger.info("\n" + "=" * 60)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 60)
    
    output_dir = config.OUTPUT_DIR
    
    # Create directories
    dirs = [
        output_dir,
        os.path.join(output_dir, 'P1_deep'),
        os.path.join(output_dir, 'P2_simple'),
        os.path.join(output_dir, 'tabular_lgbm'),
        os.path.join(output_dir, 'graphs'),
        os.path.join(output_dir, 'reports')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    # 1. Metadata
    metadata = {
        'station_list': station_list,
        'feature_list': feature_list,
        'target_list': config.TARGETS,
        'num_stations': len(station_list),
        'num_features': len(feature_list),
        'num_targets': len(config.TARGETS),
        'lookback': config.LOOKBACK,
        'horizon': config.HORIZON,
        'split_boundaries': {
            'train': {'start': config.TRAIN_START, 'end': config.TRAIN_END},
            'val': {'start': config.VAL_START, 'end': config.VAL_END},
            'test': {'start': config.TEST_START, 'end': config.TEST_END}
        },
        'cap_handling_mode': config.CAP_VALUE_MODE,
        'cap_values': config.CAP_VALUES,
        'scale_targets': config.SCALE_TARGETS,
        'wd_encoding': config.WD_ENCODING,
        'use_time_features': config.USE_TIME_FEATURES,
        'seed': config.SEED
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Saved: metadata.json")
    
    # 2. Feature and target lists
    with open(os.path.join(output_dir, 'feature_list.json'), 'w') as f:
        json.dump(feature_list, f, indent=2)
    
    with open(os.path.join(output_dir, 'target_list.json'), 'w') as f:
        json.dump(config.TARGETS, f, indent=2)
    
    logger.info(f"  Saved: feature_list.json, target_list.json")
    
    # 3. P1 Deep Learning outputs
    for split_name, data_dict in p1_processed.items():
        npz_path = os.path.join(output_dir, 'P1_deep', f'{split_name}.npz')
        np.savez_compressed(
            npz_path,
            X=data_dict['X'],
            Y=data_dict['Y'],
            X_mask=data_dict['X_mask'],
            Y_mask=data_dict['Y_mask'],
            X_flat=data_dict['X_flat'],
            Y_flat=data_dict['Y_flat'],
            datetime_origins=data_dict['datetime_origins']
        )
        logger.info(f"  Saved: P1_deep/{split_name}.npz - X:{data_dict['X'].shape}, Y:{data_dict['Y'].shape}")
    
    # Save P1 scaler
    with open(os.path.join(output_dir, 'P1_deep', 'scaler.pkl'), 'wb') as f:
        pickle.dump({'input_scaler': input_scaler, 'target_scaler': target_scaler}, f)
    
    with open(os.path.join(output_dir, 'P1_deep', 'scaler_params.json'), 'w') as f:
        json.dump(scaler_params, f, indent=2)
    
    logger.info(f"  Saved: P1_deep/scaler.pkl, scaler_params.json")
    
    # 4. P2 Simple outputs
    for split_name, data_dict in p2_processed.items():
        npz_path = os.path.join(output_dir, 'P2_simple', f'{split_name}.npz')
        np.savez_compressed(
            npz_path,
            X=data_dict['X'],
            Y=data_dict['Y'],
            X_flat=data_dict['X_flat'],
            Y_flat=data_dict['Y_flat'],
            datetime_origins=data_dict['datetime_origins']
        )
        logger.info(f"  Saved: P2_simple/{split_name}.npz - X:{data_dict['X'].shape}, Y:{data_dict['Y'].shape}")
    
    # 5. LightGBM tabular data (CSV format for compatibility)
    for split_name, df in lgbm_dfs.items():
        csv_path = os.path.join(output_dir, 'tabular_lgbm', f'lgbm_{split_name}.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"  Saved: tabular_lgbm/lgbm_{split_name}.csv - {len(df)} rows, {len(df.columns)} cols")
    
    # 6. Graph adjacency matrices
    np.save(os.path.join(output_dir, 'graphs', 'adjacency_corr_topk.npy'), adj_topk)
    np.save(os.path.join(output_dir, 'graphs', 'adjacency_corr_full.npy'), adj_full)
    
    with open(os.path.join(output_dir, 'graphs', 'station_list.json'), 'w') as f:
        json.dump(station_list, f, indent=2)
    
    logger.info(f"  Saved: graphs/adjacency_corr_topk.npy, adjacency_corr_full.npy, station_list.json")
    
    # 7. Reports
    cap_report_df.to_csv(os.path.join(output_dir, 'reports', 'cap_values_report.csv'), index=False)
    missingness_report.to_csv(os.path.join(output_dir, 'reports', 'missingness_report_by_station_feature.csv'), index=False)
    
    logger.info(f"  Saved: reports/cap_values_report.csv, missingness_report_by_station_feature.csv")


def create_missingness_report(
    data_tensor: np.ndarray,
    station_list: List[str],
    feature_list: List[str]
) -> pd.DataFrame:
    """Create missingness report by station and feature."""
    
    T, N, F = data_tensor.shape
    records = []
    
    for s_idx, station in enumerate(station_list):
        for f_idx, feature in enumerate(feature_list):
            missing_count = np.isnan(data_tensor[:, s_idx, f_idx]).sum()
            records.append({
                'station': station,
                'feature': feature,
                'missing_count': missing_count,
                'missing_pct': round(100 * missing_count / T, 2),
                'total_count': T
            })
    
    return pd.DataFrame(records)


def generate_readme(config: Config, feature_list: List[str], station_list: List[str], logger: logging.Logger):
    """Generate README.md explaining how to use the processed data."""
    
    readme_content = f"""# Beijing Air Quality Preprocessed Dataset

## Overview
This directory contains preprocessed data for the UCI Beijing Multi-Site Air Quality (PRSA) dataset,
ready for training various forecasting models.

## Task Definition
- **Time Resolution**: Hourly
- **Stations**: N = {len(station_list)}
- **Lookback Window**: L = {config.LOOKBACK} hours (7 days)
- **Forecast Horizon**: H = {config.HORIZON} hours
- **Targets**: D = {len(config.TARGETS)} pollutants

## Station Order (CRITICAL - must match adjacency matrix)
```python
station_list = {station_list}
```

## Feature Order
```python
feature_list = {feature_list}
# Total: {len(feature_list)} features
```

## Target Order
```python
target_list = {config.TARGETS}
```

## Data Splits
| Split | Start | End | Purpose |
|-------|-------|-----|---------|
| Train | {config.TRAIN_START} | {config.TRAIN_END} | Model training |
| Val | {config.VAL_START} | {config.VAL_END} | Hyperparameter tuning |
| Test | {config.TEST_START} | {config.TEST_END} | Final evaluation |

## Directory Structure
```
processed/
├── metadata.json              # All configuration and parameters
├── feature_list.json          # Input feature names in order
├── target_list.json           # Target names in order
├── P1_deep/                   # For LSTM, TCN, STGCN, GWNet
│   ├── train.npz
│   ├── val.npz
│   ├── test.npz
│   ├── scaler.pkl
│   └── scaler_params.json
├── P2_simple/                 # For Persistence, Seasonal Naive
│   ├── train.npz
│   ├── val.npz
│   └── test.npz
├── tabular_lgbm/              # For LightGBM
│   ├── lgbm_train.csv
│   ├── lgbm_val.csv
│   └── lgbm_test.csv
├── graphs/                    # For STGCN, GWNet, MTGNN
│   ├── adjacency_corr_topk.npy
│   ├── adjacency_corr_full.npy
│   └── station_list.json
├── reports/
│   ├── cap_values_report.csv
│   ├── missingness_report_by_station_feature.csv
│   └── preprocessing_log.txt
└── README.md
```

## Loading Data

### Pipeline P1 (Deep Learning with Masks)
```python
import numpy as np
import pickle

# Load data
train = np.load('processed/P1_deep/train.npz')
X_train = train['X']           # (num_samples, 168, 12, F_in)
Y_train = train['Y']           # (num_samples, 24, 12, 6)
X_mask = train['X_mask']       # (num_samples, 168, 12, F_in) - 1=observed, 0=missing
Y_mask = train['Y_mask']       # (num_samples, 24, 12, 6) - 1=observed, 0=missing

# For time-only models (no spatial dimension in input)
X_flat = train['X_flat']       # (num_samples, 168, 12*F_in)
Y_flat = train['Y_flat']       # (num_samples, 24, 12*6)

# Load scaler for inverse transform
with open('processed/P1_deep/scaler.pkl', 'rb') as f:
    scalers = pickle.load(f)
input_scaler = scalers['input_scaler']
```

### Pipeline P2 (Simple Baselines)
```python
train = np.load('processed/P2_simple/train.npz')
X_train = train['X']           # (num_samples, 168, 12, F_in)
Y_train = train['Y']           # (num_samples, 24, 12, 6)
```

### LightGBM Tabular Data
```python
import pandas as pd

train_df = pd.read_csv('processed/tabular_lgbm/lgbm_train.csv')

# Features: lag features, rolling stats, time features, station_id
# Targets: PM2.5_h1, PM2.5_h2, ..., O3_h24 (6 pollutants × 24 horizons)

# Example: Train model for PM2.5 at horizon 1
feature_cols = [c for c in train_df.columns if not c.startswith(('PM2.5_h', 'PM10_h', 'SO2_h', 'NO2_h', 'CO_h', 'O3_h')) 
                and c not in ['datetime', 'station']]
X = train_df[feature_cols]
y = train_df['PM2.5_h1']
```

### Graph Adjacency Matrix
```python
import numpy as np
import json

# Load adjacency (matches station_list order)
adj = np.load('processed/graphs/adjacency_corr_topk.npy')  # (12, 12)

with open('processed/graphs/station_list.json') as f:
    station_list = json.load(f)

# CRITICAL: Station order in adj[i,j] corresponds to station_list[i], station_list[j]
```

## Mask Usage in Training

For P1 pipeline, use masks to handle missing values properly:

```python
# In loss computation (PyTorch example)
def masked_mse_loss(pred, target, mask):
    '''
    pred: (batch, H, N, D)
    target: (batch, H, N, D)
    mask: (batch, H, N, D) - 1 where observed, 0 where missing
    '''
    squared_error = (pred - target) ** 2
    masked_se = squared_error * mask
    return masked_se.sum() / mask.sum()
```

## Inverse Transform (if targets were scaled)

```python
# Check if targets were scaled
import json
with open('processed/P1_deep/scaler_params.json') as f:
    params = json.load(f)

if params['scale_targets']:
    # Inverse transform predictions
    target_center = np.array(params['target_center'])
    target_scale = np.array(params['target_scale'])
    predictions_original = predictions * target_scale + target_center
else:
    # Targets are already in original scale
    predictions_original = predictions
```

## Reproducibility
- Random seed: {config.SEED}
- Cap value handling: Mode {config.CAP_VALUE_MODE}
- All processing is deterministic
- Scalers fitted on TRAIN only
- Imputation performed within each split separately (no leakage)

## Notes
- Wind direction encoded as sin/cos of angle
- Time features: hour and month as sin/cos
- Missing values in P1: forward-fill then back-fill (conservative)
- Missing values in P2: linear interpolation + ffill/bfill
- Adjacency matrix built from PM2.5 correlations on TRAIN data only
"""
    
    readme_path = os.path.join(config.OUTPUT_DIR, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"  Saved: README.md")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(config: Config = None):
    """Run the complete preprocessing pipeline."""
    
    if config is None:
        config = Config()
    
    # Set random seed
    np.random.seed(config.SEED)
    
    # Setup logging
    logger = setup_logging(config.OUTPUT_DIR)
    
    logger.info("=" * 80)
    logger.info("BEIJING AIR QUALITY PREPROCESSING PIPELINE")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Lookback (L): {config.LOOKBACK} hours")
    logger.info(f"  Horizon (H): {config.HORIZON} hours")
    logger.info(f"  Targets: {config.TARGETS}")
    logger.info(f"  Cap value mode: {config.CAP_VALUE_MODE}")
    
    # Step 1: Load data
    combined_df, station_list = load_station_data(config, logger)
    
    # Step 2: Define features
    combined_df, feature_list = define_features(combined_df, config, logger)
    
    # Step 3: Handle cap values
    combined_df, cap_report_df = handle_cap_values(combined_df, config, logger)
    
    # Step 4: Build spatiotemporal tensor
    data_tensor, datetime_index, debug_df = build_spatiotemporal_tensor(
        combined_df, station_list, feature_list, config, logger
    )
    
    # Create missingness report before imputation
    missingness_report = create_missingness_report(data_tensor, station_list, feature_list)
    
    # Step 5: Time-based split
    splits, boundaries = time_based_split(data_tensor, datetime_index, config, logger)
    
    # Step 6: Missing value handling
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Missing value handling")
    logger.info("=" * 60)
    
    p1_splits = process_pipeline_p1(splits, feature_list, config, logger)
    p2_splits = process_pipeline_p2(splits, config, logger)
    
    # Step 7: Scaling (fit on train only)
    input_scaler, target_scaler, scaler_params = fit_scalers(
        p1_splits['train']['data'], feature_list, config, logger
    )
    
    # Apply scaling to all splits
    logger.info("\n  Applying scalers to all splits...")
    for split_name in ['train', 'val', 'test']:
        p1_splits[split_name]['data'] = apply_scaling(
            p1_splits[split_name]['data'], input_scaler, target_scaler, feature_list, config
        )
        p2_splits[split_name]['data'] = apply_scaling(
            p2_splits[split_name]['data'], input_scaler, target_scaler, feature_list, config
        )
    
    # Step 8: Generate supervised windows
    logger.info("\n" + "=" * 60)
    logger.info("STEP 8: Generating supervised windows")
    logger.info("=" * 60)
    
    p1_windows = {}
    p2_windows = {}
    
    for split_name in ['train', 'val', 'test']:
        logger.info(f"\n  {split_name.upper()}:")
        
        # P1 with masks
        p1_windows[split_name] = generate_supervised_windows(
            p1_splits[split_name]['data'],
            p1_splits[split_name]['datetime_index'],
            feature_list,
            config,
            X_mask=p1_splits[split_name]['X_mask'],
            Y_mask_full=p1_splits[split_name]['Y_mask']
        )
        logger.info(f"    P1: X={p1_windows[split_name]['X'].shape}, Y={p1_windows[split_name]['Y'].shape}")
        
        # P2 without masks
        p2_windows[split_name] = generate_supervised_windows(
            p2_splits[split_name]['data'],
            p2_splits[split_name]['datetime_index'],
            feature_list,
            config
        )
        logger.info(f"    P2: X={p2_windows[split_name]['X'].shape}, Y={p2_windows[split_name]['Y'].shape}")
    
    # Step 9: Create LightGBM tabular data
    logger.info("\n" + "=" * 60)
    logger.info("STEP 9: Creating LightGBM tabular dataset")
    logger.info("=" * 60)
    
    lgbm_dfs = {}
    for split_name in ['train', 'val', 'test']:
        lgbm_dfs[split_name] = create_lgbm_features(
            p2_splits[split_name]['data'],
            p2_splits[split_name]['datetime_index'],
            station_list,
            feature_list,
            config,
            logger
        )
        logger.info(f"  {split_name}: {len(lgbm_dfs[split_name])} rows, {len(lgbm_dfs[split_name].columns)} columns")
    
    # Step 10: Build adjacency matrix (using TRAIN data only - after imputation)
    adj_topk, adj_full = build_adjacency_matrix(
        p2_splits['train']['data'],  # Use imputed data for clean correlation
        station_list,
        feature_list,
        config,
        logger
    )
    
    # Save all outputs
    save_outputs(
        config=config,
        station_list=station_list,
        feature_list=feature_list,
        boundaries=boundaries,
        p1_processed=p1_windows,
        p2_processed=p2_windows,
        input_scaler=input_scaler,
        target_scaler=target_scaler,
        scaler_params=scaler_params,
        lgbm_dfs=lgbm_dfs,
        adj_topk=adj_topk,
        adj_full=adj_full,
        cap_report_df=cap_report_df,
        missingness_report=missingness_report,
        logger=logger
    )
    
    # Generate README
    generate_readme(config, feature_list, station_list, logger)
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutput directory: {config.OUTPUT_DIR}/")
    logger.info(f"\nTensor shapes:")
    logger.info(f"  X: (samples, {config.LOOKBACK}, {len(station_list)}, {len(feature_list)})")
    logger.info(f"  Y: (samples, {config.HORIZON}, {len(station_list)}, {len(config.TARGETS)})")
    logger.info(f"\nSamples per split:")
    for split_name in ['train', 'val', 'test']:
        logger.info(f"  {split_name}: {len(p1_windows[split_name]['X'])}")
    
    logger.info(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'config': config,
        'station_list': station_list,
        'feature_list': feature_list,
        'p1_windows': p1_windows,
        'p2_windows': p2_windows,
        'lgbm_dfs': lgbm_dfs,
        'adj_topk': adj_topk,
        'adj_full': adj_full
    }


if __name__ == "__main__":
    # Run with default configuration
    results = run_pipeline()
    
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    # Verify shapes
    print("\nP1 Deep Learning shapes:")
    for split in ['train', 'val', 'test']:
        X = results['p1_windows'][split]['X']
        Y = results['p1_windows'][split]['Y']
        print(f"  {split}: X={X.shape}, Y={Y.shape}")
        assert X.shape[1] == 168, f"Lookback should be 168, got {X.shape[1]}"
        assert X.shape[2] == 12, f"Stations should be 12, got {X.shape[2]}"
        assert Y.shape[1] == 24, f"Horizon should be 24, got {Y.shape[1]}"
        assert Y.shape[2] == 12, f"Stations should be 12, got {Y.shape[2]}"
        assert Y.shape[3] == 6, f"Targets should be 6, got {Y.shape[3]}"
    
    print("\nAdjacency matrix shape:", results['adj_topk'].shape)
    assert results['adj_topk'].shape == (12, 12), "Adjacency should be 12x12"
    
    print("\nAll verifications passed! ✓")
