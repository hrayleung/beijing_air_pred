# Preprocessing Pipeline v2.0 - CHANGELOG

## Summary of Critical Fixes

This document summarizes all rigor/leakage issues fixed in v2.0 of the preprocessing pipeline.

---

## 1. CAUSAL IMPUTATION (No Look-Ahead)

**Problem in v1**: Used `bfill` and `interpolate()` which look at future values - this is data leakage for time series.

**Fix in v2**:
- Pipeline P1 (rigorous): Forward-fill only (`ffill`)
- Leading NaNs filled with TRAIN-only per-station/per-feature medians
- **NO `bfill` anywhere in causal pipelines**
- Pipeline P2 labeled as "NON-CAUSAL" with explicit warning

```python
# v2 causal imputation
def causal_impute(data, train_medians, ...):
    series = series.ffill()  # Forward-fill only
    series = series.fillna(train_medians[s, f])  # TRAIN median fallback
    # NO bfill!
```

---

## 2. DECOUPLED X/Y SCALING

**Problem in v1**: When `SCALE_TARGETS=False`, code restored target columns to original scale in the same tensor used for X, leaving pollutant history features unscaled.

**Fix in v2**:
- X features ALWAYS scaled (including pollutant history)
- Y scaling independently configurable (default: raw/unscaled)
- Separate data paths for X and Y in window generation

```python
# v2 window generation
def generate_windows(
    X_data,      # Scaled data for X
    Y_data_raw,  # RAW data for Y (separate tensor)
    ...
):
    X[i] = X_data[x_start:x_end]           # From scaled
    Y[i] = Y_data_raw[y_start:y_end, ...]  # From raw
```

---

## 3. RAW LABELS (Never Imputed)

**Problem in v1**: Y was constructed from imputed data, potentially contaminating labels.

**Fix in v2**:
- Y ALWAYS constructed from RAW data BEFORE imputation
- Y_mask indicates which target values are observed
- NaN in Y replaced with 0 for storage, mask indicates validity

```python
# v2: Y from raw data
Y[i] = Y_data_raw[y_start:y_end, :, target_indices]  # May contain NaN
Y_mask[i] = (~np.isnan(Y[i])).astype(np.float32)
Y = np.nan_to_num(Y, nan=0.0)  # Store 0, use mask
```

---

## 4. SPLIT-FIRST POLICY

**Problem in v1**: Some operations could potentially use cross-split information.

**Fix in v2**:
- Time split applied FIRST (Step 5)
- Then within each split:
  - Create masks
  - Apply imputation
  - Apply scaling
- Statistics (medians, scalers) computed on TRAIN only

```python
# v2 order of operations
raw_splits = split_by_time(raw_tensor, ...)  # SPLIT FIRST
train_medians = compute_train_statistics(raw_splits['train'], ...)  # TRAIN only
scaler.fit(train_imputed)  # TRAIN only
```

---

## 5. GRAPH CONSTRUCTION (Positive Correlations Only)

**Problem in v1**: Selected top-k including negative correlations, then clamped to 0, silently changing effective k.

**Fix in v2**:
- Only consider POSITIVE correlations for neighbor selection
- If fewer than k positive neighbors exist, keep only those (sparse)
- Clearly documented method

```python
# v2 graph construction
if config.GRAPH_USE_POSITIVE_ONLY:
    positive_mask = corrs > 0
    positive_indices = np.where(positive_mask)[0]
    # Select top-k from positive only
```

---

## 6. LIGHTGBM LAG CONVENTIONS (Corrected)

**Problem in v1**: Lag definitions were inconsistent (`shift(lag-1)` instead of `shift(lag)`).

**Fix in v2**:
- Standard convention: `lag_k = value at t-k` using `shift(k)`
- Target convention: `target_h_k = value at t+k` using `shift(-k)`
- Clearly documented

```python
# v2 lag features (CORRECT)
for lag in config.LGBM_LAGS:
    df[f'{f_name}_lag{lag}'] = series.shift(lag).values  # lag1 = t-1

# v2 targets (CORRECT)  
for h in range(1, H + 1):
    df[f'{target}_h{h}'] = series.shift(-h).values  # h1 = t+1
```

---

## 7. PERFORMANCE FIX (debug_df Removed)

**Problem in v1**: Created huge debug_df with nested Python loops - slow and memory-heavy.

**Fix in v2**:
- `CREATE_DEBUG_DF = False` by default
- All operations vectorized using pandas/numpy
- LightGBM tabular creation uses vectorized `shift()` and `rolling()`

---

## Validation Tests Added

The pipeline now runs 5 validation tests:

1. **X is scaled**: Verifies pollutant channels are transformed
2. **No bfill**: Verified by assertions in `causal_impute()`
3. **Y equals raw**: Confirms labels from raw data with proper masking
4. **No boundary crossing**: Windows stay within split bounds
5. **Station order**: Adjacency axes match station list

---

## Output Structure (Unchanged)

```
processed/
├── metadata.json
├── feature_list.json
├── target_list.json
├── README.md
├── P1_deep/
│   ├── train.npz (X, Y, X_mask, Y_mask, X_flat, Y_flat, datetime_origins)
│   ├── val.npz
│   ├── test.npz
│   ├── scaler.pkl
│   └── scaler_params.json
├── P2_simple/
│   ├── train.npz (X, Y, X_flat, Y_flat, datetime_origins)
│   ├── val.npz
│   └── test.npz
├── tabular_lgbm/
│   ├── lgbm_train.csv
│   ├── lgbm_val.csv
│   └── lgbm_test.csv
├── graphs/
│   ├── adjacency_corr_topk.npy
│   ├── adjacency_corr_full.npy
│   └── station_list.json
└── reports/
    ├── cap_values_report.csv
    ├── missingness_report_by_station_feature.csv
    └── preprocessing_log.txt
```

---

## How to Run

```bash
python preprocessing_pipeline_v2.py
```

Expected output:
- All validation tests should pass
- Tensor shapes: X=(samples, 168, 12, 17), Y=(samples, 24, 12, 6)
- Train: ~26,113 samples, Val: ~5,689 samples, Test: ~2,689 samples

---

## Key Differences Summary

| Aspect | v1 | v2 |
|--------|----|----|
| Imputation | ffill + bfill | ffill only (causal) |
| Leading NaN | bfill | TRAIN median |
| X scaling | Inconsistent when SCALE_TARGETS=False | Always scaled |
| Y source | Imputed data | Raw data |
| Y default | Potentially imputed | Never imputed |
| Split timing | After some processing | FIRST |
| Graph neighbors | Top-k then clamp | Positive-only top-k |
| Lag convention | shift(lag-1) | shift(lag) |
| debug_df | Always created | Disabled by default |
| Validation | None | 5 automated tests |
