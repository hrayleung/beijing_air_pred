# Beijing Air Quality Preprocessed Dataset v2.1

## CHANGELOG from v2.0
- **FIX A**: SCALE_TARGETS properly implemented - separate target scaler on observed values only
- **FIX B**: LightGBM valid_start respects max(LOOKBACK-1, max_lag, max_roll) = 168
- **FIX C**: P2 outputs now include Y_mask (prevents evaluation contamination)
- **FIX D**: Validation Test #3 strengthened with rigorous spot-check comparisons

## Task Definition
| Parameter | Value |
|-----------|-------|
| Lookback (L) | 168 hours |
| Horizon (H) | 24 hours |
| Stations (N) | 12 |
| Targets (D) | 6 pollutants |

## Y Scaling Configuration
- **SCALE_TARGETS = False**
- If False: Y is in original units (μg/m³ etc.)
- If True: Y is scaled using RobustScaler fitted on TRAIN observed values only

## Station Order
```python
station_list = ['Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan', 'Gucheng', 'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong']
```

## Feature Order (17 features)
```python
feature_list = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'wd_sin', 'wd_cos', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
```

## Pipeline P1 (Deep Learning)

```python
import numpy as np

data = np.load('processed/P1_deep/train.npz')
X = data['X']           # (samples, 168, 12, 17) - SCALED
Y = data['Y']           # (samples, 24, 12, 6) - RAW (0 where missing)
X_mask = data['X_mask'] # 1=observed, 0=missing
Y_mask = data['Y_mask'] # 1=observed, 0=missing - MUST USE IN LOSS/METRICS

# CRITICAL: Y contains 0 at missing positions. Use Y_mask:
def masked_mse(pred, target, mask):
    se = (pred - target) ** 2
    return (se * mask).sum() / (mask.sum() + 1e-8)
```

## Pipeline P2 (Simple Baselines)

**FIX C: P2 now includes Y_mask**

```python
data = np.load('processed/P2_simple/train.npz')
X = data['X']       # (samples, 168, 12, 17)
Y = data['Y']       # (samples, 24, 12, 6) - 0 where missing
Y_mask = data['Y_mask']  # MUST USE - 1=observed, 0=missing
```

## LightGBM Tabular Data

**FIX B: valid_start = dt_index[168]**

This ensures all lag features (up to lag168) and rolling windows 
(up to 168h) have sufficient history.

**Lag Convention (FIX E clarification)**:
- `lag1` = value at t-1 (shift(1))
- `lag168` = value at t-168 (shift(168))
- **NO lag0** - current value not included as feature
- All features use values from time < t (strictly causal)

**Target Convention**:
- `PM2.5_h1` = value at t+1 (shift(-1))
- `PM2.5_h24` = value at t+24 (shift(-24))

```python
import pandas as pd

df = pd.read_csv('processed/tabular_lgbm/lgbm_train.csv')

# Feature columns
target_cols = [c for c in df.columns if '_h' in c]
meta_cols = ['datetime', 'station']
feature_cols = [c for c in df.columns if c not in target_cols + meta_cols]
```

## Inverse Transform (if SCALE_TARGETS=True)

```python
import pickle
import json

with open('processed/P1_deep/scaler.pkl', 'rb') as f:
    scalers = pickle.load(f)

with open('processed/P1_deep/scaler_params.json') as f:
    params = json.load(f)

if params['scale_targets']:
    target_scaler = scalers['target_scaler']
    # Inverse: pred_original = pred_scaled * scale + center
    pred_original = pred_scaled * target_scaler.scale_ + target_scaler.center_
```

## Validation Tests (all must pass)
1. X is scaled for pollutant channels ✓
2. No bfill used in causal pipelines ✓
3. Y equals raw targets (rigorous spot-check with 200 samples) ✓
4. Windows don't cross split boundaries ✓
5. Station order matches adjacency axes ✓
