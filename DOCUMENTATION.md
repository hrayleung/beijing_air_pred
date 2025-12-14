# Beijing Multi-Site Air Quality Dataset: EDA & Preprocessing Documentation

## Overview

This document describes the complete Exploratory Data Analysis (EDA) and preprocessing pipeline for the UCI Beijing Multi-Site Air Quality (PRSA) dataset, designed to support multiple forecasting baselines.

**Dataset**: UCI Beijing Multi-Site Air Quality Data (2013-2017)  
**Source**: 12 monitoring stations across Beijing  
**Period**: March 1, 2013 - February 28, 2017 (4 years, hourly data)

---

## Part 1: Exploratory Data Analysis (EDA)

**Script**: `eda_beijing_air_quality.py`  
**Output**: `eda_output/`

### 1.1 Dataset Inventory

| Metric | Value |
|--------|-------|
| Total Stations | 12 |
| Total Rows | 420,768 (35,064 per station) |
| Time Range | 2013-03-01 00:00 to 2017-02-28 23:00 |
| Hourly Timestamps | 35,064 |
| Columns per File | 18 |

**Stations (alphabetical order)**:
1. Aotizhongxin
2. Changping
3. Dingling
4. Dongsi
5. Guanyuan
6. Gucheng
7. Huairou
8. Nongzhanguan
9. Shunyi
10. Tiantan
11. Wanliu
12. Wanshouxigong

### 1.2 Variables

**Pollutants (6)**:
| Variable | Unit | Description |
|----------|------|-------------|
| PM2.5 | μg/m³ | Fine particulate matter |
| PM10 | μg/m³ | Coarse particulate matter |
| SO2 | μg/m³ | Sulfur dioxide |
| NO2 | μg/m³ | Nitrogen dioxide |
| CO | μg/m³ | Carbon monoxide |
| O3 | μg/m³ | Ozone |

**Meteorology (5)**:
| Variable | Unit | Description |
|----------|------|-------------|
| TEMP | °C | Temperature |
| PRES | hPa | Atmospheric pressure |
| DEWP | °C | Dew point temperature |
| RAIN | mm | Precipitation |
| WSPM | m/s | Wind speed |

**Other**:
- `wd`: Wind direction (16 compass points + calm)
- `year`, `month`, `day`, `hour`: Time components

### 1.3 Missing Value Analysis

**Overall Missingness**: ~1.06% of total cells (75,909 missing values)

**By Feature** (highest to lowest):
- O3: ~1.5%
- PM2.5, PM10: ~1.2%
- SO2, NO2, CO: ~1.0%
- Meteorology variables: <0.5%

**Key Findings**:
- Missing values are distributed across all stations
- No systematic patterns (random missingness)
- All stations have complete hourly coverage (no timestamp gaps)

### 1.4 Descriptive Statistics

**PM2.5 Statistics**:
| Statistic | Value |
|-----------|-------|
| Mean | ~73 μg/m³ |
| Median | ~51 μg/m³ |
| Std | ~75 μg/m³ |
| Min | 2 μg/m³ |
| Max | 957 μg/m³ |
| P95 | ~220 μg/m³ |
| P99 | ~400 μg/m³ |

### 1.5 Temporal Patterns

**Seasonal Pattern**:
- Winter (Dec-Feb): Highest PM2.5 levels (~100+ μg/m³ mean)
- Summer (Jun-Aug): Lowest PM2.5 levels (~50 μg/m³ mean)
- Clear heating season effect

**Diurnal Pattern**:
- PM2.5: Peaks in late evening/night (traffic + boundary layer)
- O3: Peaks in afternoon (photochemical production)
- Inverse relationship between PM2.5 and O3

### 1.6 Station Variability

- Urban stations (Dongsi, Guanyuan, Wanshouxigong): Higher PM2.5
- Suburban stations (Dingling, Huairou): Lower PM2.5
- Strong inter-station correlations (r > 0.8 for most pairs)

### 1.7 EDA Outputs

```
eda_output/
├── eda_report.html              # Complete HTML report
├── station_summary.csv          # File-level inventory
├── missingness_by_station_feature.csv
├── stats_overall.csv            # Descriptive statistics
├── stats_by_station.csv         # Per-station statistics
├── missingness_analysis.png     # Heatmaps
├── distributions_seasonality.png
├── station_comparison.png       # Boxplots
└── correlation_matrix.png
```

---

## Part 2: Preprocessing Pipeline

**Script**: `preprocessing_pipeline.py`  
**Output**: `processed/`

### 2.1 Task Definition

| Parameter | Value |
|-----------|-------|
| Time Resolution | Hourly |
| Stations (N) | 12 |
| Lookback Window (L) | 168 hours (7 days) |
| Forecast Horizon (H) | 24 hours |
| Targets (D) | 6 pollutants |

**Supported Baselines**:
- B0: Persistence / Naive
- B1: Seasonal Naive (24-hour)
- B2: LightGBM (tabular regression)
- B3: LSTM Seq2Seq
- B4: TCN
- B5: STGCN (fixed graph)
- B6: Graph WaveNet / MTGNN (adaptive graph)

### 2.2 Pipeline Steps

#### Step 1: Load & Clean Data
- Load all 12 station CSV files
- Construct datetime from year/month/day/hour
- Sort by datetime
- Drop "No" column
- Verify hourly coverage (35,064 rows per station ✓)

#### Step 2: Feature Engineering

**Input Features (17 total)**:
```python
feature_list = [
    # Pollutants (6)
    'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
    # Meteorology (5)
    'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM',
    # Wind direction - sin/cos encoded (2)
    'wd_sin', 'wd_cos',
    # Time features - cyclical encoding (4)
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
]
```

**Wind Direction Encoding**:
- 16 compass points mapped to angles (0-360°)
- Converted to sin/cos for continuity
- Calm/variable treated as missing

**Time Features**:
- Hour: sin(2π × hour/24), cos(2π × hour/24)
- Month: sin(2π × month/12), cos(2π × month/12)

#### Step 3: Cap Value Handling

**Sensor Cap Values Detected & Converted to NaN**:
| Variable | Cap Value | Count |
|----------|-----------|-------|
| PM2.5 | 999 | 1 |
| PM10 | 999 | 3 |
| CO | 10000 | 57 |

**Outlier Detection** (>99.9th percentile, reported only):
- PM2.5: 413 values > 564 (max: 957)
- PM10: 414 values > 665 (max: 995)
- O3: 392 values > 312 (max: 1071)

#### Step 4: Spatiotemporal Tensor

**Tensor Shape**: `(35064, 12, 17)` = (time, stations, features)

- Aligned all stations to common datetime index
- Station order: alphabetical (consistent across all outputs)
- Missing cells: 75,909 (1.06%)

#### Step 5: Time-Based Split

| Split | Start | End | Hours | Samples |
|-------|-------|-----|-------|---------|
| Train | 2013-03-01 00:00 | 2016-02-29 23:00 | 26,304 | 26,113 |
| Val | 2016-03-01 00:00 | 2016-10-31 23:00 | 5,880 | 5,689 |
| Test | 2016-11-01 00:00 | 2017-02-28 23:00 | 2,880 | 2,689 |

**Note**: Sample count = Hours - Lookback - Horizon + 1 (no cross-boundary windows)

#### Step 6: Missing Value Handling

**Two Pipelines Produced**:

**Pipeline P1 (Deep Learning - with masks)**:
- Creates binary masks: `X_mask`, `Y_mask` (1=observed, 0=missing)
- Imputation: Forward-fill then back-fill (conservative)
- Masks enable proper loss computation during training

**Pipeline P2 (Simple Baselines)**:
- Imputation: Linear interpolation (max 6-hour gaps) + ffill/bfill
- No masks (fully imputed data)

#### Step 7: Scaling

- **Method**: RobustScaler (median/IQR based, robust to outliers)
- **Fitted on**: TRAIN split only
- **Applied to**: All splits (train/val/test)
- **Targets**: Kept in original scale (configurable)

#### Step 8: Supervised Window Generation

**Window Shapes**:
```
X: (num_samples, 168, 12, 17)  # Lookback × Stations × Features
Y: (num_samples, 24, 12, 6)    # Horizon × Stations × Targets
```

**Flattened Versions** (for time-only models):
```
X_flat: (num_samples, 168, 204)  # 12 × 17 = 204
Y_flat: (num_samples, 24, 72)    # 12 × 6 = 72
```

#### Step 9: LightGBM Tabular Dataset

**Approach**: Global model with station_id as categorical feature

**Features (350 columns)**:
- Lag features: lags 1-24, 48, 72, 168 for all 11 variables
- Rolling statistics: mean/std over 24h, 72h, 168h windows
- Time features: hour_sin/cos, month_sin/cos, dayofweek
- station_id (categorical)

**Targets**: 144 columns (6 pollutants × 24 horizons)
- Format: `{pollutant}_h{horizon}` (e.g., PM2.5_h1, PM2.5_h24)

#### Step 10: Graph Construction

**Method**: Station-to-station Pearson correlation on PM2.5 (TRAIN only)

**Adjacency Matrix**:
- Full correlation matrix: 12×12
- Top-k sparse graph: k=4 neighbors per node
- Symmetric, diagonal = 1
- Non-negative (negative correlations clipped to 0)

**Correlation Range**: 0.84 - 0.97 (high inter-station correlation)

### 2.3 Output Structure

```
processed/
├── metadata.json                 # All configuration parameters
├── feature_list.json             # 17 input features (ordered)
├── target_list.json              # 6 target pollutants (ordered)
├── README.md                     # Usage instructions
│
├── P1_deep/                      # For LSTM, TCN, STGCN, GWNet
│   ├── train.npz                 # X, Y, X_mask, Y_mask, X_flat, Y_flat
│   ├── val.npz
│   ├── test.npz
│   ├── scaler.pkl                # Fitted RobustScaler
│   └── scaler_params.json        # Scaler parameters (JSON)
│
├── P2_simple/                    # For Persistence, Seasonal Naive
│   ├── train.npz                 # X, Y, X_flat, Y_flat (no masks)
│   ├── val.npz
│   └── test.npz
│
├── tabular_lgbm/                 # For LightGBM
│   ├── lgbm_train.csv            # 313,356 rows × 350 cols
│   ├── lgbm_val.csv              # 68,268 rows
│   └── lgbm_test.csv             # 32,268 rows
│
├── graphs/                       # For graph neural networks
│   ├── adjacency_corr_topk.npy   # 12×12 sparse adjacency
│   ├── adjacency_corr_full.npy   # 12×12 full correlation
│   └── station_list.json         # Station order (matches adj)
│
└── reports/
    ├── cap_values_report.csv
    ├── missingness_report_by_station_feature.csv
    └── preprocessing_log.txt
```

### 2.4 Data Loading Examples

**Deep Learning (PyTorch)**:
```python
import numpy as np
import torch

# Load P1 data
train = np.load('processed/P1_deep/train.npz')
X_train = torch.FloatTensor(train['X'])      # (26113, 168, 12, 17)
Y_train = torch.FloatTensor(train['Y'])      # (26113, 24, 12, 6)
X_mask = torch.FloatTensor(train['X_mask'])  # (26113, 168, 12, 17)
Y_mask = torch.FloatTensor(train['Y_mask'])  # (26113, 24, 12, 6)

# Load adjacency for GNN
adj = np.load('processed/graphs/adjacency_corr_topk.npy')  # (12, 12)
```

**LightGBM**:
```python
import pandas as pd
import lightgbm as lgb

train_df = pd.read_csv('processed/tabular_lgbm/lgbm_train.csv')

# Get feature columns (exclude targets and metadata)
target_cols = [c for c in train_df.columns if '_h' in c]
meta_cols = ['datetime', 'station']
feature_cols = [c for c in train_df.columns if c not in target_cols + meta_cols]

# Train model for PM2.5 horizon 1
X = train_df[feature_cols]
y = train_df['PM2.5_h1']
```

### 2.5 Reproducibility Guarantees

1. **Random Seed**: Fixed at 42
2. **No Data Leakage**:
   - Chronological split (no shuffling)
   - Imputation within each split only
   - Scalers fitted on TRAIN only
   - Adjacency computed on TRAIN only
3. **Deterministic Processing**: All operations are deterministic
4. **Station Ordering**: Alphabetical, consistent across all outputs

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Raw Data Points | 7,153,056 |
| Missing Values | 75,909 (1.06%) |
| Cap Values Converted | 61 |
| Training Samples | 26,113 |
| Validation Samples | 5,689 |
| Test Samples | 2,689 |
| Input Features | 17 |
| Target Variables | 6 |
| Lookback Window | 168 hours |
| Forecast Horizon | 24 hours |

---

## Files Reference

| File | Purpose |
|------|---------|
| `eda_beijing_air_quality.py` | EDA script |
| `preprocessing_pipeline.py` | Preprocessing pipeline |
| `eda_output/eda_report.html` | Interactive EDA report |
| `processed/README.md` | Data loading instructions |
| `processed/metadata.json` | All configuration parameters |

---

*Generated: December 14, 2025*
