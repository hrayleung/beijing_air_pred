# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an end-to-end air quality forecasting research project for the UCI Beijing Multi-Site Air Quality dataset. It includes exploratory data analysis, leakage-safe preprocessing, a comprehensive baseline suite (7 models), and a custom spatio-temporal graph neural network (WG-DGTM).

**Task**: Forecast 6 pollutants (PM2.5, PM10, SO2, NO2, CO, O3) for 12 Beijing stations over the next 24 hours using the past 7 days of observations.

## Key Commands

### Setup
```bash
pip install -r baseline/requirements.txt
```

### Data Pipeline
```bash
# Run exploratory data analysis (outputs to eda_output/)
python eda_beijing_air_quality.py

# Run preprocessing pipeline (outputs to processed/)
python preprocessing_pipeline_v2.1.py
```

### Baseline Models
```bash
# Run specific baseline model
python -m baseline.scripts.run --model <name> --config baseline/configs/<name>.yaml

# Available models: naive, seasonal, lgbm, lstm, tcn, stgcn, gwnet, all
# Examples:
python -m baseline.scripts.run --model naive --config baseline/configs/default.yaml
python -m baseline.scripts.run --model lstm --config baseline/configs/lstm.yaml
python -m baseline.scripts.run --model all --config baseline/configs/default.yaml

# Run debug/validation checks
python -m baseline.scripts.debug_checks --config baseline/configs/default.yaml
```

### Custom Model (WG-DGTM)
```bash
# Train WG-DGTM
python -m model.scripts.run_train --config model/configs/wgdgtm.yaml

# Train with residual forecasting + multi-head decoder
python -m model.scripts.run_train --config model/configs/wgdgtm_residual_multihead.yaml

# Evaluate trained model
python -m model.scripts.run_eval --config model/configs/wgdgtm.yaml --ckpt model/results/checkpoints/best.pt

# Run unit tests
python -m unittest -v model.tests.test_residual_multihead
```

### Utilities
```bash
# Update auto-generated sections in PROJECT_REPORT.md
python scripts/update_project_report.py
```

## Architecture & Code Organization

### High-Level Structure
```
.
├── eda_beijing_air_quality.py           # EDA script
├── preprocessing_pipeline_v2.1.py        # Main preprocessing pipeline
├── PRSA_Data_20130301-20170228/          # Raw 12-station CSV files
├── processed/                            # Preprocessed datasets (P1_deep, P2_simple, tabular_lgbm, graphs)
├── baseline/                             # Baseline models package
│   ├── configs/                          # YAML configs per model
│   ├── data/                             # Data loaders (NPZ, LGBM CSV, graphs)
│   ├── models/                           # 7 baseline implementations
│   ├── training/                         # PyTorch + LightGBM trainers
│   ├── evaluation/                       # Masked metrics, evaluation, plots
│   ├── scripts/run.py                    # Main CLI entrypoint
│   └── results/                          # Output: metrics, plots, checkpoints
└── model/                                # WG-DGTM custom model package
    ├── configs/                          # YAML configs
    ├── data/                             # NPZ dataset loader
    ├── modules/                          # Model components (TCN, dynamic graph, decoders)
    ├── models/wgdgtm.py                  # Full model assembly
    ├── losses/                           # Std-weighted masked MAE
    ├── metrics/                          # Masked metrics
    ├── training/trainer.py               # Training loop
    ├── evaluation/evaluate.py            # Evaluation script
    ├── scripts/                          # run_train.py, run_eval.py
    └── results/                          # Output: checkpoints, logs, metrics, plots
```

### Critical Data Shapes and Conventions

**Fixed task parameters** (see `processed/metadata.json`):
- Lookback window `L=168` hours (7 days)
- Forecast horizon `H=24` hours
- Stations `N=12` (alphabetically ordered)
- Targets `D=6`: PM2.5, PM10, SO2, NO2, CO, O3
- Input features `F=17`: 6 pollutants + 5 meteorology + 2 wind (sin/cos) + 4 time (hour/month sin/cos)

**Tensor shapes**:
- `X`: `(samples, 168, 12, 17)` — **SCALED** inputs (RobustScaler fit on TRAIN only)
- `Y`: `(samples, 24, 12, 6)` — **RAW** target units, with **zeros at missing positions**
- `Y_mask`: `(samples, 24, 12, 6)` — 1=observed, 0=missing (CRITICAL for loss/metrics)

**Splits** (time-based, no shuffling):
- Train: 2013-03-01 to 2016-02-29 (26,113 samples)
- Val: 2016-03-01 to 2016-10-31 (5,689 samples)
- Test: 2016-11-01 to 2017-02-28 (2,689 samples)

### Preprocessing Architecture

The preprocessing pipeline (`preprocessing_pipeline_v2.1.py`) implements a **leakage-safe design**:

1. **Split first**: time-based boundaries before imputation/scaling/graph building
2. **Fit on TRAIN only**: scalers, adjacency matrices, median fallbacks
3. **Window generation**: no cross-boundary windows; each sample uses `X[t-167:t]` to predict `Y[t+1:t+24]`
4. **Two pipelines**:
   - **P1_deep** (`processed/P1_deep/`): causal forward-fill + median imputation, with masks (for deep models)
   - **P2_simple** (`processed/P2_simple/`): non-causal interpolation+ffill+bfill (for naive baselines)
5. **Outputs**:
   - `processed/P1_deep/{train,val,test}.npz` with keys: X, Y, X_mask, Y_mask, X_flat, Y_flat
   - `processed/P2_simple/{train,val,test}.npz` with keys: X, Y, X_flat, Y_flat
   - `processed/tabular_lgbm/lgbm_{train,val,test}.csv` with lagged/rolling features
   - `processed/graphs/adjacency_corr_topk.npy` (12×12 top-k=4 correlation graph)
   - `processed/metadata.json`, `processed/feature_list.json`, `processed/target_list.json`

### Baseline Suite Architecture

7 models with unified interface (`baseline/models/`):
- **B0 Naive Persistence**: y(t+h) = y(t)
- **B1 Seasonal Naive 24h**: y(t+h) = y(t+h-24)
- **B2 LightGBM**: tabular ML with lag/rolling features
- **B3 LSTM**: direct multi-horizon seq2seq
- **B4 TCN**: dilated temporal convolutions
- **B5 STGCN**: spatio-temporal graph convnet (fixed graph)
- **B6 Graph WaveNet**: adaptive graph temporal model

**Evaluation protocol** (`baseline/evaluation/evaluate.py`):
- All metrics are **masked** using `Y_mask` (missing targets stored as 0)
- Computes overall, per-horizon (h=1,6,12,24), and per-pollutant metrics
- Exports `baseline/results/metrics_overall.csv` and `baseline/results/metrics_per_pollutant.csv`
- Includes macro-averages to prevent CO magnitude from dominating overall metrics

### Custom Model (WG-DGTM) Architecture

**Wind-Gated Dynamic Graph + TCN Model** (`model/models/wgdgtm.py`):

1. **Feature encoder**: linear projection + LayerNorm + GELU per (time, station)
2. **Dynamic graph builder** (`model/modules/dynamic_graph.py`):
   - Static prior: top-k correlation adjacency (TRAIN-only)
   - Learned graph: from trainable node embeddings
   - Dynamic graph: attention-based, wind-gated using (u=WSPM×cos, v=WSPM×sin, WSPM)
   - Fusion: `A_t = RowNorm(softplus(a)·A_static + softplus(b)·A_learn + softplus(c)·A_dyn(t) + I)`
3. **Spatial layer**: graph message passing per timestep
4. **TCN backbone** (`model/modules/tcn.py`): causal dilated convolutions per station
5. **Horizon decoder**: uses learnable horizon embeddings to prevent collapsed forecasts

**Two upgrade modes** (see `model/configs/wgdgtm_residual_multihead.yaml`):
- **Residual forecasting**: predict Δ over persistence baseline y_base(t+h)=y(t)
- **Multi-head decoder**: one output head per pollutant to reduce cross-pollutant interference

**Training details**:
- Loss: std-weighted masked MAE (weights computed from TRAIN observed targets only)
- AdamW optimizer (lr=1e-3, weight_decay=1e-4)
- Batch size 64, 50 epochs, early stopping patience 8
- DataParallel support for multi-GPU

## Critical Implementation Details

### 1. Masking is Mandatory
**ALWAYS** use `Y_mask` when computing loss or metrics. Missing targets are stored as `0` in `Y`, so unmasked operations will be incorrect.

```python
# Correct masked MAE
def masked_mae(pred, target, mask):
    abs_error = torch.abs(pred - target) * mask
    return abs_error.sum() / (mask.sum() + 1e-8)

# WRONG (will include zeros from missing values)
mae = torch.abs(pred - target).mean()
```

### 2. Naive Baselines Must Inverse-Transform
Naive/seasonal models use pollutant values from `X`, which are **scaled**. They must inverse-transform to raw units before predicting:

```python
# X is scaled, need raw values for prediction
last_values_scaled = X[:, -1, :, :]  # (B, N, F=17)
scaler = joblib.load('processed/P1_deep/scaler.pkl')
last_raw = scaler.inverse_transform(last_values_scaled.reshape(-1, 17))
last_raw = last_raw.reshape(B, N, 17)
predictions = last_raw[:, :, :6]  # Extract first 6 pollutant features
```

### 3. Horizon Indexing Convention
- Horizon 1 (1-hour-ahead) is stored at index 0
- Horizon 24 (24-hour-ahead) is stored at index 23
- Per-horizon metrics use this 0-indexed convention

### 4. Wind Gating Requires Unscaling
WG-DGTM uses wind features (wd_sin, wd_cos, WSPM) for gating. Since inputs are scaled, the model **unscales only the wind channels** before gating (see `model/models/wgdgtm.py:_unscale_wind_features()`).

### 5. Station and Feature Ordering
- Stations: **alphabetical order** (see `processed/graphs/station_list.json`)
- Features: see `processed/feature_list.json` (pollutants, meteo, wind sin/cos, time sin/cos)
- Targets: see `processed/target_list.json` (PM2.5, PM10, SO2, NO2, CO, O3)

Ordering is consistent across all datasets and adjacency matrices.

### 6. No Data Leakage
- Scalers fit on TRAIN only
- Adjacency matrices built from TRAIN correlations only
- Window generation does not cross split boundaries
- LightGBM features use strict lagging (lags: 1,2,3,6,12,24,48,72,168; no lag-0)

### 7. CO Scale Dominance
CO has much larger magnitude (~1000s μg/m³) than other pollutants (~10s-100s μg/m³). This causes:
- Unweighted MAE to be dominated by CO errors
- Solution: use **std-weighted loss** during training and **macro-averaged metrics** (equal weight per pollutant) for evaluation

## Common Workflows

### Adding a New Baseline Model
1. Implement model class in `baseline/models/<name>.py` following existing interface
2. Add config in `baseline/configs/<name>.yaml`
3. Register model in `baseline/scripts/run.py`
4. Ensure predictions are shape `(samples, 24, 12, 6)` in raw units
5. Run evaluation with `python -m baseline.scripts.run --model <name> --config baseline/configs/<name>.yaml`

### Modifying Preprocessing
1. Edit `preprocessing_pipeline_v2.1.py`
2. Delete `processed/` directory
3. Re-run `python preprocessing_pipeline_v2.1.py`
4. Verify `processed/metadata.json` is updated
5. Re-run at least one baseline to confirm compatibility

### Changing Task Parameters (Not Recommended)
If you must change lookback/horizon:
1. Update `preprocessing_pipeline_v2.1.py` constants
2. Re-run full preprocessing pipeline
3. Update all model configs (baseline and WG-DGTM)
4. Re-train all models from scratch

### Debugging Metric Discrepancies
1. Check `Y_mask` is being used correctly
2. Verify predictions are in raw units (not scaled)
3. Confirm horizon indexing (h=1 is index 0)
4. Check station/feature ordering matches `processed/{feature,target}_list.json`
5. Run `python -m baseline.scripts.debug_checks` to verify data integrity

## Important Files

- `processed/metadata.json` — task parameters, split dates, window counts
- `processed/feature_list.json` — ordered list of 17 input features
- `processed/target_list.json` — ordered list of 6 target pollutants
- `baseline/results/metrics_overall.csv` — baseline comparison table
- `model/DESIGN_NOTE.md` — design rationale for WG-DGTM
- `PROJECT_REPORT.md` — comprehensive project report with auto-generated results

## Notes on Generated Artifacts

The following directories are build outputs and should not be edited manually:
- `eda_output/` — EDA reports and plots
- `processed/` — preprocessed datasets
- `baseline/results/` — baseline metrics, plots, checkpoints
- `model/results/` — WG-DGTM checkpoints, logs, metrics, plots

To regenerate: re-run the corresponding script (EDA, preprocessing, or model training).
