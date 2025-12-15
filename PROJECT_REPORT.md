# PRSA Beijing Multi‑Site Air Quality Forecasting — Project Report

This repository builds an end‑to‑end forecasting pipeline for the UCI PRSA Beijing Multi‑Site Air Quality dataset: exploratory analysis, leakage‑safe preprocessing, a baseline suite, and a custom “strong” spatio‑temporal model.

---

## 1) Problem Statement & Task Setup

**Goal:** forecast 6 pollutants for 12 stations in Beijing for the next 24 hours using the past 7 days of observations.

**Fixed task parameters** (see `processed/metadata.json`):
- Stations `N=12`
- Lookback `L=168` hours
- Horizon `H=24` hours
- Targets `D=6`: `PM2.5, PM10, SO2, NO2, CO, O3`
- Inputs `F=17`: 6 pollutants + 5 meteorology + wind dir sin/cos + hour/month cyclical features (see `processed/feature_list.json`)

**Shapes used across the codebase**:
- `X`: `(samples, 168, 12, 17)` — **scaled** (RobustScaler, TRAIN‑fit only)
- `Y`: `(samples, 24, 12, 6)` — **raw units**, but **zeros at missing**
- `Y_mask`: `(samples, 24, 12, 6)` — 1 observed, 0 missing (**must mask loss/metrics**)

---

## 2) Exploratory Data Analysis (EDA)

EDA is implemented in `eda_beijing_air_quality.py` and writes a full report to `eda_output/` (notably `eda_output/eda_report.html`).

### 2.0 Methods (what was analyzed)
- Loaded all 12 station CSVs, constructed a unified `datetime` index, and verified hourly continuity and row counts per station.
- Quantified missingness per station/feature and exported CSV summaries + heatmaps.
- Computed descriptive statistics (count/mean/std/percentiles) and distribution/seasonality plots for each feature.
- Compared station relationships using inter-station correlation (PM2.5 as a reference signal) to motivate graph-based modeling.

### 2.1 Dataset inventory
From `DOCUMENTATION.md` / EDA script outputs:
- 12 station CSVs, each with **35,064** hourly rows
- Time range: **2013‑03‑01 00:00** to **2017‑02‑28 23:00** (4 years)
- Total rows across stations: **420,768**

### 2.2 Missingness and data integrity
Key outcomes (also visualized in `eda_output/missingness_analysis.png` and tables like `eda_output/missingness_by_station_feature.csv`):
- All stations share full hourly coverage (no timestamp gaps).
- Missingness is low (~1% overall) and broadly distributed (pollutants have the most missingness; meteorology is nearly complete).
- Because missing targets are stored as `0` in tensors, the project treats **masking** as first‑class: metrics and training losses must use `Y_mask`.

**Missingness by feature (all stations, raw)** from `eda_output/missingness_by_station_feature.csv`:

| feature | missing_pct |
|---|---:|
| PM2.5 | 2.08% |
| PM10 | 1.53% |
| SO2 | 2.14% |
| NO2 | 2.88% |
| CO | 4.92% |
| O3 | 3.16% |
| TEMP | 0.09% |
| PRES | 0.09% |
| DEWP | 0.10% |
| RAIN | 0.09% |
| wd | 0.43% |
| WSPM | 0.08% |

### 2.3 Distribution & temporal patterns
From `eda_output/distributions_seasonality.png` / EDA report:
- Strong seasonality (winter higher PM, summer lower).
- Diurnal cycles:
  - PM2.5 tends to peak at night/evening.
  - O3 peaks during afternoon (photochemical production).
- Heavy tails / outliers exist, motivating robust scaling.

**Target summary stats (raw units, pooled across stations)** from `eda_output/stats_overall.csv`:

| pollutant | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|
| PM2.5 | 79.79 | 55 | 242 | 999 |
| PM10 | 104.6 | 82 | 279 | 999 |
| SO2 | 15.83 | 7 | 60 | 500 |
| NO2 | 50.64 | 43 | 117 | 290 |
| CO | 1230.77 | 900 | 3500 | 10000 |
| O3 | 57.37 | 45 | 177 | 1071 |

Note: the maxima `PM2.5=999`, `PM10=999`, `CO=10000` match known sensor cap values; preprocessing converts these to `NaN` before imputation.

### 2.4 Station relationships
From `eda_output/correlation_matrix.png` and the preprocessing graph build:
- Stations are strongly correlated (especially for PM2.5), supporting graph‑based modeling.

---

## 3) Preprocessing Pipeline (v2.1)

Preprocessing is implemented in `preprocessing_pipeline_v2.1.py` and produces versioned artifacts under `processed/` (see `processed/README.md`).

### 3.0 End‑to‑end pipeline steps (methods + outputs)
The full run is logged in `processed/reports/preprocessing_log.txt`. Key steps and current-snapshot outputs:

1. **Load raw station CSVs** → 12 stations × 35,064 hours; combined **420,768** rows.
2. **Cap value handling (mode A)** → convert sensor caps to `NaN` (`PM2.5=999`, `PM10=999`, `CO=10000`; current snapshot: **60** total; see `processed/reports/cap_values_report.csv`).
3. **Feature engineering** → 17 features (pollutants + meteo + `wd_sin/wd_cos` + cyclic time; stored in `processed/feature_list.json`).
4. **Build raw tensor** → `(T=35,064, N=12, F=17)`; overall missing **75,909 / 7,153,056 ≈ 1.06%** across all tensor entries.
5. **Split-first** (time boundaries) → TRAIN **26,304** hours; VAL **5,880** hours; TEST **2,880** hours.
6. **TRAIN-only medians** (per station × feature) → `(12, 17)` fallback values for imputation.
7. **Imputation** → causal ffill-only + median fallback for P1/LightGBM; non-causal interpolation+ffill+bfill for P2 (both preserve masks).
8. **Scaling** → `RobustScaler` fit on TRAIN only for `X` (median/IQR; params saved to `processed/P1_deep/scaler_params.json`).
9. **Window generation** → P1 window counts: TRAIN **26,113**; VAL **5,689**; TEST **2,689** (each sample produces `X:(168,12,17)`, `Y:(24,12,6)`).
10. **LightGBM tabular dataset** → `processed/tabular_lgbm/lgbm_train.csv`: **313,344 rows × 317 cols** (plus val/test); strict lagging avoids leakage (lags: 1/2/3/6/12/24/48/72/168, rolling windows: 24/72/168; no lag0).
11. **Static graph build (TRAIN only)** → top‑k (k=4) correlation adjacency with **78** non‑zero entries (see `processed/graphs/adjacency_corr_topk.npy`).
12. **Validation tests** → all preprocessing checks passed (see the “VALIDATION TESTS” block in the log).

### 3.1 Leakage‑safe design principles
The pipeline is designed to avoid subtle leakage:
- **Split first** (time‑based boundaries) before imputation/scaling/graph building.
- Fit scalers on **TRAIN only**.
- Build the static graph on **TRAIN only**.
- Window generation does not cross split boundaries.

### 3.2 Feature engineering
The input feature list (ordered) is stored in `processed/feature_list.json`:
- Pollutants: `PM2.5, PM10, SO2, NO2, CO, O3`
- Meteorology: `TEMP, PRES, DEWP, RAIN, WSPM`
- Wind direction: `wd_sin, wd_cos`
- Time features: `hour_sin, hour_cos, month_sin, month_cos`

### 3.3 Masks and imputation
Two pipelines are produced:

**P1_deep (`processed/P1_deep/`)** — used by deep models:
- Produces `X_mask` and `Y_mask`.
- **Causal imputation for inputs**: forward-fill only, then TRAIN-median fallback for any leading gaps (**no bfill**; see `causal_impute()` in `preprocessing_pipeline_v2.1.py`).
- **Targets are stored in raw units** with `NaN → 0` for storage; `Y_mask` preserves what was truly observed and must be used in loss/metrics.

**P2_simple (`processed/P2_simple/`)** — used for simple baselines:
- **Non-causal full imputation**: interpolation (limit 6) + ffill/bfill (see `non_causal_impute()`); this is convenient for simple models but uses future values during imputation.
- Still ships `Y_mask` (important for evaluation correctness, since `Y` stores 0 at missing).

### 3.4 Scaling
- RobustScaler is fit on TRAIN only and applied to all splits.
- In the current configuration, `Y` remains in **raw units** (`scale_targets=false` in `processed/metadata.json`).

### 3.5 Supervised window generation and splits
Output shapes:
- `X`: `(samples, 168, 12, 17)`
- `Y`: `(samples, 24, 12, 6)`

**Window origin convention (leakage-safe)**:
- Let `t` be the “origin” timestamp (the last timestamp included in `X`).
- `X` uses the last `L=168` hours ending at `t`: `[t-L+1, …, t]`.
- `Y` predicts the next `H=24` hours starting at `t+1`: `[t+1, …, t+H]`.

Current snapshot window counts (from `processed/reports/preprocessing_log.txt`):
- Train: **26,113** samples
- Val: **5,689** samples
- Test: **2,689** samples

Time splits (see `processed/metadata.json`):
- Train: 2013‑03‑01 → 2016‑02‑29
- Val: 2016‑03‑01 → 2016‑10‑31
- Test: 2016‑11‑01 → 2017‑02‑28

### 3.6 Graph construction
Static adjacency is built from TRAIN correlations (PM2.5):
- `processed/graphs/adjacency_corr_topk.npy`: top‑k sparse adjacency (k=4)
- `processed/graphs/station_list.json`: station ordering for adjacency axes

The current snapshot graph build reports **78** non-zero entries (k=4) in `processed/reports/preprocessing_log.txt`.

---

## 4) Baseline Suite

Baselines live in `baseline/` and are runnable via `baseline/scripts/run.py` (see `baseline/README.md`).

### 4.0 Methods (data + training setup)
- **Data sources**:
  - Deep baselines (LSTM/TCN/STGCN/GWNet) load `processed/P1_deep/*.npz` (scaled `X`, raw `Y`, and masks).
  - LightGBM uses `processed/tabular_lgbm/lgbm_{split}.csv` with strictly causal lag/rolling features.
  - Graph baselines load the fixed adjacency from `processed/graphs/adjacency_corr_topk.npy`.
- **Leakage & masking**:
  - All reported metrics are computed on the TEST set and are masked by `Y_mask`.
  - Naive/seasonal baselines must inverse-transform the scaled pollutant channels in `X` back to raw units before predicting.

### 4.1 Models included
- **B0 Naive Persistence**: `y(t+h)=y(t)`
- **B1 Seasonal Naive 24h**: `y(t+h)=y(t+h-24)`
- **B2 LightGBM**: tabular autoregressive features (lags/rolling + station id)
- **B3 LSTM**: direct multi‑horizon sequence model
- **B4 TCN**: temporal convolutional network
- **B5 STGCN**: spatio‑temporal graph conv (fixed graph)
- **B6 Graph WaveNet**: adaptive graph temporal model

### 4.2 Evaluation protocol (critical)
All baseline evaluation is centralized in `baseline/evaluation/evaluate.py`:
- Predictions must be `(samples, 24, 12, 6)`.
- All metrics are **masked** by `Y_mask` (since missing targets are stored as 0).
- Metrics include overall, per‑horizon, and per‑pollutant values.
- Per‑pollutant reporting writes:
  - `baseline/results/metrics_per_pollutant.csv`
  - `baseline/results/metrics_overall.csv` (includes macro averages to reduce CO dominance)

---

## 5) Baseline Results (Current Snapshot)

The latest baseline exports are under `baseline/results/`.

### 5.0 Methods (how results are produced)
- All baselines write predictions in shape `(samples, 24, 12, 6)` and evaluate with `baseline/evaluation/evaluate.py`.
- Reported metrics are **masked** MAE/RMSE/sMAPE plus macro-averages (equal weight per pollutant) to prevent CO magnitude from dominating.

### 5.1 Overall comparison
From `baseline/results/metrics_overall.csv` (masked):
- LightGBM is the best **overall** baseline by macro_MAE (182.237) in this snapshot.
- Naive persistence is strongest at **h=1** (MAE_h1 60.819) but degrades by **h=24** (MAE_h24 255.227).
- Seasonal naive is competitive only when diurnal patterns dominate; otherwise it can be worse than persistence.

### 5.2 Per‑pollutant reporting
From `baseline/results/metrics_per_pollutant.csv`:
- CO has much larger magnitude, so:
  - “Overall MAE” across pollutants can be dominated by CO.
  - Macro‑averaged metrics provide a fairer cross‑pollutant comparison.

### 5.3 Plots
Representative plots are stored under `baseline/results/plots/`, including:
- `*_mae_vs_horizon.png` and `*_mae_by_pollutant.png`
- `seasonal_naive_sanity.png` (true vs seasonal baseline across horizons)

<!-- BASELINE_RESULTS_START -->

### 5.4 Latest baseline results (auto-generated)

This section is generated from:
- `baseline/results/metrics_overall.csv`
- `baseline/results/metrics_per_pollutant.csv`

Regenerate after reruns with:

```bash
python scripts/update_project_report.py
```

**Overall + macro MAE (masked)**

| model | MAE | macro_MAE | MAE_h1 | MAE_h6 | MAE_h12 | MAE_h24 |
|---|---:|---:|---:|---:|---:|---:|
| lightgbm | 180.994 | 182.237 | 87.205 | 155.131 | 188.603 | 225.830 |
| naive_persistence | 195.757 | 197.094 | 60.819 | 162.775 | 208.991 | 255.227 |
| tcn | 197.514 | 198.905 | 158.952 | 173.278 | 197.184 | 235.474 |
| lstm | 227.942 | 229.522 | 209.124 | 216.625 | 228.896 | 245.183 |
| gwnet | 249.602 | 251.325 | 246.335 | 247.532 | 249.442 | 252.665 |
| seasonal_naive_24h | 254.928 | 256.660 | 254.494 | 254.583 | 254.923 | 255.227 |
| stgcn | 333.515 | 335.951 | 333.646 | 333.867 | 333.394 | 333.377 |

**CO focus (masked MAE, raw units)**

| model | CO_MAE | CO_MAE_h1 | CO_MAE_h6 | CO_MAE_h12 | CO_MAE_h24 |
|---|---:|---:|---:|---:|---:|
| lightgbm | 922.082 | 453.580 | 792.321 | 958.347 | 1148.320 |
| naive_persistence | 993.039 | 318.087 | 831.433 | 1058.164 | 1290.773 |
| tcn | 1024.914 | 840.672 | 907.614 | 1021.261 | 1212.051 |
| lstm | 1169.578 | 1100.652 | 1126.277 | 1168.251 | 1246.810 |
| gwnet | 1277.015 | 1260.946 | 1266.943 | 1276.417 | 1291.409 |
| seasonal_naive_24h | 1289.223 | 1286.848 | 1287.481 | 1289.263 | 1290.773 |
| stgcn | 1781.227 | 1782.146 | 1783.873 | 1781.296 | 1779.038 |

<!-- BASELINE_RESULTS_END -->

---

## 6) Custom Strong Model: WG‑DGTM

The custom PyTorch model package is under `model/` with training/eval CLIs:
- Train: `python -m model.scripts.run_train --config <cfg>`
- Eval: `python -m model.scripts.run_eval --config <cfg> --ckpt <path>`

### 6.1 High‑level architecture
WG‑DGTM combines:
1) **Feature encoder** (per time/station): `x(t,i) → h(t,i)` (Linear + LayerNorm + GELU)
2) **Wind‑gated dynamic graph** per timestep:
   - `A_static`: top‑k correlation adjacency (TRAIN‑only)
   - `A_learn`: learned adjacency from node embeddings
   - `A_dyn(t)`: attention adjacency from node states
   - fuse: `A_t = RowNormalize( softplus(a)A_static + softplus(b)A_learn + softplus(c)A_dyn(t) + I )`
3) **Spatial message passing**: `z(t) = A_t · h(t) · W`
4) **Temporal backbone (TCN)** per station:
   - causal, dilated temporal blocks
5) **Horizon‑aware decoder**:
   - horizon embeddings prevent degenerate constant forecasts

Implementation references:
- Graph builder: `model/modules/dynamic_graph.py`
- TCN backbone: `model/modules/tcn.py`
- Model assembly: `model/models/wgdgtm.py`
- Design rationale: `model/DESIGN_NOTE.md`

### 6.2 Wind gating details
WG‑DGTM uses physically interpretable wind components:
- `u = WSPM * wd_cos`, `v = WSPM * wd_sin`
- gate `g(t,i) = sigmoid(MLP([u, v, WSPM]))`
- bias dynamic attention logits: `logits(t,i,*) += λ · g(t,i)`

Because inputs are scaled, the model unscales only the wind channels for gating (see `model/models/wgdgtm.py`).

### 6.3 Loss, masking, and scale handling
Training uses **std‑weighted masked MAE**:
- Compute per‑pollutant std on TRAIN observed targets only.
- Weight each pollutant by `1/(std + eps)` to reduce CO dominance.
- Compute loss on observed positions only using `Y_mask`.

Default training config (see `model/configs/wgdgtm.yaml`): AdamW (`lr=1e-3`, `weight_decay=1e-4`), `batch_size=64`, `epochs=50`, grad clip `5.0`, early stopping patience `8`.

Loss code: `model/losses/masked_losses.py`.

### 6.4 Results (current snapshot, masked TEST metrics)
Metrics are written by `model/evaluation/evaluate.py` under `model/results/**/metrics/`.

Training snapshots (VAL macro_MAE from `model/results/**/logs/train_history.csv`):
- WG‑DGTM best val_macro_MAE: **70.979** (epoch 22)
- WG‑DGTM residual+multihead best val_macro_MAE: **69.004** (epoch 8)

**Macro averages (equal weight per pollutant)**:

| run | macro_MAE | macro_RMSE | macro_sMAPE |
|---|---:|---:|---:|
| WG‑DGTM (`model/results/metrics/`) | 179.533 | 269.082 | 29.422 |
| WG‑DGTM residual+multihead (`model/results/wgdgtm_residual_multihead/metrics/`) | 184.897 | 278.924 | 30.026 |

**Macro MAE at selected horizons** (computed as the mean of per‑pollutant `MAE_h*`):

| model | macro_MAE | macro_MAE_h1 | macro_MAE_h6 | macro_MAE_h12 | macro_MAE_h24 |
|---|---:|---:|---:|---:|---:|
| lightgbm (best baseline) | 182.237 | 87.819 | 156.200 | 189.895 | 227.379 |
| WG‑DGTM | 179.533 | 132.354 | 155.645 | 182.838 | 216.645 |
| WG‑DGTM residual+multihead | 184.897 | 67.734 | 155.974 | 193.889 | 237.722 |

**WG‑DGTM vs LightGBM (per‑pollutant MAE, masked)**:

| pollutant | lightgbm | WG‑DGTM | Δ (WG‑DGTM − lightgbm) |
|---|---:|---:|---:|
| PM2.5 | 56.505 | 52.279 | -4.226 |
| PM10 | 65.302 | 60.752 | -4.550 |
| SO2 | 9.411 | 9.187 | -0.224 |
| NO2 | 23.949 | 23.023 | -0.926 |
| CO | 922.082 | 916.276 | -5.806 |
| O3 | 16.171 | 15.678 | -0.493 |

---

## 7) WG‑DGTM Upgrades: Residual Forecasting + Multi‑Head Output

To improve short‑horizon behavior (especially for large‑scale CO) while keeping multi‑horizon stability, the model supports two upgrades (enabled in `model/configs/wgdgtm_residual_multihead.yaml`).

### 7.1 Residual forecasting over persistence (Upgrade #1)
Instead of predicting `Y` directly:
- baseline (raw): `y_base(t+h) = y(t)` for all horizons
- model predicts residual: `Δ_hat(t+h)`
- final: `y_hat = y_base + Δ_hat`

The baseline uses the **last lookback timestep** `X[:, L-1]` and inverse‑transforms only pollutant channels to raw units.

Code: `model/modules/residual_baseline.py` + integrated in `model/models/wgdgtm.py` via `use_residual_forecasting`.

### 7.2 Multi‑head decoder (Upgrade #2)
Instead of one shared output head for all pollutants, the decoder can use **one small head per pollutant**:
- each head outputs `(B, H, N, 1)`
- concatenate → `(B, H, N, 6)`

This reduces cross‑pollutant interference while preserving a shared backbone.

Code: `model/modules/multihead_decoder.py` + integrated in `model/models/wgdgtm.py` via `decoder_type`.

### 7.3 Sanity tests
Lightweight unit tests validate:
- persistence baseline invariants and shapes
- multihead output shape

Run: `python -m unittest -v model.tests.test_residual_multihead`

### 7.4 Upgrade results (current snapshot)
Residual forecasting + multi-head output substantially improves **h=1** behavior but is worse at longer horizons in this snapshot:
- macro_MAE_h1: **132.354 → 67.734**
- macro_MAE_h24: **216.645 → 237.722**
- overall macro_MAE: **179.533 → 184.897**

---

## 8) Reproducibility & How to Run

### 8.1 EDA
```bash
python eda_beijing_air_quality.py
```

### 8.2 Preprocessing
```bash
python preprocessing_pipeline_v2.1.py
```

### 8.3 Baselines
```bash
python -m baseline.scripts.run --model all --config baseline/configs/default.yaml
```

### 8.4 WG‑DGTM
```bash
# Original WG-DGTM
python -m model.scripts.run_train --config model/configs/wgdgtm.yaml
python -m model.scripts.run_eval  --config model/configs/wgdgtm.yaml --ckpt model/results/wgdgtm/checkpoints/best.pt

# Upgraded residual + multi-head WG-DGTM
python -m model.scripts.run_train --config model/configs/wgdgtm_residual_multihead.yaml
python -m model.scripts.run_eval  --config model/configs/wgdgtm_residual_multihead.yaml --ckpt model/results/wgdgtm_residual_multihead/checkpoints/best.pt
```

Outputs are written under:
- `baseline/results/`
- `model/results/<experiment_name>/` (checkpoints, logs, metrics, plots)

Note: older snapshots may place WG‑DGTM outputs directly under `model/results/` (e.g., `model/results/checkpoints/best.pt`, `model/results/metrics/`).

---

## 9) Notes / Common Pitfalls

- **Never compute loss/metrics without `Y_mask`**: missing targets are stored as 0 in `Y`.
- **Naive baselines must operate in raw units**: pollutant channels in `X` are scaled; inverse‑transform before using them as predictions/baselines.
- **Horizon correctness**: horizon 1 is index 0; horizon 24 is index 23.
