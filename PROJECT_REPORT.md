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

### 2.1 Dataset inventory
From `DOCUMENTATION.md` / EDA script outputs:
- 12 station CSVs, each with **35,064** hourly rows
- Time range: **2013‑03‑01 00:00** to **2017‑02‑28 23:00** (4 years)
- Total rows across stations: **420,768**

### 2.2 Missingness and data integrity
Key outcomes (also visualized in `eda_output/missingness_analysis.png` and tables like `eda_output/missingness_by_station_feature.csv`):
- All stations share full hourly coverage (no timestamp gaps).
- Missingness is low (~1% overall) and broadly distributed.
- Because missing targets are stored as `0` in tensors, the project treats **masking** as first‑class: metrics and training losses must use `Y_mask`.

### 2.3 Distribution & temporal patterns
From `eda_output/distributions_seasonality.png` / EDA report:
- Strong seasonality (winter higher PM, summer lower).
- Diurnal cycles:
  - PM2.5 tends to peak at night/evening.
  - O3 peaks during afternoon (photochemical production).
- Heavy tails / outliers exist, motivating robust scaling.

### 2.4 Station relationships
From `eda_output/correlation_matrix.png` and the preprocessing graph build:
- Stations are strongly correlated (especially for PM2.5), supporting graph‑based modeling.

---

## 3) Preprocessing Pipeline (v2.1)

Preprocessing is implemented in `preprocessing_pipeline_v2.1.py` and produces versioned artifacts under `processed/` (see `processed/README.md`).

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
- Performs conservative imputation (ffill then bfill) while keeping masks so training/eval ignore imputed targets.

**P2_simple (`processed/P2_simple/`)** — used for simple baselines:
- Fully imputes with interpolation/ffill/bfill.
- Still ships `Y_mask` (important for evaluation correctness).

### 3.4 Scaling
- RobustScaler is fit on TRAIN only and applied to all splits.
- In the current configuration, `Y` remains in **raw units** (`scale_targets=false` in `processed/metadata.json`).

### 3.5 Supervised window generation and splits
Output shapes:
- `X`: `(samples, 168, 12, 17)`
- `Y`: `(samples, 24, 12, 6)`

Time splits (see `processed/metadata.json`):
- Train: 2013‑03‑01 → 2016‑02‑29
- Val: 2016‑03‑01 → 2016‑10‑31
- Test: 2016‑11‑01 → 2017‑02‑28

### 3.6 Graph construction
Static adjacency is built from TRAIN correlations (PM2.5):
- `processed/graphs/adjacency_corr_topk.npy`: top‑k sparse adjacency (k=4)
- `processed/graphs/station_list.json`: station ordering for adjacency axes

---

## 4) Baseline Suite

Baselines live in `baseline/` and are runnable via `baseline/scripts/run.py` (see `baseline/README.md`).

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

### 5.1 Overall comparison
From `baseline/results/metrics_overall.csv` (masked):
- LightGBM currently gives the best overall MAE among baselines listed.
- Naive persistence is strong at short horizons (as expected).
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

Loss code: `model/losses/masked_losses.py`.

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

---

## 9) Notes / Common Pitfalls

- **Never compute loss/metrics without `Y_mask`**: missing targets are stored as 0 in `Y`.
- **Naive baselines must operate in raw units**: pollutant channels in `X` are scaled; inverse‑transform before using them as predictions/baselines.
- **Horizon correctness**: horizon 1 is index 0; horizon 24 is index 23.
