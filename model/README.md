# WG-DGTM (Wind-Gated Dynamic Graph + TCN) — Repository Add-on

This package implements a “strong” spatio-temporal forecasting model for the PRSA Beijing multi-site air-quality task:

- **Input**: `X` with shape `(B, L=168, N=12, F)`
- **Output**: `Yhat` with shape `(B, H=24, N=12, D=6)` in **RAW** target units
- **Targets (fixed order)**: `[PM2.5, PM10, SO2, NO2, CO, O3]`
- **Strict setting A**: uses only past features from the lookback window (no future meteorology/covariates)

## How to Train

```bash
python -m model.scripts.run_train --config model/configs/wgdgtm.yaml
```

Outputs (default): `model/results/` (checkpoints, logs, metrics, plots).

### Multi-GPU

By default the provided config enables `torch.nn.DataParallel` when multiple GPUs are visible. Control this via:

- `training.use_data_parallel` (true/false)
- `training.gpus` (null for all GPUs, or e.g. `[0,1]`)

## How to Evaluate

```bash
python -m model.scripts.run_eval --config model/configs/wgdgtm.yaml --ckpt model/results/checkpoints/best.pt
```

## Model Overview (WG-DGTM)

At each time step `t` the model builds a directed adjacency matrix:

1. **Static prior** `A_static` loaded from `processed/graphs/adjacency_corr_topk.npy`
2. **Learned graph** `A_learn` from trainable node embeddings `e_i`
3. **Dynamic graph** `A_dyn(t)` via node-state attention, wind-gated:
   - wind features from `X`: `wd_sin`, `wd_cos`, `WSPM`
   - `u = WSPM * wd_cos`, `v = WSPM * wd_sin`
   - gate `g(t,i) = sigmoid(MLP([u,v,WSPM]))`
   - attention logits include a source-node bias: `logits += lambda * g(t,i)`

The fused adjacency is:

`A_t = RowNormalize( softplus(a)*A_static + softplus(b)*A_learn + softplus(c)*A_dyn(t) + I )`

Spatial message passing is applied per time step, then a **dilated TCN** runs over time per station. A **horizon embedding** is used in the decoder so outputs differ by horizon and avoid collapsed constant forecasts.

## Upgrades (Residual + Multi-head)

### Residual forecasting over persistence

Optionally, the model predicts residuals over a persistence baseline:

- `y_base(t+h) = y(t)` for all `h=1..H` (computed from the **last lookback step** in `X`)
- model predicts `Δ_hat(t+h)`
- final prediction: `y_hat(t+h) = y_base(t+h) + Δ_hat(t+h)`

`y_base` is computed in **raw units** by inverse-transforming only the 6 pollutant channels from `X[:, L-1]` using `processed/P1_deep/scaler.pkl`.

Enable via `model.use_residual_forecasting: true`.

### Multi-head decoder (one head per pollutant)

To reduce cross-pollutant interference, the decoder can use one small head per pollutant and concatenate outputs to `(B,H,N,6)`.

Enable via `model.decoder.type: "multihead"` (default remains `"shared"` in `model/configs/wgdgtm.yaml`).

## Loss / Metrics

Training uses **pollutant-wise std-weighted masked MAE** (observed positions only):

- compute `std[d]` from TRAIN targets where `Y_mask==1`
- `weight[d] = 1 / (std[d] + eps)`
- `loss = sum(mask * weight[d] * |pred - y|) / sum(mask)`

Evaluation exports per-pollutant metrics (masked): MAE/RMSE/sMAPE and MAE at horizons {1,6,12,24}.

## Sanity tests

Run lightweight component checks:

```bash
python -m unittest -v model.tests.test_residual_multihead
```
