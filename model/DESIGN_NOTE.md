# Design Note — Wind-Gated Dynamic Graph + TCN (WG‑DGTM)

## Why a Dynamic Graph?
Air-quality dynamics are **spatially coupled**: stations correlate due to shared regional emissions, transport, and local meteorology. A fixed graph captures average correlation structure, but real propagation is **time-varying** (e.g., wind-driven advection).

WG‑DGTM builds a **directed** adjacency at every time step:
- `A_static` anchors the model to known spatial structure (top‑k correlation graph).
- `A_learn` captures persistent residual structure not present in `A_static`.
- `A_dyn(t)` adapts connectivity based on the current node states (attention).

## Why Wind Gating?
Wind direction and speed are a first-order driver of pollutant transport. We compute:
- `u = WSPM * wd_cos`, `v = WSPM * wd_sin` (wind components)
- a scalar gate `g(t,i) ∈ (0,1)` per node/time

This gate biases dynamic attention logits for outgoing edges from node `i`:
`logits(t,i,*) += λ · g(t,i)`

Intuition: when wind is strong/consistent, the model can increase directed propagation from the source node; when wind is calm/uncertain, the model can fall back to `A_static`/`A_learn`.

## Why a TCN Backbone?
TCNs provide:
- **causal** receptive fields (no future leakage)
- **long-range** modeling via dilation
- efficient direct multi-horizon forecasting (no recursive rollouts)

## Preventing Horizon Collapse
Direct multi-horizon heads can collapse to near-constant outputs if the decoder is not horizon-aware. WG‑DGTM uses learnable **horizon embeddings** and a horizon-conditioned MLP decoder to produce distinct `h=1..24` predictions.

## Handling Scale Differences Across Pollutants
CO has much larger magnitude than other pollutants. A naive MAE loss is dominated by CO and encourages median-like forecasts for all outputs. WG‑DGTM trains with a **std-weighted masked MAE** so each pollutant contributes comparably (computed using TRAIN observed targets only), while evaluation remains in raw units and reports per-pollutant metrics.

