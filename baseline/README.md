# Beijing Air Quality Baseline Models

Modular baseline codebase for multi-horizon air quality forecasting.

## Task Definition (FIXED)

| Parameter | Value | Description |
|-----------|-------|-------------|
| L | 168 | Lookback window (7 days) |
| H | 24 | Forecast horizon (24 hours) |
| N | 12 | Number of stations |
| D | 6 | Number of targets (PM2.5, PM10, SO2, NO2, CO, O3) |
| F | 17 | Number of input features |

## Models

| ID | Model | Type | Description |
|----|-------|------|-------------|
| B0 | `naive` | Baseline | Persistence: y(t+h) = y(t) |
| B1 | `seasonal` | Baseline | Seasonal 24h: y(t+h) = y(t-24+h) |
| B2 | `lgbm` | ML | LightGBM with lag features |
| B3 | `lstm` | DL | LSTM direct multi-horizon |
| B4 | `tcn` | DL | Temporal Convolutional Network |
| B5 | `stgcn` | DL | Spatio-Temporal Graph ConvNet |
| B6 | `gwnet` | DL | Graph WaveNet |

## Quick Start

```bash
# Run naive baseline
python -m baseline.scripts.run --model naive --config baseline/configs/default.yaml

# Run LSTM
python -m baseline.scripts.run --model lstm --config baseline/configs/lstm.yaml

# Run all models
python -m baseline.scripts.run --model all --config baseline/configs/default.yaml
```

## CLI Usage

```bash
python -m baseline.scripts.run --model <name> --config <path> [--seed <int>]
```

Arguments:
- `--model`: Model name (`naive`, `seasonal`, `lgbm`, `lstm`, `tcn`, `stgcn`, `gwnet`, `all`)
- `--config`: Path to YAML config file
- `--seed`: Random seed override (optional)

## Data Format

### Input (X)
- Shape: `(samples, 168, 12, 17)` - SCALED
- Features: pollutants + meteorology + cyclical encodings
- Scaling: RobustScaler fitted on training data

### Target (Y)
- Shape: `(samples, 24, 12, 6)` - RAW units
- Contains 0 at missing positions
- **MUST use Y_mask for loss/metrics**

### Mask (Y_mask)
- Shape: `(samples, 24, 12, 6)`
- 1 = observed (valid), 0 = missing
- **Critical**: All losses and metrics must be masked

## Masked Loss Example

```python
def masked_mae_loss(pred, target, mask):
    abs_error = torch.abs(pred - target) * mask
    return abs_error.sum() / (mask.sum() + 1e-8)
```

## Naive Baselines

Naive models must inverse-transform pollutant channels from scaled X:

```python
# X is scaled, need raw values for prediction
last_values_scaled = X[:, -1, :, :]  # (B, N, F)
last_raw = input_scaler.inverse_transform(last_values_scaled.reshape(-1, F))
predictions = last_raw[:, :, target_indices]  # Extract pollutants only
```

## Directory Structure

```
baseline/
├── configs/
│   ├── default.yaml      # Base config
│   ├── lstm.yaml         # LSTM-specific
│   ├── tcn.yaml          # TCN-specific
│   ├── stgcn.yaml        # STGCN-specific
│   ├── gwnet.yaml        # Graph WaveNet-specific
│   └── lgbm.yaml         # LightGBM-specific
├── data/
│   ├── loader_npz.py     # NPZ data loader
│   ├── loader_lgbm.py    # LightGBM CSV loader
│   └── graph.py          # Adjacency matrix loader
├── models/
│   ├── naive_persistence.py
│   ├── naive_seasonal24.py
│   ├── lgbm_multioutput.py
│   ├── lstm_seq2seq.py
│   ├── tcn.py
│   ├── stgcn.py
│   └── graph_wavenet.py
├── training/
│   ├── trainer_torch.py  # PyTorch trainer
│   ├── trainer_lgbm.py   # LightGBM trainer
│   ├── early_stopping.py
│   └── checkpointing.py
├── evaluation/
│   ├── masked_metrics.py # MAE, RMSE, sMAPE
│   ├── evaluate.py       # Evaluation entrypoint
│   └── plots.py          # Visualization
├── scripts/
│   └── run.py            # Main CLI
└── results/              # Output directory
    ├── metrics_summary.csv
    ├── plots/
    └── checkpoints/
```

## Config File Format

```yaml
seed: 42

data:
  processed_dir: "processed"
  p1_deep_dir: "processed/P1_deep"
  tabular_dir: "processed/tabular_lgbm"
  graphs_dir: "processed/graphs"

task:
  lookback: 168
  horizon: 24
  num_stations: 12
  num_targets: 6

training:
  batch_size: 64
  epochs: 100
  learning_rate: 0.001
  patience: 10

# Model-specific section (e.g., lstm, tcn, stgcn, gwnet, lgbm)
lstm:
  hidden_dim: 256
  num_layers: 2
  dropout: 0.2

output:
  results_dir: "baseline/results"
```

## Metrics

All metrics are computed with masking:

| Metric | Formula |
|--------|---------|
| MAE | `sum(|pred - target| * mask) / sum(mask)` |
| RMSE | `sqrt(sum((pred - target)² * mask) / sum(mask))` |
| sMAPE | `100 * sum(|pred - target| / (|pred| + |target|) * mask) / sum(mask)` |

Results are saved to `baseline/results/metrics_summary.csv` with breakdowns by:
- Overall
- Per-horizon (h=1..24)
- Per-pollutant (PM2.5, PM10, SO2, NO2, CO, O3)

## Requirements

```
torch>=1.9.0
numpy>=1.20.0
pandas>=1.3.0
lightgbm>=3.2.0
scikit-learn>=0.24.0
pyyaml>=5.4.0
tqdm>=4.60.0
matplotlib>=3.4.0
```

Install: `pip install -r baseline/requirements.txt`
