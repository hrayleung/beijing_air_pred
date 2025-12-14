# Repository Guidelines

## Project Structure & Module Organization

- `eda_beijing_air_quality.py`: Exploratory analysis of the raw PRSA station CSVs. Writes reports to `eda_output/`.
- `preprocessing_pipeline_v2.1.py`: Primary preprocessing pipeline that builds model-ready datasets in `processed/` (see `processed/README.md`).
- `baseline/`: Modular baselines package (configs, data loaders, models, training, evaluation). Outputs to `baseline/results/`.
- `model/`: “Strong” PyTorch model package (WG‑DGTM) with its own training/eval entrypoints. Outputs to `model/results/`.
- `PRSA_Data_20130301-20170228/`: Raw dataset directory expected by the scripts (12 station CSV files).
- Generated artifacts: `processed/`, `processed_scaled/`, `eda_output/`, `baseline/results/`, `model/results/` (treat as build outputs).

## Build, Test, and Development Commands

- Install dependencies: `python -m pip install -r baseline/requirements.txt`
- Run EDA (creates `eda_output/`): `python eda_beijing_air_quality.py`
- Run preprocessing (creates/overwrites `processed/`): `python preprocessing_pipeline_v2.1.py`
- Train/evaluate a baseline model:
  - `python -m baseline.scripts.run --model naive --config baseline/configs/default.yaml`
  - `python -m baseline.scripts.run --model lstm --config baseline/configs/lstm.yaml`
- Run baseline debug checks (writes logs/plots under `baseline/results/`): `python -m baseline.scripts.debug_checks --config baseline/configs/default.yaml`
- Train/evaluate the WG‑DGTM strong model:
  - `python -m model.scripts.run_train --config model/configs/wgdgtm.yaml`
  - `python -m model.scripts.run_eval --config model/configs/wgdgtm.yaml --ckpt model/results/checkpoints/best.pt`

## Coding Style & Naming Conventions

- Python: 4-space indentation, `snake_case` for files/functions, `CapWords` for classes.
- Keep module boundaries aligned with package layout (`baseline/{data,models,training,evaluation}` and `model/{data,modules,models,training,evaluation}`).
- Configs live in `baseline/configs/*.yaml` and `model/configs/*.yaml`.
- Prefer small, testable functions and keep station/feature ordering explicit (see `processed/README.md`).

## Testing Guidelines

- No dedicated unit-test suite is included. Treat pipeline “validation tests” and baseline evaluation outputs as required checks.
- After changes that affect data semantics, regenerate outputs and confirm:
  - `processed/metadata.json` is updated and consistent.
  - Masked metrics use `Y_mask` (see `baseline/evaluation/masked_metrics.py`).
  - Multi-horizon shapes stay consistent (`Y`/pred: `(S, 24, 12, 6)`).
  - Results are produced in `baseline/results/metrics_summary.csv` and `baseline/results/metrics_per_pollutant.csv`.

## Commit & Pull Request Guidelines

- Git history is minimal (e.g., `initial commit`). Use a clear conventional format: `feat: …`, `fix: …`, `docs: …`, `refactor: …`.
- PRs should state which scripts changed, how outputs were validated (commands + key files), and include metric deltas and/or plots when relevant.

## Configuration & Data Safety

- Avoid changing fixed task parameters (lookback `168`, horizon `24`) unless the repository explicitly intends a new dataset version.
- Do not commit large raw/derived datasets in normal code reviews; prefer documenting how to reproduce them.
