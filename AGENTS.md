# Repository Guidelines

## Project Structure & Module Organization

- `eda_beijing_air_quality.py`: Exploratory analysis of the raw PRSA station CSVs. Writes reports to `eda_output/`.
- `preprocessing_pipeline_v2.1.py`: Primary preprocessing pipeline that builds model-ready datasets in `processed/` (see `processed/README.md`).
- `baseline/`: Modular baselines package (configs, data loaders, models, training, evaluation). Outputs to `baseline/results/`.
- `PRSA_Data_20130301-20170228/`: Raw dataset directory expected by the scripts (12 station CSV files).
- Generated artifacts: `processed/`, `processed_scaled/`, `eda_output/`, `baseline/results/` (treat as build outputs).

## Build, Test, and Development Commands

- Install dependencies: `python -m pip install -r baseline/requirements.txt`
- Run EDA (creates `eda_output/`): `python eda_beijing_air_quality.py`
- Run preprocessing (creates/overwrites `processed/`): `python preprocessing_pipeline_v2.1.py`
- Train/evaluate a baseline model:
  - `python -m baseline.scripts.run --model naive --config baseline/configs/default.yaml`
  - `python -m baseline.scripts.run --model lstm --config baseline/configs/lstm.yaml`

## Coding Style & Naming Conventions

- Python: 4-space indentation, `snake_case` for files/functions, `CapWords` for classes.
- Keep module boundaries aligned with `baseline/{data,models,training,evaluation}`; configs live in `baseline/configs/*.yaml`.
- Prefer small, testable functions and keep station/feature ordering explicit (see `processed/README.md`).

## Testing Guidelines

- No dedicated unit-test suite is included. Treat pipeline “validation tests” and baseline evaluation outputs as required checks.
- After changes that affect data semantics, regenerate outputs and confirm:
  - `processed/metadata.json` is updated and consistent.
  - Masked metrics use `Y_mask` (see `baseline/evaluation/masked_metrics.py`).
  - Results are produced in `baseline/results/metrics_summary.csv`.

## Commit & Pull Request Guidelines

- This workspace does not include git history; use a clear conventional format if you are versioning changes: `feat: …`, `fix: …`, `docs: …`, `refactor: …`.
- PRs should state which script(s) changed, how outputs were validated, and include any metric deltas or schema changes (e.g., shapes, `station_list`, `feature_list`).

## Configuration & Data Safety

- Avoid changing fixed task parameters (lookback `168`, horizon `24`) unless the repository explicitly intends a new dataset version.
- Do not commit large raw/derived datasets in normal code reviews; prefer documenting how to reproduce them.
