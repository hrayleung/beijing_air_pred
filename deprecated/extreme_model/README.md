# ST-Former (Proposed Model)

This folder contains a spatio-temporal Transformer (encoder-decoder) for PRSA Beijing air quality forecasting.

## Commands (root conda env `dl`)

- Train:
  - `python -m extreme_model.scripts.run_train --config extreme_model/configs/stformer.yaml`
  - `python -m extreme_model.scripts.run_train --config extreme_model/configs/stformer_residual.yaml`
- Evaluate:
  - `python -m extreme_model.scripts.run_eval --config extreme_model/configs/stformer.yaml --ckpt extreme_model/results/stformer/checkpoints/best.pt`
  - `python -m extreme_model.scripts.run_eval --config extreme_model/configs/stformer_residual.yaml --ckpt extreme_model/results/stformer_residual/checkpoints/best.pt`

Outputs are written under `extreme_model/results/<experiment_name>/`:
- `config.yaml`
- `checkpoints/{best.pt,stformer_best.pt}`
- `logs/{run_metadata_train.json,run_metadata_eval.json,train_history.json,train_history.csv,stformer_training_log.csv}`
- `metrics/{macro_avg_metrics.csv,target_std_weights.json}`
- `metrics_{overall,per_pollutant,summary}.csv`
- `model_comparison.csv`
- `plots/{stformer_loss_curve.png,stformer_mae_vs_horizon.png,stformer_mae_by_pollutant.png,stformer_sample_predictions.png,all_models_mae_vs_horizon.png,all_models_rmse_vs_horizon.png,train_history.png}`
