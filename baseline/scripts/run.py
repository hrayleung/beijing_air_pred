"""
Main CLI entrypoint for training and evaluating baselines.

Usage:
    python -m baseline.scripts.run --model naive --config baseline/configs/default.yaml
    python -m baseline.scripts.run --model lstm --config baseline/configs/lstm.yaml
"""
import os
import sys
import argparse
import gc
import yaml
import json
from typing import List, Optional, Tuple
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _parse_gpu_ids(gpu_arg: Optional[str]) -> Optional[List[int]]:
    if gpu_arg is None:
        return None
    gpu_arg = str(gpu_arg).strip()
    if gpu_arg == "":
        return None
    return [int(g.strip()) for g in gpu_arg.split(',') if g.strip() != ""]


def _restrict_cuda_visible_devices(requested: List[int]) -> Tuple[bool, Optional[List[int]]]:
    """
    Restrict CUDA to a subset of devices *before* importing torch.

    Behavior:
      - If CUDA_VISIBLE_DEVICES is set to an integer list:
          * If all `requested` entries are contained in that list, treat them as physical GPU IDs.
          * Otherwise, treat `requested` entries as indices into that visible list.
      - If CUDA_VISIBLE_DEVICES is not set, treat `requested` as physical GPU IDs
        and set CUDA_VISIBLE_DEVICES to that list.

    Returns:
      (restricted, effective_physical_ids_or_none)
    """
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing:
        try:
            visible_physical = [int(x.strip()) for x in existing.split(",") if x.strip() != ""]
        except ValueError:
            return False, None

        if not visible_physical:
            return False, None

        if all(g in visible_physical for g in requested):
            effective_physical = requested
        else:
            try:
                effective_physical = [visible_physical[i] for i in requested]
            except IndexError:
                return False, None

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in effective_physical)
        return True, effective_physical

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in requested)
    return True, requested


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cleanup_between_models():
    """
    Best-effort cleanup to reduce cross-library interference when running multiple models
    in a single process (e.g., `--model all`).
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _fit_target_normalizer(train_y: np.ndarray, train_mask: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a simple robust normalizer per target (pollutant) using TRAIN observed values only.
    Returns (center, scale) arrays of shape (D,).
    """
    if train_y.shape != train_mask.shape:
        raise ValueError(f"train_y/train_mask shape mismatch: {train_y.shape} vs {train_mask.shape}")
    if train_y.ndim != 4:
        raise ValueError(f"Expected train_y with shape (S,H,N,D); got {train_y.shape}")
    D = train_y.shape[3]
    center = np.zeros((D,), dtype=np.float32)
    scale = np.ones((D,), dtype=np.float32)
    for d in range(D):
        vals = train_y[:, :, :, d][train_mask[:, :, :, d] == 1]
        if vals.size == 0:
            center[d] = 0.0
            scale[d] = 1.0
            continue
        center[d] = np.median(vals).astype(np.float32)
        q25, q75 = np.percentile(vals, [25, 75])
        iqr = float(q75 - q25)
        scale[d] = np.float32(max(iqr, eps))
    return center, scale


def run_naive(
    config: dict,
    data: dict,
    scaler_dict: dict,
    feature_list: list,
    target_list: list,
    *,
    scale_targets: bool = False,
    target_scaler=None,
):
    """Run naive persistence baseline."""
    from baseline.models import NaivePersistence
    from baseline.evaluation import evaluate_predictions

    print("\n" + "="*60)
    print("Running Naive Persistence (B0)")
    print("="*60)
    
    model = NaivePersistence(
        input_scaler=scaler_dict['input_scaler'],
        feature_list=feature_list,
        target_list=target_list
    )
    
    # Generate predictions on test set
    X_test = data['test']['X']
    Y_test = data['test']['Y']
    Y_mask = data['test']['Y_mask']
    
    pred = model.predict(X_test)
    
    # Y is already in raw units (or needs inverse transform if scaled)
    # For now assume Y is raw
    results = evaluate_predictions(
        pred, Y_test, Y_mask,
        model_name='naive_persistence',
        split='test',
        pollutant_names=target_list,
        results_dir=config['output']['results_dir'],
        scale_targets=scale_targets,
        target_scaler=target_scaler,
        pred_is_scaled=False,
        expected_shape=Y_test.shape,
        require_horizon_variation=False,
    )
    
    return results, pred


def run_seasonal(
    config: dict,
    data: dict,
    scaler_dict: dict,
    feature_list: list,
    target_list: list,
    *,
    scale_targets: bool = False,
    target_scaler=None,
):
    """Run seasonal naive baseline."""
    from baseline.models import SeasonalNaive24
    from baseline.evaluation import evaluate_predictions
    from baseline.evaluation.per_pollutant_report import plot_seasonal_naive_sanity

    print("\n" + "="*60)
    print("Running Seasonal Naive 24h (B1)")
    print("="*60)
    
    model = SeasonalNaive24(
        input_scaler=scaler_dict['input_scaler'],
        feature_list=feature_list,
        target_list=target_list
    )
    
    X_test = data['test']['X']
    Y_test = data['test']['Y']
    Y_mask = data['test']['Y_mask']
    
    pred = model.predict(X_test)
    
    results = evaluate_predictions(
        pred, Y_test, Y_mask,
        model_name='seasonal_naive_24h',
        split='test',
        pollutant_names=target_list,
        results_dir=config['output']['results_dir'],
        scale_targets=scale_targets,
        target_scaler=target_scaler,
        pred_is_scaled=False,
        expected_shape=Y_test.shape,
        require_horizon_variation=True,
    )

    # Save a small seasonal sanity plot (true vs seasonal naive, PM2.5 @ station 0).
    try:
        plot_path = plot_seasonal_naive_sanity(
            pred,
            Y_test,
            Y_mask,
            results_dir=config['output']['results_dir'],
            pollutant_idx=0,
            pollutant_name=target_list[0] if target_list else "PM2.5",
            station_idx=0,
        )
        print(f"Seasonal sanity plot saved to {plot_path}")
    except Exception as e:
        print(f"Warning: failed to save seasonal sanity plot: {e}")
    
    return results, pred


def run_lgbm(
    config: dict,
    lgbm_data: dict,
    npz_data: dict,
    station_list: list,
    target_list: list,
    *,
    scale_targets: bool = False,
    target_scaler=None,
):
    """Run LightGBM baseline."""
    from baseline.training import train_lgbm
    from baseline.evaluation import evaluate_predictions
    from baseline.data.loader_lgbm import align_lgbm_predictions_to_npz

    print("\n" + "="*60)
    print("Running LightGBM (B2)")
    print("="*60)
    
    lgbm_params = config.get('lgbm', {})
    
    model = train_lgbm(
        train_df=lgbm_data['train'],
        val_df=lgbm_data['val'],
        lgbm_params=lgbm_params,
        save_dir=os.path.join(config['output']['results_dir'], 'checkpoints', 'lgbm'),
        verbose=True
    )
    
    # Align predictions by datetime_origins (LightGBM tabular may skip some origins due to valid_start filtering).
    raw_pred = model.predict(lgbm_data['test'])  # (num_rows, 24, 6)
    pred, present_mask = align_lgbm_predictions_to_npz(raw_pred, lgbm_data['test'], npz_data['test'], station_list)

    Y_test = npz_data['test']['Y'][present_mask]
    Y_mask = npz_data['test']['Y_mask'][present_mask]
    pred = pred[present_mask]
    
    results = evaluate_predictions(
        pred, Y_test, Y_mask,
        model_name='lightgbm',
        split='test',
        pollutant_names=target_list,
        results_dir=config['output']['results_dir'],
        scale_targets=scale_targets,
        target_scaler=target_scaler,
        pred_is_scaled=False,
        expected_shape=Y_test.shape,
        require_horizon_variation=True,
    )
    
    return results, pred


def run_lstm(
    config: dict,
    data: dict,
    target_list: list,
    gpu_ids: list = None,
    *,
    scale_targets: bool = False,
    target_scaler=None,
):
    """Run LSTM baseline."""
    from baseline.data.loader_npz import create_dataloaders
    from baseline.models.lstm_seq2seq import LSTMDirect
    from baseline.training import TorchTrainer
    from baseline.evaluation import evaluate_predictions

    print("\n" + "="*60)
    print("Running LSTM Seq2Seq (B3)")
    print("="*60)
    
    # Get dimensions
    _, L, N, F = data['train']['X'].shape
    H = data['train']['Y'].shape[1]
    D = len(target_list)
    
    input_dim = N * F  # 12 * 17 = 204
    output_dim = N * D  # 12 * 6 = 72
    
    # Create model
    model_config = config.get('lstm', {})
    model = LSTMDirect(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=model_config.get('hidden_dim', 256),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.2),
        horizon=H
    )
    
    # Scale batch size by number of GPUs
    num_gpus = len(gpu_ids) if gpu_ids else 1
    batch_size = config['training']['batch_size'] * num_gpus
    
    # Create dataloaders with flattened X (num_workers=0 to avoid multiprocessing issues)
    loaders = create_dataloaders(data, batch_size=batch_size, flatten_x=True, num_workers=0, pin_memory=False)
    
    # Train
    normalize_targets = bool(config.get('training', {}).get('normalize_targets', True)) and (not scale_targets)
    target_center = None
    target_scale = None
    if normalize_targets:
        target_center, target_scale = _fit_target_normalizer(data['train']['Y'], data['train']['Y_mask'])

    trainer = TorchTrainer(
        model=model,
        gpu_ids=gpu_ids,
        learning_rate=config['training']['learning_rate'],
        checkpoint_dir=os.path.join(config['output']['results_dir'], 'checkpoints'),
        target_center=target_center,
        target_scale=target_scale,
    )
    
    trainer.fit(
        loaders['train'], loaders['val'],
        epochs=config['training']['epochs'],
        patience=config['training']['patience'],
        model_name='lstm'
    )
    
    # Predict
    pred_flat = trainer.predict(loaders['test'])  # (samples, H, N*D)
    pred = pred_flat.reshape(-1, H, N, D)
    
    Y_test = data['test']['Y']
    Y_mask = data['test']['Y_mask']
    
    results = evaluate_predictions(
        pred, Y_test, Y_mask,
        model_name='lstm',
        split='test',
        pollutant_names=target_list,
        results_dir=config['output']['results_dir'],
        scale_targets=scale_targets,
        target_scaler=target_scaler,
        pred_is_scaled=scale_targets,
        expected_shape=Y_test.shape,
        require_horizon_variation=True,
    )
    
    return results, pred


def run_tcn(
    config: dict,
    data: dict,
    target_list: list,
    gpu_ids: list = None,
    *,
    scale_targets: bool = False,
    target_scaler=None,
):
    """Run TCN baseline."""
    from baseline.data.loader_npz import create_dataloaders
    from baseline.models import TCN
    from baseline.training import TorchTrainer
    from baseline.evaluation import evaluate_predictions

    print("\n" + "="*60)
    print("Running TCN (B4)")
    print("="*60)
    
    _, L, N, F = data['train']['X'].shape
    H = data['train']['Y'].shape[1]
    D = len(target_list)
    
    input_dim = N * F
    output_dim = N * D
    
    model_config = config.get('tcn', {})
    model = TCN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_channels=model_config.get('hidden_channels', 64),
        num_layers=model_config.get('num_layers', 6),
        kernel_size=model_config.get('kernel_size', 3),
        dropout=model_config.get('dropout', 0.2),
        horizon=H
    )
    
    num_gpus = len(gpu_ids) if gpu_ids else 1
    batch_size = config['training']['batch_size'] * num_gpus
    loaders = create_dataloaders(data, batch_size=batch_size, flatten_x=True, num_workers=0, pin_memory=False)
    
    normalize_targets = bool(config.get('training', {}).get('normalize_targets', True)) and (not scale_targets)
    target_center = None
    target_scale = None
    if normalize_targets:
        target_center, target_scale = _fit_target_normalizer(data['train']['Y'], data['train']['Y_mask'])

    trainer = TorchTrainer(
        model=model,
        gpu_ids=gpu_ids,
        learning_rate=config['training']['learning_rate'],
        checkpoint_dir=os.path.join(config['output']['results_dir'], 'checkpoints'),
        target_center=target_center,
        target_scale=target_scale,
    )
    
    trainer.fit(
        loaders['train'], loaders['val'],
        epochs=config['training']['epochs'],
        patience=config['training']['patience'],
        model_name='tcn'
    )
    
    pred_flat = trainer.predict(loaders['test'])
    pred = pred_flat.reshape(-1, H, N, D)
    
    Y_test = data['test']['Y']
    Y_mask = data['test']['Y_mask']
    
    results = evaluate_predictions(
        pred, Y_test, Y_mask,
        model_name='tcn',
        split='test',
        pollutant_names=target_list,
        results_dir=config['output']['results_dir'],
        scale_targets=scale_targets,
        target_scaler=target_scaler,
        pred_is_scaled=scale_targets,
        expected_shape=Y_test.shape,
        require_horizon_variation=True,
    )
    
    return results, pred


def run_stgcn(
    config: dict,
    data: dict,
    adj: np.ndarray,
    target_list: list,
    gpu_ids: list = None,
    *,
    scale_targets: bool = False,
    target_scaler=None,
):
    """Run STGCN baseline."""
    from baseline.data.loader_npz import create_dataloaders
    from baseline.models import STGCN
    from baseline.training import TorchTrainer
    from baseline.evaluation import evaluate_predictions

    print("\n" + "="*60)
    print("Running STGCN (B5)")
    print("="*60)
    
    _, L, N, F = data['train']['X'].shape
    H = data['train']['Y'].shape[1]
    D = len(target_list)
    
    model_config = config.get('stgcn', {})
    model = STGCN(
        num_nodes=N,
        in_channels=F,
        out_channels=D,
        hidden_channels=model_config.get('hidden_channels', 64),
        num_layers=model_config.get('num_layers', 2),
        kernel_size=model_config.get('kernel_size', 3),
        K=model_config.get('K', 3),
        horizon=H,
        dropout=model_config.get('dropout', 0.2),
        time_pool=model_config.get('time_pool', 'mean'),
    )
    model.set_adjacency(adj)
    
    num_gpus = len(gpu_ids) if gpu_ids else 1
    batch_size = config['training']['batch_size'] * num_gpus
    loaders = create_dataloaders(data, batch_size=batch_size, flatten_x=False, num_workers=0, pin_memory=False)
    
    normalize_targets = bool(config.get('training', {}).get('normalize_targets', True)) and (not scale_targets)
    target_center = None
    target_scale = None
    if normalize_targets:
        target_center, target_scale = _fit_target_normalizer(data['train']['Y'], data['train']['Y_mask'])

    trainer = TorchTrainer(
        model=model,
        gpu_ids=gpu_ids,
        learning_rate=config['training']['learning_rate'],
        checkpoint_dir=os.path.join(config['output']['results_dir'], 'checkpoints'),
        target_center=target_center,
        target_scale=target_scale,
    )
    
    trainer.fit(
        loaders['train'], loaders['val'],
        epochs=config['training']['epochs'],
        patience=config['training']['patience'],
        model_name='stgcn'
    )
    
    pred = trainer.predict(loaders['test'])
    
    Y_test = data['test']['Y']
    Y_mask = data['test']['Y_mask']
    
    results = evaluate_predictions(
        pred, Y_test, Y_mask,
        model_name='stgcn',
        split='test',
        pollutant_names=target_list,
        results_dir=config['output']['results_dir'],
        scale_targets=scale_targets,
        target_scaler=target_scaler,
        pred_is_scaled=scale_targets,
        expected_shape=Y_test.shape,
        require_horizon_variation=True,
    )
    
    return results, pred


def run_gwnet(
    config: dict,
    data: dict,
    adj: np.ndarray,
    target_list: list,
    gpu_ids: list = None,
    *,
    scale_targets: bool = False,
    target_scaler=None,
):
    """Run Graph WaveNet baseline."""
    from baseline.data.loader_npz import create_dataloaders
    from baseline.models import GraphWaveNet
    from baseline.training import TorchTrainer
    from baseline.evaluation import evaluate_predictions

    print("\n" + "="*60)
    print("Running Graph WaveNet (B6)")
    print("="*60)
    
    _, L, N, F = data['train']['X'].shape
    H = data['train']['Y'].shape[1]
    D = len(target_list)
    
    model_config = config.get('gwnet', {})
    model = GraphWaveNet(
        num_nodes=N,
        in_channels=F,
        out_channels=D,
        hidden_channels=model_config.get('hidden_channels', 32),
        skip_channels=model_config.get('skip_channels', 64),
        num_layers=model_config.get('num_layers', 4),
        horizon=H,
        dropout=model_config.get('dropout', 0.2),
        use_adaptive_adj=model_config.get('use_adaptive_adj', True),
        time_pool=model_config.get('time_pool', 'mean'),
    )
    model.set_adjacency(adj)
    
    num_gpus = len(gpu_ids) if gpu_ids else 1
    batch_size = config['training']['batch_size'] * num_gpus
    loaders = create_dataloaders(data, batch_size=batch_size, flatten_x=False, num_workers=0, pin_memory=False)
    
    normalize_targets = bool(config.get('training', {}).get('normalize_targets', True)) and (not scale_targets)
    target_center = None
    target_scale = None
    if normalize_targets:
        target_center, target_scale = _fit_target_normalizer(data['train']['Y'], data['train']['Y_mask'])

    trainer = TorchTrainer(
        model=model,
        gpu_ids=gpu_ids,
        learning_rate=config['training']['learning_rate'],
        checkpoint_dir=os.path.join(config['output']['results_dir'], 'checkpoints'),
        target_center=target_center,
        target_scale=target_scale,
    )
    
    trainer.fit(
        loaders['train'], loaders['val'],
        epochs=config['training']['epochs'],
        patience=config['training']['patience'],
        model_name='gwnet'
    )
    
    pred = trainer.predict(loaders['test'])
    
    Y_test = data['test']['Y']
    Y_mask = data['test']['Y_mask']
    
    results = evaluate_predictions(
        pred, Y_test, Y_mask,
        model_name='gwnet',
        split='test',
        pollutant_names=target_list,
        results_dir=config['output']['results_dir'],
        scale_targets=scale_targets,
        target_scaler=target_scaler,
        pred_is_scaled=scale_targets,
        expected_shape=Y_test.shape,
        require_horizon_variation=True,
    )
    
    return results, pred


def main():
    parser = argparse.ArgumentParser(description='PRSA Baseline Training and Evaluation')
    parser.add_argument('--model', type=str, required=True,
                       choices=['naive', 'seasonal', 'lgbm', 'lstm', 'tcn', 'stgcn', 'gwnet', 'all'],
                       help='Model to run')
    parser.add_argument('--config', type=str, default='baseline/configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--seed', type=int, default=None, help='Random seed override')
    parser.add_argument(
        '--gpus',
        type=str,
        default=None,
        help=(
            'GPU IDs to use (comma-separated). If CUDA_VISIBLE_DEVICES is set, values may be either '
            'physical GPU IDs (recommended) or indices within the visible set.'
        ),
    )
    parser.add_argument('--gpu', type=str, default=None, help='Alias for --gpus (single GPU index)')
    
    args = parser.parse_args()

    if args.gpus is not None and args.gpu is not None:
        parser.error("Use only one of --gpu or --gpus")

    requested_gpu_arg = args.gpus if args.gpus is not None else args.gpu
    requested_gpu_ids = _parse_gpu_ids(requested_gpu_arg)

    restricted = False
    effective_physical = None
    if requested_gpu_ids is not None:
        restricted, effective_physical = _restrict_cuda_visible_devices(requested_gpu_ids)

    import torch

    if requested_gpu_ids is not None and restricted:
        gpu_ids = list(range(len(requested_gpu_ids)))
    elif requested_gpu_ids is not None:
        gpu_ids = requested_gpu_ids
    elif torch.cuda.is_available():
        gpu_ids = list(range(torch.cuda.device_count()))
        print(f"Auto-detected {len(gpu_ids)} GPUs: {gpu_ids}")
    else:
        gpu_ids = None

    if torch.cuda.is_available():
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
        print(f"Visible GPU count: {torch.cuda.device_count()}")
        if requested_gpu_ids is not None and restricted and effective_physical is not None:
            print(f"Requested GPUs: {requested_gpu_ids} -> physical GPUs: {effective_physical}")
        if gpu_ids is not None:
            print(f"Using GPU ids within this process: {gpu_ids}")
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    seed = args.seed if args.seed is not None else config.get('seed', 42)
    set_seed(seed)
    print(f"Random seed: {seed}")
    
    # Create output directories
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['output']['results_dir'], 'plots'), exist_ok=True)
    os.makedirs(os.path.join(config['output']['results_dir'], 'checkpoints'), exist_ok=True)
    
    # Load data
    from baseline.data.loader_npz import (
        load_npz_data,
        load_scaler,
        load_metadata,
        load_feature_list,
        load_target_list,
    )
    from baseline.data.loader_lgbm import load_lgbm_data
    from baseline.data.graph import load_adjacency
    from baseline.evaluation.plots import plot_error_vs_horizon

    print("\nLoading data...")
    npz_data = load_npz_data(config['data']['p1_deep_dir'])
    scaler_dict = load_scaler(config['data']['p1_deep_dir'])
    metadata = load_metadata(config['data']['processed_dir'])
    feature_list = load_feature_list(config['data']['processed_dir'])
    target_list = load_target_list(config['data']['processed_dir'])

    scale_targets = bool(metadata.get("scale_targets", False))
    target_scaler = scaler_dict.get("target_scaler")
    
    print(f"Train: {npz_data['train']['X'].shape}")
    print(f"Val: {npz_data['val']['X'].shape}")
    print(f"Test: {npz_data['test']['X'].shape}")
    
    all_results = {}
    
    # Run selected model(s)
    if args.model == 'all':
        # Prefer running GPU-heavy Torch models before LightGBM (which may spawn workers).
        results, pred = run_naive(
            config,
            npz_data,
            scaler_dict,
            feature_list,
            target_list,
            scale_targets=scale_targets,
            target_scaler=target_scaler,
        )
        all_results['naive_persistence'] = results
        _cleanup_between_models()

        results, pred = run_seasonal(
            config,
            npz_data,
            scaler_dict,
            feature_list,
            target_list,
            scale_targets=scale_targets,
            target_scaler=target_scaler,
        )
        all_results['seasonal_naive_24h'] = results
        _cleanup_between_models()

        results, pred = run_lstm(
            config,
            npz_data,
            target_list,
            gpu_ids,
            scale_targets=scale_targets,
            target_scaler=target_scaler,
        )
        all_results['lstm'] = results
        _cleanup_between_models()

        results, pred = run_tcn(
            config,
            npz_data,
            target_list,
            gpu_ids,
            scale_targets=scale_targets,
            target_scaler=target_scaler,
        )
        all_results['tcn'] = results
        _cleanup_between_models()

        adj, station_list = load_adjacency(config['data']['graphs_dir'])

        results, pred = run_stgcn(
            config,
            npz_data,
            adj,
            target_list,
            gpu_ids,
            scale_targets=scale_targets,
            target_scaler=target_scaler,
        )
        all_results['stgcn'] = results
        _cleanup_between_models()

        results, pred = run_gwnet(
            config,
            npz_data,
            adj,
            target_list,
            gpu_ids,
            scale_targets=scale_targets,
            target_scaler=target_scaler,
        )
        all_results['gwnet'] = results
        _cleanup_between_models()

        lgbm_data = load_lgbm_data(config['data']['tabular_dir'])
        results, pred = run_lgbm(
            config,
            lgbm_data,
            npz_data,
            station_list,
            target_list,
            scale_targets=scale_targets,
            target_scaler=target_scaler,
        )
        all_results['lightgbm'] = results
        _cleanup_between_models()
    else:
        if args.model == 'naive':
            results, pred = run_naive(
                config,
                npz_data,
                scaler_dict,
                feature_list,
                target_list,
                scale_targets=scale_targets,
                target_scaler=target_scaler,
            )
            all_results['naive_persistence'] = results

        if args.model == 'seasonal':
            results, pred = run_seasonal(
                config,
                npz_data,
                scaler_dict,
                feature_list,
                target_list,
                scale_targets=scale_targets,
                target_scaler=target_scaler,
            )
            all_results['seasonal_naive_24h'] = results

        if args.model == 'lgbm':
            lgbm_data = load_lgbm_data(config['data']['tabular_dir'])
            adj, station_list = load_adjacency(config['data']['graphs_dir'])
            results, pred = run_lgbm(
                config,
                lgbm_data,
                npz_data,
                station_list,
                target_list,
                scale_targets=scale_targets,
                target_scaler=target_scaler,
            )
            all_results['lightgbm'] = results

        if args.model == 'lstm':
            results, pred = run_lstm(
                config,
                npz_data,
                target_list,
                gpu_ids,
                scale_targets=scale_targets,
                target_scaler=target_scaler,
            )
            all_results['lstm'] = results

        if args.model == 'tcn':
            results, pred = run_tcn(
                config,
                npz_data,
                target_list,
                gpu_ids,
                scale_targets=scale_targets,
                target_scaler=target_scaler,
            )
            all_results['tcn'] = results

        if args.model == 'stgcn':
            adj, station_list = load_adjacency(config['data']['graphs_dir'])
            results, pred = run_stgcn(
                config,
                npz_data,
                adj,
                target_list,
                gpu_ids,
                scale_targets=scale_targets,
                target_scaler=target_scaler,
            )
            all_results['stgcn'] = results

        if args.model == 'gwnet':
            adj, station_list = load_adjacency(config['data']['graphs_dir'])
            results, pred = run_gwnet(
                config,
                npz_data,
                adj,
                target_list,
                gpu_ids,
                scale_targets=scale_targets,
                target_scaler=target_scaler,
            )
            all_results['gwnet'] = results
    
    # Generate comparison plots if multiple models
    if len(all_results) > 1:
        plot_error_vs_horizon(
            all_results,
            metric='MAE',
            save_path=os.path.join(config['output']['results_dir'], 'plots', 'all_models_mae_vs_horizon.png')
        )
        plot_error_vs_horizon(
            all_results,
            metric='RMSE',
            save_path=os.path.join(config['output']['results_dir'], 'plots', 'all_models_rmse_vs_horizon.png')
        )
        # Save comparison summary
        save_comparison_summary(all_results, config['output']['results_dir'])
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Results saved to: {config['output']['results_dir']}")
    print(f"  - Metrics: {config['output']['results_dir']}/metrics_summary.csv")
    print(f"  - Logs: {config['output']['results_dir']}/logs/")
    print(f"  - Plots: {config['output']['results_dir']}/plots/")
    print(f"  - Checkpoints: {config['output']['results_dir']}/checkpoints/")


def save_comparison_summary(all_results: dict, results_dir: str):
    """Save a comparison summary table."""
    import pandas as pd
    
    rows = []
    for model_name, results in all_results.items():
        rows.append({
            'Model': model_name,
            'MAE': results['overall']['MAE'],
            'RMSE': results['overall']['RMSE'],
            'sMAPE': results['overall']['sMAPE'],
            'MAE_h1': results['per_horizon'][1]['MAE'],
            'MAE_h6': results['per_horizon'][6]['MAE'],
            'MAE_h12': results['per_horizon'][12]['MAE'],
            'MAE_h24': results['per_horizon'][24]['MAE'],
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('MAE')
    
    summary_path = os.path.join(results_dir, 'model_comparison.csv')
    df.to_csv(summary_path, index=False)
    print(f"\nModel comparison saved to {summary_path}")
    
    # Print table
    print("\n" + "="*80)
    print("MODEL COMPARISON (sorted by MAE)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)


if __name__ == '__main__':
    main()
