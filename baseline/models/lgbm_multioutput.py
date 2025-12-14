"""
B2: LightGBM Multi-Horizon Multi-Output Model
Trains one MultiOutputRegressor per horizon (24 models total).
"""
import os
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor


class LightGBMMultiHorizon:
    """
    LightGBM model for multi-horizon forecasting.
    
    Trains 24 separate MultiOutputRegressor models, one per horizon.
    Each model predicts 6 pollutants simultaneously.
    """
    
    def __init__(
        self,
        lgbm_params: Dict = None,
        n_jobs: int = -1
    ):
        """
        Args:
            lgbm_params: LightGBM parameters
            n_jobs: Number of parallel jobs
        """
        self.lgbm_params = lgbm_params or {
            'n_estimators': 100,
            'max_depth': 8,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1
        }
        self.n_jobs = n_jobs
        self.models = {}  # horizon -> MultiOutputRegressor
        self.feature_cols = None
        self.target_names = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    
    def _get_feature_target_cols(self, df: pd.DataFrame) -> Tuple[List[str], Dict[int, List[str]]]:
        """Get feature columns and target columns per horizon."""
        # Target columns: {pollutant}_h{horizon}
        target_cols_by_horizon = {}
        for h in range(1, 25):
            target_cols_by_horizon[h] = [f'{t}_h{h}' for t in self.target_names]
        
        # All target columns
        all_target_cols = [c for c in df.columns if any(f'{t}_h' in c for t in self.target_names)]
        
        # Feature columns: exclude targets and metadata
        meta_cols = ['datetime', 'station']
        feature_cols = [c for c in df.columns if c not in all_target_cols and c not in meta_cols]
        
        return feature_cols, target_cols_by_horizon
    
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        verbose: bool = True
    ):
        """
        Train models for all horizons.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame (for early stopping)
            verbose: Print progress
        """
        self.feature_cols, target_cols_by_horizon = self._get_feature_target_cols(train_df)
        
        X_train = train_df[self.feature_cols].values
        
        if val_df is not None:
            X_val = val_df[self.feature_cols].values
        
        for h in range(1, 25):
            if verbose:
                print(f"Training horizon {h}/24...", end='\r')
            
            target_cols = target_cols_by_horizon[h]
            y_train = train_df[target_cols].values
            
            # Handle NaN in targets (drop rows with any NaN)
            valid_mask = ~np.isnan(y_train).any(axis=1)
            X_train_h = X_train[valid_mask]
            y_train_h = y_train[valid_mask]
            
            # Create and train model
            base_model = lgb.LGBMRegressor(**self.lgbm_params)
            model = MultiOutputRegressor(base_model, n_jobs=self.n_jobs)
            
            model.fit(X_train_h, y_train_h)
            self.models[h] = model
        
        if verbose:
            print(f"Training complete: 24 models trained")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for all horizons.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Predictions (num_rows, 24, 6)
        """
        X = df[self.feature_cols].values
        num_rows = len(df)
        
        predictions = np.zeros((num_rows, 24, 6), dtype=np.float32)
        
        for h in range(1, 25):
            pred_h = self.models[h].predict(X)  # (num_rows, 6)
            predictions[:, h-1, :] = pred_h
        
        return predictions
    
    def predict_aligned(
        self,
        df: pd.DataFrame,
        station_list: List[str]
    ) -> np.ndarray:
        """
        Generate predictions aligned to (samples, H, N, D) format.
        
        Groups by datetime and arranges stations in correct order.
        
        Args:
            df: DataFrame with datetime and station columns
            station_list: Ordered list of station names
            
        Returns:
            Predictions (num_datetimes, 24, 12, 6)
        """
        # Get raw predictions
        raw_pred = self.predict(df)  # (num_rows, 24, 6)
        
        # Create station index mapping
        station_to_idx = {s: i for i, s in enumerate(station_list)}
        
        # Get unique datetimes (sorted to ensure consistent ordering)
        df = df.copy().reset_index(drop=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        unique_datetimes = sorted(df['datetime'].unique())
        num_samples = len(unique_datetimes)
        
        # Initialize aligned output
        aligned = np.zeros((num_samples, 24, 12, 6), dtype=np.float32)
        
        # Map datetime to sample index
        dt_to_idx = {dt: i for i, dt in enumerate(unique_datetimes)}
        
        # Fill aligned array using positional index
        datetimes = df['datetime'].values
        stations = df['station'].values
        
        for pos_idx in range(len(df)):
            dt = datetimes[pos_idx]
            station = stations[pos_idx]
            
            sample_idx = dt_to_idx.get(dt)
            station_idx = station_to_idx.get(station)
            
            if sample_idx is not None and station_idx is not None:
                aligned[sample_idx, :, station_idx, :] = raw_pred[pos_idx]
        
        return aligned
    
    def save(self, path: str):
        """Save all models to directory."""
        os.makedirs(path, exist_ok=True)
        
        # Save models
        for h, model in self.models.items():
            model_path = os.path.join(path, f'lgbm_h{h}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save metadata
        meta = {
            'feature_cols': self.feature_cols,
            'target_names': self.target_names,
            'lgbm_params': self.lgbm_params
        }
        with open(os.path.join(path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(meta, f)
    
    def load(self, path: str):
        """Load models from directory."""
        # Load metadata
        with open(os.path.join(path, 'metadata.pkl'), 'rb') as f:
            meta = pickle.load(f)
        
        self.feature_cols = meta['feature_cols']
        self.target_names = meta['target_names']
        self.lgbm_params = meta['lgbm_params']
        
        # Load models
        self.models = {}
        for h in range(1, 25):
            model_path = os.path.join(path, f'lgbm_h{h}.pkl')
            with open(model_path, 'rb') as f:
                self.models[h] = pickle.load(f)
    
    def __repr__(self):
        return f"LightGBMMultiHorizon(n_models={len(self.models)})"
