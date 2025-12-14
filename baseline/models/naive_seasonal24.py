"""
B1: Seasonal Naive (24-hour) Baseline
Predicts y(t+h) = y(t+h-24), i.e., same hour yesterday
"""
import numpy as np
from typing import Dict


class SeasonalNaive24:
    """
    Seasonal naive model with 24-hour seasonality.
    
    For horizon h, predict using value from 24 hours before:
    y(t+h) = y(t+h-24) = X[t-(24-h)]
    
    Since X contains the lookback window [t-L+1, t], we can access:
    - For h=1: need y(t-23), which is X[:, L-24, :, :]
    - For h=24: need y(t), which is X[:, L-1, :, :]
    - General: for h in 1..24, use X[:, L-24+h-1, :, :]
    """
    
    def __init__(self, input_scaler, feature_list: list, target_list: list):
        """
        Args:
            input_scaler: RobustScaler used for X
            feature_list: List of feature names in X
            target_list: List of target names (pollutants)
        """
        self.input_scaler = input_scaler
        self.feature_list = feature_list
        self.target_list = target_list
        self.target_indices = [feature_list.index(t) for t in target_list]
        self._center = np.asarray(getattr(input_scaler, "center_", None), dtype=np.float32)
        self._scale = np.asarray(getattr(input_scaler, "scale_", None), dtype=np.float32)
        if self._center.ndim != 1 or self._scale.ndim != 1 or self._center.shape != self._scale.shape:
            raise ValueError("input_scaler must provide 1D center_ and scale_ arrays")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Input tensor (samples, L=168, N=12, F=17) - SCALED
            
        Returns:
            Predictions (samples, H=24, N=12, D=6) - RAW units
        """
        samples, L, N, F = X.shape
        H = 24
        D = len(self.target_list)
        
        # Inverse-transform ONLY the target pollutant channels from X (RobustScaler is per-feature).
        target_center = self._center[self.target_indices].reshape(1, 1, 1, D)
        target_scale = self._scale[self.target_indices].reshape(1, 1, 1, D)
        X_targets_raw = X[:, :, :, self.target_indices].astype(np.float32) * target_scale + target_center  # (S, L, N, D)

        predictions = np.empty((samples, H, N, D), dtype=np.float32)

        # Seasonal mapping:
        # idx = (L-1) - (24-h) = L - 25 + h, for h=1..24
        for h in range(1, H + 1):
            lookback_idx = L - 25 + h
            predictions[:, h - 1, :, :] = X_targets_raw[:, lookback_idx, :, :]

        # Quick sanity: ensure not constant across horizons.
        if float(np.mean(np.abs(predictions[:, 0] - predictions[:, -1]))) <= 0.0:
            raise ValueError("SeasonalNaive24 predictions appear constant across horizons (unexpected)")

        return predictions
    
    def __repr__(self):
        return "SeasonalNaive24()"
