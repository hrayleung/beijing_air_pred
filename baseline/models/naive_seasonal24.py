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
        
        predictions = np.zeros((samples, H, N, D), dtype=np.float32)
        
        for h in range(1, H + 1):
            # Index in lookback window for value 24 hours before target
            # Target is at t+h, we want t+h-24
            # X covers [t-L+1, t], so index for time t+h-24 is:
            # (t+h-24) - (t-L+1) = L + h - 25
            lookback_idx = L + h - 25  # = L - 24 + h - 1
            
            if lookback_idx < 0 or lookback_idx >= L:
                # Fallback to persistence if out of range
                lookback_idx = L - 1
            
            # Get scaled values at this timestep
            values_scaled = X[:, lookback_idx, :, :]  # (samples, N, F)
            
            # Inverse transform
            values_flat = values_scaled.reshape(-1, F)
            values_raw = self.input_scaler.inverse_transform(values_flat)
            values_raw = values_raw.reshape(samples, N, F)
            
            # Extract target pollutants
            predictions[:, h-1, :, :] = values_raw[:, :, self.target_indices]
        
        return predictions
    
    def __repr__(self):
        return "SeasonalNaive24()"
