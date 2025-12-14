"""
B0: Naive Persistence Baseline
Predicts y(t+h) = y(t) for all h=1..24
"""
import numpy as np
from typing import Dict


class NaivePersistence:
    """
    Naive persistence model: predict last observed value for all horizons.
    
    Since X is scaled, we need to inverse-transform pollutant channels
    to get predictions in raw units.
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
        
        # Find indices of target pollutants in feature list
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
        
        # Get last timestep values for all features
        # X[:, -1, :, :] shape: (samples, N, F)
        last_values_scaled = X[:, -1, :, :]  # (samples, N, F)
        
        # Inverse transform to raw units
        # Reshape for scaler: (samples * N, F)
        last_flat = last_values_scaled.reshape(-1, F)
        last_raw = self.input_scaler.inverse_transform(last_flat)
        last_raw = last_raw.reshape(samples, N, F)
        
        # Extract only target pollutants
        # last_raw[:, :, target_indices] shape: (samples, N, D)
        last_targets = last_raw[:, :, self.target_indices]  # (samples, N, D)
        
        # Repeat for all horizons
        # predictions shape: (samples, H, N, D)
        predictions = np.tile(last_targets[:, np.newaxis, :, :], (1, H, 1, 1))
        
        return predictions.astype(np.float32)
    
    def __repr__(self):
        return "NaivePersistence()"
