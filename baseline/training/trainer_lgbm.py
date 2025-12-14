"""
LightGBM training utilities.
"""
import os
from typing import Dict, Optional

import pandas as pd
import numpy as np

from baseline.models.lgbm_multioutput import LightGBMMultiHorizon


def train_lgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    lgbm_params: Dict = None,
    save_dir: str = 'baseline/results/checkpoints/lgbm',
    verbose: bool = True
) -> LightGBMMultiHorizon:
    """
    Train LightGBM multi-horizon model.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        lgbm_params: LightGBM parameters
        save_dir: Directory to save models
        verbose: Print progress
        
    Returns:
        Trained LightGBMMultiHorizon model
    """
    model = LightGBMMultiHorizon(lgbm_params=lgbm_params)
    
    if verbose:
        print("Training LightGBM models...")
    
    model.fit(train_df, val_df, verbose=verbose)
    
    # Save models
    if save_dir:
        model.save(save_dir)
        if verbose:
            print(f"Models saved to {save_dir}")
    
    return model
