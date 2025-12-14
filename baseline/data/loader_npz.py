"""
NPZ Data Loader for P1_deep and P2_simple splits.
"""
import os
import json
import pickle
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_npz_data(
    data_dir: str = "processed/P1_deep",
    splits: Tuple[str, ...] = ("train", "val", "test")
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load NPZ data for specified splits.
    
    Returns:
        Dict with keys 'train', 'val', 'test', each containing:
        - X: (samples, 168, 12, 17)
        - Y: (samples, 24, 12, 6)
        - X_mask: (samples, 168, 12, 17)
        - Y_mask: (samples, 24, 12, 6)
        - datetime_origins: (samples,)
    """
    data = {}
    for split in splits:
        path = os.path.join(data_dir, f"{split}.npz")
        loaded = np.load(path, allow_pickle=True)
        data[split] = {key: loaded[key] for key in loaded.files}
    return data


def load_scaler(data_dir: str = "processed/P1_deep") -> Dict:
    """Load scalers from pickle file."""
    path = os.path.join(data_dir, "scaler.pkl")
    with open(path, 'rb') as f:
        scalers = pickle.load(f)
    # Handle both old (single scaler) and new (dict) formats
    if isinstance(scalers, dict):
        return scalers
    else:
        return {'input_scaler': scalers, 'target_scaler': None}


def load_metadata(processed_dir: str = "processed") -> Dict:
    """Load metadata JSON."""
    with open(os.path.join(processed_dir, "metadata.json")) as f:
        return json.load(f)


def load_feature_list(processed_dir: str = "processed") -> list:
    """Load feature list."""
    with open(os.path.join(processed_dir, "feature_list.json")) as f:
        return json.load(f)


def load_target_list(processed_dir: str = "processed") -> list:
    """Load target list."""
    with open(os.path.join(processed_dir, "target_list.json")) as f:
        return json.load(f)


class PRSADataset(Dataset):
    """PyTorch Dataset for PRSA data."""
    
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        X_mask: Optional[np.ndarray] = None,
        Y_mask: Optional[np.ndarray] = None,
        flatten_x: bool = False
    ):
        """
        Args:
            X: Input tensor (samples, L, N, F)
            Y: Target tensor (samples, H, N, D)
            X_mask: Input mask (samples, L, N, F)
            Y_mask: Target mask (samples, H, N, D)
            flatten_x: If True, flatten spatial dim: (samples, L, N*F)
        """
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        self.X_mask = torch.FloatTensor(X_mask) if X_mask is not None else None
        self.Y_mask = torch.FloatTensor(Y_mask) if Y_mask is not None else None
        self.flatten_x = flatten_x
        
        if flatten_x:
            B, L, N, F = self.X.shape
            self.X = self.X.reshape(B, L, N * F)
            if self.X_mask is not None:
                self.X_mask = self.X_mask.reshape(B, L, N * F)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        item = {
            'X': self.X[idx],
            'Y': self.Y[idx]
        }
        if self.X_mask is not None:
            item['X_mask'] = self.X_mask[idx]
        if self.Y_mask is not None:
            item['Y_mask'] = self.Y_mask[idx]
        return item


def create_dataloaders(
    data: Dict[str, Dict[str, np.ndarray]],
    batch_size: int = 64,
    flatten_x: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Dict[str, DataLoader]:
    """Create DataLoaders for all splits."""
    loaders = {}
    for split, split_data in data.items():
        dataset = PRSADataset(
            X=split_data['X'],
            Y=split_data['Y'],
            X_mask=split_data.get('X_mask'),
            Y_mask=split_data.get('Y_mask'),
            flatten_x=flatten_x
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    return loaders
