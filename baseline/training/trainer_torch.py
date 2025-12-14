"""
Generic PyTorch trainer for time series models.
Supports multi-GPU training with DataParallel.
"""
import os
import time
from typing import Dict, Optional, Callable, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from tqdm import tqdm

from .early_stopping import EarlyStopping
from .checkpointing import save_checkpoint, load_checkpoint


def masked_mae_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked MAE loss."""
    abs_error = torch.abs(pred - target) * mask
    return abs_error.sum() / (mask.sum() + 1e-8)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked MSE loss."""
    sq_error = ((pred - target) ** 2) * mask
    return sq_error.sum() / (mask.sum() + 1e-8)


def get_available_gpus() -> List[int]:
    """Get list of available GPU indices."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


class TorchTrainer:
    """Generic trainer for PyTorch models with multi-GPU support."""
    
    def __init__(
        self,
        model: nn.Module,
        gpu_ids: List[int] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        loss_fn: str = 'mae',
        checkpoint_dir: str = 'baseline/results/checkpoints',
        target_center: Optional[np.ndarray] = None,
        target_scale: Optional[np.ndarray] = None,
    ):
        """
        Args:
            model: PyTorch model
            gpu_ids: List of GPU IDs to use (None = auto-detect all)
            learning_rate: Learning rate
            weight_decay: L2 regularization
            loss_fn: 'mae' or 'mse'
            checkpoint_dir: Directory for checkpoints
            target_center: Optional per-target center values (D,)
            target_scale: Optional per-target scale values (D,), must be > 0
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self._target_center_np = None if target_center is None else np.asarray(target_center, dtype=np.float32)
        self._target_scale_np = None if target_scale is None else np.asarray(target_scale, dtype=np.float32)
        
        # Debug CUDA status
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Multi-GPU setup
        if torch.cuda.is_available():
            available_gpus = get_available_gpus()
            
            if gpu_ids is None:
                gpu_ids = available_gpus
            else:
                gpu_ids = [g for g in gpu_ids if g in available_gpus]
            
            self.gpu_ids = gpu_ids
            
            if len(gpu_ids) >= 1:
                primary_device = f'cuda:{gpu_ids[0]}'
                self.device = primary_device
                
                if len(gpu_ids) > 1:
                    print(f">>> Using DataParallel with {len(gpu_ids)} GPUs: {gpu_ids}")
                    model = model.to(primary_device)
                    self.model = DataParallel(model, device_ids=gpu_ids)
                else:
                    print(f">>> Using single GPU: cuda:{gpu_ids[0]}")
                    self.model = model.to(primary_device)
            else:
                print(">>> No valid GPU IDs, falling back to CPU")
                self.device = 'cpu'
                self.gpu_ids = []
                self.model = model.to('cpu')
        else:
            print(">>> CUDA not available, using CPU")
            self.device = 'cpu'
            self.gpu_ids = []
            self.model = model.to('cpu')

        if self._target_center_np is not None or self._target_scale_np is not None:
            if self._target_center_np is None or self._target_scale_np is None:
                raise ValueError("target_center and target_scale must be provided together")
            if self._target_center_np.ndim != 1 or self._target_scale_np.ndim != 1:
                raise ValueError("target_center/target_scale must be 1D arrays of length D")
            if self._target_center_np.shape != self._target_scale_np.shape:
                raise ValueError("target_center and target_scale must have the same shape")
            if np.any(self._target_scale_np <= 0):
                raise ValueError("target_scale must be strictly positive")

            self._target_center = torch.tensor(self._target_center_np, device=self.device).view(1, 1, 1, -1)
            self._target_scale = torch.tensor(self._target_scale_np, device=self.device).view(1, 1, 1, -1)
            print(">>> Using per-target normalization for loss (robust center/scale)")
        else:
            self._target_center = None
            self._target_scale = None
        
        # Get actual model for optimizer (unwrap DataParallel if needed)
        actual_model = self.model.module if isinstance(self.model, DataParallel) else self.model
        
        self.optimizer = torch.optim.Adam(
            actual_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        if loss_fn == 'mae':
            self.loss_fn = masked_mae_loss
        else:
            self.loss_fn = masked_mse_loss
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, dataloader: DataLoader, flatten_x: bool = False) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            X = batch['X'].to(self.device)
            Y = batch['Y'].to(self.device)
            Y_mask = batch['Y_mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            pred = self.model(X)
            
            # Reshape prediction if needed to match Y shape
            if pred.shape != Y.shape:
                # Assume pred is (B, H, N*D) and Y is (B, H, N, D)
                B, H, N, D = Y.shape
                pred = pred.view(B, H, N, D)
            
            if self._target_center is not None and self._target_scale is not None:
                pred_for_loss = (pred - self._target_center) / self._target_scale
                target_for_loss = (Y - self._target_center) / self._target_scale
            else:
                pred_for_loss = pred
                target_for_loss = Y

            loss = self.loss_fn(pred_for_loss, target_for_loss, Y_mask)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            X = batch['X'].to(self.device)
            Y = batch['Y'].to(self.device)
            Y_mask = batch['Y_mask'].to(self.device)
            
            pred = self.model(X)
            
            if pred.shape != Y.shape:
                B, H, N, D = Y.shape
                pred = pred.view(B, H, N, D)

            if self._target_center is not None and self._target_scale is not None:
                pred_for_loss = (pred - self._target_center) / self._target_scale
                target_for_loss = (Y - self._target_center) / self._target_scale
            else:
                pred_for_loss = pred
                target_for_loss = Y

            loss = self.loss_fn(pred_for_loss, target_for_loss, Y_mask)
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 10,
        model_name: str = 'model'
    ) -> Dict:
        """
        Train model with early stopping.
        
        Returns:
            Training history dict
        """
        early_stopping = EarlyStopping(patience=patience, mode='min')
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Checkpointing (save unwrapped model for DataParallel)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                model_to_save = self.model.module if isinstance(self.model, DataParallel) else self.model
                save_checkpoint(
                    model_to_save, self.optimizer, epoch, val_loss,
                    os.path.join(self.checkpoint_dir, f'{model_name}_best.pt')
                )
            
            # Logging
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Best: {self.best_val_loss:.4f} | "
                  f"Time: {elapsed:.1f}s")
            
            # Early stopping
            if early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model (handle DataParallel)
        model_to_load = self.model.module if isinstance(self.model, DataParallel) else self.model
        load_checkpoint(
            os.path.join(self.checkpoint_dir, f'{model_name}_best.pt'),
            model_to_load, device=self.device
        )
        
        # Save training log to file
        self._save_training_log(model_name)
        
        # Save loss curve plot
        self._save_loss_plot(model_name)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
    
    def _save_training_log(self, model_name: str):
        """Save training log to CSV file."""
        import pandas as pd
        
        log_dir = os.path.join(os.path.dirname(self.checkpoint_dir), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_data = {
            'epoch': list(range(1, len(self.train_losses) + 1)),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }
        df = pd.DataFrame(log_data)
        log_path = os.path.join(log_dir, f'{model_name}_training_log.csv')
        df.to_csv(log_path, index=False)
        print(f"Training log saved to {log_path}")
    
    def _save_loss_plot(self, model_name: str):
        """Save training/validation loss curve plot."""
        import matplotlib.pyplot as plt
        
        plot_dir = os.path.join(os.path.dirname(self.checkpoint_dir), 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        plt.axhline(y=self.best_val_loss, color='g', linestyle='--', label=f'Best Val: {self.best_val_loss:.4f}')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (Masked MAE)', fontsize=12)
        plt.title(f'{model_name.upper()} Training Curve', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(plot_dir, f'{model_name}_loss_curve.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Loss curve saved to {plot_path}")
    
    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """Generate predictions for a dataloader."""
        self.model.eval()
        predictions = []
        
        for batch in dataloader:
            X = batch['X'].to(self.device)
            pred = self.model(X)
            predictions.append(pred.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
