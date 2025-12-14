"""
B4: Temporal Convolutional Network (TCN) for Direct Multi-Horizon Forecasting
"""
import torch
import torch.nn as nn
from typing import List


class CausalConv1d(nn.Module):
    """Causal convolution with proper padding."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
    
    def forward(self, x):
        # x: (B, C, L)
        out = self.conv(x)
        # Remove future padding
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """Single TCN residual block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        # x: (B, C, L)
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        return self.relu(out + residual)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for time series forecasting.
    
    Input: (B, L=168, N*F) or (B, L, N, F)
    Output: (B, H=24, N*D)
    """
    
    def __init__(
        self,
        input_dim: int,      # N * F = 204
        output_dim: int,     # N * D = 72
        hidden_channels: int = 64,
        num_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.2,
        horizon: int = 24
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, hidden_channels, 1)
        
        # TCN layers with exponentially increasing dilation
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(TCNBlock(
                hidden_channels, hidden_channels,
                kernel_size, dilation, dropout
            ))
        self.tcn = nn.Sequential(*layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, horizon * output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, input_dim)
        Returns:
            (B, H, output_dim)
        """
        batch_size = x.size(0)
        
        # (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # TCN layers
        x = self.tcn(x)
        
        # Use last timestep
        x = x[:, :, -1]  # (B, hidden_channels)
        
        # Output projection
        out = self.output_proj(x)  # (B, H * output_dim)
        out = out.view(batch_size, self.horizon, self.output_dim)
        
        return out
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)
