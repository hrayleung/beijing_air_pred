"""
B5: Spatio-Temporal Graph Convolutional Network (STGCN)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChebConv(nn.Module):
    """Chebyshev spectral graph convolution."""
    
    def __init__(self, in_channels: int, out_channels: int, K: int = 3):
        super().__init__()
        self.K = K
        self.weights = nn.Parameter(torch.FloatTensor(K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) node features
            L: (N, N) normalized Laplacian
        Returns:
            (B, N, out_channels)
        """
        B, N, C = x.shape
        
        # Chebyshev polynomials
        T_0 = x  # (B, N, C)
        out = torch.einsum('bnc,co->bno', T_0, self.weights[0])
        
        if self.K > 1:
            T_1 = torch.einsum('nm,bmc->bnc', L, x)
            out = out + torch.einsum('bnc,co->bno', T_1, self.weights[1])
        
        for k in range(2, self.K):
            T_2 = 2 * torch.einsum('nm,bmc->bnc', L, T_1) - T_0
            out = out + torch.einsum('bnc,co->bno', T_2, self.weights[k])
            T_0, T_1 = T_1, T_2
        
        return out + self.bias


class STConvBlock(nn.Module):
    """Spatio-temporal convolution block: Temporal -> Spatial -> Temporal."""
    
    def __init__(
        self,
        in_channels: int,
        spatial_channels: int,
        out_channels: int,
        num_nodes: int,
        kernel_size: int = 3,
        K: int = 3
    ):
        super().__init__()
        
        # Temporal conv 1
        self.temporal1 = nn.Conv2d(in_channels, spatial_channels, (1, kernel_size))
        
        # Spatial conv (graph)
        self.spatial = ChebConv(spatial_channels, spatial_channels, K)
        
        # Temporal conv 2
        self.temporal2 = nn.Conv2d(spatial_channels, out_channels, (1, kernel_size))
        
        # Batch norms
        self.bn1 = nn.BatchNorm2d(spatial_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Time reduction from two temporal convs
        self.time_reduction = 2 * (kernel_size - 1)
    
    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, N, T) input
            L: (N, N) normalized Laplacian
        Returns:
            (B, out_channels, N, T')
        """
        # Temporal conv 1: (B, C, N, T) -> (B, spatial_channels, N, T-k+1)
        out = self.temporal1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Spatial conv: process each timestep
        B, C, N, T = out.shape
        out = out.permute(0, 3, 2, 1)  # (B, T, N, C)
        out = out.reshape(B * T, N, C)
        out = self.spatial(out, L)
        out = out.reshape(B, T, N, -1)
        out = out.permute(0, 3, 2, 1)  # (B, C, N, T)
        out = F.relu(out)
        
        # Temporal conv 2
        out = self.temporal2(out)
        out = self.bn2(out)
        
        return out


class STGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network.
    
    Input: (B, L=168, N=12, F=17)
    Output: (B, H=24, N=12, D=6)
    """
    
    def __init__(
        self,
        num_nodes: int = 12,
        in_channels: int = 17,
        out_channels: int = 6,
        hidden_channels: int = 64,
        num_layers: int = 2,
        kernel_size: int = 3,
        K: int = 3,
        horizon: int = 24,
        dropout: float = 0.2,
        time_pool: str = "mean",
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.horizon = horizon
        self.time_pool = str(time_pool).lower().strip()
        
        # ST blocks
        self.blocks = nn.ModuleList()
        
        # First block
        self.blocks.append(STConvBlock(
            in_channels, hidden_channels, hidden_channels,
            num_nodes, kernel_size, K
        ))
        
        # Additional blocks
        for _ in range(num_layers - 1):
            self.blocks.append(STConvBlock(
                hidden_channels, hidden_channels, hidden_channels,
                num_nodes, kernel_size, K
            ))
        
        # Output layer
        self.output_conv = nn.Conv2d(hidden_channels, out_channels, (1, 1))
        
        # Final projection to horizon
        self.fc = nn.Linear(hidden_channels, horizon * out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Placeholder for Laplacian
        self.register_buffer('L', None)
    
    def set_adjacency(self, adj: np.ndarray):
        """Set adjacency matrix and compute normalized Laplacian."""
        # Symmetric normalization
        adj = adj + np.eye(adj.shape[0])
        d = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(d, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat = np.diag(d_inv_sqrt)
        adj_norm = d_mat @ adj @ d_mat
        
        # Laplacian: L = I - A_norm
        L = np.eye(adj.shape[0]) - adj_norm
        
        # Scale to [-1, 1]
        L = torch.FloatTensor(L)
        self.register_buffer('L', L)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, N, F)
        Returns:
            (B, H, N, D)
        """
        B, L, N, F = x.shape
        
        # (B, L, N, F) -> (B, F, N, L)
        x = x.permute(0, 3, 2, 1)
        
        # ST blocks
        for block in self.blocks:
            x = block(x, self.L)
            x = self.dropout(x)
        
        if self.time_pool == "mean":
            x = x.mean(dim=-1)  # (B, C, N)
        elif self.time_pool == "last":
            x = x[:, :, :, -1]  # (B, C, N)
        else:
            raise ValueError(f"Invalid time_pool={self.time_pool!r} (expected 'mean' or 'last')")
        
        # Per-node output projection
        x = x.permute(0, 2, 1)  # (B, N, C)
        x = self.fc(x)  # (B, N, H * D)
        x = x.view(B, N, self.horizon, self.out_channels)
        x = x.permute(0, 2, 1, 3)  # (B, H, N, D)
        
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)
