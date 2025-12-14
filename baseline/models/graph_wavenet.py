"""
B6: Graph WaveNet for Spatio-Temporal Forecasting
Supports both fixed and adaptive adjacency matrices.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DilatedCausalConv(nn.Module):
    """Dilated causal convolution."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
    
    def forward(self, x):
        # x: (B, C, L)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class GraphConv(nn.Module):
    """Graph convolution with support for multiple adjacency matrices."""
    
    def __init__(self, in_channels: int, out_channels: int, num_supports: int = 2):
        super().__init__()
        self.num_supports = num_supports
        self.weights = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False)
            for _ in range(num_supports)
        ])
        self.bias = nn.Parameter(torch.zeros(out_channels))
    
    def forward(self, x: torch.Tensor, supports: list) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)
            supports: list of (N, N) adjacency matrices
        Returns:
            (B, N, out_channels)
        """
        out = 0
        for i, (support, weight) in enumerate(zip(supports, self.weights)):
            # support: (N, N), x: (B, N, C)
            h = torch.einsum('nm,bmc->bnc', support, x)
            out = out + weight(h)
        return out + self.bias


class GWNetBlock(nn.Module):
    """Graph WaveNet block with gated TCN and graph convolution."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_nodes: int,
        kernel_size: int = 2,
        dilation: int = 1,
        num_supports: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Gated TCN
        self.filter_conv = DilatedCausalConv(in_channels, out_channels, kernel_size, dilation)
        self.gate_conv = DilatedCausalConv(in_channels, out_channels, kernel_size, dilation)
        
        # Graph convolution
        self.graph_conv = GraphConv(out_channels, out_channels, num_supports)
        
        # Skip and residual
        self.skip_conv = nn.Conv1d(out_channels, out_channels, 1)
        self.residual_conv = nn.Conv1d(out_channels, in_channels, 1)
        
        self.bn = nn.BatchNorm1d(in_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, supports: list) -> tuple:
        """
        Args:
            x: (B, C, N, L)
            supports: list of (N, N) adjacency matrices
        Returns:
            residual: (B, C, N, L)
            skip: (B, out_channels, N, L)
        """
        B, C, N, L = x.shape
        
        # Reshape for temporal conv: (B*N, C, L)
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, C, L)
        
        # Gated TCN
        filter_out = torch.tanh(self.filter_conv(x_flat))
        gate_out = torch.sigmoid(self.gate_conv(x_flat))
        h = filter_out * gate_out  # (B*N, out_channels, L)
        
        # Reshape for graph conv: (B, L, N, out_channels)
        out_channels = h.size(1)
        L_out = h.size(2)
        h = h.reshape(B, N, out_channels, L_out).permute(0, 3, 2, 1)  # (B, L, out_channels, N)
        h = h.permute(0, 1, 3, 2)  # (B, L, N, out_channels)
        
        # Graph conv per timestep
        h_list = []
        for t in range(L_out):
            h_t = self.graph_conv(h[:, t], supports)  # (B, N, out_channels)
            h_list.append(h_t)
        h = torch.stack(h_list, dim=1)  # (B, L, N, out_channels)
        
        # Reshape back: (B, out_channels, N, L)
        h = h.permute(0, 3, 2, 1)
        
        # Skip connection
        skip = self.skip_conv(h.reshape(B * N, out_channels, L_out))
        skip = skip.reshape(B, N, out_channels, L_out).permute(0, 2, 1, 3)
        
        # Residual
        residual = self.residual_conv(h.reshape(B * N, out_channels, L_out))
        residual = residual.reshape(B, N, C, L_out).permute(0, 2, 1, 3)
        
        # Pad to match input length if needed
        if L_out < L:
            pad = L - L_out
            residual = F.pad(residual, (pad, 0))
            skip = F.pad(skip, (pad, 0))
        
        residual = self.bn(residual.reshape(B * N, C, L)).reshape(B, N, C, L).permute(0, 2, 1, 3)
        residual = self.dropout(residual)
        
        return x + residual, skip


class GraphWaveNet(nn.Module):
    """
    Graph WaveNet for spatio-temporal forecasting.
    
    Input: (B, L=168, N=12, F=17)
    Output: (B, H=24, N=12, D=6)
    """
    
    def __init__(
        self,
        num_nodes: int = 12,
        in_channels: int = 17,
        out_channels: int = 6,
        hidden_channels: int = 32,
        skip_channels: int = 64,
        num_layers: int = 4,
        kernel_size: int = 2,
        horizon: int = 24,
        dropout: float = 0.2,
        use_adaptive_adj: bool = True,
        time_pool: str = "mean",
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.horizon = horizon
        self.use_adaptive_adj = use_adaptive_adj
        self.time_pool = str(time_pool).lower().strip()
        
        # Input projection
        self.input_conv = nn.Conv2d(in_channels, hidden_channels, (1, 1))
        
        # Adaptive adjacency
        if use_adaptive_adj:
            self.node_emb1 = nn.Parameter(torch.randn(num_nodes, 10))
            self.node_emb2 = nn.Parameter(torch.randn(10, num_nodes))
        
        # WaveNet blocks
        self.blocks = nn.ModuleList()
        receptive_field = 1
        
        for layer in range(num_layers):
            dilation = 2 ** layer
            self.blocks.append(GWNetBlock(
                hidden_channels, hidden_channels, num_nodes,
                kernel_size, dilation, num_supports=3 if use_adaptive_adj else 2,
                dropout=dropout
            ))
            receptive_field += (kernel_size - 1) * dilation
        
        # Output layers
        self.end_conv1 = nn.Conv2d(skip_channels, skip_channels, (1, 1))
        self.end_conv2 = nn.Conv2d(skip_channels, horizon * out_channels, (1, 1))
        
        # Skip aggregation
        self.skip_conv = nn.Conv2d(hidden_channels, skip_channels, (1, 1))
        
        # Placeholders for adjacency
        self.register_buffer('adj_forward', None)
        self.register_buffer('adj_backward', None)
    
    def set_adjacency(self, adj: np.ndarray):
        """Set fixed adjacency matrices."""
        # Forward and backward random walk normalization
        adj = adj + np.eye(adj.shape[0])
        
        # Forward: D^{-1} A
        d = np.sum(adj, axis=1)
        d_inv = np.power(d, -1)
        d_inv[np.isinf(d_inv)] = 0.0
        adj_forward = np.diag(d_inv) @ adj
        
        # Backward: A D^{-1}
        adj_backward = adj @ np.diag(d_inv)
        
        self.register_buffer('adj_forward', torch.FloatTensor(adj_forward))
        self.register_buffer('adj_backward', torch.FloatTensor(adj_backward))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, N, F)
        Returns:
            (B, H, N, D)
        """
        B, L, N, in_features = x.shape
        
        # (B, L, N, F) -> (B, F, N, L)
        x = x.permute(0, 3, 2, 1)
        
        # Input projection
        x = self.input_conv(x)  # (B, hidden, N, L)
        
        # Build supports
        supports = [self.adj_forward, self.adj_backward]
        if self.use_adaptive_adj:
            adp = F.softmax(F.relu(self.node_emb1 @ self.node_emb2), dim=1)
            supports.append(adp)
        
        # WaveNet blocks
        skip_sum = 0
        for block in self.blocks:
            x, skip = block(x, supports)
            skip_sum = skip_sum + skip
        
        # Output
        x = F.relu(skip_sum)
        x = self.skip_conv(x)
        x = F.relu(self.end_conv1(x))
        x = self.end_conv2(x)  # (B, H*D, N, L)
        
        if self.time_pool == "mean":
            x = x.mean(dim=-1)  # (B, H*D, N)
        elif self.time_pool == "last":
            x = x[:, :, :, -1]  # (B, H*D, N)
        else:
            raise ValueError(f"Invalid time_pool={self.time_pool!r} (expected 'mean' or 'last')")
        x = x.permute(0, 2, 1)  # (B, N, H*D)
        x = x.view(B, N, self.horizon, self.out_channels)
        x = x.permute(0, 2, 1, 3)  # (B, H, N, D)
        
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)
