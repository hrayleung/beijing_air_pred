"""
Graph utilities for spatial models (STGCN, Graph WaveNet).
"""
import os
import json
from typing import Tuple

import numpy as np
import torch


def load_adjacency(
    graphs_dir: str = "processed/graphs",
    adj_type: str = "topk"
) -> Tuple[np.ndarray, list]:
    """
    Load adjacency matrix and station list.
    
    Args:
        graphs_dir: Directory containing graph files
        adj_type: 'topk' or 'full'
        
    Returns:
        adj: Adjacency matrix (N, N)
        station_list: List of station names in order
    """
    if adj_type == "topk":
        adj_path = os.path.join(graphs_dir, "adjacency_corr_topk.npy")
    else:
        adj_path = os.path.join(graphs_dir, "adjacency_corr_full.npy")
    
    adj = np.load(adj_path)
    
    with open(os.path.join(graphs_dir, "station_list.json")) as f:
        station_list = json.load(f)
    
    return adj, station_list


def normalize_adjacency(adj: np.ndarray, method: str = "sym") -> np.ndarray:
    """
    Normalize adjacency matrix.
    
    Args:
        adj: Raw adjacency matrix (N, N)
        method: 'sym' for symmetric normalization, 'rw' for random walk
        
    Returns:
        Normalized adjacency matrix
    """
    # Add self-loops if not present
    adj = adj + np.eye(adj.shape[0]) * (1 - np.diag(adj))
    
    # Compute degree matrix
    d = np.sum(adj, axis=1)
    
    if method == "sym":
        # D^{-1/2} A D^{-1/2}
        d_inv_sqrt = np.power(d, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat = np.diag(d_inv_sqrt)
        return d_mat @ adj @ d_mat
    elif method == "rw":
        # D^{-1} A
        d_inv = np.power(d, -1)
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = np.diag(d_inv)
        return d_mat @ adj
    else:
        return adj


def adjacency_to_edge_index(adj: np.ndarray) -> torch.LongTensor:
    """Convert adjacency matrix to edge index format for PyG."""
    rows, cols = np.where(adj > 0)
    edge_index = torch.LongTensor(np.stack([rows, cols], axis=0))
    return edge_index


def adjacency_to_edge_weight(adj: np.ndarray) -> torch.FloatTensor:
    """Extract edge weights from adjacency matrix."""
    rows, cols = np.where(adj > 0)
    weights = adj[rows, cols]
    return torch.FloatTensor(weights)


def compute_chebyshev_polynomials(adj: np.ndarray, K: int) -> list:
    """
    Compute Chebyshev polynomials of the graph Laplacian.
    
    Args:
        adj: Normalized adjacency matrix
        K: Order of polynomials
        
    Returns:
        List of K+1 matrices [T_0, T_1, ..., T_K]
    """
    N = adj.shape[0]
    
    # Compute Laplacian: L = I - A (for normalized adj)
    L = np.eye(N) - adj
    
    # Scale to [-1, 1]: L_scaled = 2*L/lambda_max - I
    # For normalized Laplacian, lambda_max <= 2
    L_scaled = L - np.eye(N)
    
    # Chebyshev polynomials
    T = [np.eye(N), L_scaled]
    
    for k in range(2, K + 1):
        T_k = 2 * L_scaled @ T[-1] - T[-2]
        T.append(T_k)
    
    return T[:K + 1]
