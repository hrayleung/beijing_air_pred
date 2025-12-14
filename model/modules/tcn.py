from __future__ import annotations

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=self.pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.pad > 0:
            y = y[:, :, :-self.pad]
        return y


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act(y)
        y = self.drop(y)
        return y + res


class TemporalConvNet(nn.Module):
    """
    TCN over time for each station independently.
    Expects input shaped (B*N, C, L).
    """

    def __init__(self, channels: int, num_layers: int, kernel_size: int, dropout: float):
        super().__init__()
        blocks = []
        for i in range(num_layers):
            blocks.append(TemporalBlock(channels, kernel_size, dilation=2**i, dropout=dropout))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

