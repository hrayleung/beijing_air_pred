"""
B3: LSTM Seq2Seq for Direct Multi-Horizon Forecasting
"""
import torch
import torch.nn as nn
from typing import Optional


class LSTMSeq2Seq(nn.Module):
    """
    LSTM Encoder-Decoder for direct 24-step forecasting.
    
    Input: (B, L=168, N*F) flattened spatial features
    Output: (B, H=24, N*D) flattened predictions
    """
    
    def __init__(
        self,
        input_dim: int,      # N * F = 12 * 17 = 204
        output_dim: int,     # N * D = 12 * 6 = 72
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 24
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output projection
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        # Initial decoder input projection
        self.fc_init = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        teacher_forcing_ratio: float = 0.0,
        target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input (B, L, input_dim)
            teacher_forcing_ratio: Probability of using ground truth as decoder input
            target: Ground truth (B, H, output_dim) for teacher forcing
            
        Returns:
            Predictions (B, H, output_dim)
        """
        batch_size = x.size(0)
        
        # Encode
        _, (hidden, cell) = self.encoder(x)
        
        # Initialize decoder input from last encoder hidden state
        decoder_input = self.fc_init(hidden[-1]).unsqueeze(1)  # (B, 1, output_dim)
        
        # Decode step by step
        outputs = []
        for t in range(self.horizon):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            prediction = self.fc_out(self.dropout(decoder_output))  # (B, 1, output_dim)
            outputs.append(prediction)
            
            # Prepare next input
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target[:, t:t+1, :]
            else:
                decoder_input = prediction
        
        # Stack outputs: (B, H, output_dim)
        outputs = torch.cat(outputs, dim=1)
        
        return outputs
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference mode (no teacher forcing)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, teacher_forcing_ratio=0.0)


class LSTMDirect(nn.Module):
    """
    Simpler LSTM that directly outputs all horizons at once.
    
    Input: (B, L=168, N*F)
    Output: (B, H=24, N*D)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 24
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Project final hidden state to all horizons
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, input_dim)
        Returns:
            (B, H, output_dim)
        """
        batch_size = x.size(0)
        
        # Encode sequence
        _, (hidden, _) = self.lstm(x)
        
        # Use last layer hidden state
        h = hidden[-1]  # (B, hidden_dim)
        
        # Project to all outputs
        out = self.fc(h)  # (B, H * output_dim)
        out = out.view(batch_size, self.horizon, self.output_dim)
        
        return out
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)
