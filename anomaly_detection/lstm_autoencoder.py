"""LSTM Autoencoder for unsupervised anomaly detection.

Architecture
------------
Input   [batch, sequence_length, num_features]
  |
  | LSTM Encoder (optional bidirectional)
  v
Latent  [batch, latent_dim]   (last hidden state -> Linear)
  |
  | LSTM Decoder (teacher-free: learns from a zero-start state + latent)
  v
Reconstructed [batch, sequence_length, num_features]

Trained ONLY on normal-operation windows using reconstruction loss (MSE).
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        num_layers: int,
        latent_dim: int,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        dir_factor = 2 if bidirectional else 1
        self.project = nn.Linear(hidden_size * dir_factor, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        hidden = h_n[-1]  # use last layer's hidden state (already dir-concat)
        return self.project(hidden)


class LSTMDecoder(nn.Module):
    """Decode a latent vector back into a [B, W, F] sequence.

    The decoder receives the latent as its initial hidden/cell state and as the
    repeated per-step input. Feeding zeros at every step often collapses to a
    mean reconstruction on standardized sensor data, because the decoder has
    very little signal after initialization.
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        num_layers: int,
        latent_dim: int,
        sequence_length: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.latent_proj = nn.Linear(latent_dim, hidden_size * num_layers)
        self.cell_proj = nn.Linear(latent_dim, hidden_size * num_layers)
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(hidden_size, num_features)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch = latent.shape[0]
        h0 = self.latent_proj(latent).view(self.num_layers, batch, self.hidden_size)
        c0 = self.cell_proj(latent).view(self.num_layers, batch, self.hidden_size)
        decoder_input = latent[:, None, :].repeat(1, self.sequence_length, 1)
        out, _ = self.lstm(decoder_input, (h0, c0))
        return self.out_proj(out)


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        sequence_length: int,
        hidden_size: int = 32,
        num_layers: int = 2,
        latent_dim: int = 16,
        dropout: float = 0.1,
        bidirectional_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.encoder = LSTMEncoder(
            num_features, hidden_size, num_layers, latent_dim, dropout,
            bidirectional_encoder,
        )
        self.decoder = LSTMDecoder(
            num_features, hidden_size, num_layers, latent_dim, sequence_length, dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def build_autoencoder(config: Dict, num_features: int, sequence_length: int) -> LSTMAutoencoder:
    """Instantiate an LSTMAutoencoder from the ``anomaly_detector.lstm`` config."""
    lstm_cfg = config["anomaly_detector"]["lstm"]
    return LSTMAutoencoder(
        num_features=num_features,
        sequence_length=sequence_length,
        hidden_size=int(lstm_cfg.get("hidden_size", 32)),
        num_layers=int(lstm_cfg.get("num_layers", 2)),
        latent_dim=int(lstm_cfg.get("latent_dim", 16)),
        dropout=float(lstm_cfg.get("dropout", 0.1)),
        bidirectional_encoder=bool(lstm_cfg.get("bidirectional_encoder", False)),
    )