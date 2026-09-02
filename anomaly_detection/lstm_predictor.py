"""Minimal LSTM predictor for next-timestep prediction.
Input: 60x52, Output: 52 (next timestep)
Trained only on normal data to test if prediction error separates faults 3,9,15.
"""
import torch
import torch.nn as nn
from typing import Dict

class LSTMPredictor(nn.Module):
    def __init__(self, num_features=52, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(hidden_size, num_features)

    def forward(self, x):
        # x: [B,60,52]
        _, (h_n, _) = self.lstm(x)
        # Use last hidden of last layer
        h = h_n[-1]  # [B, hidden]
        pred = self.out_proj(h)  # [B,52]
        return pred

def build_predictor(config: Dict, num_features=52):
    hidden = int(config.get("hidden_size", 64))
    layers = int(config.get("num_layers", 1))
    dropout = float(config.get("dropout", 0.0))
    return LSTMPredictor(num_features, hidden, layers, dropout)
