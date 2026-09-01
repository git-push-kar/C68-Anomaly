"""Inference wrapper for the trained LSTM autoencoder.

The anomaly detector is fully independent of InternVL: it loads its own model,
the separately saved scaler, and the threshold, and produces per-window scores
plus per-sensor reconstruction errors.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from anomaly_detection.lstm_autoencoder import LSTMAutoencoder, build_autoencoder
from anomaly_detection.threshold import ThresholdConfig, load_threshold
from preprocessing.scaler import BaselineScaler, load_scaler
from utils import ensure_dir, get_device, load_json

logger = logging.getLogger(__name__)

MODEL_FILE = "model.pt"


class AnomalyDetector:
    """Loads scaler + autoencoder + threshold and scores windows."""

    def __init__(
        self,
        model: LSTMAutoencoder,
        scaler: BaselineScaler,
        threshold: ThresholdConfig,
        device=None,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.device = device or get_device()
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_artifacts(
        cls,
        model_dir,
        scaler_dir,
        threshold_dir: Optional[Union[str, Path]] = None,
        device=None,
    ) -> "AnomalyDetector":
        model_dir = Path(model_dir)
        scaler_dir = Path(scaler_dir)
        threshold_dir = Path(threshold_dir) if threshold_dir else model_dir

        model_path = model_dir / MODEL_FILE
        if not model_path.exists():
            raise FileNotFoundError(f"Anomaly detector model not found: {model_path}")
        state = torch.load(model_path, map_location="cpu")
        scaler = load_scaler(scaler_dir)
        threshold = load_threshold(threshold_dir)

        num_features = int(state["num_features"])
        sequence_length = int(state["sequence_length"])
        # Prefer lstm config saved inside the checkpoint, fall back to the
        # separate config.json (written by train.py) which contains the full
        # 128/64 architecture used for the current model.
        saved_config = {}
        config_path = model_dir / "config.json"
        if config_path.exists():
            try:
                saved_config = load_json(config_path)
            except Exception:
                saved_config = {}
        lstm_cfg = state.get("lstm") or saved_config.get("lstm") or {}
        if not lstm_cfg and saved_config:
            # config.json nests under "lstm" directly
            lstm_cfg = saved_config.get("lstm", {})
        model = build_autoencoder(
            {"anomaly_detector": {"lstm": lstm_cfg}},
            num_features,
            sequence_length,
        )
        model.load_state_dict(state["model_state_dict"])
        return cls(model, scaler, threshold, device=device)

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    @torch.no_grad()
    def score_window(self, window: np.ndarray) -> tuple:
        """Score one window (original scale, [W, F]).

        Returns:
            (anomaly_score, per_sensor_error[F]) where anomaly_score is the
            mean per-sensor MSE over the window (in scaled units).
        """
        window = np.asarray(window, dtype=np.float32)
        if window.ndim != 2:
            raise ValueError(f"Expected [W, F] window, got {window.shape}")
        scaled = self.scaler.transform(window)[None, :, :]
        tensor = torch.as_tensor(scaled, dtype=torch.float32, device=self.device)
        recon = self.model(tensor)
        err = (recon - tensor) ** 2  # [1, W, F]
        per_sensor = err.mean(dim=1).cpu().numpy()[0]  # [F]
        score = float(per_sensor.mean())
        return score, per_sensor

    @torch.no_grad()
    def score_windows(self, windows: np.ndarray) -> tuple:
        """Score a batch of windows -> (scores[N], per_sensor_errors[N, F])."""
        windows = np.asarray(windows, dtype=np.float32)
        n = windows.shape[0]
        flat = windows.reshape(-1, windows.shape[-1])
        scaled = self.scaler.transform(flat).reshape(windows.shape)
        tensor = torch.as_tensor(scaled, dtype=torch.float32, device=self.device)
        all_scores: list = []
        all_errors: list = []
        batch_size = 256
        for i in range(0, n, batch_size):
            batch = tensor[i:i + batch_size]
            recon = self.model(batch)
            err = (recon - batch) ** 2
            per_sensor = err.mean(dim=1).cpu().numpy()
            all_scores.append(per_sensor.mean(axis=1))
            all_errors.append(per_sensor)
        return (
            np.concatenate(all_scores),
            np.concatenate(all_errors, axis=0),
        )

    def is_anomalous(self, score: float) -> bool:
        return score > self.threshold.threshold

    def describe(self) -> Dict:
        return {
            "threshold_method": self.threshold.method,
            "threshold": self.threshold.threshold,
            "num_features": self.model.num_features,
            "sequence_length": self.model.sequence_length,
        }