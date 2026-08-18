"""Anomaly threshold estimation from NORMAL validation scores only.

The threshold is learned from the reconstruction-error distribution of the
normal validation split -- never from fault data and never a hardcoded 0.5.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from utils import ensure_dir, load_json, save_json

logger = logging.getLogger(__name__)

THRESHOLD_FILE = "threshold.json"


@dataclass
class ThresholdConfig:
    method: str = "percentile"          # percentile | mean_std | validation_percentile
    percentile: float = 99.0
    k_std: float = 6.0
    threshold: Optional[float] = None   # filled after estimation
    n_normal_scores: Optional[int] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ThresholdConfig":
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__})


def compute_threshold(
    normal_scores: np.ndarray, config: Dict, method: Optional[str] = None
) -> ThresholdConfig:
    """Estimate ``threshold`` from normal (validation) reconstruction scores.

    Args:
        normal_scores: per-window MSE of normal validation windows.
        config: the ``anomaly_detector.threshold`` config section.
        method: override the configured method.

    Returns:
        ThresholdConfig with a concrete ``threshold`` value.
    """
    th_cfg = config if isinstance(config, ThresholdConfig) else config["anomaly_detector"]["threshold"]
    method = (method or th_cfg.get("method", "percentile")).lower()
    scores = np.asarray(normal_scores, dtype=np.float32)
    if scores.size == 0:
        raise ValueError("Cannot estimate threshold from an empty score array.")

    if method in ("percentile", "validation_percentile"):
        pct = float(th_cfg.get("percentile", 99.0))
        threshold = float(np.percentile(scores, pct))
    elif method == "mean_std":
        mean = float(scores.mean())
        std = float(scores.std())
        k = float(th_cfg.get("k_std", 6.0))
        threshold = mean + k * std
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    result = ThresholdConfig(
        method=method,
        percentile=float(th_cfg.get("percentile", 99.0)),
        k_std=float(th_cfg.get("k_std", 6.0)),
        threshold=threshold,
        n_normal_scores=int(scores.size),
        mean=float(scores.mean()),
        std=float(scores.std()),
    )
    logger.info(
        "Threshold (method=%s): %.5f  [n=%d, mean=%.5f, std=%.5f]",
        method, threshold, scores.size, result.mean, result.std,
    )
    return result


def save_threshold(threshold: ThresholdConfig, directory) -> Path:
    directory = ensure_dir(directory)
    path = directory / THRESHOLD_FILE
    save_json(path, threshold.to_dict())
    logger.info("Saved threshold config to %s", path)
    return path


def load_threshold(directory) -> ThresholdConfig:
    path = Path(directory) / THRESHOLD_FILE
    if not path.exists():
        raise FileNotFoundError(f"Threshold file not found: {path}")
    return ThresholdConfig.from_dict(load_json(path))