"""Normalization for TEP sensor data.

The scaler must be fitted ONLY on normal-operation training data so that the
normal baseline is never contaminated by fault data. The fitted scaler and the
per-sensor baseline statistics (mean/std in the original scale) are persisted
separately so that the anomaly detector and evidence generator can load them
independently.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils import ensure_dir, load_json, save_json, to_native

logger = logging.getLogger(__name__)

SCALER_FILE = "scaler.pkl"
CONFIG_FILE = "config.json"
BASELINE_FILE = "baseline_stats.json"


@dataclass
class BaselineStats:
    """Per-sensor statistics in the ORIGINAL (unscaled) units, from normal data."""

    mean: np.ndarray
    std: np.ndarray
    min_: np.ndarray
    max_: np.ndarray
    feature_names: list
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "mean": to_native(self.mean),
            "std": to_native(self.std),
            "min": to_native(self.min_),
            "max": to_native(self.max_),
            "feature_names": list(self.feature_names),
            "n_samples": int(self.n_samples),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BaselineStats":
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
            min_=np.asarray(data["min"], dtype=np.float32),
            max_=np.asarray(data["max"], dtype=np.float32),
            feature_names=list(data["feature_names"]),
            n_samples=int(data["n_samples"]),
        )

    def percent_deviation(self, values: np.ndarray) -> np.ndarray:
        """Signed % deviation of ``values`` from the normal mean.

        Returns a float array of the same shape; when the baseline mean is
        (near) zero the z-score is returned instead so the value stays finite.
        """
        values = np.asarray(values, dtype=np.float32)
        eps = 1e-6
        safe_mean = np.where(np.abs(self.mean) > eps, self.mean, 1.0)
        percent = (values - self.mean) / safe_mean * 100.0
        zero_base = np.abs(self.mean) <= eps
        if zero_base.any():
            z = (values - self.mean) / (self.std + eps)
            percent[zero_base] = z[zero_base] * 100.0
        return percent

    def z_score(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return (values - self.mean) / (self.std + 1e-6)


class BaselineScaler:
    """Thin wrapper around sklearn scalers with baseline statistics."""

    def __init__(
        self,
        scaler: Union[StandardScaler, MinMaxScaler],
        baseline: Optional[BaselineStats] = None,
        kind: str = "standard",
    ) -> None:
        self.scaler = scaler
        self.baseline = baseline
        self.kind = kind

    @classmethod
    def build(
        cls, normal_data: pd.DataFrame, kind: str = "standard"
    ) -> "BaselineScaler":
        """Fit a scaler on normal data ONLY and compute baseline stats."""
        if kind == "standard":
            scaler = StandardScaler()
        elif kind == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler kind: {kind}")
        values = normal_data.astype(np.float32).to_numpy()
        scaler.fit(values)
        baseline = BaselineStats(
            mean=values.mean(axis=0),
            std=values.std(axis=0),
            min_=values.min(axis=0),
            max_=values.max(axis=0),
            feature_names=list(normal_data.columns),
            n_samples=int(values.shape[0]),
        )
        return cls(scaler=scaler, baseline=baseline, kind=kind)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.transform(np.asarray(data, dtype=np.float32))

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(np.asarray(data, dtype=np.float32))

    def save(self, directory: Union[str, Path]) -> None:
        import pickle

        directory = ensure_dir(directory)
        with open(directory / SCALER_FILE, "wb") as handle:
            pickle.dump(self.scaler, handle)
        save_json(
            directory / CONFIG_FILE,
            {"kind": self.kind, "feature_names": list(self.baseline.feature_names)
             if self.baseline else None},
        )
        if self.baseline is not None:
            save_json(directory / BASELINE_FILE, self.baseline.to_dict())
        logger.info("Saved scaler + baseline to %s", directory)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "BaselineScaler":
        import pickle

        directory = Path(directory)
        with open(directory / SCALER_FILE, "rb") as handle:
            scaler = pickle.load(handle)
        config = load_json(directory / CONFIG_FILE) if (directory / CONFIG_FILE).exists() else {}
        baseline = None
        baseline_path = directory / BASELINE_FILE
        if baseline_path.exists():
            baseline = BaselineStats.from_dict(load_json(baseline_path))
        return cls(scaler=scaler, baseline=baseline, kind=config.get("kind", "standard"))


def build_and_save_scaler(
    normal_data: pd.DataFrame, config: dict, output_dir: Union[str, Path]
) -> BaselineScaler:
    """Fit (normal-only) and persist the scaler plus baseline statistics."""
    scaler = BaselineScaler.build(normal_data, config["preprocessing"].get("scaler", "standard"))
    scaler.save(output_dir)
    return scaler


def load_scaler(directory: Union[str, Path]) -> BaselineScaler:
    return BaselineScaler.load(directory)