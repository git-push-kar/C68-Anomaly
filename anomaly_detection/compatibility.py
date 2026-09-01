"""Guards against loading a detector with incompatible live sensor data."""
from __future__ import annotations

from typing import Sequence

import numpy as np


class ArtifactCompatibilityError(RuntimeError):
    """Raised when a live stream cannot safely be scored by saved artifacts."""


def validate_feature_schema(actual: Sequence[str], expected: Sequence[str]) -> None:
    """Fail before scoring when source columns differ from training columns."""
    if list(actual) != list(expected):
        raise ArtifactCompatibilityError(
            "Live sensor feature order does not match the detector baseline. "
            f"Expected {list(expected)}, received {list(actual)}. Re-prepare and "
            "retrain using the same canonical sensor schema."
        )


def validate_startup_window(window, baseline, config: dict) -> None:
    """Detect an obvious raw-scale/schema mismatch before emitting false alerts.

    This is deliberately a startup-only guard. It compares the first normal
    operating window against the normal baseline; it is not used to reject
    later faults, which must be detected and reported.
    """
    guard = config.get("runtime_guard", {})
    if not bool(guard.get("enabled", True)):
        return
    values = np.asarray(window, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(baseline.mean):
        raise ArtifactCompatibilityError(
            f"Live window has shape {values.shape}; detector expects [T, {len(baseline.mean)}]."
        )
    representative = np.median(values, axis=0)
    z = np.abs((representative - baseline.mean) / (baseline.std + 1e-6))
    median_abs_z = float(np.median(z))
    outlier_fraction = float(np.mean(z > float(guard.get("max_feature_abs_z", 50.0))))
    max_median = float(guard.get("max_median_abs_z", 12.0))
    max_fraction = float(guard.get("max_outlier_fraction", 0.35))
    if median_abs_z > max_median or outlier_fraction > max_fraction:
        raise ArtifactCompatibilityError(
            "Live startup data is incompatible with the saved normal baseline "
            f"(median |z|={median_abs_z:.1f}, fraction |z|>{guard.get('max_feature_abs_z', 50.0)} "
            f"is {outlier_fraction:.0%}). This usually means stale artifacts or a "
            "different feature order/unit. Do not trust anomaly reports; run "
            "prepare_tep.py and train_anomaly_detector.py on this source data."
        )
