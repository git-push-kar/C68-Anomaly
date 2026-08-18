"""Temporal analysis of sensor behaviour across an anomaly event.

Produces per-sensor onset ordering, trends, and pre/post anomaly context. All
statements are descriptive (temporal evidence); causation is NOT asserted here.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

from evidence.process_relationships import sensor_display_name

logger = logging.getLogger(__name__)


def sensor_trend(series: np.ndarray, smoothing: int = 8) -> str:
    """Classify a 1D series as increasing / decreasing / stable.

    Uses a simple linear regression slope normalized by the series scale.
    """
    series = np.asarray(series, dtype=np.float32)
    if series.ndim != 1 or series.size < 3:
        return "stable"
    if smoothing > 1 and series.size > smoothing:
        kernel = np.ones(smoothing) / smoothing
        series = np.convolve(series, kernel, mode="same")
    x = np.arange(series.size, dtype=np.float32)
    slope = np.polyfit(x, series, 1)[0]
    scale = float(np.abs(series).max()) or 1.0
    rel_slope = slope / scale * series.size
    if rel_slope > 0.02:
        return "increasing"
    if rel_slope < -0.02:
        return "decreasing"
    return "stable"


def detect_onsets(
    per_sensor_error_series: np.ndarray,
    onset_sensitivity: float = 1.5,
    min_relative_error: float = 0.25,
) -> List[Dict]:
    """Detect, for each sensor, the first window it becomes anomalous.

    Args:
        per_sensor_error_series: [T_windows, F] per-window per-sensor errors.
        onset_sensitivity: multiple of the sensor's error std (over the series)
            that must be exceeded for onset.
        min_relative_error: a sensor qualifies only if its maximum error is at
            least this fraction of the series-wide maximum error, so sensors
            with negligible absolute error never claim an early onset.

    Returns:
        List of {sensor_index, onset_window, onset_time} sorted by onset_time.
    """
    series = np.asarray(per_sensor_error_series, dtype=np.float32)
    if series.ndim != 2:
        raise ValueError(f"Expected [T, F] error series, got {series.shape}")
    global_max = float(series.max())
    out = []
    for f in range(series.shape[1]):
        col = series[:, f]
        std = float(col.std())
        col_max = float(col.max())
        if col_max < min_relative_error * global_max:
            continue
        # Require the sensor to have a real "bump" relative to its own error
        # profile; sensors that barely change must not claim an early onset.
        threshold = max(onset_sensitivity * (std + 1e-9), 0.5 * (col_max + 1e-9))
        onset = int(np.argmax(col > threshold)) if (col > threshold).any() else -1
        if onset >= 0:
            out.append(
                {
                    "sensor_index": int(f),
                    "onset_window": onset,
                    "onset_time": onset,  # caller converts to seconds
                    "mean_error": float(col[onset:].mean()) if onset < col.size else 0.0,
                }
            )
    out.sort(key=lambda d: d["onset_time"])
    return out


def analyze_temporal_sequence(
    per_sensor_error_series: np.ndarray,
    feature_names: Sequence[str],
    window_size: int,
    stride: int,
    samples_per_minute: float = 1.0,
    onset_sensitivity: float = 1.5,
    max_events: int = 6,
    min_relative_error: float = 0.25,
) -> Dict:
    """Build the temporal sequence of sensor changes (evidence, not causation).

    relative_time is measured in minutes from the first detected onset
    (converted via ``samples_per_minute`` from window indices).
    """
    onsets = detect_onsets(per_sensor_error_series, onset_sensitivity,
                           min_relative_error=min_relative_error)
    if not onsets:
        return {"sequence": [], "first_onset_minutes": None}
    first_time = onsets[0]["onset_time"]
    seconds_per_window = stride / max(samples_per_minute, 1e-9) / 60.0
    sequence = []
    for entry in onsets[:max_events]:
        relative = (entry["onset_time"] - first_time) * seconds_per_window
        sequence.append(
            {
                "sensor": feature_names[entry["sensor_index"]],
                "display_name": sensor_display_name(feature_names[entry["sensor_index"]]),
                "event": "became_anomalous",
                "relative_time_minutes": round(float(relative), 2),
                "onset_window": entry["onset_window"],
            }
        )
    return {
        "sequence": sequence,
        "first_onset_minutes": float(first_time * seconds_per_window),
        "seconds_per_window": seconds_per_window,
    }


def pre_post_context(
    event_windows: np.ndarray,
    baseline_mean: np.ndarray,
    n_baseline: int = 20,
    dtype: str = "float32",
) -> Dict:
    """Pre-anomaly baseline vs post-anomaly window mean, per sensor.

    Args:
        event_windows: [N, W, F] windows of the anomaly event (original scale).
        baseline_mean: per-sensor mean of the normal baseline.
        n_baseline: number of pre-event windows used for the baseline window.

    Returns:
        {pre_mean[F], post_mean[F], pre_status, post_status, delta[F]}.
    """
    windows = np.asarray(event_windows, dtype=dtype)
    pre = windows[:max(n_baseline, 1)].mean(axis=0).mean(axis=0)
    post = windows.mean(axis=0).mean(axis=0)
    delta = post - pre
    return {
        "pre_mean": pre.tolist(),
        "post_mean": post.tolist(),
        "delta": delta.tolist(),
        "pre_status": "normal",
        "post_status": "anomalous",
        "baseline_mean": baseline_mean.tolist(),
    }