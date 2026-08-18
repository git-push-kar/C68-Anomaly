"""Per-sensor anomaly contribution and deviation from the normal baseline."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

from evidence.process_relationships import sensor_display_name

logger = logging.getLogger(__name__)


def compute_sensor_contributions(per_sensor_error: np.ndarray) -> Dict:
    """Normalize per-sensor reconstruction error into relative contributions.

    Args:
        per_sensor_error: [F] per-sensor MSE for one window/event.

    Returns:
        Dict with absolute errors and softmax-style contributions summing to 1.
    """
    errors = np.asarray(per_sensor_error, dtype=np.float32)
    if errors.ndim != 1:
        errors = errors.mean(axis=0)
    total = float(errors.sum())
    if total <= 0:
        contributions = np.zeros_like(errors)
    else:
        contributions = errors / total
    return {
        "errors": errors.tolist(),
        "contributions": contributions.tolist(),
        "top_contributor_index": int(np.argmax(errors)) if errors.size else None,
    }


def top_k_sensors(
    per_sensor_error: np.ndarray,
    feature_names: Sequence[str],
    k: int = 5,
) -> List[Dict]:
    """Return the ``k`` sensors with the highest reconstruction error."""
    errors = np.asarray(per_sensor_error, dtype=np.float32)
    if errors.ndim != 1:
        errors = errors.mean(axis=0)
    k = min(int(k), len(errors))
    order = np.argsort(errors)[::-1][:k]
    total = float(errors.sum()) or 1.0
    result = []
    for idx in order:
        raw = feature_names[idx]
        result.append(
            {
                "index": int(idx),
                "name": raw,
                "display_name": sensor_display_name(raw),
                "error": float(errors[idx]),
                "contribution": float(errors[idx] / total),
            }
        )
    return result


def compute_percent_deviations(
    values: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_std: np.ndarray,
    feature_names: Sequence[str],
    top_sensors: Optional[Sequence[Dict]] = None,
    min_deviation_percent: float = 3.0,
) -> List[Dict]:
    """Percentage deviation of ``values`` from the normal baseline.

    ``values`` are in the ORIGINAL (unscaled) units. Sensors whose deviation is
    below ``min_deviation_percent`` are dropped, unless explicitly in
    ``top_sensors``.
    """
    values = np.asarray(values, dtype=np.float32)
    baseline_mean = np.asarray(baseline_mean, dtype=np.float32)
    baseline_std = np.asarray(baseline_std, dtype=np.float32)
    eps = 1e-6

    safe_mean = np.where(np.abs(baseline_mean) > eps, baseline_mean, 1.0)
    deviation = (values - baseline_mean) / safe_mean * 100.0
    zero_base = np.abs(baseline_mean) <= eps
    if zero_base.any():
        deviation[zero_base] = (values - baseline_mean)[zero_base] / (
            baseline_std[zero_base] + eps
        ) * 100.0

    force_indices = {int(t["index"]) for t in (top_sensors or [])}
    keep = np.abs(deviation) >= min_deviation_percent
    selected = np.flatnonzero(np.logical_or(keep, np.isin(np.arange(len(values)), list(force_indices))))

    out = []
    for idx in selected:
        out.append(
            {
                "index": int(idx),
                "name": feature_names[idx],
                "display_name": sensor_display_name(feature_names[idx]),
                "current_value": float(values[idx]),
                "baseline_value": float(baseline_mean[idx]),
                "deviation_percent": float(deviation[idx]),
                "z_score": float((values[idx] - baseline_mean[idx]) / (baseline_std[idx] + eps)),
            }
        )
    return sorted(out, key=lambda d: abs(d["deviation_percent"]), reverse=True)


def event_level_deviations(
    event_errors: np.ndarray,
    event_windows_mean: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_std: np.ndarray,
    feature_names: Sequence[str],
    top_k: int = 5,
    min_deviation_percent: float = 3.0,
) -> List[Dict]:
    """Combine reconstruction-error ranking with % deviation for an event."""
    top = top_k_sensors(event_errors, feature_names, k=top_k)
    deviations = compute_percent_deviations(
        event_windows_mean, baseline_mean, baseline_std, feature_names,
        top_sensors=top, min_deviation_percent=min_deviation_percent,
    )
    dev_by_name = {d["display_name"]: d for d in deviations}
    merged = []
    for entry in top:
        merged.append({**entry, **dev_by_name.get(entry["display_name"], {})})
    return merged