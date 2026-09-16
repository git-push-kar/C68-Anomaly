"""Thresholds and smoothing for dynamic statistics.
Empirical percentiles on held-out normal runs, plus EWMA/CUSUM.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

from utils import ensure_dir


def empirical_threshold(
    normal_stats: np.ndarray, target_far: float
) -> float:
    """Percentile threshold to hit target FAR on held-out normal data."""
    if normal_stats.size == 0:
        raise ValueError("Empty normal_stats for threshold")
    # FAR 0.01 => 99th percentile
    q = 100.0 * (1.0 - target_far)
    return float(np.percentile(normal_stats, q))


def apply_ewma(x: np.ndarray, lam: float = 0.1) -> np.ndarray:
    """EWMA smoothing: s_t = lam*x_t + (1-lam)*s_{t-1}, s_0 = x_0."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    s = np.empty_like(x, dtype=np.float64)
    s[0] = x[0]
    for t in range(1, x.size):
        s[t] = lam * x[t] + (1.0 - lam) * s[t - 1]
    return s


def apply_cusum(
    x: np.ndarray, k: float = 0.5, h: float = 5.0, target: Optional[float] = None
) -> np.ndarray:
    """Two-sided CUSUM. Returns cumulative sum statistic; threshold at h."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    if target is None:
        target = float(np.mean(x))
    g_pos = np.zeros_like(x, dtype=np.float64)
    g_neg = np.zeros_like(x, dtype=np.float64)
    s = np.zeros_like(x, dtype=np.float64)
    for t in range(1, x.size):
        g_pos[t] = max(0.0, g_pos[t - 1] + x[t] - target - k)
        g_neg[t] = max(0.0, g_neg[t - 1] + target - x[t] - k)
        s[t] = max(g_pos[t], g_neg[t])
    return s


def smooth_stats(
    stats: Dict[str, np.ndarray],
    mode: str = "none",
    ewma_lambda: float = 0.1,
    cusum_k: float = 0.5,
    cusum_h: float = 5.0,
) -> Dict[str, np.ndarray]:
    """Apply smoothing to each statistic stream."""
    if mode == "none" or mode is None:
        return stats
    out: Dict[str, np.ndarray] = {}
    for name, arr in stats.items():
        if mode == "ewma":
            out[name] = apply_ewma(arr, lam=ewma_lambda)
        elif mode == "cusum":
            out[name] = apply_cusum(arr, k=cusum_k, h=cusum_h)
        else:
            raise ValueError(f"Unknown smoothing mode: {mode}")
    return out


def compute_thresholds(
    normal_stats_dict: Dict[str, np.ndarray],
    target_far: float = 0.01,
    holdout_label: str = "fault_free_training_holdout",
) -> Dict[str, float]:
    """Compute per-statistic thresholds from held-out normal stats."""
    thresholds: Dict[str, float] = {}
    for name, arr in normal_stats_dict.items():
        thr = empirical_threshold(arr, target_far)
        thresholds[name] = thr
    return thresholds


def save_thresholds(
    thresholds: Dict[str, float],
    stats_dict: Dict[str, np.ndarray],
    path: Path,
    config: Optional[dict] = None,
) -> None:
    """Save thresholds alongside stats summary."""
    out = {
        "thresholds": thresholds,
        "stats_summary": {
            k: {
                "mean": float(np.mean(v)) if v.size else None,
                "std": float(np.std(v)) if v.size else None,
                "p50": float(np.percentile(v, 50)) if v.size else None,
                "p99": float(np.percentile(v, 99)) if v.size else None,
                "max": float(np.max(v)) if v.size else None,
            }
            for k, v in stats_dict.items()
        },
    }
    if config is not None:
        out["config"] = config
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def load_thresholds(path: Path) -> Dict[str, float]:
    """Load thresholds from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("thresholds", data)
