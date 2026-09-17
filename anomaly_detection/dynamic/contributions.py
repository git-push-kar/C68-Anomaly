"""Back-projection of lag-stacked residuals to 52 sensors.
"""
from __future__ import annotations

import numpy as np


def sensor_contributions(
    e_t: np.ndarray, n_lags: int, n_sensors: int = 52, mode: str = "sum"
) -> tuple[np.ndarray, np.ndarray]:
    """Fold lag-stacked residual back to per-sensor contributions.

    Args:
        e_t: residual vector in lag-stacked space, length n_sensors*n_lags
              (or n_sensors*(n_lags+1) for DPCA). For CVA, n_lags = n_past.
        n_lags: number of lag blocks in the stacked vector.
        n_sensors: original sensor count (52).
        mode: "sum" (sum of squares over lags) or "mean".

    Returns:
        reduced: (n_sensors,) per-sensor contribution (sum of squares by default).
        lag_profile: (n_lags, n_sensors) unreduced per-lag breakdown.
    """
    e_t = np.asarray(e_t, dtype=np.float64)
    expected = n_sensors * n_lags
    if e_t.size != expected:
        # Handle n_lags+1 case (DPCA includes lag 0) — infer n_lags from size
        # If mismatch, try to reshape with inferred n_lags
        inferred = e_t.size // n_sensors
        if e_t.size % n_sensors == 0:
            n_lags = inferred
        else:
            raise ValueError(f"Residual size {e_t.size} != {expected} (n_lags={n_lags}, n_sensors={n_sensors})")
    # Reshape to (n_lags, n_sensors) — block 0 is most recent lag
    lag_profile = e_t.reshape(n_lags, n_sensors)
    # Per-sensor contribution: sum of squares over lags (default)
    if mode == "sum":
        reduced = np.sum(lag_profile**2, axis=0)
    elif mode == "mean":
        reduced = np.mean(lag_profile**2, axis=0)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return reduced, lag_profile


def batch_sensor_contributions(
    E: np.ndarray, n_lags: int, n_sensors: int = 52, mode: str = "sum"
) -> tuple[np.ndarray, np.ndarray]:
    """Batch version: E (N, n_sensors*n_lags) -> reduced (N, n_sensors), profiles (N, n_lags, n_sensors)."""
    E = np.asarray(E, dtype=np.float64)
    N = E.shape[0]
    reduced = np.empty((N, n_sensors), dtype=np.float64)
    profiles = np.empty((N, n_lags, n_sensors), dtype=np.float64)
    for i in range(N):
        r, p = sensor_contributions(E[i], n_lags, n_sensors, mode)
        reduced[i] = r
        profiles[i] = p
    return reduced, profiles
