"""Hankel / lag utilities for CVA and DPCA.
Builds strictly within one simulationRun, never straddling a run boundary.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def build_past_future(
    X: np.ndarray, n_past: int, n_future: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build past/future stacked vectors from a single run.

    Args:
        X: (T, 52) float64, already scaled with NORMAL-ONLY scaler.
        n_past: lags in past vector.
        n_future: lags in future vector.

    Returns:
        P: (N, 52*n_past) past vectors, p_t = [y_{t-1}, ..., y_{t-n_past}]
        F: (N, 52*n_future) future vectors, f_t = [y_t, ..., y_{t+n_future-1}]
        t_index: (N,) absolute sample index t for each row.
        where N = T - n_past - n_future + 1.
        Vectors are ordered with most recent first: y_{t-1} is first block.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D [T, F], got {X.shape}")
    T, F = X.shape
    N = T - n_past - n_future + 1
    if N <= 0:
        return (
            np.empty((0, F * n_past), dtype=np.float64),
            np.empty((0, F * n_future), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )
    P = np.empty((N, F * n_past), dtype=np.float64)
    Future = np.empty((N, F * n_future), dtype=np.float64)
    t_index = np.empty((N,), dtype=np.int64)
    for i in range(N):
        t = n_past + i  # absolute t
        # Past: y_{t-1} .. y_{t-n_past} (most recent first)
        # Slice X[t-n_past : t] gives y_{t-n_past} .. y_{t-1} in order;
        # we need reversed block order so y_{t-1} first.
        past_block = X[t - n_past : t][::-1]  # (n_past, F) with y_{t-1} first
        P[i] = past_block.reshape(-1)
        # Future: y_t .. y_{t+n_future-1}
        Future[i] = X[t : t + n_future].reshape(-1)
        t_index[i] = t
    return P, Future, t_index


def build_past_future_runs(
    runs: Dict[int, np.ndarray], n_past: int, n_future: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build past/future across multiple runs without crossing boundaries.

    Args:
        runs: dict run_id -> (T_run, 52) array.
        n_past, n_future: lags.

    Returns:
        P_all: (N_total, 52*n_past)
        F_all: (N_total, 52*n_future)
        t_index_all: (N_total,) absolute global sample index (run-offset based)
        run_ids: (N_total,) run id for each row (for debugging / per-run stats)
    """
    Ps: List[np.ndarray] = []
    Fs: List[np.ndarray] = []
    Ts: List[np.ndarray] = []
    Rs: List[np.ndarray] = []
    offset = 0
    for run_id in sorted(runs.keys()):
        X = runs[run_id]
        P, F, t_idx = build_past_future(X, n_past, n_future)
        if P.shape[0] == 0:
            offset += X.shape[0]
            continue
        # t_idx is absolute within run (n_past .. T-n_future); make global by offset
        Ps.append(P)
        Fs.append(F)
        Ts.append(t_idx + offset)
        Rs.append(np.full(P.shape[0], run_id, dtype=np.int64))
        offset += X.shape[0]
    if not Ps:
        F_dim_p = 52 * n_past if runs else 0
        F_dim_f = 52 * n_future if runs else 0
        return (
            np.empty((0, F_dim_p), dtype=np.float64),
            np.empty((0, F_dim_f), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    return np.concatenate(Ps, axis=0), np.concatenate(Fs, axis=0), np.concatenate(Ts, axis=0), np.concatenate(Rs, axis=0)


def build_augmented_matrix(
    X: np.ndarray, n_lags: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Build DPCA augmented matrix X_aug (T-n_lags, 52*(n_lags+1)).

    Includes lag 0 (current) through n_lags. Returns X_aug and t_index.
    Per-run, never straddles boundary — caller must handle runs separately
    via build_augmented_runs if needed.
    """
    X = np.asarray(X, dtype=np.float64)
    T, F = X.shape
    N = T - n_lags
    if N <= 0:
        return np.empty((0, F * (n_lags + 1)), dtype=np.float64), np.empty((0,), dtype=np.int64)
    X_aug = np.empty((N, F * (n_lags + 1)), dtype=np.float64)
    t_index = np.empty((N,), dtype=np.int64)
    for i in range(N):
        # X_aug[t] = [y_{t+n_lags}, y_{t+n_lags-1}, ..., y_t] with most recent first
        # Equivalent to lag 0..n_lags
        block = X[i : i + n_lags + 1][::-1]  # most recent first
        X_aug[i] = block.reshape(-1)
        t_index[i] = i + n_lags
    return X_aug, t_index
