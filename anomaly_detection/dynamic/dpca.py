"""DPCA detector — baseline for dynamic faults.
Augmented X_aug (T-n_lags, 52*(n_lags+1)), PCA on normal only.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh

from .lag_builder import build_augmented_matrix
from utils import ensure_dir

logger = logging.getLogger(__name__)


class DPCAModel:
    """Fitted DPCA model."""

    def __init__(
        self,
        mean_: np.ndarray,
        loadings: np.ndarray,
        eigenvalues: np.ndarray,
        n_components: int,
        n_lags: int,
        n_sensors: int = 52,
    ):
        self.mean_ = mean_  # (F*(n_lags+1),)
        self.loadings = loadings  # (F*(n_lags+1), n_components)
        self.eigenvalues = eigenvalues  # (n_components,)
        self.n_components = n_components
        self.n_lags = n_lags
        self.n_sensors = n_sensors
        # Precompute for fast scoring
        self._inv_eig = 1.0 / np.maximum(eigenvalues, 1e-12)

    def score(self, X_aug: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute T2 and SPE for each row of X_aug (already centered?).

        Args:
            X_aug: (N, F*(n_lags+1)) already scaled? We center internally.
        Returns:
            dict with T2, SPE arrays (N,)
        """
        Xc = X_aug - self.mean_[None, :]
        # Scores in PC space
        scores = Xc @ self.loadings  # (N, n_components)
        # T2: Hotelling in PC space (normalized by eigenvalues)
        T2 = np.sum((scores**2) * self._inv_eig[None, :], axis=1)
        # Reconstruction and SPE
        X_recon = scores @ self.loadings.T
        resid = Xc - X_recon
        SPE = np.sum(resid**2, axis=1)
        return {"T2": T2, "SPE": SPE, "scores": scores, "resid": resid}

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            mean_=self.mean_,
            loadings=self.loadings,
            eigenvalues=self.eigenvalues,
            n_components=np.array(self.n_components),
            n_lags=np.array(self.n_lags),
            n_sensors=np.array(self.n_sensors),
        )
        logger.info("Saved DPCA model to %s (n_components=%d, n_lags=%d)", path, self.n_components, self.n_lags)

    @classmethod
    def load(cls, path: Path) -> "DPCAModel":
        data = np.load(path, allow_pickle=True)
        return cls(
            mean_=data["mean_"],
            loadings=data["loadings"],
            eigenvalues=data["eigenvalues"],
            n_components=int(data["n_components"]),
            n_lags=int(data["n_lags"]),
            n_sensors=int(data["n_sensors"]),
        )


def fit_dpca(
    runs: Dict[int, np.ndarray],
    n_lags: int = 3,
    variance_threshold: float = 0.90,
    eps: float = 1e-6,
) -> DPCAModel:
    """Fit DPCA on normal training runs (per-run, never straddling).

    Args:
        runs: dict run_id -> (T_run, 52) scaled float64.
        n_lags: lags (X_aug has n_lags+1 blocks).
        variance_threshold: retain components to explain this variance.
    """
    # Streaming covariance accumulation for X_aug
    dim = 52 * (n_lags + 1)
    # Accumulate sums for mean and covariance without storing full matrix
    n_total = 0
    sum_x = np.zeros(dim, dtype=np.float64)
    sum_xx = np.zeros((dim, dim), dtype=np.float64)
    for run_id, X in runs.items():
        X = np.asarray(X, dtype=np.float64)
        X_aug, _ = build_augmented_matrix(X, n_lags)
        if X_aug.shape[0] == 0:
            continue
        n_total += X_aug.shape[0]
        sum_x += X_aug.sum(axis=0)
        # Use float64, accumulate outer product streaming
        # For memory, do batch accumulation per run
        sum_xx += X_aug.T @ X_aug
    if n_total == 0:
        raise ValueError("No samples for DPCA fit (all runs too short)")
    mean_ = sum_x / n_total
    # Covariance (unbiased)
    S = (sum_xx - n_total * np.outer(mean_, mean_)) / (n_total - 1)
    # Regularise trace-based
    trace = np.trace(S)
    eps_reg = eps * trace / dim if dim else eps
    S += eps_reg * np.eye(dim, dtype=np.float64)
    # Eigendecomposition (symmetric)
    eigvals, eigvecs = eigh(S)
    # eigh returns ascending; sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Clip small eigenvalues
    eigvals = np.maximum(eigvals, 1e-12)
    # Select components by variance
    total_var = eigvals.sum()
    cumsum = np.cumsum(eigvals) / total_var
    n_components = int(np.searchsorted(cumsum, variance_threshold) + 1)
    n_components = max(1, min(n_components, dim))
    loadings = eigvecs[:, :n_components]
    eigenvalues = eigvals[:n_components]
    logger.info(
        "DPCA fit: dim %d, n_total %d, retained %d components (variance %.3f), eig spectrum head %s",
        dim, n_total, n_components, cumsum[n_components - 1], eigvals[:5],
    )
    return DPCAModel(mean_, loadings, eigenvalues, n_components, n_lags)
