"""CVA detector — primary method for dynamic faults 3/9/15.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh

from .lag_builder import build_past_future
from utils import ensure_dir

logger = logging.getLogger(__name__)


class CVAModel:
    """Fitted CVA model with projections J (states) and L (residual)."""

    def __init__(
        self,
        mean_p: np.ndarray,
        mean_f: np.ndarray,
        inv_sqrt_Spp: np.ndarray,
        inv_sqrt_Sff: np.ndarray,
        Vt: np.ndarray,
        singular_values: np.ndarray,
        r: int,
        n_past: int,
        n_future: int,
        n_sensors: int = 52,
    ):
        self.mean_p = mean_p  # (52*n_past,)
        self.mean_f = mean_f  # (52*n_future,)
        self.inv_sqrt_Spp = inv_sqrt_Spp  # (52*n_past, 52*n_past)
        self.inv_sqrt_Sff = inv_sqrt_Sff  # (52*n_future, 52*n_future)
        self.Vt = Vt  # (min_dim, 52*n_past)
        self.s = singular_values  # (min_dim,)
        self.r = r
        self.n_past = n_past
        self.n_future = n_future
        self.n_sensors = n_sensors
        # Precompute projections
        # J: (r, 52*n_past)
        self.J = Vt[:r, :] @ inv_sqrt_Spp  # type: ignore
        # L: (52*n_past, 52*n_past) residual projector
        # L = (I - Vt_r^T Vt_r) @ Spp^{-1/2}
        Vr = Vt[:r, :]  # (r, dim_p)
        proj = Vr.T @ Vr  # (dim_p, dim_p)
        I = np.eye(proj.shape[0], dtype=np.float64)
        self.L = (I - proj) @ inv_sqrt_Spp
        # For Tr (residual Hotelling), we need the discarded subspace's
        # singular values? For now compute Tr as sum of squared standardized
        # residual in discarded canonical variates (unit var assumed).
        # We expose Q (SPE) as primary; Tr is secondary.
        # Store Vt for Tr if needed
        self._Vt_full = Vt

    def score(self, P: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute T2, Q, Tr for each past vector row.

        Args:
            P: (N, 52*n_past) past vectors (already centered externally? we center here).

        Returns:
            dict with T2, Q, Tr arrays (N,)
        """
        # Center
        Pc = P - self.mean_p[None, :]
        # States: z = J @ p  -> (N, r)
        # J is (r, dim_p), Pc is (N, dim_p) -> z = Pc @ J.T
        z = Pc @ self.J.T  # (N, r)
        T2 = np.sum(z**2, axis=1)  # unit var
        # Residual: e = L @ p  -> (N, dim_p)
        # L is (dim_p, dim_p)
        e = Pc @ self.L.T  # (N, dim_p)
        Q = np.sum(e**2, axis=1)
        # Tr: Hotelling in residual canonical space (discarded directions)
        # Project onto discarded Vt: z_disc = Vt_disc @ Spp^{-1/2} @ p
        # Vt_disc is Vt[r:, :] (dim_p - r, dim_p)
        dim_p = self.inv_sqrt_Spp.shape[0]
        if self.r < dim_p:
            Vt_disc = self.Vt[self.r :, :]  # (dim_p - r, dim_p)
            # z_disc = Vt_disc @ Spp^{-1/2} @ p = Vt_disc @ (Spp^{-1/2} p)
            # But Spp^{-1/2} p is Pc @ inv_sqrt_Spp.T? Actually inv_sqrt_Spp is symmetric, so Pc @ inv_sqrt_Spp
            # Vt_disc @ inv_sqrt_Spp @ p  -> p centred
            # Compute p_whitened = Pc @ inv_sqrt_Spp (since inv_sqrt is symmetric)
            p_white = Pc @ self.inv_sqrt_Spp  # (N, dim_p)
            z_disc = p_white @ Vt_disc.T  # (N, dim_p - r)
            # Tr is sum of squares of discarded canonical variates (unit var)
            Tr = np.sum(z_disc**2, axis=1)
        else:
            Tr = np.zeros(P.shape[0], dtype=np.float64)
        return {"T2": T2, "Q": Q, "Tr": Tr, "z": z, "e": e}

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            mean_p=self.mean_p,
            mean_f=self.mean_f,
            inv_sqrt_Spp=self.inv_sqrt_Spp,
            inv_sqrt_Sff=self.inv_sqrt_Sff,
            Vt=self.Vt,
            s=self.s,
            r=np.array(self.r),
            n_past=np.array(self.n_past),
            n_future=np.array(self.n_future),
            n_sensors=np.array(self.n_sensors),
        )
        logger.info("Saved CVA model to %s (r=%d, n_past=%d, n_future=%d)", path, self.r, self.n_past, self.n_future)

    @classmethod
    def load(cls, path: Path) -> "CVAModel":
        data = np.load(path, allow_pickle=True)
        return cls(
            mean_p=data["mean_p"],
            mean_f=data["mean_f"],
            inv_sqrt_Spp=data["inv_sqrt_Spp"],
            inv_sqrt_Sff=data["inv_sqrt_Sff"],
            Vt=data["Vt"],
            singular_values=data["s"],
            r=int(data["r"]),
            n_past=int(data["n_past"]),
            n_future=int(data["n_future"]),
            n_sensors=int(data["n_sensors"]),
        )


def _inverse_sqrt_sym(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute S^{-1/2} via eigh, clipping eigenvalues at eps."""
    # Regularisation already applied
    eigvals, eigvecs = eigh(S)
    # eigh ascending
    eigvals = np.maximum(eigvals, eps)
    inv_sqrt_vals = 1.0 / np.sqrt(eigvals)
    # S^{-1/2} = V @ diag(inv_sqrt) @ V.T
    return (eigvecs * inv_sqrt_vals[None, :]) @ eigvecs.T


def fit_cva(
    runs: Dict[int, np.ndarray],
    n_past: int = 5,
    n_future: int = 5,
    state_order_mode: str = "energy",
    state_order: int = 20,
    energy_threshold: float = 0.90,
    eps: float = 1e-6,
) -> CVAModel:
    """Fit CVA on normal training runs (per-run, streaming).

    Args:
        runs: dict run_id -> (T_run, 52) scaled float64.
        n_past, n_future: lags.
        state_order_mode: "energy" or "fixed".
        state_order: used when mode == fixed.
        energy_threshold: for energy mode.
    """
    dim_p = 52 * n_past
    dim_f = 52 * n_future
    # Streaming accumulation
    n_total = 0
    sum_p = np.zeros(dim_p, dtype=np.float64)
    sum_f = np.zeros(dim_f, dtype=np.float64)
    Spp = np.zeros((dim_p, dim_p), dtype=np.float64)
    Sff = np.zeros((dim_f, dim_f), dtype=np.float64)
    Sfp = np.zeros((dim_f, dim_p), dtype=np.float64)  # F^T P
    # Also need Sfp as F^T P, Spp as P^T P, Sff as F^T F
    for run_id, X in runs.items():
        X = np.asarray(X, dtype=np.float64)
        P, F, _ = build_past_future(X, n_past, n_future)
        if P.shape[0] == 0:
            continue
        n_total += P.shape[0]
        sum_p += P.sum(axis=0)
        sum_f += F.sum(axis=0)
        Spp += P.T @ P
        Sff += F.T @ F
        Sfp += F.T @ P
    if n_total == 0:
        raise ValueError("No samples for CVA fit (all runs too short for n_past/n_future)")
    # Means
    mean_p = sum_p / n_total
    mean_f = sum_f / n_total
    # Centered covariances (unbiased)
    Spp = (Spp - n_total * np.outer(mean_p, mean_p)) / (n_total - 1)
    Sff = (Sff - n_total * np.outer(mean_f, mean_f)) / (n_total - 1)
    Sfp = (Sfp - n_total * np.outer(mean_f, mean_p)) / (n_total - 1)  # Note: F^T P
    # S_pf = Sfp.T, but we need Sfp as defined
    # Regularise: eps * trace/dim
    trace_p = np.trace(Spp)
    trace_f = np.trace(Sff)
    eps_p = eps * trace_p / dim_p if dim_p else eps
    eps_f = eps * trace_f / dim_f if dim_f else eps
    Spp += eps_p * np.eye(dim_p, dtype=np.float64)
    Sff += eps_f * np.eye(dim_f, dtype=np.float64)
    # Inverse square roots
    inv_sqrt_Spp = _inverse_sqrt_sym(Spp, eps=1e-12)
    inv_sqrt_Sff = _inverse_sqrt_sym(Sff, eps=1e-12)
    # Hankel: H = Sff^{-1/2} @ Sfp @ Spp^{-1/2}
    H = inv_sqrt_Sff @ Sfp @ inv_sqrt_Spp
    # SVD: H = U @ diag(s) @ Vt ; we need Vt (V^T)
    # Use full_matrices=False for efficiency
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    # Log spectrum
    total_s = s.sum()
    cumsum = np.cumsum(s) / total_s if total_s > 0 else np.zeros_like(s)
    if state_order_mode == "energy":
        r = int(np.searchsorted(cumsum, energy_threshold) + 1)
        r = max(1, min(r, min(dim_p, dim_f)))
    elif state_order_mode == "fixed":
        r = int(state_order)
        r = max(1, min(r, min(dim_p, dim_f)))
    else:
        raise ValueError(f"Unknown state_order_mode: {state_order_mode}")
    logger.info(
        "CVA fit: dim_p %d dim_f %d n_total %d, r=%d (mode %s thr %.2f), singular head %s, cumsum@r %.3f",
        dim_p, dim_f, n_total, r, state_order_mode, energy_threshold, s[:5], cumsum[r - 1] if r <= len(cumsum) else 0.0,
    )
    logger.info("CVA singular values full (first 20): %s", s[:20])
    return CVAModel(
        mean_p=mean_p,
        mean_f=mean_f,
        inv_sqrt_Spp=inv_sqrt_Spp,
        inv_sqrt_Sff=inv_sqrt_Sff,
        Vt=Vt,
        singular_values=s,
        r=r,
        n_past=n_past,
        n_future=n_future,
    )
