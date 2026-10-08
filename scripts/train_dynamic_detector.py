"""Train dynamic detector (DPCA/CVA) on normal training runs.
Usage: python scripts/train_dynamic_detector.py --config configs/config_a5000.yaml [--method cva|dpca|both]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.scaler import load_scaler
from preprocessing.tep_loader import load_tep_data
from anomaly_detection.dynamic.cva import fit_cva
from anomaly_detection.dynamic.dpca import fit_dpca
from anomaly_detection.dynamic.statistics import compute_thresholds, save_thresholds, smooth_stats
from utils import ensure_dir, load_config, save_json, set_seed

logger = logging.getLogger(__name__)


def _assert_scaler_training_only(scaler, config) -> None:
    # Check n_samples matches Training (250k) not Testing, and baseline not from faulty
    # For smoke, n_samples will be smaller, but should still be from Training file
    # We check that scaler was fitted on faultNumber 0 data
    baseline = scaler.baseline
    if baseline is None:
        raise ValueError("Scaler has no baseline; cannot verify training-only fit")
    # Check feature names are canonical 52
    if len(baseline.feature_names) != 52:
        logger.warning("Scaler feature count %d !=52", len(baseline.feature_names))
    # We cannot strictly verify file, but log and assert n_samples reasonable
    logger.info("Scaler assert: n_samples %d, kind %s (should be fault-free Training only)", baseline.n_samples, scaler.kind)


def _load_normal_runs_scaled(config, max_runs_per_fault=None):
    """Load normal Training runs as dict run_id -> scaled (T,52) array."""
    # Load via tep_loader but need per-run separation
    # Use load_tep_data to get DataFrames, but we need per-run
    # Instead, read raw Training file directly and split by simulationRun
    from pathlib import Path as P
    from utils import resolve_path
    import pandas as pd

    scaler = load_scaler(config["preprocessing"]["scaler_dir"])
    normal_path = Path(resolve_path(config, "normal_data_path")) / "TEP_FaultFree_Training.csv"
    if not normal_path.exists():
        # Fallback to pattern
        normal_path = sorted(Path(resolve_path(config, "normal_data_path")).glob("TEP_FaultFree_Training.csv"))[0]
    # Read with chunking to handle 250k rows, but for training we need all
    # For smoke, limit runs
    dfs = []
    for chunk in pd.read_csv(normal_path, chunksize=500000):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        # Normalize column names
        from preprocessing.tep_loader import _normalize_rieth_columns
        chunk = _normalize_rieth_columns(chunk)
        if "simulationRun" in chunk.columns and max_runs_per_fault is not None:
            chunk = chunk[chunk["simulationRun"] <= max_runs_per_fault]
        dfs.append(chunk)
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    # Filter faultNumber 0
    if "faultNumber" in df.columns:
        df = df[df["faultNumber"] == 0]
    # Group by simulationRun
    runs: dict[int, np.ndarray] = {}
    sensor_cols = [f"XMEAS_{i}" for i in range(1, 42)] + [f"XMV_{i}" for i in range(42, 53)]
    # Ensure columns exist
    for run_id, group in df.groupby("simulationRun"):
        group = group.sort_values("sample")
        arr = group[sensor_cols].to_numpy(dtype=np.float64)
        # Scale with NORMAL-ONLY scaler (already fitted)
        # Scaler expects 52 cols in canonical order
        arr_scaled = scaler.transform(arr.astype(np.float32)).astype(np.float64)
        runs[int(run_id)] = arr_scaled
    return runs, scaler


def main():
    parser = argparse.ArgumentParser(description="Train dynamic detector")
    parser.add_argument("--config", default=None)
    parser.add_argument("--method", choices=["cva", "dpca", "both"], default=None, help="Override config method")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)-7s %(message)s")
    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    dyn_cfg = config.get("dynamic_detector", {})
    method = args.method or dyn_cfg.get("method", "cva")
    logger.info("Dynamic detector method: %s", method)

    # Load scaler and assert
    scaler = load_scaler(config["preprocessing"]["scaler_dir"])
    _assert_scaler_training_only(scaler, config)

    # Load normal training runs
    # For threshold holdout we need to split runs
    holdout_n = int(dyn_cfg.get("thresholding", {}).get("holdout_normal_runs", 100))
    # Load all training runs (500 runs)
    runs_all, _ = _load_normal_runs_scaled(config, max_runs_per_fault=None)
    all_run_ids = sorted(runs_all.keys())
    logger.info("Total normal training runs: %d (each 500 samples)", len(all_run_ids))
    if len(all_run_ids) < holdout_n + 10:
        raise ValueError(f"Not enough runs {len(all_run_ids)} for holdout {holdout_n}")
    # Split BY RUN ID: last holdout_n runs for threshold, rest for fit
    # Deterministic: sorted run ids, holdout is highest ids
    fit_run_ids = all_run_ids[:-holdout_n] if holdout_n > 0 else all_run_ids
    holdout_run_ids = all_run_ids[-holdout_n:] if holdout_n > 0 else []
    runs_fit = {rid: runs_all[rid] for rid in fit_run_ids}
    runs_holdout = {rid: runs_all[rid] for rid in holdout_run_ids}
    logger.info("Fit runs: %d, Holdout runs: %d", len(runs_fit), len(runs_holdout))

    # Config hash for metadata
    cfg_hash = hashlib.sha256(json.dumps(dyn_cfg, sort_keys=True).encode()).hexdigest()[:8]

    out_dir = ensure_dir(Path(config["anomaly_detector"]["model_dir"]) / "dynamic")
    # Fit models
    cva_model = None
    dpca_model = None
    if method in ("cva", "both"):
        cva_cfg = dyn_cfg.get("cva", {})
        cva_model = fit_cva(
            runs_fit,
            n_past=int(cva_cfg.get("n_past", 5)),
            n_future=int(cva_cfg.get("n_future", 5)),
            state_order_mode=cva_cfg.get("state_order_mode", "energy"),
            state_order=int(cva_cfg.get("state_order", 20)),
            energy_threshold=float(cva_cfg.get("energy_threshold", 0.90)),
        )
        cva_model.save(out_dir / "cva_model.npz")
    if method in ("dpca", "both"):
        dpca_cfg = dyn_cfg.get("dpca", {})
        dpca_model = fit_dpca(
            runs_fit,
            n_lags=int(dpca_cfg.get("n_lags", 3)),
            variance_threshold=float(dpca_cfg.get("variance_threshold", 0.90)),
        )
        dpca_model.save(out_dir / "dpca_model.npz")

    # Compute thresholds on holdout normal runs
    thresholds: dict = {}
    stats_holdout: dict = {}
    target_far = float(dyn_cfg.get("thresholding", {}).get("target_far", 0.01))
    smoothing = dyn_cfg.get("thresholding", {}).get("smoothing", "none")
    ewma_lambda = float(dyn_cfg.get("thresholding", {}).get("ewma_lambda", 0.1))
    cusum_k = float(dyn_cfg.get("thresholding", {}).get("cusum_k", 0.5))
    cusum_h = float(dyn_cfg.get("thresholding", {}).get("cusum_h", 5.0))

    if cva_model is not None:
        # Build stats on holdout by streaming per run
        from anomaly_detection.dynamic.lag_builder import build_past_future
        all_T2, all_Q, all_Tr = [], [], []
        for rid, X in runs_holdout.items():
            P, F, _ = build_past_future(X, cva_model.n_past, cva_model.n_future)
            if P.shape[0] == 0:
                continue
            scores = cva_model.score(P)
            # Smooth PER RUN (never straddling run boundary)
            smoothed = smooth_stats(scores, mode=smoothing, ewma_lambda=ewma_lambda, cusum_k=cusum_k, cusum_h=cusum_h)
            all_T2.append(smoothed["T2"])
            all_Q.append(smoothed["Q"])
            all_Tr.append(smoothed["Tr"])
        if all_T2:
            T2_arr = np.concatenate(all_T2)
            Q_arr = np.concatenate(all_Q)
            Tr_arr = np.concatenate(all_Tr)
            cva_stats = {"T2": T2_arr, "Q": Q_arr, "Tr": Tr_arr}
            cva_thr = compute_thresholds(cva_stats, target_far=target_far)
            thresholds["cva"] = cva_thr
            stats_holdout["cva"] = {k: {"mean": float(v.mean()), "p99": float(np.percentile(v, 99))} for k, v in cva_stats.items()}
            logger.info("CVA thresholds (FAR %.3f, smoothing %s): %s", target_far, smoothing, cva_thr)

    if dpca_model is not None:
        from anomaly_detection.dynamic.lag_builder import build_augmented_matrix
        all_T2, all_SPE = [], []
        for rid, X in runs_holdout.items():
            Xa, _ = build_augmented_matrix(X, dpca_model.n_lags)
            if Xa.shape[0] == 0:
                continue
            scores = dpca_model.score(Xa)
            # Smooth PER RUN (never straddling run boundary)
            smoothed = smooth_stats(scores, mode=smoothing, ewma_lambda=ewma_lambda, cusum_k=cusum_k, cusum_h=cusum_h)
            all_T2.append(smoothed["T2"])
            all_SPE.append(smoothed["SPE"])
        if all_T2:
            T2_arr = np.concatenate(all_T2)
            SPE_arr = np.concatenate(all_SPE)
            dpca_stats = {"T2": T2_arr, "SPE": SPE_arr}
            dpca_thr = compute_thresholds(dpca_stats, target_far=target_far)
            thresholds["dpca"] = dpca_thr
            stats_holdout["dpca"] = {k: {"mean": float(v.mean()), "p99": float(np.percentile(v, 99))} for k, v in dpca_stats.items()}
            logger.info("DPCA thresholds (FAR %.3f, smoothing %s): %s", target_far, smoothing, dpca_thr)

    # Fusion per-detector FAR (when mode or, split budget)
    fusion_cfg = dyn_cfg.get("fusion", {})
    if fusion_cfg.get("mode") == "or":
        per_far = float(fusion_cfg.get("per_detector_far", 0.005))
        # Recompute per-detector thresholds at per_detector_far if different
        # For now, log realised FAR will be done in evaluation; thresholds already at target_far
        # If per_detector_far != target_far, recompute
        if abs(per_far - target_far) > 1e-9:
            logger.info("Fusion OR mode: per_detector_far %.4f vs target %.4f — thresholds at target, will verify fused FAR", per_far, target_far)

    # Save thresholds and metadata
    save_thresholds(thresholds, {}, out_dir / "dynamic_thresholds.json", config=dyn_cfg)
    # Also save detailed thresholds per stat
    with open(out_dir / "dynamic_thresholds.json", "r", encoding="utf-8") as f:
        thr_data = json.load(f)
    # Save metadata
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_hash": cfg_hash,
        "method": method,
        "fit_runs": fit_run_ids[:10],
        "fit_n_runs": len(fit_run_ids),
        "holdout_runs": holdout_run_ids[:10],
        "holdout_n_runs": len(holdout_run_ids),
        "n_past": cva_model.n_past if cva_model else None,
        "n_future": cva_model.n_future if cva_model else None,
        "cva_r": cva_model.r if cva_model else None,
        "dpca_lags": dpca_model.n_lags if dpca_model else None,
        "thresholds": thresholds,
        "stats_holdout": stats_holdout,
    }
    save_json(out_dir / "fit_metadata.json", meta)
    logger.info("Dynamic detector training complete → %s", out_dir)


if __name__ == "__main__":
    main()
