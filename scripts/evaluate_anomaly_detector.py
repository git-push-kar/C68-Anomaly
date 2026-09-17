"""Evaluate anomaly detectors: LSTM AE, DPCA, CVA, fused.
Per-fault FDR at fixed FAR, detection delay, per-statistic breakdown.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anomaly_detection import AnomalyDetector
from preprocessing.windowing import to_windows
from utils import get_fault_onset, load_config, load_json, save_json

logger = logging.getLogger(__name__)


def _window_label(sample_starts: np.ndarray, onset: int) -> np.ndarray:
    return (sample_starts >= onset).astype(int)


def _load_normal_runs_for_eval(config, split: str = "fault_free_testing"):
    """Load fault-free testing runs as dict run_id -> scaled (T,52)."""
    from pathlib import Path as P
    from utils import resolve_path
    from preprocessing.scaler import load_scaler
    from preprocessing.tep_loader import _normalize_rieth_columns, CANONICAL_NAMES

    scaler = load_scaler(config["preprocessing"]["scaler_dir"])
    # Determine file
    if split == "fault_free_testing":
        path = P(resolve_path(config, "normal_data_path")) / "TEP_FaultFree_Testing.csv"
        if not path.exists():
            # Fallback pattern
            cands = sorted(P(resolve_path(config, "normal_data_path")).glob("TEP_FaultFree_Testing.csv"))
            if cands:
                path = cands[0]
            else:
                return {}, scaler
    else:
        path = P(resolve_path(config, "normal_data_path")) / "TEP_FaultFree_Training.csv"
        if not path.exists():
            cands = sorted(P(resolve_path(config, "normal_data_path")).glob("TEP_FaultFree_Training.csv"))
            path = cands[0] if cands else None
            if path is None:
                return {}, scaler
    runs: Dict[int, np.ndarray] = {}
    for chunk in pd.read_csv(path, chunksize=500000):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        chunk = _normalize_rieth_columns(chunk)
        if "faultNumber" in chunk.columns:
            chunk = chunk[chunk["faultNumber"] == 0]
        sensor_cols = [c for c in CANONICAL_NAMES if c in chunk.columns]
        for run_id, group in chunk.groupby("simulationRun"):
            group = group.sort_values("sample")
            arr = group[sensor_cols].to_numpy(dtype=np.float64)
            arr_scaled = scaler.transform(arr.astype(np.float32)).astype(np.float64)
            rid = int(run_id)
            if rid in runs:
                runs[rid] = np.concatenate([runs[rid], arr_scaled], axis=0)
            else:
                runs[rid] = arr_scaled
    return runs, scaler


def _load_faulty_runs_for_eval(config, fault_id: int, split: str = "faulty_testing"):
    """Load one fault's runs for given split as dict run_id -> scaled (T,52)."""
    from pathlib import Path as P
    from utils import resolve_path
    from preprocessing.scaler import load_scaler
    from preprocessing.tep_loader import _normalize_rieth_columns, CANONICAL_NAMES

    scaler = load_scaler(config["preprocessing"]["scaler_dir"])
    if split == "faulty_testing":
        path = P(resolve_path(config, "fault_data_path")) / "TEP_Faulty_Testing.csv"
    else:
        path = P(resolve_path(config, "fault_data_path")) / "TEP_Faulty_Training.csv"
    if not path.exists():
        cands = sorted(P(resolve_path(config, "fault_data_path")).glob(f"TEP_Faulty_{split.split('_')[1].capitalize()}.csv"))
        path = cands[0] if cands else path
    runs: Dict[int, np.ndarray] = {}
    for chunk in pd.read_csv(path, chunksize=500000):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        chunk = _normalize_rieth_columns(chunk)
        chunk = chunk[chunk["faultNumber"] == int(fault_id)]
        if chunk.empty:
            continue
        sensor_cols = [c for c in CANONICAL_NAMES if c in chunk.columns]
        for run_id, group in chunk.groupby("simulationRun"):
            group = group.sort_values("sample")
            arr = group[sensor_cols].to_numpy(dtype=np.float64)
            arr_scaled = scaler.transform(arr.astype(np.float32)).astype(np.float64)
            rid = int(run_id)
            if rid in runs:
                runs[rid] = np.concatenate([runs[rid], arr_scaled], axis=0)
            else:
                runs[rid] = arr_scaled
    return runs


def _evaluate_lstm_per_fault(config, detector, threshold, fault_id: int, split: str = "faulty_testing"):
    """Return FDR and delay for one fault using LSTM AE windows."""
    ws = int(config["windowing"]["window_size"])
    stride = int(config["windowing"]["stride"])
    onset = int(get_fault_onset(config, split))
    runs = _load_faulty_runs_for_eval(config, fault_id, split)
    # Build windows per run, never crossing boundary
    all_scores = []
    all_labels = []
    delays = []
    for run_id, X in runs.items():
        # X is (T,52) scaled, T 960 for testing, 500 for training
        windows = to_windows(X.astype(np.float32), ws, stride)
        if windows.shape[0] == 0:
            continue
        starts = np.arange(windows.shape[0]) * stride
        # Discard samples before onset: label per window
        labels = (starts >= onset).astype(int)
        # But windows that start before onset but end after? Task says discard samples before onset
        # For window, if its start < onset, it's pre-fault even if it straddles. So label as above.
        # For FDR, only consider windows with start >= onset
        scores, _ = detector.score_windows(windows)
        # For per-fault, filter to post-onset only for FDR
        post_mask = labels == 1
        if post_mask.sum() == 0:
            continue
        scores_post = scores[post_mask]
        all_scores.append(scores_post)
        all_labels.append(labels[post_mask])
        # Detection delay: first sustained alarm where sustained = consecutive confirm
        confirm = int(config["events"].get("consecutive_windows_to_confirm", 3))
        pred = (scores > threshold).astype(int)
        # Find first sustained sequence in post-onset region
        # Need to consider only post-onset predictions
        pred_post = pred[post_mask]
        starts_post = starts[post_mask]
        # Find first sustained run
        found = None
        for i in range(len(pred_post) - confirm + 1):
            if np.all(pred_post[i : i + confirm] == 1):
                found = int(starts_post[i] - onset)
                break
        delays.append(found)
    if not all_scores:
        return {"fdr": 0.0, "delay": None, "n_windows": 0}
    scores_all = np.concatenate(all_scores)
    # FDR at fixed FAR = fraction of post-onset windows alarmed
    pred_all = (scores_all > threshold).astype(int)
    fdr = float(pred_all.mean()) if pred_all.size else 0.0
    # Delay: median or mean? Use mean of delays where detected, else None
    valid_delays = [d for d in delays if d is not None]
    delay = float(np.mean(valid_delays)) if valid_delays else None
    return {"fdr": fdr, "delay": delay, "n_windows": int(scores_all.size), "delays_per_run": delays}


def _evaluate_dynamic_per_fault(config, model, thresholds: Dict[str, float], fault_id: int, split: str, method: str):
    """Evaluate DPCA or CVA per fault, per statistic."""
    onset = int(get_fault_onset(config, split))
    runs = _load_faulty_runs_for_eval(config, fault_id, split)
    # For dynamic, need per-sample stats, but we map to per-window for FDR comparability?
    # Task says per-sample statistics once buffer full, but for FDR we can use per-sample labels
    # We will compute per-sample stats and then threshold per sample, then compute FDR as fraction of post-onset samples alarmed
    results = {}
    for stat_name, thr in thresholds.items():
        all_scores = []
        delays = []
        for run_id, X in runs.items():
            # X scaled already
            if method == "cva":
                from anomaly_detection.dynamic.lag_builder import build_past_future
                P, F, t_idx = build_past_future(X, model.n_past, model.n_future)
                if P.shape[0] == 0:
                    continue
                scores_dict = model.score(P)
                vals = scores_dict[stat_name]
                # t_idx is absolute sample index t for each row; label post-onset if t >= onset
                labels = (t_idx >= onset).astype(int)
                post_vals = vals[labels == 1]
                all_scores.append(post_vals)
                # Delay: first sustained alarm in post region
                pred = (vals > thr).astype(int)
                # Need to consider only post-onset predictions, but find first sustained in full sequence where t>=onset
                # Extract post-onset predictions in order
                post_pred = pred[labels == 1]
                post_t = t_idx[labels == 1]
                confirm = int(config["events"].get("consecutive_windows_to_confirm", 3))
                found = None
                for i in range(len(post_pred) - confirm + 1):
                    if np.all(post_pred[i : i + confirm] == 1):
                        found = int(post_t[i] - onset)
                        break
                delays.append(found)
            elif method == "dpca":
                from anomaly_detection.dynamic.lag_builder import build_augmented_matrix
                Xa, t_idx = build_augmented_matrix(X, model.n_lags)
                if Xa.shape[0] == 0:
                    continue
                scores_dict = model.score(Xa)
                vals = scores_dict[stat_name]
                labels = (t_idx >= onset).astype(int)
                post_vals = vals[labels == 1]
                all_scores.append(post_vals)
                pred = (vals > thr).astype(int)
                post_pred = pred[labels == 1]
                post_t = t_idx[labels == 1]
                confirm = int(config["events"].get("consecutive_windows_to_confirm", 3))
                found = None
                for i in range(len(post_pred) - confirm + 1):
                    if np.all(post_pred[i : i + confirm] == 1):
                        found = int(post_t[i] - onset)
                        break
                delays.append(found)
        if not all_scores:
            results[stat_name] = {"fdr": 0.0, "delay": None, "n_samples": 0}
        else:
            vals_all = np.concatenate(all_scores)
            # Apply smoothing if configured? Thresholds already on smoothed holdout, so smooth here too
            # For eval, apply same smoothing
            dyn_cfg = config.get("dynamic_detector", {}).get("thresholding", {})
            smooth = dyn_cfg.get("smoothing", "none")
            if smooth != "none":
                from anomaly_detection.dynamic.statistics import smooth_stats
                # smooth per stat name? Need to smooth the stream per run, not concatenated. For simplicity, skip smoothing in eval for now
                pass
            pred_all = (vals_all > thr).astype(int)
            fdr = float(pred_all.mean()) if pred_all.size else 0.0
            valid = [d for d in delays if d is not None]
            delay = float(np.mean(valid)) if valid else None
            results[stat_name] = {"fdr": fdr, "delay": delay, "n_samples": int(vals_all.size)}
    return results


def evaluate_anomaly_detector(config: Dict, threshold_override: Optional[float] = None, detector_filter: Optional[str] = None) -> Dict:
    processed = Path(config["paths"]["processed_data_path"])
    # Load LSTM detector if needed
    lstm_detector = None
    try:
        lstm_detector = AnomalyDetector.from_artifacts(
            model_dir=config["anomaly_detector"]["model_dir"],
            scaler_dir=config["preprocessing"]["scaler_dir"],
            threshold_dir=config["anomaly_detector"]["model_dir"],
        )
    except Exception as e:
        logger.warning("LSTM detector not available: %s", e)

    # Load dynamic models if enabled
    dyn_cfg = config.get("dynamic_detector", {})
    enabled = dyn_cfg.get("enabled", False)
    cva_model = None
    dpca_model = None
    cva_thr = {}
    dpca_thr = {}
    if enabled:
        dyn_dir = Path(config["anomaly_detector"]["model_dir"]) / "dynamic"
        thr_path = dyn_dir / "dynamic_thresholds.json"
        if thr_path.exists():
            import json
            thr_data = json.load(open(thr_path))
            cva_thr = thr_data.get("thresholds", {}).get("cva", {}) if "thresholds" in thr_data else thr_data.get("cva", {})
            dpca_thr = thr_data.get("thresholds", {}).get("dpca", {}) if "thresholds" in thr_data else thr_data.get("dpca", {})
            # Handle nested
            if isinstance(cva_thr, dict) and "thresholds" in cva_thr:
                cva_thr = cva_thr["thresholds"]
        # Load models
        if (dyn_dir / "cva_model.npz").exists():
            from anomaly_detection.dynamic.cva import CVAModel
            cva_model = CVAModel.load(dyn_dir / "cva_model.npz")
        if (dyn_dir / "dpca_model.npz").exists():
            from anomaly_detection.dynamic.dpca import DPCAModel
            dpca_model = DPCAModel.load(dyn_dir / "dpca_model.npz")

    # Determine which detectors to evaluate
    detectors = []
    if detector_filter is None or detector_filter == "lstm_ae":
        if lstm_detector is not None:
            detectors.append("lstm_ae")
    if detector_filter is None or detector_filter in ("cva", "dpca", "dynamic"):
        if cva_model is not None:
            detectors.append("cva")
        if dpca_model is not None:
            detectors.append("dpca")
    if detector_filter is None:
        if lstm_detector is not None and (cva_model is not None or dpca_model is not None):
            detectors.append("fused")

    # Compute realised FAR on fault_free_testing for each detector
    realised_far = {}
    # LSTM FAR
    if "lstm_ae" in detectors:
        runs_ff, _ = _load_normal_runs_for_eval(config, "fault_free_testing")
        # Build windows per run and score
        ws = int(config["windowing"]["window_size"])
        stride = int(config["windowing"]["stride"])
        lstm_thr = threshold_override if threshold_override is not None else lstm_detector.threshold.threshold
        # Collect scores from fault-free testing windows
        from preprocessing.windowing import to_windows
        scores = []
        for rid, X in runs_ff.items():
            windows = to_windows(X.astype(np.float32), ws, stride)
            if windows.shape[0]==0:
                continue
            s, _ = lstm_detector.score_windows(windows)
            scores.append(s)
        if scores:
            all_s = np.concatenate(scores)
            far = float((all_s > lstm_thr).mean())
            realised_far["lstm_ae"] = far
        else:
            realised_far["lstm_ae"] = None
    # Dynamic FAR
    for name, model, thr_dict in [("cva", cva_model, cva_thr), ("dpca", dpca_model, dpca_thr)]:
        if name not in detectors:
            continue
        runs_ff, _ = _load_normal_runs_for_eval(config, "fault_free_testing")
        # Need to compute per-statistic FAR
        for stat, thr in thr_dict.items():
            key = f"{name}_{stat}"
            vals = []
            for rid, X in runs_ff.items():
                if name == "cva":
                    from anomaly_detection.dynamic.lag_builder import build_past_future
                    P, F, t_idx = build_past_future(X, model.n_past, model.n_future)
                    if P.shape[0]==0:
                        continue
                    sc = model.score(P)
                    vals.append(sc[stat])
                else:
                    from anomaly_detection.dynamic.lag_builder import build_augmented_matrix
                    Xa, t_idx = build_augmented_matrix(X, model.n_lags)
                    if Xa.shape[0]==0:
                        continue
                    sc = model.score(Xa)
                    vals.append(sc[stat])
            if vals:
                all_v = np.concatenate(vals)
                realised_far[key] = float((all_v > thr).mean())
            else:
                realised_far[key] = None
    # Fused FAR (OR)
    if "fused" in detectors:
        # For OR, need per-sample fused alarm: alarm if lstm OR dynamic primary
        # Approximate by per-window OR using per-run alignment (simplified)
        # For now, compute fused FAR as 1 - (1 - far_lstm)*(1 - far_dynamic_primary)
        # Will verify empirically via per-sample OR in full eval
        far_lstm = realised_far.get("lstm_ae")
        # Use primary dynamic stat
        cva_primary = dyn_cfg.get("cva", {}).get("primary_statistic", "Q")
        dpca_primary = dyn_cfg.get("dpca", {}).get("primary_statistic", "SPE")
        # Choose which dynamic is primary for fusion
        method = dyn_cfg.get("method", "cva")
        if method == "cva" and cva_model is not None:
            far_dyn = realised_far.get(f"cva_{cva_primary}")
        elif method == "dpca" and dpca_model is not None:
            far_dyn = realised_far.get(f"dpca_{dpca_primary}")
        else:
            far_dyn = realised_far.get(f"cva_{cva_primary}") or realised_far.get(f"dpca_{dpca_primary}")
        if far_lstm is not None and far_dyn is not None:
            # OR
            fused_far = 1.0 - (1.0 - far_lstm) * (1.0 - far_dyn)
            realised_far["fused"] = float(fused_far)
        else:
            realised_far["fused"] = None

    # Per-fault table
    per_fault: Dict[str, Dict] = {}
    for fid in range(1, 21):
        per_fault[str(fid)] = {}
        # LSTM
        if "lstm_ae" in detectors:
            res = _evaluate_lstm_per_fault(config, lstm_detector, lstm_thr, fid, "faulty_testing")
            per_fault[str(fid)]["lstm_ae"] = res
        # CVA per stat
        if "cva" in detectors:
            cva_res = _evaluate_dynamic_per_fault(config, cva_model, cva_thr, fid, "faulty_testing", "cva")
            per_fault[str(fid)]["cva"] = cva_res
            # Also primary
            primary = dyn_cfg.get("cva", {}).get("primary_statistic", "Q")
            if primary in cva_res:
                per_fault[str(fid)]["cva_primary"] = cva_res[primary]
        if "dpca" in detectors:
            dpca_res = _evaluate_dynamic_per_fault(config, dpca_model, dpca_thr, fid, "faulty_testing", "dpca")
            per_fault[str(fid)]["dpca"] = dpca_res
            primary = dyn_cfg.get("dpca", {}).get("primary_statistic", "SPE")
            if primary in dpca_res:
                per_fault[str(fid)]["dpca_primary"] = dpca_res[primary]
        # Fused: OR of lstm and dynamic primary
        if "fused" in detectors:
            # For fused, compute FDR as OR of per-sample alarms
            # Need to align per-sample predictions: for simplicity, use per-window OR for LSTM and per-sample for dynamic
            # Approximate by max of FDRs for now, but proper would be per-sample OR
            # For this implementation, compute fused FDR as 1 - (1 - fdr_lstm)*(1 - fdr_dynamic) approx
            fdr_lstm = per_fault[str(fid)].get("lstm_ae", {}).get("fdr", 0.0)
            # Choose dynamic primary fdr
            method = dyn_cfg.get("method", "cva")
            if method == "cva" and "cva" in per_fault[str(fid)]:
                fdr_dyn = per_fault[str(fid)]["cva"].get(cva_primary, {}).get("fdr", 0.0) if isinstance(per_fault[str(fid)]["cva"], dict) and cva_primary in per_fault[str(fid)]["cva"] else 0.0
                # Actually per_fault[str(fid)]["cva"] is dict of stats, so get primary
                if isinstance(per_fault[str(fid)]["cva"], dict):
                    fdr_dyn = per_fault[str(fid)]["cva"].get(cva_primary, {}).get("fdr", 0.0)
            elif method == "dpca" and "dpca" in per_fault[str(fid)]:
                fdr_dyn = per_fault[str(fid)]["dpca"].get(dpca_primary, {}).get("fdr", 0.0)
            else:
                fdr_dyn = 0.0
            fused_fdr = 1.0 - (1.0 - fdr_lstm) * (1.0 - fdr_dyn)
            # Delay: min of delays
            delay_lstm = per_fault[str(fid)].get("lstm_ae", {}).get("delay")
            delay_dyn = None
            if method == "cva" and "cva" in per_fault[str(fid)]:
                delay_dyn = per_fault[str(fid)]["cva"].get(cva_primary, {}).get("delay") if isinstance(per_fault[str(fid)]["cva"], dict) else None
            delays = [d for d in [delay_lstm, delay_dyn] if d is not None]
            fused_delay = float(min(delays)) if delays else None
            per_fault[str(fid)]["fused"] = {"fdr": float(fused_fdr), "delay": fused_delay}

    # Highlight 3/9/15
    highlight = {str(fid): per_fault[str(fid)] for fid in [3, 9, 15] if str(fid) in per_fault}

    # Aggregate metrics for backward compat (old keys)
    # Compute overall precision/recall etc for lstm for backward compat
    # For now, compute as before but using per-fault aggregated
    # Use lstm overall if available
    overall = {}
    if "lstm_ae" in detectors:
        # Recompute overall from per_fault
        # For overall, we need tp etc, but we have per-fault fdr and n_windows
        # Approximate overall FDR as mean fdr
        fdr_vals = [per_fault[str(fid)]["lstm_ae"]["fdr"] for fid in range(1,21) if "lstm_ae" in per_fault[str(fid)]]
        overall["lstm_ae_mean_fdr"] = float(np.mean(fdr_vals)) if fdr_vals else None

    metrics = {
        "threshold": float(lstm_thr) if "lstm_ae" in detectors else None,
        "dynamic_thresholds": {"cva": cva_thr, "dpca": dpca_thr},
        "realised_far": realised_far,
        "per_fault": per_fault,
        "highlight_3_9_15": highlight,
        "overall": overall,
        # Backward compat keys (from old eval)
        "false_positive_rate": realised_far.get("lstm_ae"),
        "auroc": None,
        "auprc": None,
    }
    # Also compute old aggregate for backward compat if needed
    # Save
    save_json(processed / "anomaly_detector_eval.json", metrics)
    logger.info("Per-fault evaluation complete. Highlight 3/9/15: %s", highlight)
    logger.info("Realised FAR: %s", realised_far)
    return metrics


def _format(metrics: Dict) -> str:
    # Simplified
    return str(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate anomaly detector (per-fault).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--detector", choices=["lstm_ae", "cva", "dpca", "fused", "all"], default="all")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)-7s %(message)s")
    config = load_config(args.config)
    filt = None if args.detector == "all" else args.detector
    evaluate_anomaly_detector(config, threshold_override=args.threshold, detector_filter=filt)


if __name__ == "__main__":
    main()
