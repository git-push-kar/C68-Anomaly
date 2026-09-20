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


def _load_normal_runs_for_eval(config, split: str = "fault_free_testing", max_runs: Optional[int] = None):
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
            cands = sorted(P(resolve_path(config, "normal_data_path")).glob("TEP_FaultFree_Testing.csv"))
            path = cands[0] if cands else None
            if path is None:
                return {}, scaler
    else:
        path = P(resolve_path(config, "normal_data_path")) / "TEP_FaultFree_Training.csv"
        if not path.exists():
            cands = sorted(P(resolve_path(config, "normal_data_path")).glob("TEP_FaultFree_Training.csv"))
            path = cands[0] if cands else None
            if path is None:
                return {}, scaler
    runs: Dict[int, np.ndarray] = {}
    expected_samples = 960 if "Testing" in str(path) else 500
    for chunk in pd.read_csv(path, chunksize=250000):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        chunk = _normalize_rieth_columns(chunk)
        if "faultNumber" in chunk.columns:
            chunk = chunk[chunk["faultNumber"] == 0]
        sensor_cols = [c for c in CANONICAL_NAMES if c in chunk.columns]
        for run_id, group in chunk.groupby("simulationRun"):
            rid = int(run_id)
            if max_runs is not None and rid > max_runs:
                continue
            group = group.sort_values("sample")
            arr = group[sensor_cols].to_numpy(dtype=np.float64)
            arr_scaled = scaler.transform(arr.astype(np.float32)).astype(np.float64)
            if rid in runs:
                runs[rid] = np.concatenate([runs[rid], arr_scaled], axis=0)
            else:
                runs[rid] = arr_scaled
        if max_runs is not None and len(runs) >= max_runs:
            if all(runs[r].shape[0] >= expected_samples for r in range(1, max_runs + 1) if r in runs):
                break
    return runs, scaler


def _load_faulty_runs_for_eval(config, fault_id: int, split: str = "faulty_testing", max_runs: Optional[int] = None):
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
    expected_samples = 960 if "Testing" in str(path) else 500
    for chunk in pd.read_csv(path, chunksize=250000):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        chunk = _normalize_rieth_columns(chunk)
        chunk = chunk[chunk["faultNumber"] == int(fault_id)]
        if chunk.empty:
            continue
        sensor_cols = [c for c in CANONICAL_NAMES if c in chunk.columns]
        for run_id, group in chunk.groupby("simulationRun"):
            rid = int(run_id)
            if max_runs is not None and rid > max_runs:
                continue
            group = group.sort_values("sample")
            arr = group[sensor_cols].to_numpy(dtype=np.float64)
            arr_scaled = scaler.transform(arr.astype(np.float32)).astype(np.float64)
            if rid in runs:
                runs[rid] = np.concatenate([runs[rid], arr_scaled], axis=0)
            else:
                runs[rid] = arr_scaled
        if max_runs is not None and len(runs) >= max_runs:
            if all(runs[r].shape[0] >= expected_samples for r in range(1, max_runs + 1) if r in runs):
                break
    return runs


def _evaluate_lstm_per_fault(config, detector, threshold, fault_id: int, split: str = "faulty_testing", max_runs: Optional[int] = None):
    """Return FDR and delay for one fault using LSTM AE windows."""
    ws = int(config["windowing"]["window_size"])
    stride = int(config["windowing"]["stride"])
    onset = int(get_fault_onset(config, split))
    runs = _load_faulty_runs_for_eval(config, fault_id, split, max_runs=max_runs)
    all_scores = []
    all_labels = []
    delays = []
    for run_id, X in runs.items():
        windows = to_windows(X.astype(np.float32), ws, stride)
        if windows.shape[0] == 0:
            continue
        starts = np.arange(windows.shape[0]) * stride
        labels = (starts >= onset).astype(int)
        scores, _ = detector.score_windows(windows, already_scaled=True)
        post_mask = labels == 1
        if post_mask.sum() == 0:
            continue
        scores_post = scores[post_mask]
        all_scores.append(scores_post)
        all_labels.append(labels[post_mask])
        confirm = int(config["events"].get("consecutive_windows_to_confirm", 3))
        pred = (scores > threshold).astype(int)
        pred_post = pred[post_mask]
        starts_post = starts[post_mask]
        found = None
        for i in range(len(pred_post) - confirm + 1):
            if np.all(pred_post[i : i + confirm] == 1):
                found = int(starts_post[i] - onset)
                break
        delays.append(found)
    if not all_scores:
        return {"fdr": 0.0, "delay": None, "n_windows": 0}
    scores_all = np.concatenate(all_scores)
    pred_all = (scores_all > threshold).astype(int)
    fdr = float(pred_all.mean()) if pred_all.size else 0.0
    valid_delays = [d for d in delays if d is not None]
    delay = float(np.mean(valid_delays)) if valid_delays else None
    return {"fdr": fdr, "delay": delay, "n_windows": int(scores_all.size), "delays_per_run": delays}


def _evaluate_dynamic_per_fault(config, model, thresholds: Dict[str, float], fault_id: int, split: str, method: str, max_runs: Optional[int] = None):
    """Evaluate DPCA or CVA per fault, per statistic with per-run smoothing."""
    onset = int(get_fault_onset(config, split))
    runs = _load_faulty_runs_for_eval(config, fault_id, split, max_runs=max_runs)
    dyn_cfg = config.get("dynamic_detector", {}).get("thresholding", {})
    smooth = dyn_cfg.get("smoothing", "none")
    ewma_lambda = float(dyn_cfg.get("ewma_lambda", 0.1))
    cusum_k = float(dyn_cfg.get("cusum_k", 0.5))
    cusum_h = float(dyn_cfg.get("cusum_h", 5.0))
    confirm = int(config["events"].get("consecutive_windows_to_confirm", 3))

    results = {}
    for stat_name, thr in thresholds.items():
        all_scores = []
        delays = []
        for run_id, X in runs.items():
            if method == "cva":
                from anomaly_detection.dynamic.lag_builder import build_past_future
                P, F, t_idx = build_past_future(X, model.n_past, model.n_future)
                if P.shape[0] == 0:
                    continue
                scores_dict = model.score(P)
            elif method == "dpca":
                from anomaly_detection.dynamic.lag_builder import build_augmented_matrix
                Xa, t_idx = build_augmented_matrix(X, model.n_lags)
                if Xa.shape[0] == 0:
                    continue
                scores_dict = model.score(Xa)
            else:
                continue

            if smooth != "none":
                from anomaly_detection.dynamic.statistics import smooth_stats
                scores_dict = smooth_stats(scores_dict, mode=smooth, ewma_lambda=ewma_lambda, cusum_k=cusum_k, cusum_h=cusum_h)

            vals = scores_dict[stat_name]
            labels = (t_idx >= onset).astype(int)
            post_vals = vals[labels == 1]
            all_scores.append(post_vals)
            pred = (vals > thr).astype(int)
            post_pred = pred[labels == 1]
            post_t = t_idx[labels == 1]
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
            pred_all = (vals_all > thr).astype(int)
            fdr = float(pred_all.mean()) if pred_all.size else 0.0
            valid = [d for d in delays if d is not None]
            delay = float(np.mean(valid)) if valid else None
            results[stat_name] = {"fdr": fdr, "delay": delay, "n_samples": int(vals_all.size)}
    return results


def evaluate_anomaly_detector(
    config: Dict,
    threshold_override: Optional[float] = None,
    detector_filter: Optional[str] = None,
    faults: Optional[List[int]] = None,
    max_runs: Optional[int] = None,
    split: str = "faulty_testing",
) -> Dict:
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
    smooth = dyn_cfg.get("thresholding", {}).get("smoothing", "none")
    ewma_lambda = float(dyn_cfg.get("thresholding", {}).get("ewma_lambda", 0.1))
    cusum_k = float(dyn_cfg.get("thresholding", {}).get("cusum_k", 0.5))
    cusum_h = float(dyn_cfg.get("thresholding", {}).get("cusum_h", 5.0))

    if enabled:
        dyn_dir = Path(config["anomaly_detector"]["model_dir"]) / "dynamic"
        thr_path = dyn_dir / "dynamic_thresholds.json"
        if thr_path.exists():
            import json
            thr_data = json.load(open(thr_path))
            cva_thr = thr_data.get("thresholds", {}).get("cva", {}) if "thresholds" in thr_data else thr_data.get("cva", {})
            dpca_thr = thr_data.get("thresholds", {}).get("dpca", {}) if "thresholds" in thr_data else thr_data.get("dpca", {})
            if isinstance(cva_thr, dict) and "thresholds" in cva_thr:
                cva_thr = cva_thr["thresholds"]
            if isinstance(dpca_thr, dict) and "thresholds" in dpca_thr:
                dpca_thr = dpca_thr["thresholds"]
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

    # Load normal test runs ONCE for FAR calculation
    normal_split = "fault_free_testing" if "testing" in split else "fault_free_training"
    runs_ff, _ = _load_normal_runs_for_eval(config, normal_split, max_runs=max_runs)

    # Compute realised FAR on fault_free data for each detector
    realised_far = {}
    # LSTM FAR
    if "lstm_ae" in detectors:
        ws = int(config["windowing"]["window_size"])
        stride = int(config["windowing"]["stride"])
        lstm_thr = threshold_override if threshold_override is not None else lstm_detector.threshold.threshold
        from preprocessing.windowing import to_windows
        scores = []
        for rid, X in runs_ff.items():
            windows = to_windows(X.astype(np.float32), ws, stride)
            if windows.shape[0] == 0:
                continue
            s, _ = lstm_detector.score_windows(windows, already_scaled=True)
            scores.append(s)
        if scores:
            all_s = np.concatenate(scores)
            far = float((all_s > lstm_thr).mean())
            realised_far["lstm_ae"] = far
        else:
            realised_far["lstm_ae"] = None

    # Dynamic FAR (with per-run smoothing)
    for name, model, thr_dict in [("cva", cva_model, cva_thr), ("dpca", dpca_model, dpca_thr)]:
        if name not in detectors:
            continue
        for stat, thr in thr_dict.items():
            key = f"{name}_{stat}"
            vals = []
            for rid, X in runs_ff.items():
                if name == "cva":
                    from anomaly_detection.dynamic.lag_builder import build_past_future
                    P, F, t_idx = build_past_future(X, model.n_past, model.n_future)
                    if P.shape[0] == 0:
                        continue
                    sc = model.score(P)
                else:
                    from anomaly_detection.dynamic.lag_builder import build_augmented_matrix
                    Xa, t_idx = build_augmented_matrix(X, model.n_lags)
                    if Xa.shape[0] == 0:
                        continue
                    sc = model.score(Xa)

                if smooth != "none":
                    from anomaly_detection.dynamic.statistics import smooth_stats
                    sc = smooth_stats(sc, mode=smooth, ewma_lambda=ewma_lambda, cusum_k=cusum_k, cusum_h=cusum_h)
                vals.append(sc[stat])
            if vals:
                all_v = np.concatenate(vals)
                realised_far[key] = float((all_v > thr).mean())
            else:
                realised_far[key] = None

    # Fused FAR (OR)
    if "fused" in detectors:
        far_lstm = realised_far.get("lstm_ae")
        cva_primary = dyn_cfg.get("cva", {}).get("primary_statistic", "Q")
        dpca_primary = dyn_cfg.get("dpca", {}).get("primary_statistic", "SPE")
        method = dyn_cfg.get("method", "cva")
        if method == "cva" and cva_model is not None:
            far_dyn = realised_far.get(f"cva_{cva_primary}")
        elif method == "dpca" and dpca_model is not None:
            far_dyn = realised_far.get(f"dpca_{dpca_primary}")
        else:
            far_dyn = realised_far.get(f"cva_{cva_primary}") or realised_far.get(f"dpca_{dpca_primary}")
        if far_lstm is not None and far_dyn is not None:
            fused_far = 1.0 - (1.0 - far_lstm) * (1.0 - far_dyn)
            realised_far["fused"] = float(fused_far)
        else:
            realised_far["fused"] = None

    # Per-fault evaluation
    faults_to_eval = faults if faults is not None else list(range(1, 21))
    per_fault: Dict[str, Dict] = {}
    for fid in faults_to_eval:
        per_fault[str(fid)] = {}
        # LSTM
        if "lstm_ae" in detectors:
            res = _evaluate_lstm_per_fault(config, lstm_detector, lstm_thr, fid, split, max_runs=max_runs)
            per_fault[str(fid)]["lstm_ae"] = res
        # CVA per stat
        if "cva" in detectors:
            cva_res = _evaluate_dynamic_per_fault(config, cva_model, cva_thr, fid, split, "cva", max_runs=max_runs)
            per_fault[str(fid)]["cva"] = cva_res
            primary = dyn_cfg.get("cva", {}).get("primary_statistic", "Q")
            if primary in cva_res:
                per_fault[str(fid)]["cva_primary"] = cva_res[primary]
        if "dpca" in detectors:
            dpca_res = _evaluate_dynamic_per_fault(config, dpca_model, dpca_thr, fid, split, "dpca", max_runs=max_runs)
            per_fault[str(fid)]["dpca"] = dpca_res
            primary = dyn_cfg.get("dpca", {}).get("primary_statistic", "SPE")
            if primary in dpca_res:
                per_fault[str(fid)]["dpca_primary"] = dpca_res[primary]
        # Fused: OR of lstm and dynamic primary
        if "fused" in detectors:
            fdr_lstm = per_fault[str(fid)].get("lstm_ae", {}).get("fdr", 0.0)
            method = dyn_cfg.get("method", "cva")
            if method == "cva" and "cva" in per_fault[str(fid)]:
                cva_dict = per_fault[str(fid)]["cva"]
                fdr_dyn = cva_dict.get(cva_primary, {}).get("fdr", 0.0) if isinstance(cva_dict, dict) and cva_primary in cva_dict else 0.0
            elif method == "dpca" and "dpca" in per_fault[str(fid)]:
                dpca_dict = per_fault[str(fid)]["dpca"]
                fdr_dyn = dpca_dict.get(dpca_primary, {}).get("fdr", 0.0) if isinstance(dpca_dict, dict) and dpca_primary in dpca_dict else 0.0
            else:
                fdr_dyn = 0.0
            fused_fdr = 1.0 - (1.0 - fdr_lstm) * (1.0 - fdr_dyn)
            delay_lstm = per_fault[str(fid)].get("lstm_ae", {}).get("delay")
            delay_dyn = None
            if method == "cva" and "cva" in per_fault[str(fid)] and isinstance(per_fault[str(fid)]["cva"], dict):
                delay_dyn = per_fault[str(fid)]["cva"].get(cva_primary, {}).get("delay")
            delays = [d for d in [delay_lstm, delay_dyn] if d is not None]
            fused_delay = float(min(delays)) if delays else None
            per_fault[str(fid)]["fused"] = {"fdr": float(fused_fdr), "delay": fused_delay}

    highlight = {str(fid): per_fault[str(fid)] for fid in [3, 9, 15] if str(fid) in per_fault}

    overall = {}
    if "lstm_ae" in detectors:
        fdr_vals = [per_fault[str(fid)]["lstm_ae"]["fdr"] for fid in faults_to_eval if "lstm_ae" in per_fault.get(str(fid), {})]
        overall["lstm_ae_mean_fdr"] = float(np.mean(fdr_vals)) if fdr_vals else None

    metrics = {
        "threshold": float(lstm_thr) if "lstm_ae" in detectors else None,
        "dynamic_thresholds": {"cva": cva_thr, "dpca": dpca_thr},
        "realised_far": realised_far,
        "per_fault": per_fault,
        "highlight_3_9_15": highlight,
        "overall": overall,
        "false_positive_rate": realised_far.get("lstm_ae"),
        "auroc": None,
        "auprc": None,
    }
    save_json(processed / "anomaly_detector_eval.json", metrics)
    print("\n" + "=" * 70)
    print(f"EVALUATION SUMMARY ({split.upper()})")
    print("=" * 70)
    print(f"Realised FAR: {realised_far}")
    for fid in faults_to_eval:
        pf = per_fault.get(str(fid), {})
        fused_fdr = pf.get("fused", {}).get("fdr", pf.get("lstm_ae", {}).get("fdr", 0.0))
        cva_q = pf.get("cva", {}).get("Q", {}).get("fdr", None) if isinstance(pf.get("cva"), dict) else None
        lstm_fdr = pf.get("lstm_ae", {}).get("fdr", None)
        print(f"Fault {fid:2d}: Fused FDR={fused_fdr*100:5.1f}% | LSTM FDR={lstm_fdr*100 if lstm_fdr is not None else 0:5.1f}% | CVA Q FDR={cva_q*100 if cva_q is not None else 0:5.1f}%")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate anomaly detector (per-fault).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--detector", choices=["lstm_ae", "cva", "dpca", "fused", "all"], default="all")
    parser.add_argument("--split", choices=["faulty_testing", "faulty_training"], default="faulty_testing", help="Data split to evaluate")
    parser.add_argument("--faults", type=int, nargs="+", default=None, help="Subset of fault numbers to evaluate (e.g. 1 3 4 9 14 15 21)")
    parser.add_argument("--max-runs", type=int, default=None, help="Maximum number of runs per fault to evaluate (default: all)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)-7s %(message)s")
    config = load_config(args.config)
    filt = None if args.detector == "all" else args.detector
    evaluate_anomaly_detector(
        config,
        threshold_override=args.threshold,
        detector_filter=filt,
        faults=args.faults,
        max_runs=args.max_runs,
        split=args.split,
    )


if __name__ == "__main__":
    main()
