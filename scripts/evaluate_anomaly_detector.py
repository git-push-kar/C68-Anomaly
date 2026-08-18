"""Evaluate the anomaly detector: precision/recall/F1, FPR/FNR, AUROC/AUPRC,
and detection delay, using held-out fault scenarios with known onset times.

Usage:
    python scripts/evaluate_anomaly_detector.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anomaly_detection import AnomalyDetector  # noqa: E402
from preprocessing.windowing import to_windows  # noqa: E402
from utils import load_config, load_json, save_json  # noqa: E402

logger = logging.getLogger(__name__)


def _window_label(sample_starts: np.ndarray, onset: int) -> np.ndarray:
    return (sample_starts >= onset).astype(int)


def evaluate_anomaly_detector(config: Dict, threshold_override: Optional[float] = None) -> Dict:
    processed = Path(config["paths"]["processed_data_path"])
    detector = AnomalyDetector.from_artifacts(
        model_dir=config["anomaly_detector"]["model_dir"],
        scaler_dir=config["preprocessing"]["scaler_dir"],
        threshold_dir=config["anomaly_detector"]["model_dir"],
    )
    threshold = threshold_override if threshold_override is not None else detector.threshold.threshold
    ws = int(config["windowing"]["window_size"])
    stride = int(config["windowing"]["stride"])

    normal_test = np.load(processed / "normal_windows_test.npy") if (processed / "normal_windows_test.npy").exists() else None

    fault_meta = load_json(processed / "fault_metadata.json")
    normal_scores: List[np.ndarray] = []
    fault_scores: List[np.ndarray] = []
    delays: List[Optional[int]] = []
    tp = fp = tn = fn = 0

    if normal_test is not None:
        normal_scores_arr, _ = detector.score_windows(normal_test)
        normal_scores.append(normal_scores_arr)
        tn += int((normal_scores_arr <= threshold).sum())
        fp += int((normal_scores_arr > threshold).sum())

    for name, meta in fault_meta.items():
        path = processed / "fault_values" / f"{name}.npy"
        if not path.exists():
            continue
        values = np.load(path)
        windows = to_windows(values, ws, stride)
        starts = np.arange(windows.shape[0]) * stride
        onset = int(meta.get("onset_index") or config["dataset"].get("fault_onset_index", 161))
        labels = _window_label(starts, onset)
        scores, _ = detector.score_windows(windows)
        fault_scores.append(scores)
        pred = (scores > threshold).astype(int)
        tp += int(((pred == 1) & (labels == 1)).sum())
        fn += int(((pred == 0) & (labels == 1)).sum())
        if np.all(labels == 1):
            fp += int(((pred == 1) & (labels == 0)).sum())  # no normal part in window
        fault_windows = starts[labels == 1]
        detected = np.flatnonzero((pred == 1) & (labels == 1))
        delays.append(int(starts[detected[0]] - onset) if len(detected) else None)

    total_pos = tp + fn
    total_neg = tn + fp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / total_pos if total_pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / total_neg if total_neg else 0.0
    fnr = fn / total_pos if total_pos else 0.0

    # AUROC / AUPRC
    auroc = auprc = None
    if normal_test is not None and fault_scores:
        from sklearn.metrics import average_precision_score, roc_auc_score

        all_scores = np.concatenate(normal_scores + fault_scores)
        y = np.concatenate([np.zeros_like(np.concatenate(normal_scores)),
                            np.ones_like(np.concatenate(fault_scores))])
        x = all_scores
        auroc = float(roc_auc_score(y, x))
        auprc = float(average_precision_score(y, x))

    metrics = {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "auroc": auroc,
        "auprc": auprc,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n_normal_test_windows": int(np.concatenate(normal_scores).size) if normal_scores else 0,
        "n_fault_windows": int(np.concatenate(fault_scores).size) if fault_scores else 0,
        "detection_delay_samples": {
            fault_name: d for fault_name, d in zip(fault_meta.keys(), delays)
        },
        "mean_detection_delay_samples": float(np.mean([d for d in delays if d is not None])) if any(d is not None for d in delays) else None,
        "missed_faults": sum(1 for d in delays if d is None),
    }
    save_json(processed / "anomaly_detector_eval.json", metrics)
    logger.info("Anomaly detector evaluation:\n%s", _format(metrics))
    return metrics


def _format(metrics: Dict) -> str:
    lines = [f"  Precision        : {metrics['precision']:.4f}",
             f"  Recall           : {metrics['recall']:.4f}",
             f"  F1               : {metrics['f1']:.4f}",
             f"  FPR              : {metrics['false_positive_rate']:.4f}",
             f"  FNR              : {metrics['false_negative_rate']:.4f}",
             f"  AUROC            : {metrics['auroc']}",
             f"  AUPRC            : {metrics['auprc']}",
             f"  Mean delay (samples): {metrics['mean_detection_delay_samples']}",
             f"  Missed faults    : {metrics['missed_faults']}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the anomaly detector.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)-7s %(message)s")
    config = load_config(args.config)
    evaluate_anomaly_detector(config, threshold_override=args.threshold)


if __name__ == "__main__":
    main()