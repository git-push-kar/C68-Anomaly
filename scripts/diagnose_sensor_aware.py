"""Sensor-aware scoring experiment for Fault 15.
Uses frozen detector, no retraining.
Compares global MSE vs top-k per-sensor error vs z-score based scoring.
Thresholds selected on TRAIN+VAL only, evaluated on TEST (normal test + Fault 15 test).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
from utils import load_config, load_json
from anomaly_detection.inference import AnomalyDetector
from preprocessing.windowing import to_windows

SENSORS = [f"XMEAS_{i}" for i in range(1, 42)] + [f"XMV_{i}" for i in range(1, 12)]

def score_run_sensor_aware(df, detector, method="global"):
    # df: [T,52] sensor DataFrame
    arr = df.to_numpy(dtype=np.float32)
    windows = to_windows(arr, 60, 5)
    scores, per_sensor = detector.score_windows(windows)
    # per_sensor: [N,52]
    # Compute sensor-aware scores per window
    sensor_aware = {}
    for wi in range(len(scores)):
        sorted_err = np.sort(per_sensor[wi])[::-1]
        sensor_aware[wi] = {
            "global": float(scores[wi]),
            "top1": float(sorted_err[0]),
            "top3": float(sorted_err[:3].mean()),
            "top5": float(sorted_err[:5].mean()),
            "top10": float(sorted_err[:10].mean()),
        }
    # For run-level, we need max, mean, etc. per method
    result = {}
    for method in ["global", "top1", "top3", "top5", "top10"]:
        vals = np.array([sensor_aware[wi][method] for wi in range(len(scores))])
        result[method] = {
            "max": float(vals.max()) if len(vals) else 0,
            "mean": float(vals.mean()) if len(vals) else 0,
            "p95": float(np.percentile(vals,95)) if len(vals) else 0,
            "scores": vals,
        }
    return result

def main():
    config = load_config()
    detector = AnomalyDetector.from_artifacts(
        config["anomaly_detector"]["model_dir"],
        config["preprocessing"]["scaler_dir"],
        config["anomaly_detector"]["model_dir"]
    )
    print(f"Loaded detector, threshold {detector.threshold.threshold:.4f}")

    # Load manifests
    det_manifest = load_json(Path("data/processed/manifests/detector_split.json"))
    normal_train_runs = det_manifest["train_runs"]
    normal_val_runs = det_manifest["validation_runs"]
    normal_test_runs = det_manifest["test_runs"]
    print(f"Normal: train {len(normal_train_runs)}, val {len(normal_val_runs)}, test {len(normal_test_runs)}")

    # Pre-load data
    print("Pre-loading data...")
    normal_all = pd.read_csv("data/raw/normal/TEP_FaultFree_Training.csv")
    fault_all = pd.read_csv("data/raw/faults/TEP_Faulty_Training.csv")
    fault15_all = fault_all[fault_all["faultNumber"]==15]
    fault15_runs = sorted(fault15_all["simulationRun"].unique())
    # For this experiment, use 50 fault15 runs as test (like before), and normal test as test
    # For threshold selection, use normal train+val

    # Collect scores for threshold selection (train+val normal)
    print("\nCollecting normal train+val scores for threshold selection...")
    normal_train_val_scores = {m: [] for m in ["global","top1","top3","top5","top10"]}
    for run in normal_train_runs[:50] + normal_val_runs[:20]:  # sample 70 runs for speed
        df_run = normal_all[normal_all["simulationRun"]==run]
        # Need sensor cols in correct order
        sensor_cols = [c for c in df_run.columns if c.startswith("xmeas") or c.startswith("xmv")]
        # Ensure order is canonical
        # Reorder to SENSORS lower case
        ordered_cols = [s.lower().replace("xmeas","xmeas").replace("xmv","xmv") for s in SENSORS]
        # Filter to only those that exist
        ordered_cols = [c for c in ordered_cols if c in df_run.columns]
        arr = df_run[ordered_cols].to_numpy(dtype=np.float32)
        windows = to_windows(arr, 60, 5)
        scores, per_sensor = detector.score_windows(windows)
        for wi in range(len(scores)):
            sorted_err = np.sort(per_sensor[wi])[::-1]
            normal_train_val_scores["global"].append(float(scores[wi]))
            normal_train_val_scores["top1"].append(float(sorted_err[0]))
            normal_train_val_scores["top3"].append(float(sorted_err[:3].mean()))
            normal_train_val_scores["top5"].append(float(sorted_err[:5].mean()))
            normal_train_val_scores["top10"].append(float(sorted_err[:10].mean()))

    # Select thresholds at p99, p99.5, p99.9, max for each method
    thresholds = {}
    for method in ["global","top1","top3","top5","top10"]:
        vals = np.array(normal_train_val_scores[method])
        thresholds[method] = {
            "p99": float(np.percentile(vals,99)),
            "p99.5": float(np.percentile(vals,99.5)),
            "p99.9": float(np.percentile(vals,99.9)),
            "max": float(vals.max()),
            "mean+3std": float(vals.mean() + 3*vals.std()),
        }
        print(f"{method}: p99 {thresholds[method]['p99']:.4f}, p99.5 {thresholds[method]['p99.5']:.4f}, max {thresholds[method]['max']:.4f}")

    # Evaluate on normal test and fault15
    print("\nEvaluating on normal test (75 runs) and fault15 (500 runs)...")
    # For each method and each threshold, compute detection rate
    methods = ["global","top1","top3","top5","top10"]
    thresh_keys = ["p99","p99.5","p99.9","max"]
    results = []

    # Pre-collect test scores
    normal_test_max = {m: [] for m in methods}
    fault15_test_max = {m: [] for m in methods}

    for run in normal_test_runs:
        df_run = normal_all[normal_all["simulationRun"]==run]
        ordered_cols = [s.lower().replace("XMEAS","xmeas").replace("XMV","xmv") for s in SENSORS]
        ordered_cols = [c for c in ordered_cols if c in df_run.columns]
        arr = df_run[ordered_cols].to_numpy(dtype=np.float32)
        windows = to_windows(arr, 60, 5)
        scores, per_sensor = detector.score_windows(windows)
        for method in methods:
            if method == "global":
                vals = scores
            elif method == "top1":
                vals = np.array([np.sort(per_sensor[wi])[::-1][0] for wi in range(len(scores))])
            elif method == "top3":
                vals = np.array([np.sort(per_sensor[wi])[::-1][:3].mean() for wi in range(len(scores))])
            elif method == "top5":
                vals = np.array([np.sort(per_sensor[wi])[::-1][:5].mean() for wi in range(len(scores))])
            elif method == "top10":
                vals = np.array([np.sort(per_sensor[wi])[::-1][:10].mean() for wi in range(len(scores))])
            normal_test_max[method].append(float(vals.max()))

    for run in fault15_runs:
        df_run = fault15_all[fault15_all["simulationRun"]==run]
        ordered_cols = [s.lower().replace("XMEAS","xmeas").replace("XMV","xmv") for s in SENSORS]
        ordered_cols = [c for c in ordered_cols if c in df_run.columns]
        arr = df_run[ordered_cols].to_numpy(dtype=np.float32)
        windows = to_windows(arr, 60, 5)
        scores, per_sensor = detector.score_windows(windows)
        for method in methods:
            if method == "global":
                vals = scores
            elif method == "top1":
                vals = np.array([np.sort(per_sensor[wi])[::-1][0] for wi in range(len(scores))])
            elif method == "top3":
                vals = np.array([np.sort(per_sensor[wi])[::-1][:3].mean() for wi in range(len(scores))])
            elif method == "top5":
                vals = np.array([np.sort(per_sensor[wi])[::-1][:5].mean() for wi in range(len(scores))])
            elif method == "top10":
                vals = np.array([np.sort(per_sensor[wi])[::-1][:10].mean() for wi in range(len(scores))])
            fault15_test_max[method].append(float(vals.max()))

    # For each method/threshold, compute FPR and TPR
    for method in methods:
        for tk in thresh_keys:
            th = thresholds[method][tk]
            # FPR: normal test runs with max > th
            fpr = float(np.mean(np.array(normal_test_max[method]) > th))
            # TPR: fault15 runs with max > th
            tpr = float(np.mean(np.array(fault15_test_max[method]) > th))
            results.append({
                "method": method,
                "threshold_key": tk,
                "threshold_value": th,
                "normal_FPR": fpr,
                "fault15_TPR": tpr,
            })
            print(f"{method:6s} {tk:6s} th {th:.4f} -> normal FPR {fpr:.3f} ({int(fpr*75)}/75), fault15 TPR {tpr:.3f} ({int(tpr*500)}/500)")

    # Save results
    import csv
    out_path = Path("outputs/evaluation/fault15_sensor_aware_evaluation.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {out_path}")

    # Also print best per method (max TPR with FPR <=0.05)
    print("\nBest per method with FPR <=0.05:")
    for method in methods:
        best = max([r for r in results if r["method"]==method and r["normal_FPR"]<=0.05], key=lambda x: x["fault15_TPR"], default=None)
        if best:
            print(f"{method:6s}: TPR {best['fault15_TPR']:.3f} at {best['threshold_key']} {best['threshold_value']:.4f} (FPR {best['normal_FPR']:.3f})")
        else:
            print(f"{method:6s}: No threshold with FPR<=0.05")

if __name__ == "__main__":
    main()
