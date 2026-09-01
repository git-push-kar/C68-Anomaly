"""Diagnostic: run CURRENT FROZEN detector over 75 normal test runs and all Fault 15 runs.

Saves: outputs/evaluation/fault15_score_distribution.csv
Prints: summary stats for normal vs Fault15 (max/mean/p95/p99, windows above threshold, consecutive, first anomalous)
Does NOT retrain, modify threshold/scaler/model.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
from utils import ensure_dir, load_config, load_json
from anomaly_detection.inference import AnomalyDetector
from preprocessing.tep_loader import load_single_csv
from preprocessing.windowing import to_windows

def score_run(df: pd.DataFrame, detector, window_size=60, stride=5, threshold=0.687):
    arr = df.to_numpy(dtype=np.float32)
    # to_windows expects [T,F] -> windows [N,W,F]
    from preprocessing.windowing import to_windows
    windows = to_windows(arr, window_size, stride)
    scores, _ = detector.score_windows(windows)
    # scores per window
    max_s = float(scores.max()) if len(scores) else 0.0
    mean_s = float(scores.mean()) if len(scores) else 0.0
    p95 = float(np.percentile(scores, 95)) if len(scores) else 0.0
    p99 = float(np.percentile(scores, 99)) if len(scores) else 0.0
    n_above = int((scores > threshold).sum())
    # first anomalous window and sample
    first_win = int(np.argmax(scores > threshold)) if n_above > 0 else -1
    first_sample = first_win * stride if first_win >= 0 else -1
    # max consecutive anomalous windows
    is_anom = scores > threshold
    max_consec = 0
    cur = 0
    for v in is_anom:
        if v:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0
    return {
        "max": max_s,
        "mean": mean_s,
        "p95": p95,
        "p99": p99,
        "n_above": n_above,
        "first_win": first_win,
        "first_sample": first_sample,
        "max_consec": max_consec,
        "n_windows": len(scores),
    }

def main():
    config = load_config()
    detector = AnomalyDetector.from_artifacts(
        config["anomaly_detector"]["model_dir"],
        config["preprocessing"]["scaler_dir"],
        config["anomaly_detector"]["model_dir"]
    )
    threshold = float(detector.threshold.threshold)
    print(f"Frozen detector threshold: {threshold:.4f} (mean {detector.threshold.mean:.4f} std {detector.threshold.std:.4f})")

    # Load manifests
    det_manifest = load_json(Path("data/processed/manifests/detector_split.json"))
    normal_test_runs = det_manifest["test_runs"]
    print(f"Normal test runs: {len(normal_test_runs)} (e.g. {normal_test_runs[:5]})")

    # Pre-load once for speed
    print("Pre-loading normal and fault15 data (once)...")
    normal_all = pd.read_csv("data/raw/normal/TEP_FaultFree_Training.csv")
    fault_all = pd.read_csv("data/raw/faults/TEP_Faulty_Training.csv")
    # Keep only fault 15 for fault set
    fault15_all = fault_all[fault_all["faultNumber"]==15]
    fault15_runs = sorted(fault15_all["simulationRun"].unique())
    print(f"Fault 15 available runs in Training: {len(fault15_runs)}")
    # Sensor cols
    sensor_cols = [c for c in fault15_all.columns if c.startswith("xmeas") or c.startswith("xmv")]

    out_dir = ensure_dir(Path("outputs/evaluation"))
    rows = []

    # Normal test runs
    print("\nScoring normal test runs...")
    for run in normal_test_runs:
        df_run = normal_all[normal_all["simulationRun"]==run]
        df = df_run[sensor_cols].reset_index(drop=True)
        metrics = score_run(df, detector, threshold=threshold)
        rows.append({"type": "normal", "simulationRun": run, **metrics})
        if run % 25 == 0:
            print(f"  normal run {run}: max {metrics['max']:.3f} n_above {metrics['n_above']}")

    # Fault 15 runs
    print("\nScoring Fault 15 runs...")
    for run in fault15_runs:
        df_run = fault15_all[fault15_all["simulationRun"]==run]
        df = df_run[sensor_cols].reset_index(drop=True)
        metrics = score_run(df, detector, threshold=threshold)
        rows.append({"type": "fault15", "simulationRun": run, **metrics})
        if run % 50 == 0:
            print(f"  fault15 run {run}: max {metrics['max']:.3f} n_above {metrics['n_above']}")

    # Save CSV
    import csv
    out_path = out_dir / "fault15_score_distribution.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type","simulationRun","max","mean","p95","p99","n_above","first_win","first_sample","max_consec","n_windows"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    print(f"\nSaved {len(rows)} rows to {out_path}")

    # Summary stats
    df_all = pd.DataFrame(rows)
    for t in ["normal", "fault15"]:
        sub = df_all[df_all["type"]==t]
        print(f"\n=== {t.upper()} (n={len(sub)}) ===")
        for col in ["max","mean","p95","p99","n_above","max_consec"]:
            print(f"{col:12s}: mean {sub[col].mean():.4f}  std {sub[col].std():.4f}  min {sub[col].min():.4f}  max {sub[col].max():.4f}  median {sub[col].median():.4f}")
        # Also first_sample stats for fault15 where detected
        if t=="fault15":
            detected = sub[sub["n_above"]>0]
            print(f"Detected runs: {len(detected)}/{len(sub)} ({len(detected)/len(sub):.1%})")
            if len(detected):
                print(f"First win (detected only): mean {detected['first_win'].mean():.1f} median {detected['first_win'].median():.1f}")

if __name__ == "__main__":
    main()
