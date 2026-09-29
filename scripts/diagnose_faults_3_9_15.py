"""Comparative deep-dive: Faults 3,9,15 vs Normal - reconstruction failure mode.
Focus: temporal transitions and cross-sensor relationships.
Frozen detector, no retraining.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
from utils import ensure_dir, load_config, load_json
from anomaly_detection.inference import AnomalyDetector
from preprocessing.windowing import to_windows

SENSORS = [f"XMEAS_{i}" for i in range(1, 42)] + [f"XMV_{i}" for i in range(1, 12)]
SENSOR_COLS = [s.lower().replace("XMEAS","xmeas").replace("XMV","xmv") for s in SENSORS]

def max_consecutive(arr):
    m=c=0
    for v in arr:
        if v:
            c+=1
            m=max(m,c)
        else:
            c=0
    return m

def main():
    config = load_config("configs/config_a5000.yaml")
    detector = AnomalyDetector.from_artifacts(
        config["anomaly_detector"]["model_dir"],
        config["preprocessing"]["scaler_dir"],
        config["anomaly_detector"]["model_dir"]
    )
    threshold = float(detector.threshold.threshold)
    print(f"Threshold {threshold:.4f}")

    det_manifest = load_json(Path("data/processed/manifests/detector_split.json"))
    normal_test_runs = det_manifest["test_runs"]
    print(f"Normal test: {len(normal_test_runs)} runs")

    # Pre-load
    print("Pre-loading...")
    normal_all = pd.read_csv("data/raw/normal/TEP_FaultFree_Training.csv")
    fault_all = pd.read_csv("data/raw/faults/TEP_Faulty_Training.csv")
    from preprocessing.scaler import load_scaler
    scaler = load_scaler(config["preprocessing"]["scaler_dir"])
    baseline_mean = np.array(scaler.baseline.mean, dtype=np.float32)
    baseline_std = np.array(scaler.baseline.std, dtype=np.float32)

    out_dir = ensure_dir(Path("outputs/evaluation"))
    # For each fault 3,9,15 and normal, collect temporal and cross-sensor metrics
    faults = [3,9,15]
    results = {}

    for fid in faults + [0]:
        if fid == 0:
            runs = normal_test_runs
            df_source = normal_all
            is_fault = False
            label = "normal"
        else:
            df_f = fault_all[fault_all["faultNumber"]==fid]
            runs = sorted(df_f["simulationRun"].unique())
            df_source = fault_all
            is_fault = True
            label = f"fault{fid}"
        print(f"\n=== {label} ({len(runs)} runs) ===")
        # Collect per-run metrics
        run_metrics = []
        # For cross-sensor: compute correlation matrix per run (52x52) for normal vs fault
        # For temporal: compute per-sensor slope and transition
        for run in runs:
            if fid == 0:
                df_run = df_source[df_source["simulationRun"]==run]
            else:
                df_run = df_source[(df_source["faultNumber"]==fid) & (df_source["simulationRun"]==run)]
            df_run = df_run.sort_values("sample")
            arr = df_run[SENSOR_COLS].to_numpy(dtype=np.float32)
            # Temporal: compute per-sensor temporal gradient and cross-sensor correlation
            # For the whole run (500 samples), compute:
            # 1. Temporal transition: mean absolute difference between consecutive samples per sensor
            grad = np.abs(np.diff(arr, axis=0)).mean(axis=0)  # [52]
            # 2. Cross-sensor correlation matrix for this run
            # Use raw values
            corr = np.corrcoef(arr, rowvar=False)  # [52,52]
            # For fault onset, compute pre (0-160) vs post (160-500) correlation change
            pre = arr[:160]
            post = arr[160:]
            corr_pre = np.corrcoef(pre, rowvar=False) if len(pre)>10 else np.zeros((52,52))
            corr_post = np.corrcoef(post, rowvar=False) if len(post)>10 else np.zeros((52,52))
            corr_change = np.abs(corr_post - corr_pre).mean()
            # Also compute per-sensor z-score post vs pre
            z_post = np.abs((post.mean(axis=0) - baseline_mean) / (baseline_std+1e-9))
            z_pre = np.abs((pre.mean(axis=0) - baseline_mean) / (baseline_std+1e-9))
            z_diff = (z_post - z_pre).mean()
            # Detector scores for this run
            windows = to_windows(arr, 60, 5)
            scores, per_sensor = detector.score_windows(windows)
            n_above = int((scores>threshold).sum())
            max_consec = max_consecutive(scores>threshold)
            # For temporal of scores: score gradient
            score_grad = np.abs(np.diff(scores)).mean() if len(scores)>1 else 0
            run_metrics.append({
                "run": run,
                "grad_mean": float(grad.mean()),
                "grad_max": float(grad.max()),
                "corr_change": float(corr_change),
                "z_post_mean": float(z_post.mean()),
                "z_post_max": float(z_post.max()),
                "z_diff": float(z_diff),
                "score_max": float(scores.max()),
                "score_mean": float(scores.mean()),
                "n_above": n_above,
                "max_consec": int(max_consec),
                "score_grad": float(score_grad),
                "corr_pre_mean": float(np.abs(corr_pre).mean()),
                "corr_post_mean": float(np.abs(corr_post).mean()),
            })
        # Aggregate
        df_metrics = pd.DataFrame(run_metrics)
        results[label] = df_metrics
        print(f"  grad_mean {df_metrics['grad_mean'].mean():.4f}±{df_metrics['grad_mean'].std():.4f}")
        print(f"  corr_change {df_metrics['corr_change'].mean():.4f}±{df_metrics['corr_change'].std():.4f}")
        print(f"  z_post_max {df_metrics['z_post_max'].mean():.4f}±{df_metrics['z_post_max'].std():.4f}")
        print(f"  score_max {df_metrics['score_max'].mean():.4f}±{df_metrics['score_max'].std():.4f}")
        print(f"  n_above {df_metrics['n_above'].mean():.2f} max_consec {df_metrics['max_consec'].mean():.2f}")

    # Save per-run metrics
    all_rows = []
    for label, df in results.items():
        for _, row in df.iterrows():
            all_rows.append({"fault": label, **row.to_dict()})
    import csv
    with open(out_dir / "faults_3_9_15_temporal_cross.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved {out_dir/'faults_3_9_15_temporal_cross.csv'}")

    # Comparative summary: normal vs faults 3,9,15
    print("\n=== Comparative Summary ===")
    for fid in [0,3,9,15]:
        label = "normal" if fid==0 else f"fault{fid}"
        df = results[label]
        print(f"{label:8s}: grad {df['grad_mean'].mean():.4f} | corr_change {df['corr_change'].mean():.4f} | z_post_max {df['z_post_max'].mean():.2f} | score_max {df['score_max'].mean():.3f} | n_above {df['n_above'].mean():.1f} | max_consec {df['max_consec'].mean():.1f}")

    # Also check per-sensor cross-correlation for top sensors
    # For each fault, find which sensor pairs have largest correlation change
    print("\n=== Top sensor pairs by correlation change (fault vs normal) ===")
    for fid in [3,9,15]:
        # Compute mean correlation change per sensor pair across runs
        # For brevity, compute for fault vs normal mean corr
        # Use the per-run corr matrices already computed? For now, approximate by z_diff
        df = results[f"fault{fid}"]
        print(f"Fault {fid}: mean z_post_max {df['z_post_max'].mean():.2f}, corr_change {df['corr_change'].mean():.4f}, score_grad {df['score_grad'].mean():.4f}")

    # Save a markdown report
    with open(out_dir / "faults_3_9_15_diagnostic_report.md", "w") as f:
        f.write("# Faults 3,9,15 vs Normal - Temporal & Cross-Sensor Diagnostic\n\n")
        f.write(f"Threshold {threshold:.4f}, Window 60/5, 52 sensors\n\n")
        f.write("## Per-Run Metrics (mean ± std)\n\n")
        f.write("| Fault | Grad Mean | Corr Change | z_post_max | Score Max | n_above | max_consec |\n")
        f.write("|-------|-----------|-------------|------------|-----------|---------|------------|\n")
        for fid in [0,3,9,15]:
            label = "Normal" if fid==0 else f"Fault {fid}"
            df = results["normal" if fid==0 else f"fault{fid}"]
            f.write(f"| {label} | {df['grad_mean'].mean():.4f}±{df['grad_mean'].std():.2f} | {df['corr_change'].mean():.4f}±{df['corr_change'].std():.2f} | {df['z_post_max'].mean():.2f}±{df['z_post_max'].std():.2f} | {df['score_max'].mean():.3f}±{df['score_max'].std():.2f} | {df['n_above'].mean():.1f} | {df['max_consec'].mean():.1f} |\n")
        f.write("\n## Interpretation\n")
        f.write("- Grad Mean: mean absolute temporal gradient per sensor (how much sensors change per sample)\n")
        f.write("- Corr Change: mean absolute change in cross-sensor correlation (pre vs post onset)\n")
        f.write("- z_post_max: maximum |z| across sensors post-onset\n")
        f.write("- If faults 3,9,15 have similar grad/corr_change to normal, they are not temporal/relationship anomalies\n")
        f.write("- If they have higher grad/corr_change, they are temporal/relationship anomalies that global MSE misses\n")
    print(f"Saved {out_dir/'faults_3_9_15_diagnostic_report.md'}")

if __name__ == "__main__":
    main()
