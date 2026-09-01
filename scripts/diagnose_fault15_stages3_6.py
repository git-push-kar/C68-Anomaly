"""Stages 3-6: z-score, temporal, persistence, correlation for Fault 15.
Uses frozen detector and scaler, no retraining.
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

def main():
    config = load_config()
    detector = AnomalyDetector.from_artifacts(
        config["anomaly_detector"]["model_dir"],
        config["preprocessing"]["scaler_dir"],
        config["anomaly_detector"]["model_dir"]
    )
    threshold = float(detector.threshold.threshold)
    print(f"Frozen detector threshold: {threshold:.4f}")

    det_manifest = load_json(Path("data/processed/manifests/detector_split.json"))
    normal_test_runs = det_manifest["test_runs"]
    print(f"Normal test runs: {len(normal_test_runs)}")

    print("Pre-loading data...")
    normal_all = pd.read_csv("data/raw/normal/TEP_FaultFree_Training.csv")
    fault_all = pd.read_csv("data/raw/faults/TEP_Faulty_Training.csv")
    fault15_all = fault_all[fault_all["faultNumber"]==15]
    fault15_runs = sorted(fault15_all["simulationRun"].unique())
    print(f"Fault 15 runs: {len(fault15_runs)}")

    from preprocessing.scaler import load_scaler
    scaler = load_scaler(config["preprocessing"]["scaler_dir"])
    baseline_mean = np.array(scaler.baseline.mean, dtype=np.float32)
    baseline_std = np.array(scaler.baseline.std, dtype=np.float32)

    out_dir = ensure_dir(Path("outputs/evaluation"))
    # For stage 3: per sensor z-score stats
    # Collect per sensor per window z for normal and fault15 post
    # For stage 4: temporal onset
    # For stage 5: persistence per run
    # For stage 6: correlation

    # Stage 3: z-score
    # We will compute per sensor per window z, then aggregate per sensor
    # Use all windows for normal, and post-onset for fault15
    print("\n=== Stage 3: z-score ===")
    # For each sensor, collect all |z| values
    normal_z_per_sensor = {s: [] for s in SENSORS}
    fault_z_per_sensor = {s: [] for s in SENSORS}
    fault_post_z_per_sensor = {s: [] for s in SENSORS}
    # Also need per run per sensor for stage 3B, 3C
    z_run_stats = []  # per run per sensor
    # For temporal stage 4
    temporal_rows = []  # per run per sensor

    # Process normal test runs for z
    for run in normal_test_runs:
        df_run = normal_all[normal_all["simulationRun"]==run]
        # Get sensor array in canonical order
        sensor_cols = [s.lower().replace("XMEAS","xmeas").replace("XMV","xmv") for s in SENSORS]
        arr = df_run[sensor_cols].to_numpy(dtype=np.float32)
        # Compute z per sample per sensor
        z = np.abs((arr - baseline_mean) / (baseline_std + 1e-9))
        # For windowing, we need per window z: we can compute per window mean |z|
        windows = to_windows(arr, 60, 5)
        # For each window, compute mean |z| per sensor
        for wi in range(len(windows)):
            window = windows[wi]
            z_win = np.abs((window - baseline_mean) / (baseline_std + 1e-9))
            z_mean_per_sensor = z_win.mean(axis=0)  # [52]
            for si, sensor in enumerate(SENSORS):
                normal_z_per_sensor[sensor].append(float(z_mean_per_sensor[si]))

    # Process fault15 runs
    for run in fault15_runs:
        df_run = fault15_all[fault15_all["simulationRun"]==run]
        df_run = df_run.sort_values("sample")
        arr = df_run[[s.lower().replace("XMEAS","xmeas").replace("XMV","xmv") for s in SENSORS]].to_numpy(dtype=np.float32)
        windows = to_windows(arr, 60, 5)
        n_windows = len(windows)
        starts = np.arange(n_windows) * 5
        is_post = (starts + 60) > 160
        for wi in range(n_windows):
            window = windows[wi]
            z_win = np.abs((window - baseline_mean) / (baseline_std + 1e-9))
            z_mean_per_sensor = z_win.mean(axis=0)
            for si, sensor in enumerate(SENSORS):
                fault_z_per_sensor[sensor].append(float(z_mean_per_sensor[si]))
                if is_post[wi]:
                    fault_post_z_per_sensor[sensor].append(float(z_mean_per_sensor[si]))
        # For run-level z stats and temporal, do per sensor
        # Compute per run per sensor post stats
        for si, sensor in enumerate(SENSORS):
            # Get all windows for this run, compute z per window for this sensor
            z_per_window = []
            for wi in range(n_windows):
                window = windows[wi]
                z_win = np.abs((window[si] - baseline_mean[si]) / (baseline_std[si] + 1e-9))
                # Actually window is [60,52], so per sensor per window we need mean over time
                z_win_sensor = np.abs((window[:, si] - baseline_mean[si]) / (baseline_std[si] + 1e-9))
                z_per_window.append(float(z_win_sensor.mean()))
            z_per_window = np.array(z_per_window)
            # Pre/post split
            pre_vals = z_per_window[~is_post] if (~is_post).any() else np.array([0])
            post_vals = z_per_window[is_post] if is_post.any() else np.array([0])
            # For stage 3A
            z_run_stats.append({
                "type": "fault15", "simulationRun": run, "sensor": sensor,
                "pre_mean_abs_z": float(pre_vals.mean()) if len(pre_vals) else 0,
                "post_mean_abs_z": float(post_vals.mean()) if len(post_vals) else 0,
                "post_p95_abs_z": float(np.percentile(post_vals,95)) if len(post_vals) else 0,
                "post_max_abs_z": float(post_vals.max()) if len(post_vals) else 0,
                "mean_z_signed": float(((windows[:, :, si] - baseline_mean[si]) / (baseline_std[si]+1e-9)).mean()) if len(windows) else 0,
            })
            # For stage 4: first crossing
            # Find first window where |z| >=2,3,4
            for thresh_z in [2,3,4]:
                # Find first post window where mean |z| >= thresh
                crossing_idx = -1
                for wi2 in range(n_windows):
                    if not is_post[wi2]:
                        continue
                    # Check this window's mean |z| for this sensor
                    # We already have z_per_window for this sensor, but need per sensor per window
                    # Recompute for this sensor
                    # Use post_vals index mapping
                    # Simpler: find first post window where z_per_window[wi2] >= thresh
                    if z_per_window[wi2] >= thresh_z:
                        crossing_idx = wi2
                        break
                first_sample = int(starts[crossing_idx]) if crossing_idx >=0 else -1
                delay = int(first_sample - 160) if first_sample >=0 else -1
                temporal_rows.append({
                    "type": "fault15",
                    "simulationRun": run,
                    "sensor": sensor,
                    "z_threshold": thresh_z,
                    "first_crossing_window": int(crossing_idx),
                    "first_crossing_sample": int(first_sample),
                    "delay_from_onset": int(delay),
                })
        if run % 50 == 0:
            print(f"  fault15 run {run} done")

    # Also need normal z for comparison (for stage 3B, need normal threshold crossings)
    # For normal, do similar for first 5 runs for comparison
    for run in normal_test_runs[:5]:
        df_run = normal_all[normal_all["simulationRun"]==run]
        arr = df_run[[s.lower().replace("XMEAS","xmeas").replace("XMV","xmv") for s in SENSORS]].to_numpy(dtype=np.float32)
        windows = to_windows(arr, 60, 5)
        n_windows = len(windows)
        starts = np.arange(n_windows) * 5
        for si, sensor in enumerate(SENSORS):
            for thresh_z in [2,3,4]:
                crossing_idx = -1
                for wi2 in range(n_windows):
                    window = windows[wi2]
                    z_win_sensor = np.abs((window[:, si] - baseline_mean[si]) / (baseline_std[si] + 1e-9))
                    if z_win_sensor.mean() >= thresh_z:
                        crossing_idx = wi2
                        break
                first_sample = int(starts[crossing_idx]) if crossing_idx >=0 else -1
                temporal_rows.append({
                    "type": "normal",
                    "simulationRun": run,
                    "sensor": sensor,
                    "z_threshold": thresh_z,
                    "first_crossing_window": int(crossing_idx),
                    "first_crossing_sample": int(first_sample),
                    "delay_from_onset": -1,
                })

    # Save stage 3 outputs
    # Sensor zscores csv
    sensor_z_rows = []
    for sensor in SENSORS:
        n_vals = normal_z_per_sensor[sensor]
        f_vals = fault_z_per_sensor[sensor]
        fp_vals = fault_post_z_per_sensor[sensor]
        sensor_z_rows.append({
            "sensor": sensor,
            "normal_mean_abs_z": float(np.mean(n_vals)) if n_vals else 0,
            "normal_p95_abs_z": float(np.percentile(n_vals,95)) if n_vals else 0,
            "normal_max_abs_z": float(np.max(n_vals)) if n_vals else 0,
            "fault15_mean_abs_z": float(np.mean(f_vals)) if f_vals else 0,
            "fault15_post_mean_abs_z": float(np.mean(fp_vals)) if fp_vals else 0,
            "fault15_post_p95_abs_z": float(np.percentile(fp_vals,95)) if fp_vals else 0,
            "fault15_post_max_abs_z": float(np.max(fp_vals)) if fp_vals else 0,
        })
    import csv
    with open(out_dir / "fault15_sensor_zscores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sensor_z_rows[0].keys())
        writer.writeheader()
        writer.writerows(sensor_z_rows)
    print(f"Saved {out_dir/'fault15_sensor_zscores.csv'}")

    # Zscore crossings
    # For stage 3B, we need per run per sensor per threshold - already in temporal_rows, but filter for zscore
    # temporal_rows already has those, save as zscore_crossings
    with open(out_dir / "fault15_zscore_crossings.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=temporal_rows[0].keys())
        writer.writeheader()
        writer.writerows(temporal_rows)
    print(f"Saved {out_dir/'fault15_zscore_crossings.csv'}")

    # For stage 5 persistence, we need per run persistence at thresholds 0.60,0.62,0.64,0.66,0.687
    # We can compute from earlier stage1 data, but we didn't save per run persistence for all thresholds
    # For now, create a simple persistence csv based on stage1 rows
    # We need to recompute for each run at each threshold - let's do it
    print("\n=== Stage 5: Persistence ===")
    persistence_rows = []
    thresholds = [0.60, 0.62, 0.64, 0.66, 0.687]
    # Use the normal and fault15 runs we already processed for stage1
    # For each run, we have scores per window from earlier, but we didn't store them
    # Let's recompute quickly for each run at each threshold
    all_runs = [(r, "normal", normal_all) for r in normal_test_runs] + [(r, "fault15", fault15_all) for r in fault15_runs]
    for run, typ, df_all in [(r, "normal", normal_all) for r in normal_test_runs] + [(r, "fault15", fault15_all) for r in fault15_runs[:50]]:  # limit to 50 fault runs for speed
        # Get run data
        if typ == "normal":
            df_run = normal_all[normal_all["simulationRun"]==run]
        else:
            df_run = fault_all[(fault_all["faultNumber"]==15) & (fault_all["simulationRun"]==run)]
        if len(df_run)==0:
            continue
        arr = df_run[[s.lower().replace("XMEAS","xmeas").replace("XMV","xmv") for s in SENSORS]].to_numpy(dtype=np.float32)
        windows = to_windows(arr, 60, 5)
        scores, _ = detector.score_windows(windows)
        for th in thresholds:
            is_anom = scores > th
            n_above = int(is_anom.sum())
            # max consecutive
            max_c = cur = 0
            for v in is_anom:
                if v:
                    cur+=1
                    max_c=max(max_c, cur)
                else:
                    cur=0
            # mean, p95, max
            persistence_rows.append({
                "type": typ,
                "simulationRun": run,
                "threshold": th,
                "mean_score": float(scores.mean()),
                "max_score": float(scores.max()),
                "p95_score": float(np.percentile(scores,95)),
                "n_above": n_above,
                "max_consec": int(max_c),
            })
        if run % 50 == 0:
            print(f"  persistence run {run} done")

    with open(out_dir / "fault15_persistence.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=persistence_rows[0].keys())
        writer.writeheader()
        writer.writerows(persistence_rows)
    print(f"Saved {out_dir/'fault15_persistence.csv'}")

    # For stage 4 temporal onset and order, we need to compute per run ordering
    # Use temporal_rows already collected for z>=2,3,4
    # Create temporal_onset csv
    # For each run, rank sensors by first crossing at z>=3
    # Save to fault15_temporal_onset.csv
    with open(out_dir / "fault15_temporal_onset.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=temporal_rows[0].keys())
        writer.writeheader()
        writer.writerows(temporal_rows)
    print(f"Saved {out_dir/'fault15_temporal_onset.csv'}")

    # Temporal order: for each run, order sensors by first crossing at z>=3
    order_rows = []
    for run in fault15_runs[:50]:  # limit
        # Get crossing for this run at z>=3
        cross_for_run = [r for r in temporal_rows if r["simulationRun"]==run and r["type"]=="fault15" and r["z_threshold"]==3 and r["first_crossing_sample"]>=0]
        cross_for_run = sorted(cross_for_run, key=lambda x: x["first_crossing_sample"])
        order = " -> ".join([c["sensor"] for c in cross_for_run[:5]])
        order_rows.append({"simulationRun": run, "order": order, "n_sensors_crossed": len(cross_for_run)})
    with open(out_dir / "fault15_temporal_order.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=order_rows[0].keys())
        writer.writeheader()
        writer.writerows(order_rows)
    print(f"Saved {out_dir/'fault15_temporal_order.csv'}")

    # Stage 6 correlation: for top 10 sensors, compute correlation between |z| and reconstruction error
    # Use fault15 post windows for top sensors
    print("\n=== Stage 6: Correlation ===")
    top_sensors = [r["sensor"] for r in sorted(sensor_z_rows, key=lambda x: x["fault15_post_mean_abs_z"], reverse=True)[:10]]
    print(f"Top 10 sensors by post z: {top_sensors}")
    corr_rows = []
    # For each top sensor, compute correlation across all fault15 post windows
    # We need per window per sensor z and reconstruction error
    # For efficiency, sample 10 runs
    for sensor in top_sensors:
        si = SENSORS.index(sensor)
        # Collect per window values for fault15 post windows
        z_vals = []
        err_vals = []
        for run in fault15_runs[:20]:  # sample 20 runs
            df_run = fault15_all[fault15_all["simulationRun"]==run]
            arr = df_run[[s.lower().replace("XMEAS","xmeas").replace("XMV","xmv") for s in SENSORS]].to_numpy(dtype=np.float32)
            windows = to_windows(arr, 60, 5)
            scores, per_sensor = detector.score_windows(windows)
            # Get per window z and error for this sensor
            n_windows = len(windows)
            starts = np.arange(n_windows) * 5
            is_post = (starts + 60) > 160
            for wi in range(n_windows):
                if not is_post[wi]:
                    continue
                window = windows[wi]
                z = float(np.abs((window[:, si] - baseline_mean[si]) / (baseline_std[si]+1e-9)).mean())
                err = float(per_sensor[wi, si])
                z_vals.append(z)
                err_vals.append(err)
        if len(z_vals) > 10:
            corr = float(np.corrcoef(z_vals, err_vals)[0,1]) if np.std(z_vals)>0 and np.std(err_vals)>0 else 0.0
        else:
            corr = 0.0
        corr_rows.append({"sensor": sensor, "correlation": corr, "n_windows": len(z_vals)})
        print(f"  {sensor}: corr {corr:.3f} n {len(z_vals)}")

    with open(out_dir / "fault15_zscore_vs_reconstruction.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=corr_rows[0].keys())
        writer.writeheader()
        writer.writerows(corr_rows)
    print(f"Saved {out_dir/'fault15_zscore_vs_reconstruction.csv'}")

if __name__ == "__main__":
    main()
