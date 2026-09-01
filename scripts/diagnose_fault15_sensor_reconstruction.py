"""Diagnostic 2: Per-sensor reconstruction error for Fault 15 vs Normal.
FROZEN detector only - no retraining, no threshold change.
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

def score_windows_per_sensor(windows, detector):
    # windows: [N,60,52] already scaled inside detector.score_windows, but we need per-sensor
    # Use detector internals to get per-sensor error without re-scaling twice
    # We'll call detector.score_windows which does scaling, but we need per-sensor
    scores, per_sensor = detector.score_windows(windows)  # per_sensor [N,52]
    # Verify global = mean(per_sensor)
    return scores, per_sensor

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

    # Pre-load normal and fault15 data once
    print("Pre-loading data...")
    normal_all = pd.read_csv("data/raw/normal/TEP_FaultFree_Training.csv")
    fault_all = pd.read_csv("data/raw/faults/TEP_Faulty_Training.csv")
    fault15_all = fault_all[fault_all["faultNumber"]==15]
    sensor_cols = [c for c in fault15_all.columns if c.startswith("xmeas") or c.startswith("xmv")]
    # Map to canonical names
    # The CSV uses xmeas_1..xmv_11, convert to XMEAS_1..XMV_11 for consistency
    col_map = {c: c.replace("xmeas", "XMEAS").replace("xmv", "XMV") for c in sensor_cols}
    # Use canonical order
    canonical = SENSORS
    # Build lookup for normal_all sensor cols: normal_all has xmeas/xmv lower case
    normal_sensor_cols = [c for c in normal_all.columns if c.startswith("xmeas") or c.startswith("xmv")]
    # For fault15, same

    out_dir = ensure_dir(Path("outputs/evaluation"))
    # Collect per-window per-sensor errors
    # For normal: all windows from 75 runs
    normal_per_sensor_all = []  # list of [52] per window
    normal_scores_all = []
    # For fault15: all windows, plus post-onset only
    fault_per_sensor_all = []
    fault_post_per_sensor_all = []
    fault_scores_all = []
    fault_post_scores_all = []
    # Run-level stats
    run_stats = []  # per run per sensor
    # Global vs top-k per window/run
    global_vs_topk_rows = []

    # Process normal test runs
    print("\nScoring normal test runs...")
    for run in normal_test_runs:
        df_run = normal_all[normal_all["simulationRun"]==run]
        # Keep sensor cols in canonical order
        # normal_all has xmeas_1.. need to map
        # Build df with canonical order
        df_sensors = pd.DataFrame()
        for s in canonical:
            col = s.lower()  # xmeas_1
            # xmv -> xmv_1
            col_lower = s.replace("XMEAS", "xmeas").replace("XMV", "xmv")
            df_sensors[col_lower] = df_run[col_lower].values
        # Rename to canonical for scoring (but scaler expects canonical order via feature_names)
        # The scaler was fitted on canonical order, so we need to pass in canonical order
        arr = df_sensors.to_numpy(dtype=np.float32)  # [500,52] already in canonical order via loop
        windows = to_windows(arr, 60, 5)
        scores, per_sensor = score_windows_per_sensor(windows, detector)
        # Verify global
        if len(scores) > 0:
            recon = np.mean(per_sensor, axis=1)
            diff = np.abs(scores - recon).max()
            if diff > 1e-5 and run == normal_test_runs[0]:
                print(f"  Verify normal run {run}: max diff global vs mean(per_sensor) = {diff:.6f} (should be ~0)")
        normal_per_sensor_all.append(per_sensor)
        normal_scores_all.append(scores)
        # Run-level per sensor
        for si, sensor in enumerate(canonical):
            run_stats.append({
                "type": "normal",
                "simulationRun": run,
                "sensor": sensor,
                "pre_onset_mean": float(per_sensor.mean(axis=0)[si]),  # for normal, all pre
                "post_onset_mean": float(per_sensor.mean(axis=0)[si]),
                "post_onset_p95": float(np.percentile(per_sensor[:, si], 95)),
                "post_onset_max": float(per_sensor[:, si].max()),
            })
        # Global vs topk per window
        for wi in range(len(scores)):
            sorted_err = np.sort(per_sensor[wi])[::-1]
            global_vs_topk_rows.append({
                "type": "normal",
                "simulationRun": run,
                "window_idx": wi,
                "global_score": float(scores[wi]),
                "top1": float(sorted_err[0]),
                "top3_mean": float(sorted_err[:3].mean()),
                "top5_mean": float(sorted_err[:5].mean()),
                "top10_mean": float(sorted_err[:10].mean()),
            })

    # Process Fault 15 runs (500)
    fault15_runs = sorted(fault15_all["simulationRun"].unique())
    print(f"\nScoring Fault 15 runs ({len(fault15_runs)})...")
    for idx, run in enumerate(fault15_runs):
        df_run = fault15_all[fault15_all["simulationRun"]==run]
        df_sensors = pd.DataFrame()
        for s in canonical:
            col_lower = s.replace("XMEAS", "xmeas").replace("XMV", "xmv")
            df_sensors[col_lower] = df_run[col_lower].values
        arr = df_sensors.to_numpy(dtype=np.float32)
        windows = to_windows(arr, 60, 5)
        scores, per_sensor = score_windows_per_sensor(windows, detector)
        # Split pre/post onset: sample 160 is onset, window starting at sample s covers [s, s+60)
        # Window is anomalous post-onset if its start >= 160 or it overlaps post-onset.
        # Define post-onset windows as those with window end >160 (i.e., start+60 >160)
        # For simplicity: window start sample = window_idx*stride, so post if start >= 100? Let's use start+60 >160
        n_windows = len(scores)
        # window start samples: 0,5,10,... for this run's 500 samples -> 89 windows, start 0..440
        starts = np.arange(n_windows) * 5
        is_post = (starts + 60) > 160
        is_pre = ~is_post
        fault_per_sensor_all.append(per_sensor)
        fault_scores_all.append(scores)
        if is_post.any():
            fault_post_per_sensor_all.append(per_sensor[is_post])
            fault_post_scores_all.append(scores[is_post])
        # Verify
        if idx == 0 and len(scores) > 0:
            recon = np.mean(per_sensor, axis=1)
            diff = np.abs(scores - recon).max()
            print(f"  Verify fault15 run {run}: max diff {diff:.6f}")
        # Run-level per sensor (post-onset)
        for si, sensor in enumerate(canonical):
            pre_vals = per_sensor[is_pre, si] if is_pre.any() else np.array([0])
            post_vals = per_sensor[is_post, si] if is_post.any() else np.array([0])
            run_stats.append({
                "type": "fault15",
                "simulationRun": run,
                "sensor": sensor,
                "pre_onset_mean": float(pre_vals.mean()) if len(pre_vals) else 0.0,
                "post_onset_mean": float(post_vals.mean()) if len(post_vals) else 0.0,
                "post_onset_p95": float(np.percentile(post_vals, 95)) if len(post_vals) else 0.0,
                "post_onset_max": float(post_vals.max()) if len(post_vals) else 0.0,
            })
        for wi in range(len(scores)):
            sorted_err = np.sort(per_sensor[wi])[::-1]
            global_vs_topk_rows.append({
                "type": "fault15",
                "simulationRun": run,
                "window_idx": wi,
                "global_score": float(scores[wi]),
                "top1": float(sorted_err[0]),
                "top3_mean": float(sorted_err[:3].mean()),
                "top5_mean": float(sorted_err[:5].mean()),
                "top10_mean": float(sorted_err[:10].mean()),
            })
        if (idx+1) % 50 == 0:
            print(f"  fault15 {idx+1}/{len(fault15_runs)} done")

    # Concatenate all per-sensor
    normal_per_sensor_all = np.concatenate(normal_per_sensor_all, axis=0) if normal_per_sensor_all else np.zeros((0,52))
    fault_per_sensor_all = np.concatenate(fault_per_sensor_all, axis=0) if fault_per_sensor_all else np.zeros((0,52))
    fault_post_per_sensor_all = np.concatenate(fault_post_per_sensor_all, axis=0) if fault_post_per_sensor_all else np.zeros((0,52))
    print(f"\nNormal windows: {len(normal_per_sensor_all)}, Fault15 windows: {len(fault_per_sensor_all)}, Post-onset fault windows: {len(fault_post_per_sensor_all)}")

    # Sensor-level statistics
    sensor_rows = []
    for si, sensor in enumerate(canonical):
        n_vals = normal_per_sensor_all[:, si] if len(normal_per_sensor_all) else np.array([0])
        f_vals = fault_per_sensor_all[:, si] if len(fault_per_sensor_all) else np.array([0])
        fp_vals = fault_post_per_sensor_all[:, si] if len(fault_post_per_sensor_all) else np.array([0])
        # Handle zero denominator
        mean_diff = float(f_vals.mean() - n_vals.mean())
        mean_ratio = float(f_vals.mean() / (n_vals.mean() + 1e-9))
        p95_diff = float(np.percentile(f_vals,95) - np.percentile(n_vals,95))
        p95_ratio = float(np.percentile(f_vals,95) / (np.percentile(n_vals,95)+1e-9))
        # Post-onset mean diff
        post_mean_diff = float(fp_vals.mean() - n_vals.mean())
        sensor_rows.append({
            "sensor": sensor,
            "normal_mean": float(n_vals.mean()),
            "normal_median": float(np.median(n_vals)),
            "normal_std": float(n_vals.std()),
            "normal_p95": float(np.percentile(n_vals,95)),
            "normal_p99": float(np.percentile(n_vals,99)),
            "normal_max": float(n_vals.max()),
            "fault15_mean": float(f_vals.mean()),
            "fault15_median": float(np.median(f_vals)),
            "fault15_std": float(f_vals.std()),
            "fault15_p95": float(np.percentile(f_vals,95)),
            "fault15_p99": float(np.percentile(f_vals,99)),
            "fault15_max": float(f_vals.max()),
            "fault15_post_mean": float(fp_vals.mean()),
            "fault15_post_p95": float(np.percentile(fp_vals,95)) if len(fp_vals) else 0,
            "fault15_post_max": float(fp_vals.max()) if len(fp_vals) else 0,
            "mean_difference": mean_diff,
            "mean_ratio": mean_ratio,
            "p95_difference": p95_diff,
            "p95_ratio": p95_ratio,
            "post_mean_difference": post_mean_diff,
        })

    # Save sensor reconstruction CSV
    import csv
    out1 = Path("outputs/evaluation/fault15_sensor_reconstruction.csv")
    with open(out1, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sensor_rows[0].keys())
        writer.writeheader()
        writer.writerows(sensor_rows)
    print(f"Saved {out1}")

    # Save run-level stats
    out2 = Path("outputs/evaluation/fault15_sensor_run_statistics.csv")
    # run_stats already has pre/post per run per sensor
    with open(out2, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type","simulationRun","sensor","pre_onset_mean","post_onset_mean","post_onset_p95","post_onset_max"])
        writer.writeheader()
        writer.writerows(run_stats)
    print(f"Saved {out2}")

    # Save global vs topk
    out3 = Path("outputs/evaluation/fault15_global_vs_topk.csv")
    with open(out3, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type","simulationRun","window_idx","global_score","top1","top3_mean","top5_mean","top10_mean"])
        writer.writeheader()
        writer.writerows(global_vs_topk_rows)
    print(f"Saved {out3}")

    # Rankings
    # Define rankings
    rankings = {}
    # A: largest fault15 mean
    rankings["A_fault15_mean"] = sorted(sensor_rows, key=lambda x: x["fault15_mean"], reverse=True)
    # B: largest ratio
    rankings["B_mean_ratio"] = sorted(sensor_rows, key=lambda x: x["mean_ratio"], reverse=True)
    # C: largest post mean diff
    rankings["C_post_mean_diff"] = sorted(sensor_rows, key=lambda x: x["post_mean_difference"], reverse=True)
    # D: largest p95 ratio
    rankings["D_p95_ratio"] = sorted(sensor_rows, key=lambda x: x["p95_ratio"], reverse=True)
    # E: consistent run-level separation - for each sensor, count runs where post_onset_mean > normal p95
    # Compute normal p95 per sensor
    normal_p95_per_sensor = {r["sensor"]: r["normal_p95"] for r in sensor_rows}
    # Count for fault15
    df_runs = pd.DataFrame(run_stats)
    # For fault15 only
    fault_runs_df = df_runs[df_runs["type"]=="fault15"]
    # For each sensor, count runs elevated
    elev_counts = {}
    for sensor in canonical:
        thresh = normal_p95_per_sensor[sensor]
        # post_onset_mean > thresh
        vals = fault_runs_df[fault_runs_df["sensor"]==sensor]["post_onset_mean"]
        elev_counts[sensor] = int((vals > thresh).sum())
    rankings["E_consistent_elevation"] = sorted(sensor_rows, key=lambda x: elev_counts[x["sensor"]], reverse=True)

    out4 = Path("outputs/evaluation/fault15_sensor_rankings.txt")
    with open(out4, "w") as f:
        for name, ranked in rankings.items():
            f.write(f"\n=== Ranking {name} (top 15) ===\n")
            f.write(f"{'Rank':<4} {'Sensor':<12} {'Normal':<10} {'Fault15':<10} {'Ratio':<8} {'ElevRuns':<10}\n")
            for i, r in enumerate(ranked[:15], 1):
                elev = elev_counts[r["sensor"]] if name=="E_consistent_elevation" else "-"
                # Choose value to display based on ranking
                if name=="A_fault15_mean":
                    f.write(f"{i:<4} {r['sensor']:<12} {r['normal_mean']:<10.4f} {r['fault15_mean']:<10.4f} {r['mean_ratio']:<8.2f} {elev_counts[r['sensor']]:<10}\n")
                elif name=="B_mean_ratio":
                    f.write(f"{i:<4} {r['sensor']:<12} {r['normal_mean']:<10.4f} {r['fault15_mean']:<10.4f} {r['mean_ratio']:<8.2f} {elev_counts[r['sensor']]:<10}\n")
                elif name=="C_post_mean_diff":
                    f.write(f"{i:<4} {r['sensor']:<12} {r['normal_mean']:<10.4f} {r['fault15_post_mean']:<10.4f} {r['post_mean_difference']:<8.4f} {elev_counts[r['sensor']]:<10}\n")
                elif name=="D_p95_ratio":
                    f.write(f"{i:<4} {r['sensor']:<12} {r['normal_p95']:<10.4f} {r['fault15_p95']:<10.4f} {r['p95_ratio']:<8.2f} {elev_counts[r['sensor']]:<10}\n")
                else:
                    f.write(f"{i:<4} {r['sensor']:<12} {r['normal_mean']:<10.4f} {r['fault15_post_mean']:<10.4f} {r['mean_ratio']:<8.2f} {elev:<10}\n")
    print(f"Saved {out4}")

    # Print top 10 overall by post separation
    print("\nTop Fault-15 sensors by post-onset separation")
    print(f"{'Rank':<4} {'Sensor':<12} {'Normal':<10} {'Fault15Post':<12} {'Ratio':<8} {'RunsElev':<10}")
    top_post = rankings["C_post_mean_diff"][:10]
    for i, r in enumerate(top_post, 1):
        print(f"{i:<4} {r['sensor']:<12} {r['normal_mean']:<10.4f} {r['fault15_post_mean']:<12.4f} {r['mean_ratio']:<8.2f} {elev_counts[r['sensor']]:<10}/500")

    # Determine global vs topk dilution
    # Compare global vs topk means for fault15 post-onset windows
    # Use fault_post scores vs topk
    # Compute mean global vs topk for fault post
    # Use global_vs_topk_rows for fault15 post windows (need to filter post windows)
    # For simplicity, use fault_post_per_sensor_all already aggregated: topk not needed, use per-window topk from global_vs_topk_rows
    # Filter global_vs_topk for fault15 and post windows (we have window_idx, need to know if post)
    # For each fault15 run, post windows are those with start+60>160, which we already have as fault_post_per_sensor_all
    # Instead, compute dilution via sensor_rows: compare global mean vs top sensor means
    # Use fault15_post mean vs top sensor means from rankings
    # For illustration, compute top 3 sensor mean for fault15 post
    top3_sensors = [r["sensor"] for r in rankings["C_post_mean_diff"][:3]]
    # Get their post means
    top3_means = [next(x for x in sensor_rows if x["sensor"]==s)["fault15_post_mean"] for s in top3_sensors]
    print(f"\nTop 3 sensors post mean: {top3_sensors} -> {top3_means} mean {np.mean(top3_means):.4f}")
    print(f"Global fault15 post mean (avg over all sensors) approx: {np.mean([r['fault15_post_mean'] for r in sensor_rows]):.4f}")
    print(f"Normal global mean: {np.mean([r['normal_mean'] for r in sensor_rows]):.4f}")

    # Conclusion
    # Heuristic: if top1 ratio >2 and top3 mean >2* global, then strong dilution
    top1_ratio = rankings["B_mean_ratio"][0]["mean_ratio"]
    top3_mean_val = np.mean(top3_means)
    global_post = np.mean([r["fault15_post_mean"] for r in sensor_rows])
    if top1_ratio > 2.0 and top3_mean_val > 2 * global_post:
        conc = "A. STRONG SENSOR-LEVEL SIGNAL"
    elif top1_ratio > 1.5 and top3_mean_val > 1.5 * global_post:
        conc = "B. MODERATE SENSOR-LEVEL SIGNAL"
    else:
        conc = "C. NO SENSOR-LEVEL SIGNAL"
    print(f"\nOverall conclusion: {conc}")
    print(f"  Top1 ratio {top1_ratio:.2f}, top3 mean {top3_mean_val:.4f} vs global post {global_post:.4f}")

    # Also answer the 7 questions
    print("\n--- Final interpretation ---")
    print("1. Is Fault 15 abnormal in specific sensors? See top rankings above")
    print("2. Which sensors? Top 3 above")
    print("3. How consistently? See RunsElev counts")
    print("4. Mainly post-onset? Compare pre vs post in run_stats")
    print("5. Is global MSE diluting? Compare global vs topk")
    print("6. Does sensor-level provide better separation? See ratios")
    print("7. Next diagnostic: sensor z-score, temporal, persistence, sensor-aware scoring, or architecture - to be decided after seeing results")

if __name__ == "__main__":
    main()
