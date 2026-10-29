# Detector evaluation v2 - protocol A / test

Source: `data/raw/faults/TEP_Faulty_Testing.csv` (2000 runs, onset 160, window 60/stride 5)
Operating point: {'id': 'detector_v2_window_op_2026-10-30', 'sha256': 'e7fe73eb0b75b4d5703136e6c997e223d2a878c7e36764ed8d6ccf00e559624b', 'source': 'configs/detector_v2_operating_point.json', 'target_window_far': 0.01, 'persistence_windows': 12, 'thresholds': {'lstm_ae': 0.6870414805412293, 'cva_Q': 194.63167266845704, 'dpca_SPE': 24.042938613891604}, 'smoothing': 'none', 'secondary_statistics_at_v1': ['cva_T2', 'cva_Tr', 'dpca_T2']}
Persistence (consecutive alarmed windows): 12
Split label: `faulty_testing` (explicit, not path-inferred)

FDR is the legacy definition: fraction of post-onset windows above threshold.

| Fault | Runs | LSTM FDR | CVA Q FDR | DPCA SPE FDR | Fused FDR | Fused det. rate | Fused delay (mean, samples) | Fused AUROC | Fused PR-AUC | Events/run |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 100 | 99.4% | 99.8% | 99.4% | 99.8% | 100.0% | 5.5 | 1.000 | 1.000 | 1.00 |
| 2 | 100 | 98.3% | 98.3% | 97.6% | 98.4% | 100.0% | 17.0 | 0.994 | 0.999 | 1.00 |
| 3 | 100 | 1.8% | 3.6% | 1.0% | 4.8% | 4.0% | 420.2 | 0.648 | 0.927 | 0.04 |
| 4 | 100 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 4.0 | 1.000 | 1.000 | 1.00 |
| 5 | 100 | 99.9% | 100.0% | 99.6% | 100.0% | 100.0% | 4.0 | 1.000 | 1.000 | 1.00 |
| 6 | 100 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 4.0 | 1.000 | 1.000 | 1.00 |
| 7 | 100 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 4.0 | 1.000 | 1.000 | 1.00 |
| 8 | 100 | 97.4% | 97.8% | 97.1% | 97.8% | 100.0% | 21.8 | 0.990 | 0.999 | 1.00 |
| 9 | 100 | 2.0% | 5.1% | 1.1% | 6.3% | 10.0% | 326.5 | 0.685 | 0.937 | 0.12 |
| 10 | 100 | 76.1% | 97.0% | 90.2% | 97.0% | 100.0% | 27.9 | 0.986 | 0.998 | 1.00 |
| 11 | 100 | 98.8% | 98.9% | 99.1% | 99.0% | 100.0% | 12.1 | 0.996 | 0.999 | 1.00 |
| 12 | 100 | 98.9% | 99.4% | 98.6% | 99.4% | 100.0% | 8.8 | 0.998 | 1.000 | 1.00 |
| 13 | 100 | 94.2% | 95.1% | 94.4% | 95.1% | 100.0% | 43.3 | 0.978 | 0.997 | 1.00 |
| 14 | 100 | 100.0% | 99.7% | 99.9% | 100.0% | 100.0% | 4.0 | 1.000 | 1.000 | 1.00 |
| 15 | 100 | 2.9% | 96.8% | 1.1% | 96.8% | 100.0% | 27.4 | 0.992 | 0.999 | 1.00 |
| 16 | 100 | 77.5% | 98.1% | 95.6% | 98.1% | 100.0% | 19.1 | 0.992 | 0.999 | 1.00 |
| 17 | 100 | 95.7% | 96.0% | 95.9% | 96.1% | 100.0% | 35.7 | 0.981 | 0.998 | 1.00 |
| 18 | 100 | 93.7% | 94.5% | 94.0% | 94.5% | 100.0% | 48.0 | 0.975 | 0.997 | 1.00 |
| 19 | 100 | 97.6% | 99.7% | 97.7% | 99.7% | 100.0% | 6.4 | 0.999 | 1.000 | 1.00 |
| 20 | 100 | 93.8% | 95.4% | 94.3% | 95.5% | 100.0% | 40.8 | 0.979 | 0.997 | 1.00 |

## False alarm rate on fault-free Testing runs (never used for calibration)

| Detector | Threshold (artifact) | FAR window | FAR sample | FAR at 1% window recal | FAR (run) | Events/run |
|---|---|---|---|---|---|---|
| lstm_ae | 0.687 | 1.65% | - | - | 1.0% | 0.01 |
| cva_T2 | 127.5 | 0.00% | 0.98% | - | 0.0% | 0.00 |
| cva_Q | 194.6 | 1.02% | 26.08% | - | 0.4% | 0.00 |
| cva_Tr | 239.8 | 0.00% | 0.94% | - | 0.0% | 0.00 |
| dpca_T2 | 124.3 | 0.00% | 1.01% | - | 0.0% | 0.00 |
| dpca_SPE | 24.04 | 0.69% | 31.39% | - | 0.6% | 0.01 |
| pca_t2 | 38.68 | 1.21% | 17.01% | - | 6.4% | 0.06 |
| pca_spe | 6.331 | 1.24% | 26.62% | - | 1.8% | 0.02 |
| iforest | 0.4854 | 1.02% | 5.06% | - | 4.8% | 0.05 |
| fused | 1 | 2.33% | - | - | 2.0% | 0.02 |

## Fused FDR at the 1% window operating point

| Fault | Fused FDR (artifact thr) | Fused FDR (window-calibrated thr) |
|---|---|---|
| 1 | 99.8% | - |
| 2 | 98.4% | - |
| 3 | 4.8% | - |
| 4 | 100.0% | - |
| 5 | 100.0% | - |
| 6 | 100.0% | - |
| 7 | 100.0% | - |
| 8 | 97.8% | - |
| 9 | 6.3% | - |
| 10 | 97.0% | - |
| 11 | 99.0% | - |
| 12 | 99.4% | - |
| 13 | 95.1% | - |
| 14 | 100.0% | - |
| 15 | 96.8% | - |
| 16 | 98.1% | - |
| 17 | 96.1% | - |
| 18 | 94.5% | - |
| 19 | 99.7% | - |
| 20 | 95.5% | - |

Calibration/FAR disjointness: `{"far_source_file": "data\\raw\\normal\\TEP_FaultFree_Testing.csv", "far_runs": [1, 500], "n_far_runs": 500, "calibration_sources": {"data\\raw\\normal\\TEP_FaultFree_Training.csv": 500}, "overlapping_files": [], "overlapping_runs": [], "ok": true}`
