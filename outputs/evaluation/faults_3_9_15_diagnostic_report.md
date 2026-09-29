# Faults 3,9,15 vs Normal - Temporal & Cross-Sensor Diagnostic

Threshold 0.6874, Window 60/5, 52 sensors

## Per-Run Metrics (mean ± std)

| Fault | Grad Mean | Corr Change | z_post_max | Score Max | n_above | max_consec |
|-------|-----------|-------------|------------|-----------|---------|------------|
| Normal | 1.7066±0.04 | 0.1088±0.01 | 0.30±0.11 | 0.683±0.02 | 1.3 | 1.0 |
| Fault 3 | 1.7034±0.04 | 0.1077±0.01 | 0.34±0.10 | 0.678±0.02 | 1.1 | 0.8 |
| Fault 9 | 1.7039±0.04 | 0.1081±0.01 | 0.31±0.11 | 0.679±0.02 | 1.1 | 0.9 |
| Fault 15 | 1.7093±0.04 | 0.1081±0.01 | 0.31±0.11 | 0.684±0.02 | 1.8 | 1.4 |

## Interpretation
- Grad Mean: mean absolute temporal gradient per sensor (how much sensors change per sample)
- Corr Change: mean absolute change in cross-sensor correlation (pre vs post onset)
- z_post_max: maximum |z| across sensors post-onset
- If faults 3,9,15 have similar grad/corr_change to normal, they are not temporal/relationship anomalies
- If they have higher grad/corr_change, they are temporal/relationship anomalies that global MSE misses
