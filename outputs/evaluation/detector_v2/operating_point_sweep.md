# Detector operating point v2 — window-level sweep (calibration data only)

Selection: run-level FAR <= 5% on fault-free Testing runs 1-250, then smallest mean
delay on Protocol A validation over the 18-fault observable set. Runs 251-500 are
reported, never used. Protocol A test was not read.

| target window FAR | persistence | thr LSTM | thr CVA Q | thr DPCA SPE | run FAR sel (1-250) | run FAR rep (251-500) | events/run sel | mean FDR (18 obs) | mean det. rate | mean delay (samples) | eligible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1% | 3 | 0.7073 | 199.87 | 25.35 | 8.4% | 9.2% | 0.08 | 98.0% | 100.0% | 19.6 | no |
| 0.1% | 5 | 0.7073 | 199.87 | 25.35 | 2.8% | 4.0% | 0.03 | 98.0% | 100.0% | 19.6 | yes |
| 0.1% | 8 | 0.7073 | 199.87 | 25.35 | 0.8% | 0.8% | 0.01 | 98.0% | 100.0% | 19.6 | yes |
| 0.1% | 12 | 0.7073 | 199.87 | 25.35 | 0.0% | 0.0% | 0.00 | 98.0% | 100.0% | 19.8 | yes |
| 0.5% | 3 | 0.6934 | 196.50 | 24.46 | 27.6% | 26.8% | 0.31 | 98.1% | 100.0% | 18.8 | no |
| 0.5% | 5 | 0.6934 | 196.50 | 24.46 | 15.6% | 14.0% | 0.16 | 98.1% | 100.0% | 18.8 | no |
| 0.5% | 8 | 0.6934 | 196.50 | 24.46 | 4.4% | 6.0% | 0.04 | 98.1% | 100.0% | 18.9 | yes |
| 0.5% | 12 | 0.6934 | 196.50 | 24.46 | 0.4% | 1.2% | 0.00 | 98.1% | 100.0% | 18.9 | yes |
| 1.0% | 3 | 0.6870 | 194.63 | 24.04 | 37.6% | 44.0% | 0.45 | 98.2% | 100.0% | 18.1 | no |
| 1.0% | 5 | 0.6870 | 194.63 | 24.04 | 24.0% | 24.8% | 0.26 | 98.2% | 100.0% | 18.3 | no |
| 1.0% | 8 | 0.6870 | 194.63 | 24.04 | 11.2% | 10.0% | 0.12 | 98.2% | 100.0% | 18.3 | no |
| 1.0% | 12 | 0.6870 | 194.63 | 24.04 | 2.4% | 1.6% | 0.02 | 98.2% | 100.0% | 18.3 | yes |

**Selected:** target window FAR 1.0%, persistence 12. LSTM 0.6870, CVA Q 194.632, DPCA SPE 24.043.

Run-level FAR: 2.4% on the selection half, 1.6% on the reporting half.

## Per-fault FDR at the selected point (Protocol A validation)

| Fault | FDR | run detection rate | mean delay |
|---|---|---|---|
| 1 | 99.8% | - | - |
| 2 | 98.4% | - | - |
| 3 (unobservable) | 4.8% | - | - |
| 4 | 100.0% | - | - |
| 5 | 100.0% | - | - |
| 6 | 100.0% | - | - |
| 7 | 100.0% | - | - |
| 8 | 97.9% | - | - |
| 9 (unobservable) | 7.0% | - | - |
| 10 | 97.1% | - | - |
| 11 | 99.0% | - | - |
| 12 | 99.4% | - | - |
| 13 | 95.4% | - | - |
| 14 | 100.0% | - | - |
| 15 | 97.2% | - | - |
| 16 | 98.2% | - | - |
| 17 | 96.2% | - | - |
| 18 | 94.2% | - | - |
| 19 | 99.7% | - | - |
| 20 | 95.5% | - | - |