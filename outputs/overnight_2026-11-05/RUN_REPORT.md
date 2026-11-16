# Overnight run 2026-11-05

Started by the project owner's approval of every decision at its recommended value (`docs/judging/confirm/signed.json`). Stored text only: nothing generated, no model loaded on the CPU chain.

## Stages

| stage | status | detail |
|---|---|---|
| N1 | ok | {"rows": 520, "calibration": "46/46 at 1.00", "contradiction_rate": 0.0, "parity_disagreements": 1390, "gate": "contradiction <= 2% -> pass"} |
| N4 | ok | {"rows": 520, "cells": ["A", "B", "C", "D", "M"], "seam": {"C_b_numeric_leaked": 0, "M_a_numeric_leaked": 0, "C_donor_unused": 88, "M_donor_unused": 584, "D_fallback_rows": 142}, "mcnemar": {"b_only": 60, "a_only": 17, "p_exact": 0.0}, "gate": "520 rows per cell -> pass", "ceiling": "FAIL: D fell back on 27.3% of rows, above the 25% ceiling; report D as A with extra steps"} |
| N5 | ok | {"invariance": {"items": 520, "identical_defect_profile_all_orders": 368, "rate": 0.7077}, "dialogue_rows": 97, "T_deferred": true} |
| N6 | ok | {"files": 16} |
| N2 | deferred | {"reason": "the arm frontier needs the evaluate_detector_v2 scoring path (window aggregation, per-arm threshold derivation) which is not re-derivable from the cache without changing the protocol's rule; left for the morning rather than approximated"} |
| N3 | deferred | {"reason": "the merged classifier row needs the stored per-item val predictions and labels joined; not attempted unattended"} |
| G1 | not_run | {"reason": "D's retries need generation on the adapter; deliberately left after the CPU chain to avoid two GPU jobs at once"} |
| G2 | launched | {"detail": "E seeds 43 and 44 via scripts/overnight_gpu.py; log in gpu_E.log"} |

## Human work now due

1. Freeze the 46 cases (`n1_calibration.json`) - 5 min.
2. Hand-check 30 mechanism sentences (`n5_examples.md`) - 15-20 min.
3. Restore or re-score `audit_ab_100_v5_scored_rater1.csv` - 10 min decision.
4. Two-rater preference study, 120 items x 3 pairings, kappa >= 0.60 - 2 days.
5. Release the one-shot confirmatory pass after reading `n4_cells.json` - 10 min.
6. Pull a non-Qwen judge model before any judge run (D5) - 15 min.
