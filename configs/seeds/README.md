# LSTM detector seed configurations (Phase 1, task 4)

Three copies of `configs/config_a5000.yaml` that differ only in `seed`
(42 / 123 / 999). They are written so a reviewer can see exactly what would be
trained; **nothing here has been trained**. The current artifacts in
`outputs/anomaly_detector` remain the frozen detector used for every number in
`docs/reports/phase_1_report.md`.

## What a seed changes

* `utils.set_seed(config["seed"])` at the start of training (weight init, batch
  order, dropout).
* The train/validation/test partition recorded in
  `data/processed/manifests/detector_split.json` is reuse-safe but is itself
  seeded; regenerate it if the split is expected to move with the seed.
* The LSTM threshold (`anomaly_detector.threshold`, percentile 99.5 of
  validation scores) moves with the trained weights, so each seed needs its own
  threshold block. Copy the value out of the seed's `threshold.json` into a
  per-seed copy of `configs/detector_thresholds_v2.json` before evaluating.

## Commands (not run)

```bash
# 1. one-off data prep (shared by all seeds; already done on this checkout)
python scripts/prepare_tep.py --config configs/seeds/config_a5000_seed42.yaml

# 2. LSTM autoencoder, one seed at a time
python scripts/train_anomaly_detector.py --config configs/seeds/config_a5000_seed42.yaml
python scripts/train_anomaly_detector.py --config configs/seeds/config_a5000_seed123.yaml
python scripts/train_anomaly_detector.py --config configs/seeds/config_a5000_seed999.yaml

# 3. dynamic detectors (CVA/DPCA) are deterministic given the normal data, but
#    their thresholds are re-fit per seed so FAR is comparable
python scripts/train_dynamic_detector.py --config configs/seeds/config_a5000_seed42.yaml

# 4. evaluation, per seed, with that seed's threshold file
python scripts/evaluate_detector_v2.py \
    --config configs/seeds/config_a5000_seed42.yaml \
    --thresholds configs/detector_thresholds_seed42.json \
    --protocol A --component val
```

Each seed writes into the same `outputs/anomaly_detector` directory, so move
or rename the artifact directory between seeds (for example
`outputs/anomaly_detector_seed42`) and point `anomaly_detector.model_dir` at it
in the seed config before training the next one.

## Estimated time (CPU, unmeasured)

Anchored on the only measured number in this phase: the frozen model scores a
60x52 window in **0.20 ms** on this CPU (batch 256). A training step costs
roughly 3x a forward pass, and the Training-file partition is about
31,000 windows per epoch (350 runs x 89 windows):

| Step | Estimate |
|---|---|
| One epoch (~31k windows x ~0.6 ms) | ~19 s |
| 60-100 epochs with early stopping | **20-35 min per seed** |
| 3 seeds, sequential | **1-1.75 h** |
| Dynamic detector fit + threshold holdout | 5-10 min per seed |

A GPU (the A5000 the artifacts were trained on) is typically several times
faster, but that factor has not been measured here and should not be quoted.
