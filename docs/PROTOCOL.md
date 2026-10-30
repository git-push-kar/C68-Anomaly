# Benchmark protocol — TEP RCA adapter (v2, Phase 0)

**Repo:** `C:\Users\Admin\Desktop\anomaly` · branch `benchmark-upgrade` · base `4d4b3be`
**Status:** Phase 0. Every success criterion below is a **PROPOSED placeholder**
awaiting approval; none is a commitment yet.

---

## 1. Fault scope

| Group | Faults | Treatment |
|---|---|---|
| Individually diagnosable | 1–15 | The adapter must name the fault. |
| Open set | 16–20 | KB names are "Unknown disturbance in …". Accepted label is `UNCLASSIFIED_OPEN_SET`, not the number. |
| Unobservable | 3, 9 | Scored and reported separately; never dropped, never pooled into the headline number. |
| Synthetic only | 21, 22 | No real run exists. Non-headline; labelled `synthetic` wherever shown. |

Source of truth: [`scripts/fault_ids.py`](../scripts/fault_ids.py) (label space,
open-set and unobservable flags) and [`data/splits_v2.json`](../data/splits_v2.json)
(source file and split ranges).

**One source file for every LLM event:** `data/raw/faults/TEP_Faulty_Testing.csv`
(960 samples/run, 0-based onset 160, 500 runs/fault). The Training file is
excluded because it is a different run shape (500 samples/run, onset 20) — see
§4 of the phase-0 report. `windowing.block_samples: 500` and `prepare_tep.py`
encode the Training shape and must not be pointed at the Testing file unchanged.

---

## 2. Split protocols

Both protocols are defined in `data/splits_v2.json`, built by
`scripts/build_splits_v2.py`, and validated by `scripts/check_splits_v2.py`.
Unique key: **`(source_file, faultNumber, simulationRun)`**. Each split records
a SHA-256 over its expanded sorted key list; the validator recomputes it.

### Protocol A — seen faults, run-disjoint (in-distribution)

| Split | Faults | Runs | Keys |
|---|---|---|---|
| train | 1–20 | 1–300 | 6,000 |
| val | 1–20 | 301–400 | 2,000 |
| test | 1–20 | 401–500 | 2,000 |

Every fault is seen in training; only the simulation runs are held out. This is
the *in-distribution* number and must always be labelled as such.

### Protocol B — leave-faults-out (generalisation)

| Fold | Unseen `test` (held out, all 500 runs) | `test_seen` (kept faults, runs 401–500) | Train / val (kept faults) | Keys (test / test_seen) |
|---|---|---|---|---|
| `fold_1_cooling_loop_holdout` | 4, 11, 15 | 1,2,3,5,6,7,8,9,10,12,13,14,16–20 | 1–300 / 301–400 | 1,500 / 1,700 |
| `fold_2_valve_sticking_family` | 14, 15 | 1–13, 16–20 | 1–300 / 301–400 | 1,000 / 1,800 |

Held-out faults are excluded from train and val entirely and form the **unseen**
component `test`. Runs 401–500 of the kept faults form **`test_seen`**, the seen
component. `test_seen` faults must be a subset of the fold's train faults and
its runs (401–500) are disjoint from train (1–300) and val (301–400); the
validator enforces both rules, and `test` stays fault-disjoint from every other
component.

**Abstention is scored on `test`** — declining to name a fault on an unseen fault
is correct behaviour. **False abstention is scored on `test_seen`** — declining
on a fault the model has trained on is an error. Both components are reported,
and `test` remains the generalisation number.

KB support for the fold design: faults 4 ↔ 14 and 5 ↔ 15 are explicit
differential pairs in the KB (each names the other as "CRITICAL DIFFERENTIAL").
Fault 11's KB differentials point to 4 and 14, **not** to 12; 12 is kept in
train only as the condenser-loop structural counterpart, and fold 1 is designed
so that 14 (the KB sibling of 4) and 5 (the KB sibling of 15) remain seen.

---

## 3. Run configuration

| Item | Value |
|---|---|
| Seeds | **42, 123, 999** — one full training run per seed; all three reported (mean ± sd) |
| Split seed | 42 (the split is fixed and hashed; it does not vary per training seed) |
| Decoding | **greedy**: `do_sample=False`, `temperature/top_p/top_k` inert, `repetition_penalty=1.12` (matches `llm/inference.py:436-443`) |
| Max new tokens | 512 for follow-ups, 768 for reports (current code defaults) |
| Events per run | **one event per simulation run** |
| Training budget | **100–150 runs per fault** → 2,000–3,000 events (Protocol A), 1,700–2,550 (fold 1), 1,800–2,700 (fold 2) |
| Checkpoint rule | select on **validation macro-F1 measured by generation on the seen faults of that fold's train set**; the held-out faults are reported, never used for selection or early stopping |
| Precision | out of scope for Phase 0 (no GPU); record whatever the training run uses |

Checkpoint selection must never look at held-out faults. If validation macro-F1
is undefined for a fold (e.g. a class with no validation items), fall back to
validation loss and say so in the report.

---

## 4. Research questions

Every criterion below is a **placeholder until Phase 4**. They will then be
replaced by relative gates calibrated on validation only — never on `test`, and
never on `test_seen`.

### RQ1 — Does the adapter name seen faults correctly under run-level holdout?

- **Metric:** macro-F1 as the headline, computed over the **observable set**
  (faults whose fused-detector FDR ≥ 5% on Protocol A validation); exact
  fault-match accuracy on the mapped label space as a secondary (faults 16–20
  scored against `UNCLASSIFIED_OPEN_SET`; 21/22 excluded). All-fault numbers are
  reported alongside the observable-set number, never instead of it.
- **Headline set (Phase 2B owner instruction):** the **v2** set — 19 faults,
  fault **3** excluded. The deployed-v1 set (18 faults, {3, 9}) is reported as a
  labelled comparison row. Fault **9** is marked `marginal` wherever it appears
  in a table (7.0% at v2, the weakest member of the set). See §8.3.
- **Data:** Protocol A `test` (faults 1–20, runs 401–500).
- **Success criterion:** `PROPOSED: exact match ≥ 0.70 and macro-F1 ≥ 0.65 (macro).`

### RQ2 — Does it generalise to faults never seen in training?

- **Metric:** macro-F1 over the held-out faults of each fold, plus per-fault
  exact-match, reported for fold 1 and fold 2 separately (never pooled);
  `test_seen` accuracy is reported next to it so false abstention is visible.
- **Data:** Protocol B `test` for both folds.
- **Success criterion:** `PROPOSED: macro-F1 ≥ 0.40 on each fold, and fold-1 exact match on faults 4 and 15 ≥ 0.30.`

### RQ3 — Are follow-up answers numerically grounded in the evidence?

- **Metric:** fraction of reference numbers reproduced verbatim (≥ 0.6 per
  answer counts as grounded), foreign-sensor rate, and markdown-table presence
  for the numeric intents; computed by a rewritten checker, not the current
  `evaluate_grounding`.
- **Data:** follow-up items generated from Protocol A `test` events, one item
  per live intent (§6).
- **Success criterion:** `PROPOSED: number-recall rate ≥ 0.85, foreign-sensor rate ≤ 0.02, table rate ≥ 0.95.`

### RQ4 — Does the system answer the intent the operator asked?

- **Metric:** intent-agreement rate between the expected intent and the live
  route (`deterministic` intent label, or `llm` when no intent matched);
  separately, format compliance per intent class.
- **Data:** the same follow-up items as RQ3, plus one adversarial item per
  intent that must *not* be routed to it.
- **Success criterion:** `PROPOSED: intent agreement ≥ 0.90; format compliance ≥ 0.95.`

### RQ5 — Is the result reproducible at acceptable cost?

- **Metric:** wall-clock train time and peak VRAM per seed; inference latency
  per answer; spread of RQ1/RQ2 metrics across the three seeds.
- **Data:** the three training runs and a fixed 200-item inference sample.
- **Success criterion:** `PROPOSED: RQ1 spread ≤ 0.05 across seeds; ≤ 6 s per answer on the A5000; peak VRAM within the 24 GB budget.`

---

## 5. Metrics that must not be cited

| Old metric | Why it cannot be cited |
|---|---|
| `severity_accuracy` | The training system prompt states the fault-number → severity mapping (`llm/dataset.py:42-49`) and every v0 row's severity equals the KB value. It measures recall of a leaked lookup. Severity is now **computed by code**, not emitted by the model. |
| `fault_classification_accuracy` | `_fault_class_match` accepts the *subsystem* as a match (`llm/evaluate.py:272-283`), and the prompt hands the model the candidate subsystem (`llm/dataset.py:215`). It therefore scores a hint, not a diagnosis. |
| `evidence_consistency` / `hallucination_rate` | The allowed set contains only `display_name` (`llm/evaluate.py:117`), so quoting a legitimate raw tag (`XMEAS_1`) is counted as a hallucination, while a broad `json_keys` allow-list excuses many tokens (`llm/evaluate.py:122`). The number is neither precision nor recall. |
| `root_cause_accuracy` (old definition) | Matching is substring **or a single overlapping token** with `top_k=1` (`llm/evaluate.py:101`, `_overlap` at `263`), so "feed" alone can pass. Replaced by exact label matching in RQ1/RQ2. |

These four may appear only in a "legacy, not comparable" appendix with this
explanation attached.

---

## 6. Follow-up intents under test

The base is the 11 live dispatcher intents (`docs/RESPONSE_SPEC.md` §2). Three
proposed additions (`hypothesis_comparison`, `out_of_scope_refusal`,
`clarification_reply`) are **not implemented** and must not be scored until they
exist in the dispatcher.

Item generation must use the operator question only — the live router receives
`user_question` (`main.py:427`), not the telemetry block. Replaying v0's
grounded_qa questions verbatim routes 100% to `sensor_scores` because the
telemetry text itself contains "sensor" and "score"; that is a dataset artefact,
not a router bug.

---

## 7. Owner decisions (recorded 2026-10-29)

These are decisions, not proposals. Phase 0 recorded them; they are open to
revision only by explicit change of mind.

1. **Success criteria stay placeholders until Phase 4.** They will then be set as
   relative gates calibrated on validation only — never on `test` or `test_seen`.
2. **Headline macro-F1 is computed over the observable set**: faults whose
   fused-detector FDR is ≥ 5% on Protocol A validation. All-fault numbers are
   reported alongside, never instead.
3. **One event per simulation run**, and a **training budget of 100–150 runs per
   fault** (2,000–3,000 events for Protocol A; 1,700–2,550 fold 1; 1,800–2,700
   fold 2). The budget is a cap, not a target to exceed.
4. **Open-set scoring is per variant.** When a candidate menu entry is present in
   the prompt, the fault ID is scored; when it is removed, `UNCLASSIFIED_OPEN_SET`
   is scored. For faults 16–20 a specific number is **wrong**; "adjacent" credit
   is reported as a separate secondary column only, never in the headline.
5. **Severity is computed in code**, not emitted by the adapter. The
   fault-number severity rubric in the prompt is removed in dataset v2.
6. **`window_size 60 / stride 5` is kept** for comparability with the existing
   detector and recorded as a limitation: it was calibrated on the 500-sample
   Training runs, not the 960-sample benchmark runs.

---

## 8. Detector operating point and the observable set (Phase 1)

Measured by `scripts/evaluate_detector_v2.py` on Protocol A (`val` = runs
301–400, `test` = runs 401–500, one run per fault, 960 samples, onset 160).
Full numbers: `docs/reports/phase_1_report.md` and
`outputs/evaluation/detector_v2/`.

### 8.1 Definitions used here

- **Window label:** a window is positive if it *contains* a post-onset sample
  (`start + 60 > 160`). The legacy evaluator labelled by `start >= 160`, which
  counts 12 windows per run as negatives while they already contain fault data,
  and then books their alarms as pre-onset false alarms. That is why the
  published `false_before_onset` was 21 for every strong fault.
- **FDR:** fraction of labelled-positive windows above threshold. This is a
  window-level detection fraction, not a run-level detection rate; both are
  recorded per fault.
- **Delay:** samples from onset to the completing sample of the first confirmed
  alarm (3 consecutive windows above threshold). The legacy delay was measured
  to the window *start*, so it is smaller by up to 59 samples.

### 8.2 The frozen detector's operating point

| Quantity | LSTM | CVA Q | DPCA SPE | PCA T² | Isolation Forest |
|---|---|---|---|---|---|
| FAR, window | 1.66% | 0.00% | 0.00% | 0.00% | 0.04% |
| FAR, sample | – | 0.94% | 0.96% | 1.07% | 1.13% |
| FAR, run (event) | 30.6% | 0.0% | 0.0% | 0.0% | 0.8% |

FAR is measured on all 500 `TEP_FaultFree_Testing` runs, which are disjoint from
every calibration run (`far_disjoint_from_calibration.ok = true`, verified by
the script, not by assumption).

**Consequence for the benchmark:** at the frozen thresholds the fused detector
raises an event on **30.6% of fault-free runs** (0.35 events per normal run).
The dynamic detectors and the baselines were thresholded per *sample* at p99;
with a 60-sample window mean their window FAR collapses to ~0, so the OR-gate is
in practice the LSTM alone. Fault 15 swings from 7.7% to 97.2% fused FDR when
the same detectors are put at a common 1% *window* FAR (thresholds fitted on
the dynamic threshold holdout, `TEP_FaultFree_Training` runs 401–500 — never on
the FAR set).

### 8.3 Observable set — HEADLINE is the frozen v2 point (superseded 2026-10-30)

Rule (owner decision 2): observable = fused-detector FDR >= 5% on Protocol A
`val`.

> **Supersedes the Phase 1 wording.** Until Phase 2B this section made the
> deployed (v1) point the headline, because every Phase 1 table had been produced
> at v1. The Phase 2B instruction sets the **v2** set as the RQ1 headline and
> demotes v1 to a labelled comparison row. The reason is the same evidence, read
> the other way round: v1 was not a *chosen* operating point but the residual of a
> deleted threshold artifact, and its fault-9 result (2.5%) is below any
> defensible sensitivity floor. Reporting v2 as headline while quoting v1 beside
> it keeps both facts visible.

| Operating point | Observable set | Excluded | Fault 9 | Fault 15 |
|---|---|---|---|---|
| **v2, frozen (HEADLINE, Phase 2B onward)** | **19 faults** | **{3}** | 7.0% `marginal` | 97.2% |
| v1, deployed (comparison row only) | 18 faults | {3, 9} | 2.5% | 7.7% |

Fault 15 sits just above the line at v1 (7.7%, `test` 8.0%) and is fully
observable at v2; its v1 number was the weakest justification for excluding it.
Fault 3 fails the rule at both points (2.4% / 4.8%) and is the only exclusion
left. Fault 9 is in the headline set but is marked `marginal` everywhere: at
7.0% it clears the floor by 2 points and its detection is concentrated in the
late portion of each run (median event-open delay 369 samples for faults 3/9
versus 59-99 for the observable families), so a reader should treat its per-fault
number as the least stable cell in the table.

The two sets differ only in faults 9 and 15, which are exactly the faults whose
detector path (dynamic vs LSTM) was mis-thresholded.

**Severity is not a benchmark metric.** `llm/response/severity.py` exists only as
a cross-check of `evidence/process_limits.compute_deterministic_severity`; the
agreement rate between the two rules is a diagnostic of the code path, is never
computed from model output, and is not reported in any RQ table.

### 8.4 Frozen v2 operating point (A3) — SECONDARY, never mixed into v1 tables

File: `configs/detector_v2_operating_point.json`,
sha256 `e7fe73eb0b75b4d5703136e6c997e223d2a878c7e36764ed8d6ccf00e559624b`
(recorded beside it in `detector_v2_operating_point.sha256.json`, because a file
cannot hash itself). Artifact hashes are inside the same file.

| Quantity | Value |
|---|---|
| window FAR target | 1.0% per detector |
| LSTM AE threshold | 0.68704 (p99.0 of normal validation windows) |
| CVA Q threshold | 194.632 (window level, Training 401-500) |
| DPCA SPE threshold | 24.043 (window level, Training 401-500) |
| persistence | **12 consecutive alarmed windows** |
| smoothing | none (thresholds were fitted unsmoothed) |
| fusion | OR of LSTM AE and CVA Q (primary dynamic), unchanged |

The secondary statistics `cva_T2`, `cva_Tr`, `dpca_T2` stay at their v1 values:
they never enter the OR-gate and never appear in a headline table, so the file
defines exactly the three numbers that are quoted.

Selection used calibration data only: thresholds from `TEP_FaultFree_Training`
runs 401-500 (dynamic) and the 75 normal validation runs (LSTM, manifest
`detector_split.json`); run-level FAR measured on `TEP_FaultFree_Testing`
runs 1-250. **Protocol A `test` was not read for any choice**, and Protocol A
`val` FDR is reported as a trade-off only.

### 8.5 Operating-point sweep (A2) — the Pareto table

12 combinations (target window FAR x persistence), all scored on the same
cached window series. "run FAR sel" is the selection half (Testing 1-250),
"rep" the reporting half (251-500).

| target | persistence | LSTM thr | CVA Q thr | run FAR sel | run FAR rep | mean FDR (18) | mean delay | eligible |
|---|---|---|---|---|---|---|---|---|
| 0.1% | 3 | 0.67798 | 178.16 | 8.4% | 9.2% | 98.0% | 19.6 | no |
| 0.1% | 5 | 0.67798 | 178.16 | 2.8% | 4.0% | 98.0% | 19.6 | yes |
| 0.1% | 8 | 0.67798 | 178.16 | 0.8% | 0.8% | 98.0% | 19.6 | yes |
| 0.1% | 12 | 0.67798 | 178.16 | 0.0% | 0.0% | 98.0% | 19.8 | yes |
| 0.5% | 3 | 0.68183 | 186.62 | 27.6% | 26.8% | 98.1% | 18.8 | no |
| 0.5% | 5 | 0.68183 | 186.62 | 15.6% | 14.0% | 98.1% | 18.8 | no |
| 0.5% | 8 | 0.68183 | 186.62 | 4.4% | 6.0% | 98.1% | 18.9 | yes |
| 0.5% | 12 | 0.68183 | 186.62 | 0.4% | 1.2% | 98.1% | 18.9 | yes |
| **1.0%** | **12** | **0.68704** | **194.63** | **2.4%** | **1.6%** | **98.2%** | **18.3** | **selected** |
| 1.0% | 3 | 0.68704 | 194.63 | 37.6% | 44.0% | 98.2% | 18.1 | no |
| 1.0% | 5 | 0.68704 | 194.63 | 24.0% | 24.8% | 98.2% | 18.3 | no |
| 1.0% | 8 | 0.68704 | 194.63 | 11.2% | 10.0% | 98.2% | 18.3 | no |

The selection half and the reporting half agree within 5.6 points everywhere and
within 1.5 points for the selected combination, so the choice is not fitted to
the half it was measured on. Full table with all columns:
`outputs/evaluation/detector_v2/operating_point_sweep.md`.

### 8.6 v1 vs v2 side by side (A4)

Protocol A `val`, 2,000 runs per point, same split, same labels, same code.

| Fault | v1 FDR | v2 FDR | v2 test FDR | v1 delay | v2 delay |
|---|---|---|---|---|---|
| 1 | 99.5% | 99.8% | 99.8% | 8.4 | 5.3 |
| 2 | 98.3% | 98.4% | 98.4% | 17.6 | 16.6 |
| 3 | 2.4% | 4.8% | 4.8% | 370.4 | 407.1 |
| 4 | 100.0% | 100.0% | 100.0% | 4.4 | 4.0 |
| 5 | 100.0% | 100.0% | 100.0% | 4.0 | 4.0 |
| 6 | 100.0% | 100.0% | 100.0% | 4.0 | 4.0 |
| 7 | 100.0% | 100.0% | 100.0% | 4.0 | 4.0 |
| 8 | 97.5% | 97.9% | 97.8% | 24.2 | 20.8 |
| 9 | 2.5% | 7.0% | 6.3% | 339.1 | 378.1 |
| 10 | 96.3% | 97.1% | 97.0% | 33.1 | 27.4 |
| 11 | 98.8% | 99.0% | 99.0% | 13.3 | 11.9 |
| 12 | 99.2% | 99.4% | 99.4% | 10.2 | 9.1 |
| 13 | 94.6% | 95.4% | 95.1% | 47.6 | 41.2 |
| 14 | 100.0% | 100.0% | 100.0% | 4.0 | 4.0 |
| 15 | 7.7% | **97.2%** | 96.8% | 315.2 | **24.9** |
| 16 | 97.3% | 98.2% | 98.1% | 25.5 | 18.7 |
| 17 | 95.8% | 96.2% | 96.1% | 37.4 | 35.2 |
| 18 | 93.5% | 94.2% | 94.5% | 55.3 | 52.0 |
| 19 | 99.1% | 99.7% | 99.7% | 11.6 | 6.8 |
| 20 | 94.9% | 95.5% | 95.5% | 44.6 | 40.5 |
| **mean (18 observable)** | **92.9%** | **98.2%** | **98.2%** | **36.9** | **18.3** |

Fault 15 is the whole story: 7.7% to 97.2% FDR and 315 to 25 samples of delay,
because the dynamic head finally sits at the operating point its threshold
claimed. Faults 3 and 9 do not move materially (they are compensated
disturbances), which is the expected null result.

Run-level FAR: fused **30.6% -> 2.0%** on the 500 fault-free Testing runs; the
selection half / reporting half split was 2.4% / 1.6%. Baselines were
recalibrated to the same 1% window FAR for the comparison (PCA T2 window 1.21% /
run 6.4%, PCA SPE 1.24% / run 1.8%, Isolation Forest 1.02% / run 4.8%), so the
baseline rows are not handicapped by their old per-sample thresholds.

One caveat that must travel with any v2 number: `far_sample` is meaningless at a
window-calibrated threshold (CVA Q reads 26.1% "sample FAR" against a threshold
that was set on window means). Sample-level FAR is only comparable at v1.

### 8.7 Event-level statistics at v2 (A5) — input to the event-builder rule

From the same window series, segmented with the frozen persistence of 12:

| Population | events/run | runs with an event |
|---|---|---|
| fault-free Testing 1-500 | 0.020 | 10/500 = 2.0% |
| faulty, Protocol A val (all 20) | 0.91 | 1,724/2,000 |

- **Events that start before the onset: 0 of 1,824 (0.0%).** With persistence 12
  and only 21 pre-onset windows in a 960-sample run at stride 5, twelve
  consecutive alarms before the fault is injected is effectively impossible.
  The v1 "pre-onset" counts were an artefact of a 3-window confirmation plus the
  legacy labelling bug (`phase_1_report.md` §10.1).
- **Fault 15: 100/100 runs produce a post-onset event, 1.00 events/run.** At v1
  the same fault showed 315-sample mean delay and 84% detection.
- Consequence for the event builder: one event per run is the correct target,
  and a pre-onset event should be suppressed rather than explained. No
  fault-free run produces more than one event (histogram: 490 runs with 0, 10
  runs with 1).

### 8.8 3-seed LSTM training — commands only, nothing run

Not trained in this patch (CPU only, and the detector is frozen by design).
`configs/seeds/config_a5000_seed{42,123,999}.yaml` exist and are identical
except `seed`:

```bash
python scripts/train_anomaly_detector.py --config configs/seeds/config_a5000_seed42.yaml
python scripts/train_anomaly_detector.py --config configs/seeds/config_a5000_seed123.yaml
python scripts/train_anomaly_detector.py --config configs/seeds/config_a5000_seed999.yaml
```

Estimated 20-35 min per seed on CPU (phase 1 report §7), 1-1.75 h for three,
plus a threshold re-derivation and a v2 re-evaluation per seed (~4 min each).

---

## 9. Response-dataset record (Phase 2B)

### 9.1 What the dataset is built from

| Property | Value |
|---|---|
| Source | `TEP_Faulty_Testing.csv`, 960 samples/run, onset 160 (0-based) |
| Detector | frozen v2 operating point (`configs/detector_v2_operating_point.json`, SHA-256 `e7fe73eb…`) |
| Aggregation | persistence 12, no smoothing, one event per run (first closed event after onset) |
| Events built | 4,438 real (train 2,168 / val 456 / test 1,814) + 20 oracle + 204 NORMAL_TRANSIENT |
| Splits | `data/splits_v2.json` (run ranges expanded and checked pairwise during the build) |

> **Superseded by §10 (2026-10-30).** The dataset now uses a fixed horizon and no
> oracle windows; the numbers below describe the full-event build, which is kept
> at `outputs/benchmark_v2/events_full/` and is **not used**.

### 9.2 Event shape — a limitation, stated plainly

Events are **long by construction**: median 735 samples (147 windows), p10 705,
p90 745, max 760. The reason is mechanical. An event closes only after
`separate` quiet windows, and a 960-sample run that alarms near sample 220 has no
room to close before the stream ends. So "event length" here measures *how much
run is left*, not how long the disturbance lasted, and the model sees a window
that spans most of the run. Resolving real disturbance duration needs the longer
runs the detector was not calibrated on; it is not available in this dataset.

Event-open delay (open minus onset 160), median by family: step/level faults
69 samples (p90 99), dynamics faults 14/15 59 (p90 94), unobservable faults 3/9
396 (p90 716). Faults 3 and 9 are late and rare (16 and 30 events from 245 runs
each); every other fault produces an event in 244/245 runs, the missing one being
the run that ends while an event is still open.

### 9.3 Misses, oracles and NORMAL_TRANSIENT

> **Superseded by §10.2.** The 20 `unclosed_at_run_end` misses were a selection
> artifact, not a stream artifact: run 86 is closed correctly in §10.2, and no
> oracle window is needed or built.

- 462 runs produced no closed event: 442 are `detector_miss` (228 fault 3, 214
  fault 9) and 20 are `unclosed_at_run_end` (exactly one per fault).
- Oracle windows are built **only** for the 20 unclosed runs. Their offset is the
  median event-open sample over the **train** split (229, i.e. +69 after onset),
  so a test-range oracle is patched with training timing. Faults 3 and 9 get no
  oracles: manufacturing events for faults the detector cannot see would make the
  response benchmark claim coverage it does not have.
- NORMAL_TRANSIENT supply: 117 train / 47 val / 40 test events from fault-free
  `TEP_FaultFree_Testing` runs, 10 of them confirmed false alarms (8 train, 2
  val, 0 test), the rest unconfirmed excursions harvested at persistence 3.
  These runs overlap the FAR selection/reporting halves; that is acceptable
  because LLM training cannot change a detector result.

### 9.4 Open-set construction

Removal is a stable function of the event id (SHA-256 of the id, 12% threshold),
so an event is an open-set episode or not regardless of which fold selects it.
Over train events of classifiable faults the rate is 12.3% (195 episodes); over
all faulty train events it is 8.9%, because faults 16-20 have no entry to remove.
Evaluation items carry **both** variants (`menu` and `removed`) for every
classifiable fault, so recall and abstention are always paired.

### 9.5 Prompt invariants (script-checked)

Every generated prompt passes `assert_prompt_clean`: 6,932/6,932 prompts contain
no `Candidate affected subsystem`, no `candidate_subsystem`, and no `severity`
fragment. Severity is absent from the prompt *and* from the target; it is
computed by the production rule at render time.

### 9.6 Teacher

Measured on this machine with `qwen3:30b` (the tag actually installed — the
expected `qwen2.5:32b` is not present, and the client refuses to substitute a
near-match): 7.8-9.2 s per call, 44/50 pilot answers accepted under the final
filter (88%). The 6 rejections are genuine: 2 invented sensor/valve names, 1
ungrounded sensor, 3 invented numbers. Teacher prose exists for **train only**;
val/test are scored against deterministic targets.

---

## 10. Phase 2B close-out (2026-10-30)

### 10.1 Evidence view — a fixed horizon, H = 20 windows

The evidence a benchmark item shows is a **fixed snapshot**: the event's open
window plus 20 windows (window_size 60, stride 5 → 100 samples), capped at the end
of the run. H was fixed before the rebuild and is not tuned on val or test.

Why: the full-event view measured *how much of the run was left* (median 149
windows, because an event that alarms near sample 220 cannot find `separate`=20
quiet windows before the 960-sample stream ends). A snapshot of fixed length makes
every item comparable and makes event closure irrelevant to whether an event
exists. The live aggregator and its config are untouched; the horizon is a
benchmark-only argument of the builder (`--max-windows`, default 20, `0` = off).

| quantity | value |
|---|---|
| events | 4,458 (train 2,188 / val 456 / test 1,814) |
| snapshot length | 20 windows for every event (5 runs truncated by stream end) |
| events whose span covers the onset | 4,458 / 4,458 |
| windows left at open | median 146, p05 137, p95 148, min 0 |
| full-event length (unused, for reference) | median 149 windows, p05/p95 149 |
| event-open delay, observable faults | median 59 (fault 1/4/14) to 94 (fault 20) samples |

The full-event build is kept at `outputs/benchmark_v2/events_full/` and is not
read by any script.

### 10.2 The "unclosed" runs — cause, fix, and why there are no oracles

All 20 `unclosed_at_run_end` misses were **run 86**, one per fault. It was not a
stream or flush artifact. `segment_events` already emits the still-open event at
the end of the stream, so run 86 produced an event correctly: open window 20,
close window 180. The builder then asked whether the **opening** window ended at
or after onset — window 20 ends at sample 159, onset is 160 — and discarded an
event that spans the entire run.

The rule is now "the first event whose span reaches the onset" (`select_event`,
`llm/benchmark/events.py`). Run 86 yields an event for every fault, open delay −1
sample, and the miss list drops from 462 to **442, all `detector_miss`** (faults 3
and 9 only). With zero unclosed runs, **oracle windows are no longer needed and
none are built**; the oracle pass remains, driven by a shared reason filter, and
is a no-op on this data.

### 10.3 Faults 3 and 9

Events from faults 3 and 9 are excluded from **train** and from teacher input:
their fused FDR is far below 5%, so an event of theirs is a chance detection, and
training on it would teach a diagnosis from evidence that does not exist. They
remain in **val/test** with `unobservable: true` so the miss is measured rather
than hidden.

| split | unobservable items | events from 3/9 |
|---|---|---|
| train | 0 (28 excluded during the build) | 0 |
| val | 12 | 6 |
| test | 28 | 14 |

### 10.4 Faults 16-20 and twin items

Faults 16-20 have no menu entry, so removing the truth removes nothing. Their
target is `UNCLASSIFIED_OPEN_SET` in **both** variants — a `menu` variant that
expects `FAULT_17` would demand an answer the menu forbids — and the two variants
are therefore identical. Evaluation items for 16-20 are emitted **once**; the
removed-twin is emitted only for faults 1-15, where removal changes the menu.

### 10.5 Class balance in train

`UNCLASSIFIED_OPEN_SET` is capped at 25% of faulty train items by seeded
subsampling of the faults 16-20 supply only; the 1-15 open-set episodes, which are
the actual abstention signal and are spread across every classifiable fault, are
never dropped. Because the share is of the faulty total *after* dropping, the
bound is `u ≤ cap·classified/(1−cap)`.

| train quantity | value |
|---|---|
| faulty items before cap | 2,160 |
| unclassified before cap | 793 (193 episodes + 600 from 16-20) |
| dropped | 338 (all 16-20) |
| unclassified after | 455 (193 episodes + 262 from 16-20) |
| share after | 0.2497 |

Test is untouched by the cap.

### 10.6 Prompt leak checks

Effort was spent to remove label-shaped tokens from the prompt, so the check is
script-enforced on **every** prompt (4,634 this build, 0 violations):

- the internal id `EVT-<split>-f<fault>-r<run>` never reaches the prompt; the id
  shown is a salted SHA-256 prefix (`EV-…`), stable per event;
- no split name, run-number token, `EVT`/`NORM` token, or `NORMAL_TRANSIENT`;
- no fault reference (`FAULT_n`, `fault n`, or a KB fault name) outside the menu
  block — the menu is exempt because naming the candidates is its job.

### 10.7 Teacher

`qwen3:30b` is the teacher, recorded in the manifest and here. It is **not** a
config value: `configs/*.yaml` are not edited for the benchmark, and the client
refuses a tag the server does not have rather than substituting a near match.

Inference: temperature 0.2, seed 42, `think: false`, JSON mode, concurrency 1
(the flag exists; Ollama serves one request at a time, so >1 buys nothing).
Output is prose only (`physical_reasoning`, `narrative`).

Provenance is recorded per call, in the cache and in the manifest: teacher tag,
temperature, seed, `filter_sha256` over the four files that decide acceptance,
`license_sha256` of the licence text the local runtime reports for the tag, and
`system_prompt_sha256`. A licence is recorded only as a hash of what the runtime
reports — no licence is asserted from memory.

Resume: every finished call is appended to `teacher_cache.jsonl` as its own line,
keyed by item id, with the verdict and rejection codes. A restart skips cached
items; a cached item produced under a different `filter_sha256` is re-run, so a
filter change cannot silently mix two regimes.

### 10.8 Confidence

Confidence is **not** a target field and **not** teacher output. The contract:

- it is computed in code, at inference, from the probability of the `fault_id`
  token, and calibrated on validation with isotonic regression;
- no calibrator is fitted yet, so `Report.confidence` is `None` and the renderer
  prints **"uncalibrated (no calibrator fitted)"** rather than a number;
- a value may not be set without `confidence_calibrated=True` and a named source,
  and a model that emits `confidence` is rejected as a schema error;
- hypothesis-level confidences in the response schema are a separate, live-path
  concern and are not reported as a benchmark metric.

### 10.9 Inputs

The four untracked detector artifacts are pinned in
`configs/benchmark_inputs.json` and verified at the start of both builds, which
fails on a missing or changed file:

| artifact | sha256 (first 12) |
|---|---|
| `outputs/anomaly_detector/model.pt` | `d0ec3f386f11` |
| `outputs/anomaly_detector/dynamic/cva_model.npz` | `97f27ae9ab10` |
| `outputs/anomaly_detector/dynamic/dpca_model.npz` | `a1b29a5b086a` |
| `outputs/preprocessing/scaler.pkl` | `6b67a6a18e74` |

### 10.10 Cause phrasing, retry policy and direction flags (close-out 2)

- **Cause phrasing.** Every fault's KB `name` describes an unmeasured cause
  (20/20); its `initiating.sensor` is a measured tag (20/20). The teacher prompt
  therefore states the disturbance in the KB's own words, and the filter accepts
  exactly that phrase (or it with the disturbance tail dropped, three words
  minimum) as a non-sensor cause mention. All other unknown tags still reject.
  Fault 5: 3/10 → 27/27.
- **Retry policy.** One retry with the rejection codes and a different seed, judged
  against the original prompt so feedback tokens cannot be quoted back; a
  twice-rejected item is dropped and logged, with per-fault coverage in the
  manifest against a 0.70 floor.
- **Direction flag.** `direction_inconsistent` is a flag, never a rejection. It
  compares direction words to the stored `trend` (movement), falling back to the
  deviation sign only when the trend is absent; `trend` and deviation disagree in
  442 of 2,151 ranked sensors, so the deviation-only version flagged 29 correct
  sentences. 8 flags remain in 7 of 172 answers.

### 10.11 Abstention, valve labels and the replay policy (close-out 3)

- **Open-set abstention contract.** On an item whose target is
  `UNCLASSIFIED_OPEN_SET` the teacher prompt no longer carries `the disturbance:`
  line; it states that no listed candidate fits and names only the nearest listed
  candidates to rule out. The filter's cause allowance is scoped to the faults the
  item's *menu* offered, so the removed fault's KB wording is rejected
  (`names_removed_fault`) while ruling out a listed candidate stays legal.
  Only **distinctive** phrases count: faults 5 and 12 share "condenser cooling
  water inlet temperature" and differ in the tail (step / random variation), and
  fault 12 must remain on the menu, so the shared prefix is not a leak.
  Bug fixed on the way: `nearest_candidates` could pad its list with the fault
  itself, which put the removed entry back into its own prompt. Before: 2 of 60
  accepted abstention answers named the removed fault and 63/63 prompts carried
  it. After: 0 of 73, and the builder now raises if any abstention prompt leaks.
- **Valve vs flow.** Eleven XMV tags have display names that read like process
  flows (`XMV_44 A_Feed_Flow` vs `XMEAS_1 A_Feed_Stream1`;
  `XMV_50 Stripper_Steam_Valve` vs `XMEAS_19 Stripper_Steam_Flow`;
  `XMV_51/52 ..._Cooling_Water_Flow` vs `XMEAS_21/22 ..._Outlet_Temperature`;
  also XMV_42/43/45/47/48/49). Every tag in the student prompt is now labelled
  `[XMV_44, valve position, manipulated variable]` or `[XMEAS_1, measurement]`,
  in the sensor list, the CVA attribution and the onset order. The teacher prompt
  says a valve position is not a flow, and `direction_inconsistent` skips XMV
  sentences entirely. Template hash changed; dataset rebuilt.
- **Replay instead of discard.** A prompt or filter edit changes the code hash of
  every cached answer. Rather than throw the prose away, the builder re-judges each
  cached raw answer with the *current* filters (`--replay-cache`, no model call).
  First rebuild: 184 of 188 still valid; after the 15-item re-run, 199 of 201.

### 10.12 The full teacher run (2026-10-30)

`qwen3:30b`, concurrency 1, final hashes. **1,914 items attempted, 2,202 API calls,
21,644 s (6.01 h)**: 1,626 accepted on the first attempt (84.9%), 198 recovered by
the single retry, **90 dropped** (4.7%), 1,849 train items with valid prose after
the 25 carried over. Every fault clears the 0.70 floor; the lowest is
`FAULT_15` at 0.858. Rejection codes are dominated by `number_not_in_store` (47)
and `direction_conflict` (28) among the dropped, and direction conflicts cluster on
the dynamics faults 14 and 7.

Dropped items stay in `train.jsonl` with `prose: null` and
`teacher_status: "rejected"` plus their codes; accepted ones carry
`teacher_status: "accepted"` (or `"accepted_replayed"`). A prose-conditioned loss
must mask on `prose is not None`; a label-only loss can use all 1,939 items.
