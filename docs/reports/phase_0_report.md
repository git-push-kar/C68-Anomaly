# Phase 0 report — benchmark upgrade

**Repo:** `C:\Users\Admin\Desktop\anomaly`
**Branch:** `benchmark-upgrade` (base `4d4b3be`; 5 commits added, listed in §3)
**Date:** 2026-10-29 · **GPU use:** none · **Training:** none · **`C68-Anomaly`:** untouched

---

## 1. Summary

Phase 0 is closed. The three requested verifications (1a run shape, 1b
file-specific assumptions, 1c grounded_qa contents) are resolved with measured
evidence, the split file now carries two protocols with recorded key hashes, the
label space and tag resolver are in code with tests, and `docs/PROTOCOL.md` fixes
the research questions, seeds, decoding and checkpoint rule.

Headline findings:

1. **The two faulty CSVs are not interchangeable.** Training: 500 samples/run,
   onset ≈ 20. Testing: 960 samples/run, onset ≈ 160. The benchmark now takes
   every LLM event from the **Testing file only**.
2. **`prepare_tep.py` is Training-shaped** (`block_samples=500`,
   `run_to_block = run_id - 1`) and would block the Testing file across run
   boundaries; the previously prepared arrays are also mutually inconsistent
   (fault arrays 730,000 rows vs 250,000 normal rows).
3. **The 7 grounded_qa intents are a different taxonomy from the 11 live
   intents**, and the dataset's question format (telemetry inline) is not what
   the live router sees. Replaying dataset questions routes 100% to
   `sensor_scores`; the operator's actual ask is what the router receives.
4. **Severity is a leaked lookup**, never a model skill; it is now declared a
   code-computed field, with the rubric replacement proposed and not applied.

---

## 2. Verifications

### 1a — Run length and onset index — **VERIFIED (mismatch confirmed)**

Measured with real commands: fault-free per-sample baseline from
`TEP_FaultFree_{Training,Testing}.csv`, fault signal as the across-run mean
z-score per sample index, onset = first of 5 consecutive samples at ≥4σ.

| File | rows/fault | runs/fault | samples/run | onset measured | onset configured |
|---|---|---|---|---|---|
| `TEP_Faulty_Training.csv` | 250,000 | 500 | **500** | **21** (faults 4/5/6/7; fault 1 at 25) | `faulty_training: 20` |
| `TEP_Faulty_Testing.csv` | 480,000 | 500 | **960** | **161** (faults 4/5/6/7) | `faulty_testing: 160` |

The one-sample offset is my 5-consecutive-sample rule; the configured 0-based
onsets (20 and 160) are correct. Weak/random faults (3, 8–12, 14–16, 19) never
reach 4σ on the across-run mean, which is consistent with faults 3 and 9 being
flagged unobservable — it does not affect the onset conclusion, because the
step faults pin both files exactly.

Code assumptions found:

* `configs/config.yaml` carries `dataset.fault_onset: {faulty_training: 20,
  faulty_testing: 160}` **and** a deprecated `fault_onset_index: 160`.
* `utils.get_fault_onset` (`utils.py:339`) resolves per split correctly, but
  `utils.resolve_onset_for_path` (`utils.py:366-372`) defaults an ambiguous path
  to `faulty_training`, silently applying onset 20.
* `scripts/prepare_tep.py:96-100` states "Each simulationRun is 500 samples" and
  slices `block_samples` (500) blocks; `:138` maps `run_id → run_id - 1`. Both
  are Training-only assumptions.
* `preprocessing/tep_loader.py:210,215` sniffs the run length from the filename
  (`500 if "Training" in path else 960`).
* `data/processed/fault_metadata.json` records `onset_index: 20` for all 20
  faults (Training shape), while `data/processed/fault_values/fault_01.npy` is
  `(730000, 52)` against `normal_values.npy` `(250000, 52)` — the prepared
  artifacts are stale/inconsistent and must not be reused for the benchmark.

**Consequence applied:** `splits_v2.json` sources all splits from
`TEP_Faulty_Testing.csv` (960 samples/run, onset 160, 500 runs/fault).

### 1b — File-specific assumptions in evaluator and evidence generator — **VERIFIED, with CORRECTED attribution**

| Component | File-specific assumption? | Evidence |
|---|---|---|
| Evaluator `llm/evaluate.py` | **No** | Text-only scoring; no window, onset or file access. Only regexes over generated text. |
| Evidence generator `evidence/event_builder.py` | **No file knowledge, but Training-calibrated** | `window_size`/`stride` from config (60/5), `baseline_windows: 20` (≈100 samples ≈ 5 h of pre-anomaly context), `onset_sensitivity: 1.5`. All in sample/window units. |
| Preprocessing `scripts/prepare_tep.py` | **Yes** | 500-sample blocks, `run_id - 1` mapping (lines 96-100, 138). |
| Loader `preprocessing/tep_loader.py` | **Yes, by filename** | run length chosen by `"Training" in path` (lines 210, 215); onset resolved from config (correct). |
| Streaming/RCA seeding `scripts/generate_real_rca_dataset.py:70-71` | **Yes** | Concatenates `normal[:onset] + df[:500-onset]` — hardcoded 500. |
| Detector eval `scripts/evaluate_anomaly_detector.py:121,161` | No | Uses `get_fault_onset(config, split)` per split. |

So the answer to "does anything assume a file?" is: the evaluator and the
evidence generator do not, but preprocessing, streaming data assembly and the
loader's run-length sniffing do — and those assumptions are all Training-shaped.

### 1c — The 2112 grounded_qa rows — **VERIFIED**

Composition of `data/llm/train.jsonl` (2,376 rows): 2,112 `grounded_qa` over 7
intents + 264 `report` (the teacher-written ones).

| Intent | n | Operator ask (the last paragraph, which is what the live router sees) | Answer opening |
|---|---|---|---|
| `sensor_values` | 528 | "What are the sensor values and ideal normal baselines?" | Markdown table: Sensor/Tag/Observed/Baseline/Dev%/z/Trend/Limit |
| `detector_evidence` | 264 | "What is the evidence and detector scores that force this to be an anomaly?" | Table: Overall score, LSTM error, CVA Q vs thresholds |
| `sensor_specific_why` | 264 | "Why did A_C_Feed_Flow decreased and how does that affect Reactor_Pressure?" | "**Direct Causal Coupling:** … first variable to deviate at t=0.0 min" |
| `valve_actuator_status` | 264 | "Which specific valves or actuators (XMV_45 …) are involved and what is their failure mode?" | "**Actuator and Valve Diagnostic Report:**" |
| `differential_diagnosis` | 264 | "How do we distinguish this event … from alternative causes or lookalike disturbances?" | "**Differential Diagnosis …** Distinguished from Fault 6 because …" |
| `sensors_inspect` | 264 | "Which sensors, valves, and equipment should I inspect first in the field?" | "**Prioritized Inspection Checklist**" |
| `severity_risk` | 264 | "What is the severity of this anomaly and what is the risk to plant safety?" | "**Severity Level:** **MEDIUM** …" |

Mapping to the live dispatcher (all 2,112 rows dispatched through
`RCAInference._dispatch_with_intent` on the operator ask only):

| Dataset intent | Live intent | Agreement |
|---|---|---|
| `sensor_values` | `sensor_values` | 528/528 |
| `detector_evidence` | `evidence_why` | 264/264 |
| `sensor_specific_why` | `overview_fallback` | 264/264 |
| `valve_actuator_status` | *(none — falls through to the LLM)* | 264/264 |
| `differential_diagnosis` | `overview` | 264/264 |
| `sensors_inspect` | `sensors_inspect` | 264/264 |
| `severity_risk` | `severity` | 264/264 |

Two facts that matter for benchmarking:

* `valve_actuator_status` has **no** live intent; `differential_diagnosis` is
  answered by the generic `overview` template, which is why
  `hypothesis_comparison` is proposed.
* If the **whole** question (telemetry included) is dispatched, every row
  routes to `sensor_scores` because the telemetry text contains "sensor" and
  "score". The live call site passes only `user_question` (`main.py:427`), so
  this is a dataset-format artefact, not a router defect — but any harness that
  replays dataset questions verbatim will measure the artefact.

---

## 3. Changes

| Commit | Area | Files |
|---|---|---|
| `881b896` | Tags: canonical resolution from the repo's convention | `scripts/tags.py` |
| `723088e` | Labels: fault label space from the KB | `scripts/fault_ids.py` |
| `db37761` | Splits: two hashed protocols + validator | `data/splits_v2.json`, `scripts/build_splits_v2.py`, `scripts/check_splits_v2.py` |
| `facf7f6` | Docs: protocol, response spec, severity-leak proposal | `docs/PROTOCOL.md`, `docs/RESPONSE_SPEC.md` |
| `9a40048` | Tests: tags, fault ids, both split protocols | `tests/conftest.py`, `tests/test_tags.py`, `tests/test_fault_ids.py`, `tests/test_splits_v2.py` |
| *(this report)* | Report | `docs/reports/phase_0_report.md` |

Split contents after the change:

| Protocol | Fold | train | val | test |
|---|---|---|---|---|
| A seen-faults, run-disjoint | — | 20 faults × 300 runs = 6,000 | 20 × 100 = 2,000 | 20 × 100 = 2,000 |
| B leave-faults-out | `fold_1` held {4,11,15} | 17 × 300 = 5,100 | 17 × 100 = 1,700 | 3 × 500 = 1,500 |
| B leave-faults-out | `fold_2` held {14,15} | 18 × 300 = 5,400 | 18 × 100 = 1,800 | 2 × 500 = 1,000 |

Recorded fingerprints (SHA-256 of each expanded sorted key list) are inside the
file; the validator recomputes them.

---

## 4. Commands and outputs

```text
$ python scripts/build_splits_v2.py --check
UP TO DATE: C:\Users\Admin\Desktop\anomaly\data\splits_v2.json

$ python scripts/check_splits_v2.py
A_seen_faults_run_disjoint  (fault_disjoint=False)
  train 20  6000  92b957f62d0e      val 20  2000  a19c647a4b29      test 20  2000  b82892ad4f73
B_leave_faults_out/fold_1_cooling_loop_holdout  (fault_disjoint=True)
  train 17  5100  deee7839bc62      val 17  1700  62e7a8862477      test  3  1500  64383da1e031
B_leave_faults_out/fold_2_valve_sticking_family  (fault_disjoint=True)
  train 18  5400  185efd2dd1be      val 18  1800  12930060c873      test  2  1000  b65240cd63cd
PASS: unique keys, no train/val/test overlap, fault-disjoint where required,
      faults 1-20 only, recorded sha256 matches.
exit 0

$ python scripts/check_splits_v2.py --verify-source     # streams the 3.6 GB CSV
PASS ... (same checks; every declared key confirmed present in the source)
exit 0

$ python -m pytest tests -q
73 passed in 2.52s
exit 0
```

Negative tests (each mutates a copy and asserts the exact rejection):
train/test run overlap, train/val run overlap, duplicate keys, held-out fault
leaked into a fold's train, SHA-256 drift, synthetic fault in a headline split,
wrong key contract, run range outside bounds — all 8 rejected.

KB tag audit (inside `tests/test_tags.py`): 22 faults, 0 invalid tags,
0 ambiguous tags, 17 distinct canonical tags referenced.

---

## 5. Gate table

| Gate | Result | Evidence |
|---|---|---|
| Split checker passes both protocols | **PASS** | `check_splits_v2.py` exit 0, 3 units (A, B/fold_1, B/fold_2) |
| Negative tests | **PASS** | 8/8 mutations rejected (pytest) |
| Tests pass | **PASS** | 73 passed, 0 failed |
| `PROTOCOL.md` complete | **PASS** | RQ1–RQ5, seeds, decoding, checkpoint rule, scope, both protocols, retired metrics |
| Verification 1a (run shape + onset) | **PASS** | measured 500/960 samples per run, onsets 20/160 |
| Verification 1b (file-specific assumptions) | **PASS** | located in prepare_tep / loader / streaming; absent in evaluator |
| Verification 1c (grounded_qa) | **PASS** | 5 examples, 7 intents, dispatcher matrix |
| Declared keys exist in the source CSV | **PASS** | `--verify-source` exit 0 |
| Model/RQ numbers (RQ1–RQ5) | **NOT VERIFIED** | no GPU, no training in Phase 0 (by instruction) |
| Proposed intents in the dispatcher | **NOT IMPLEMENTED** | proposals only in `RESPONSE_SPEC.md` §2.2 |
| Severity rubric replacement applied | **NOT APPLIED** | proposal only in `RESPONSE_SPEC.md` §7 |

---

## 6. Deviations

1. **Added `scripts/build_splits_v2.py`** (not requested). Without a builder the
   recorded SHA-256 values would be hand-pasted and could silently diverge from
   the splits; the builder makes the file reproducible and `--check` catches drift.
2. **`splits_v2.json` is protocol-structured**, not a flat list of 15,000 key
   triples; the validator expands ranges into `(source_file, faultNumber,
   simulationRun)` and hashes them. Literal triples can be emitted on request.
3. **Protocol B leaves seen-fault runs 401–500 unused.** The alternative
   (sending them to val) would have rightsized val at 200 runs but broken the
   300/100/100 convention; the alternative (test) would break fault
   disjointness, which the gate requires.
4. **Fold 1's third held-out fault (11) has no KB sibling among the kept
   faults.** The KB ties 11 to 4 and 14, not to 12; 12 is kept only as the
   condenser-loop structural counterpart. The fold design is unchanged, but the
   rationale is now stated honestly in `PROTOCOL.md` and in the JSON note.
5. **Faults 16–20 stay inside the splits** and are mapped to
   `UNCLASSIFIED_OPEN_SET` rather than removed, so the benchmark still measures
   open-set behaviour.
6. **The earlier single-protocol `splits_v2.json` was replaced** (key without
   `source_file`, Training+Testing mix): superseded within this branch.
7. **No prompt edit, no data regeneration, no retraining.** The severity rubric
   replacement is a proposal, as instructed.

---

## 7. Open questions

1. **Success criteria approval** — all five RQ thresholds are placeholders.
2. **Unobservable faults** — 3 and 9 are currently kept and reported separately;
   should they be excluded from the headline entirely?
3. **Events per run** — one event per simulation run (making Protocol A test
   2,000 items), or several windows per run? This sets evaluation cost and
   per-fault variance.
4. **Open-set scoring** — should naming the specific number for faults 16–20 be
   scored as wrong-but-adjacent, or simply wrong?
5. **Severity replacement** — approve the prompt change (and therefore data
   regeneration + retraining), or keep severity as an echo check only?
6. **Windows on 960-sample runs** — keep `window_size 60 / stride 5` (≈3 h
   window, 5 h baseline context) or recalibrate for the longer runs?

---

## 8. Estimated cost of Phases 1 and 2A

Estimates, not measurements. The only measured anchor is the existing training
run: 8,877 s (≈2 h 28 m) per seed on the A5000.

**Phase 1 — event extraction, retrain, evaluate under A and B**

| Work | Estimate |
|---|---|
| Event extraction from the Testing CSV (9.6 M rows, 20 faults) | 1–2 h CPU, no GPU |
| Training, 3 seeds × 2 h 28 m | ≈ 7.5 GPU-hours |
| Evaluation: A test 2,000 + B fold_1 1,500 + B fold_2 1,000 = 4,500 items per seed × 3 = 13,500 generations | 7–15 GPU-hours at an assumed 2–4 s/item (must be measured) |
| **Total** | **≈ 15–23 GPU-hours**, roughly 1–2 days wall-clock with review |
| Optional severity fix (regenerate answers + retrain) | +2–3 days wall-clock, +7.5 GPU-hours |

**Phase 2A — evaluator rewrite, item generation, routing evaluation**

| Work | Estimate |
|---|---|
| Evaluator rewrite (exact label matching, tag-aware allowed set, table check) + item generator + tests | 1–2 days CPU |
| Routing/format evaluation, 200-item sample × 3 seeds | < 1 GPU-hour |
| Adversarial items per intent (11 + 3 proposed) | 0.5 day CPU |

Costs scale linearly with events per run (open question 3); one event per run is
the cheapest defensible setting.
