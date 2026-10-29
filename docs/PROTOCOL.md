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
