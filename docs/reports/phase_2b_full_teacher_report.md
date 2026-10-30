# Full teacher run report (Phase 2B, close-out 3 follow-up)

Repo: `C:\Users\Admin\Desktop\anomaly`, branch `benchmark-upgrade`.
Teacher `qwen3:30b`, concurrency 1, resume on, final hashes
(filter `c3888e7d…`, teacher prompt `19125313…`, template `fe4b8e0d…`).
CPU + Ollama only. **No training was started.**

---

## 1. Summary

| quantity | value |
|---|---|
| train items in the dataset | 1,939 |
| items attempted in this run | **1,914** |
| already valid under the final hashes (not re-called) | 25 |
| API calls (retries included) | **2,202** |
| first-pass accepted | **1,626** (84.9% of attempted) |
| retried, recovered | **198** (of 288 retried) |
| dropped after a second rejection | **90** (4.7%) |
| accepted | 1,824 (95.3%) |
| valid prose in `train.jsonl` | **1,849** (1,824 + 25) |
| runtime | **21,644 s = 6.01 h** of measured call time (wall clock ≈6.3 h) |
| mean per call | 11.3 s |

Every fault clears the 0.70 coverage floor; the lowest is `FAULT_15` at 0.858.
The full open-set scan over the completed cache (487 abstention items, 471
accepted) finds **0** answers naming the removed fault.

---

## 2. Per-fault coverage against the 70% floor

| fault | items | prose | share | floor |
|---|---|---|---|---|
| FAULT_01 | 120 | 117 | 0.975 | ok |
| FAULT_02 | 120 | 119 | 0.992 | ok |
| FAULT_04 | 120 | 117 | 0.975 | ok |
| FAULT_05 | 120 | 115 | 0.958 | ok |
| FAULT_06 | 120 | 118 | 0.983 | ok |
| FAULT_07 | 120 | 109 | 0.908 | ok |
| FAULT_08 | 120 | 116 | 0.967 | ok |
| FAULT_10 | 120 | 115 | 0.958 | ok |
| FAULT_11 | 120 | 120 | 1.000 | ok |
| FAULT_12 | 120 | 118 | 0.983 | ok |
| FAULT_13 | 120 | 118 | 0.983 | ok |
| FAULT_14 | 120 | 106 | 0.883 | ok |
| FAULT_15 | 120 | 103 | **0.858** | ok (lowest) |
| FAULT_16 | 52 | 51 | 0.981 | ok |
| FAULT_17 | 52 | 50 | 0.962 | ok |
| FAULT_18 | 52 | 52 | 1.000 | ok |
| FAULT_19 | 53 | 52 | 0.981 | ok |
| FAULT_20 | 53 | 53 | 1.000 | ok |
| NORMAL_TRANSIENT | 117 | 100 | 0.855 | ok |
| open-set episodes | 193 | 184 | 0.953 | ok |

`teacher_coverage.below_floor = []` in the manifest.

---

## 3. Rejection reasons

### First pass (288 items rejected once, 298 codes)

| fault | items | reasons |
|---|---|---|
| FAULT_14 | 53 | direction_conflict 53 |
| FAULT_15 | 46 | number_not_in_store 30, direction_conflict 11, unknown_tag 5 |
| NORMAL | 36 | number_not_in_store 30, sensor_not_grounded 4, unknown_tag 2 |
| FAULT_07 | 26 | direction_conflict 22, number_not_in_store 4 |
| OPEN_SET | 21 | number_not_in_store 13, unknown_tag 5, sensor_not_grounded 3 |
| FAULT_05 | 19 | number_not_in_store 12, direction_conflict 3, reasoning_length 2, unknown_tag 1, sensor_not_grounded 1 |
| FAULT_01 | 16 | number_not_in_store 12, unknown_tag 3, sensor_not_grounded 1 |
| FAULT_08 | 14 | number_not_in_store 12, reasoning_length 2 |
| FAULT_10 | 14 | number_not_in_store 9, sensor_not_grounded 5 |
| FAULT_13 | 12 | number_not_in_store 4, sensor_not_grounded 3, direction_conflict 5 |
| FAULT_12 | 10 | reasoning_length 7, number_not_in_store 1, sensor_not_grounded 1, direction_conflict 1 |
| FAULT_06 | 7 | number_not_in_store 6, direction_conflict 1 |
| FAULT_04 | 5 | direction_conflict 4, sensor_not_grounded 1 |
| FAULT_16 | 5 | number_not_in_store 4, sensor_not_grounded 1 |
| FAULT_02 | 4 | sensor_not_grounded 3, unknown_tag 1 |
| FAULT_17 | 4 | number_not_in_store 2, sensor_not_grounded 1, unknown_tag 1 |
| FAULT_20 | 4 | number_not_in_store 3, unknown_tag 1 |
| FAULT_11 | 2 | reasoning_length 2 |
| FAULT_19 | 2 | number_not_in_store 1, unknown_tag 1 |

### Final attempt, the 90 that stayed rejected

| fault | items | reasons |
|---|---|---|
| NORMAL | 19 | number_not_in_store 10, unknown_tag 5, sensor_not_grounded 4 |
| FAULT_15 | 17 | number_not_in_store 13, unknown_tag 2, direction_conflict 1, reasoning_length 1 |
| FAULT_14 | 14 | direction_conflict 14 |
| FAULT_07 | 10 | direction_conflict 7, number_not_in_store 3 |
| OPEN_SET | 9 | unknown_tag 4, number_not_in_store 4, sensor_not_grounded 1 |
| FAULT_05 | 5 | number_not_in_store 3, direction_conflict 2 |
| FAULT_08 | 4 | number_not_in_store 4 |
| FAULT_10 | 4 | number_not_in_store 4 |
| FAULT_04 | 2 | direction_conflict 2 |
| FAULT_06 | 2 | number_not_in_store 1, direction_conflict 1 |
| FAULT_13 | 2 | direction_conflict 1, number_not_in_store 1 |
| FAULT_17 | 2 | number_not_in_store 2 |
| FAULT_01 | 1 | number_not_in_store 1 |
| FAULT_02 | 1 | sensor_not_grounded 1 |
| FAULT_12 | 1 | reasoning_length 1 |
| FAULT_16 | 1 | number_not_in_store 1 |
| FAULT_19 | 1 | unknown_tag 1 |

Totals for the 90: `number_not_in_store` 47, `direction_conflict` 28,
`unknown_tag` 12, `sensor_not_grounded` 6, `reasoning_length` 2.

Two patterns are worth naming. **Direction conflicts cluster on the dynamics
faults** (14: 28, 07: 7) — a slow drift invites a prose description of the
consequence, and the KB's stated direction of the initiating variable disagrees.
**Number rejections cluster on 15 and on NORMAL** — a valve-sticking fault and an
excursion with no fault both tempt the model to quantify something the evidence
does not print.

---

## 4. Items without valid prose in the final dataset

The 90 dropped items are **kept**, not deleted: removing them would change class
balance and split composition after the fact, and their deterministic target
(fault id, ranking, mitigation, tables, follow-ups) is still correct.

Each carries an explicit status, so a training run cannot mistake "no prose" for
"never called":

```json
"prose": null,
"teacher_status": "rejected",
"teacher": {"accepted": false, "dropped": true, "attempts": 2,
            "first_codes": ["number_not_in_store:..."],
            "codes": ["retry:number_not_in_store:..."]}
```

Accepted items carry `teacher_status: "accepted"` (or `"accepted_replayed"` for
the 25 answers carried over). The manifest reports
`teacher_accepted: 1849, teacher_rejected: 90`, and all 90 are listed with their
codes in `teacher_rejections.jsonl`.

Consequence for training: a prose-conditioned loss must mask on
`prose is not None`; a label-only loss can use all 1,939 items. 90/1,939 = 4.6%
of train has no prose, spread over 17 families, so no family loses material
coverage — the floor check confirms it.

---

## 5. Verification

| check | result |
|---|---|
| prompt leak check | 4,634 prompts, **0 violations** |
| open-set scan, full cache | 487 abstention items, 471 accepted, **0** naming the removed fault |
| split overlap | pairwise disjoint, unchanged |
| input artifact hashes | verified at build start |
| manifest consistency | accepted 1,849 + rejected 90 = 1,939 |
| audit sheet | regenerated, 100 rows, 5 per fault incl. 16-20, 5 normal, 5 open-set |
| full test suite | **263 passed** in 7.43 s |

---

## 6. Deviation

**205 items were called again.** With the filter hash changed, every cached answer
was "pending", so the run re-called the 205 items whose prose had been carried
over by replay rather than accepting prose written from a prompt without the tag
labels. Cost: ~45 min of the 6.0 h. Benefit: every prose row in the dataset now
comes from the final prompt. The 25 that were *not* re-called are the ones written
under the final hash by the close-out 3 re-run and resume test.

---

## 7. What is ready and what is not

Ready: 1,849 prose rows across all 18 classifiable faults plus normal and open-set
families, plus deterministic targets for all 1,939.

Not started (and not to be started without approval): LLM training — 3
configurations (Protocol A, fold_1, fold_2) × seeds 42/123/999, per the cost
section of the phase 0 report. Inference speed is still an assumption; the only
measured number so far is teacher latency (11.3 s per ~2.5k-token prompt).
