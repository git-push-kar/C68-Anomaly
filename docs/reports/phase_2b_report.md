# Phase 2B report-back — real-pipeline event builder and report dataset

**Repository:** `C:\Users\Admin\Desktop\anomaly`
**Branch:** `benchmark-upgrade` (base `4d4b3be`)
**Date:** 2026-10-30
**Compute:** CPU only. No LLM training. Teacher calls: 50 (pilot only, budget-limited).

---

## 1. Summary

The response benchmark now exists as data: **4,438 real events** built by streaming
`TEP_Faulty_Testing.csv` through the frozen v2 detector (2,168 train / 456 val /
1,814 test), plus **20 oracle events** for runs that end mid-event and **204
NORMAL_TRANSIENT** events from fault-free runs. Each event is rendered into a
v2 prompt carrying evidence and a KB candidate menu, with a deterministic target
whose label the teacher cannot move.

Three findings changed the design during the phase and are worth stating first:

1. **The prompt was empty for every item in the first pilot.** `build_event`
   nests its output under `event.evidence`, and the assembler read `event`.
   Nothing failed loudly — the prompt simply had no evidence, and the teacher
   answered from the menu alone. Fixed; prompts now carry detector ratios, top
   sensors, latency attribution and onset order.
2. **Dynamics faults had no usable sensor evidence.** Fault 15's LSTM sensor
   ranking is noise (z ≈ 0.09) because a sticking valve under closed-loop control
   barely moves a level. The CVA residual attribution puts `XMV_52` on top with a
   9% share, so it was added to the evidence. Without it the benchmark would have
   contained items that are unanswerable by construction.
3. **Most pilot rejections were filter bugs, not teacher failures** — truncated
   quotes of real values, a printed ratio that was not in the store, and the
   menu printing display names the validator could not resolve. All four were
   fixed at the cause. The teacher's genuine failure mode, inventing plant
   numbers, survives at ~12%.

The teacher tag actually installed is **`qwen3:30b`**, not the expected
`qwen2.5:32b`. The client refuses to substitute a near-match, so the pilot ran on
`qwen3:30b` with that label attached everywhere.

---

## 2. Changes

| Commit | What it does |
|---|---|
| `6caa54c` | Restores `outputs/preprocessing/baseline_stats.json` (+ provenance). The event builder dereferences it and it was absent, so no event could be built. Reconstructed from the fitted scaler, verified against it (max dev 1e-7). |
| `95594b9` | `canonical_tag` resolves a display name (`Condenser_Cooling_Water_Flow`) to its tag. The menu printed KB names the validator then rejected as unknown — an answer quoting its own menu failed. |
| `577c8c0` | `llm/benchmark/`: menu, prompt v2, deterministic targets, filters, teacher client, plus 44 tests. |
| `0e23cc7` | `scripts/build_benchmark_events.py`, `build_benchmark_oracles.py`, `build_benchmark_dataset.py`. |
| `802b57f` | `docs/PROTOCOL.md`: v2 headline set, §9 dataset record. |
| (this file) | Report. |

---

## 3. PART 0 — A0 fix

Verified in this session:

```
AnomalyDetector.from_artifacts("outputs/anomaly_detector", "outputs/preprocessing")
  -> ThresholdConfig(method='percentile', percentile=99.0, threshold=0.6870414805412293,
                     n_normal_scores=6675, mean=0.6192, std=0.0283)
TEPApp() constructed OK
```

`outputs/anomaly_detector/threshold.json` holds the re-derived value (0.68704,
percentile 99.0, from the 75 normal validation runs — 6,675 windows). No live
config points at the v2 operating point; `/configs/config.yaml` is untouched.

One pre-existing, unrelated limitation surfaced: `Failed to load LLM adapter (will
use fallback reports): No module named 'peft'`. The app still constructs; the
adapter path needs `peft` installed, which is an environment issue, not a code
change in this phase.

---

## 4. Inspections

### 4a. How the live path builds an event — VERIFIED

`main.py` opens an event on the first fused alarm and closes it after the
aggregator's quiet-window requirement; the report is generated **at close**.
`evidence/event_builder.py:build_event` assembles `evidence` (top sensors,
temporal sequence, pre-anomaly context, `candidate_subsystem`, `reasoning_notes`,
`detector_evidence`) and it is **nested**: `event.to_dict()["evidence"]`. That
nesting is what broke the first pilot.

**Event-length distribution under the v2 point (n = 4,438):** median 735 samples
(147 windows), p10 705, p90 745, max 760.

This is a mechanical artefact and is recorded as a limitation in `PROTOCOL.md`
§9.2: an event cannot close until `separate` quiet windows have passed, and a
960-sample run that alarms near sample 220 has no room left. Event length here
measures *how much run remains*, not disturbance duration.

### 4b. What was reused vs replaced — VERIFIED

| Reused | Replaced |
|---|---|
| `evidence/event_builder.build_event`, the aggregator, `preprocessing` stack, `scripts/fault_knowledge.py` | v0 prompt and its severity rubric (`llm/dataset.py:42-49`) |
| `scripts/tags.py`, `llm/response/allowed_set.py`, `validators.numbers_in/number_supported` | `generate_distillation_dataset.py`'s label-from-teacher flow |
| The Ollama HTTP pattern (rewritten as `llm/benchmark/teacher.py` with tag verification, `think: false`, retry on the unknown field) | — |

### 4c. Deterministic targets from the KB — VERIFIED

`FAULT_KNOWLEDGE[n]` supplies `name`, `subsystem`, `initiating` (parsed with
`ast.literal_eval`), `recommended_action`, and a `differential_diagnosis` text
that names the competing faults by number — that text, not a similarity metric,
supplies the ranked lookalikes.

---

## 5. Tasks

**1. Event builder.** 4,900 runs streamed in one CSV pass (530 s; per-fault reads
would re-scan 3.6 GB twenty times). Frozen v2 thresholds: LSTM 0.68704, CVA Q
194.63, DPCA SPE 24.04, persistence 12, no smoothing, run used as-is (960
samples, onset 160). One event per run = first closed event after onset.
Per-sample fields recorded: `source_file, fault, run, split, seed,
injection_sample, event_first_alarm_sample, event_open_sample, event_close_sample,
event_open_delay, first_alarm_delay, event_length_samples, triggered_by`.

**2. Misses and oracles.** 462 runs produced no closed event: **442
`detector_miss`** (228 fault 3, 214 fault 9) and **20 `unclosed_at_run_end`**
(exactly one per fault). Oracles were built only for the 20 unclosed runs, at
offset **229** = the median event-open sample over the **train** split. Faults 3
and 9 were deliberately not patched — see deviation D2.

**3. NORMAL_TRANSIENT.** From `TEP_FaultFree_Testing`, split by run range:
**117 train / 47 val / 40 test**. Of these, **10 are confirmed false alarms**
(8/2/0); the remaining 194 are unconfirmed excursions harvested at persistence 3
that never reach 12. These runs overlap the detector's FAR selection/reporting
halves — harmless, because LLM training cannot change a detector result.

**4. Prompt v2.** Evidence + menu. Removed: `Candidate affected subsystem`, the
`candidate_subsystem` value the event carries, and the fault→severity rubric.
Severity is neither prompted nor targeted. **Every** generated prompt is checked:
6,932/6,932 prompts contain none of the forbidden fragments.

**5. Open-set episodes.** Removal is `sha256(event_id) % 100 < 12`, so it is
stable across folds. **195 train episodes = 12.3%** of classifiable train events
(8.9% of all faulty train events, because 16-20 have no entry to remove).
Evaluation items carry both variants (`menu` and `removed`) for every classifiable
fault, so recall and abstention are always paired.

**6. Deterministic targets.** `fault_id, verdict, primary_subsystem, ranked
hypotheses (truth first, KB lookalikes next), mitigation, abstain,
table_selection, suggested_followups (router-verified), clarifying_question`.
`confidence` is `None` with an explicit note — see §9 for the proposal.
A gap found while writing tests: `truth_fault` holds the *expected answer*, so for
faults 16-20 it is `UNCLASSIFIED_OPEN_SET` and the generating fault was
unrecoverable. Added `truth_fault_number`.

**7. Teacher.** `qwen3:30b`, `think: false` (with the field omitted and retried on
older servers), temperature 0.2, seed 42, JSON mode. Filters: label unchanged,
tags resolvable and grounded, numbers in the prompt, no absolute wording, no
severity claim, no out-of-scope fault, sentence limits.

**8. Sizes.** Train ≤120 runs/fault → **2,305 items** (2,288 faulty + oracles,
117 normal); val 25/fault → **959** (456 faulty × both variants + 47 normal);
test all 100/fault → **3,668** (1,814 × both variants + 40 normal).

**9. Pilot.** 50 teacher calls, stratified one-per-family round-robin,
**7.8-9.2 s per call** (mean 7.8 s in the final run), **44/50 accepted (88%)**
under the final filter. The full-scale job is costed in §10 and **not run** —
it needs your approval.

**10. Audit sheet.** `outputs/benchmark_v2/dataset/audit_sample.csv`, 100 rows,
stratified across every fault, the normal family and the open-set episodes, with
`correct / grounded / useful` left blank. Not scored by me.

**11. Manifest.** `outputs/benchmark_v2/dataset/manifest.json`: per-file SHA-256,
counts, split-integrity proof, operating-point hash (`e7fe73eb…`), KB hash
(`f01236ff…`), prompt-template hash (`fe4b8e0d…`), teacher metadata.

---

## 6. Commands and outputs

```bash
python scripts/build_benchmark_events.py --skip-oracle      # 530 s
python scripts/build_benchmark_oracles.py                   # 58 s, 20 events
python scripts/build_benchmark_dataset.py                   # full, no teacher
python scripts/build_benchmark_dataset.py --pilot 10 --teacher \
       --model qwen3:30b --teacher-limit 50                 # 6 m 38 s
python -m pytest tests -q                                   # 194 passed
```

Split integrity, checked during the build and aborting on violation:

```
A_seen_faults_run_disjoint: train 6000, val 2000, test 2000        pairwise_disjoint true
B_leave_faults_out/fold_1_cooling_loop_holdout:
    train 5100, val 1700, test 1500, test_seen 1700                pairwise_disjoint true
    held_out_faults [4, 11, 15]                                    held_out_absent_from_seen true
B_leave_faults_out/fold_2_valve_sticking_family:
    train 5400, val 1800, test 1000, test_seen 1800                pairwise_disjoint true
    held_out_faults [14, 15]                                       held_out_absent_from_seen true
```

Event-open delay by family (median / p90, samples after onset 160):

| Family | n | median | p90 |
|---|---|---|---|
| step/level (1-8, 10-13, 16-20) | 3,904 | 69 | 99 |
| dynamics (14, 15) | 488 | 59 | 94 |
| unobservable (3, 9) | 46 | 396 | 716 |

---

## 7. Pilot: rejection reasons and examples

Measured on 50 calls, final filter, replayed over the stored raw text:

```json
{"accepted": 44, "rejected": 6,
 "codes": {"unknown_tag": 2, "sensor_not_grounded": 1, "number_not_in_store": 3}}
```

Rejection history matters here, because four earlier "rejections" were my bugs:
first run 0/3 (empty prompts — the nesting bug), then 29/50 and 33/50 while the
filters mis-classified truncated quotes (`206` for `206.82`), a printed ratio
(`643.34x`) absent from the store, and menu display names that did not resolve.

**Example 1 — accepted, normal excursion (NORM-train-r001).**

> The CVA Q-statistic alarm (195.5 vs threshold 194.6) indicates a process
> deviation, but the LSTM reconstruction error (0.6557 vs threshold 0.6870)
> remains below alarm threshold. … Dynamic residual attribution shows slow-drift
> behavior (slow-drift ratio 0.48) with dominant lag 3 window, consistent with a
> transient process adjustment rather than a fault.
>
> *narrative:* The process is showing minor, stable deviations across multiple
> streams with no significant trends. Check the A/C feed ratio and Component_D
> composition as they show the largest relative deviations, but verify these are
> part of a planned process adjustment rather than a fault.

**Example 2 — accepted, open-set episode (EVT-train-f01-r010, `FAULT_01` removed
from the menu; target `UNCLASSIFIED_OPEN_SET`).** It abstains and names the
rejected candidate instead of inventing an entry:

> … The rejected FAULT_07 (C header pressure loss) does not match the observed
> sensor trends, as it would primarily affect …

**Example 3 — accepted, dynamics fault (EVT-train-f14-r001).**

> The CVA Q-statistic alarm and high-lag residual attribution to XMEAS_21
> (reactor cooling water flow sensor) suggest a slow drift in the cooling system.
> … The rejected fault FAULT_04 was considered but ruled out because …

**Example 4 — rejected, invented number (EVT-train-f04-r001).**
`number_not_in_store:70.2` — "XMV_51 as dominant (70.2% share)". The prompt prints
shares as fractions, so a faithful rendering would be `70.2%` of `0.702`; no such
value was shown.

**Example 5 — rejected, invented name (EVT-train-f05-r002).**
`unknown_tag:Condenser_Cooling_Water_Valve` — a plausible valve that does not
exist in the tag vocabulary (`XMV_52` is the condenser cooling water *flow*).

---

## 8. Gate

| Gate item | Status | Evidence |
|---|---|---|
| A0 fix verified (app loads, tests pass) | **PASS** | `from_artifacts` + `TEPApp()` OK, threshold 0.68704 |
| Zero (source, fault, run) overlap across train/val/test per protocol and fold | **PASS** | manifest `split_integrity`, aborts on violation |
| Events built / misses per fault | **PASS** | 4,438 real + 20 oracle; 442 + 20 misses, per-fault table §5.2 |
| Event-open delay and event-length distributions | **PASS** | §4a, §6 |
| NORMAL_TRANSIENT counts per split | **PASS** | 117 / 47 / 40, 10 confirmed |
| Prompt v2 free of subsystem hint and severity rubric, script-checked on every prompt | **PASS** | 6,932 / 6,932 clean |
| Pilot rejection rate and examples reported | **PASS** | 88% accepted, 5 examples §7 |
| Full-scale command and cost estimate | **PASS** | §10 |
| Full test suite passes | **PASS** | 194 passed in 5.01 s |
| Addendum in `PROTOCOL.md` | **PASS** | §8.3 rewritten, §9 added |
| Teacher model as specified | **NOT VERIFIED** | `qwen2.5:32b` is not installed; pilot used `qwen3:30b` |
| Full teacher job executed | **NOT RUN** | awaiting approval, §10 |

---

## 9. Confidence — proposal, not implemented

Recommendation: derive it at inference from the **`fault_id` token probability** —
the softmax mass on the chosen menu id at the position where it is emitted — then
**calibrate on validation** (isotonic or temperature scaling fitted on
`val` only, per protocol, and stored beside the adapter). The schema field then
means *the calibrated probability that this label is correct for this event*,
which is what a downstream abstention rule needs.

It is deliberately not in this dataset: a number written by the teacher, or
invented by me, would look measured and be neither. Until calibration exists the
field stays `None` with `confidence_note`.

A secondary check worth adding at that point: report the calibration curve and
Brier score on `test`, since a confident wrong label is worse for an operator
than an abstention.

---

## 10. Cost — full teacher run and Phase 2C

**Teacher (needs approval — over the 30-minute threshold):**
2,305 train items × 7.8 s = **4.99 h** serial. With one retry on the 12% rejected
(+276 calls × 7.8 s) ≈ **5.6 h** wall clock. Two concurrent requests, if VRAM
allows, roughly halves it. Nothing beyond the 50-call pilot has been run.

**LLM training (Phase 2C, 3 configurations × 3 seeds = 9 runs):** measured
8,877 s per 2,376 examples = 3.74 s/example. Protocol A train 2,305 items →
2.4 h; fold trains carry 17/19 and 18/19 of the fault count → 2.2-2.3 h each.
Total ≈ **20-21 h** on the A5000. Evaluation inference is **an assumption to be
measured**: at ~1.2 s/item, one test pass over 3,668 Protocol A items is ≈ 1.2 h
per run; nine runs would add ~11 h, so the real figure is likely 30 h+ until
measured.

---

## 11. Deviations

- **D1 — teacher tag.** `qwen3:30b` (installed) instead of `qwen2.5:32b`
  (expected). Not silently substituted: `resolve_model` fails loudly, and the
  pilot label records the true tag.
- **D2 — oracles only for unclosed runs.** The brief says "for missed runs of
  faults in scope". Building 442 oracles for faults 3 and 9 would manufacture
  events for faults the detector cannot see; they are excluded from the observable
  set anyway. Their misses stay misses.
- **D3 — filter revisions after seeing the pilot.** Five corrections, each at the
  cause (empty evidence, tag classification split into fabricated vs ungrounded,
  printed-ratio grounding, percent↔fraction rendering, `Component_X` as chemistry
  not a tag). The 88% figure is a **replay over the stored raw answers**, not a
  fresh model run; the raw text is kept per item precisely so this is possible.
- **D4 — evidence addition.** CVA lag attribution was added to the evidence. It is
  detector output, not a KB hint, and without it dynamics faults are unanswerable.
- **D5 — evaluation twins.** Both variants are emitted for **every** classifiable
  val/test item, not only held-out faults (a superset of the ask).
- **D6 — `truth_fault` semantics.** Now the expected label; the simulator's number
  moved to `truth_fault_number`.
- **D7 — open-set rate.** 12.3% over eligible events, 8.9% over all faulty train
  events; both are in the manifest rather than picking whichever looks right.
- **D8 — dataset not committed.** `outputs/` is git-ignored; the manifest carries
  the hashes so the exact build can be identified and regenerated.

---

## 12. Open questions

1. **Teacher approval.** Run the full 2,305-item job with `qwen3:30b` (~5.6 h), or
   install `qwen2.5:32b` first and accept a different prose distribution?
2. **Event span.** Events span ~735 of 960 samples because they cannot close
   earlier. Should the response layer instead be trained on a fixed post-onset
   window (e.g. onset+300), trading realism for comparability across runs?
3. **Confidence calibration** (§9) — approve the token-probability + isotonic
   approach before Phase 2C?
4. **Pilot size.** 50 calls is enough to estimate timing but not per-family
   rejection rates; a 200-call pilot would cost ~26 min and tighten the estimate
   before the 5.6-hour commitment.
