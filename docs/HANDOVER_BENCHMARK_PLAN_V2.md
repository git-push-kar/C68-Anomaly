# Benchmark implementation — handover plan V2

Status: **plan and cross-check record.** Supersedes `docs/PAPER_LEVEL_BENCHMARK_PLAN_V1.md` as the
authoritative plan; V1 is retained as the discussion record. Dated `docs/PROTOCOL.md` sections still
win where they disagree with this document, and every place this document proposes to change one is
listed in §11 as a decision, never as a decision already taken.

Written for a **different machine** that holds the full checkout (including `data/`, the teacher
corpus, the base model and the training artefacts). Everything below marked *this device* was
verified here on 2026-11-05; everything marked *GPU device* needs the machine that produced
`outputs/tep_rca_v2_prose_pilot_seed42/preflight_gpu.json` (A5000-class, 23.988 GiB).

---

## 1. What this handover is for

Three questions, answered in order:

1. **What is already built** — verified by reading the tree, not by trusting the docs (§2).
2. **What remains, in what order, and what proves each step done** — phases P0–P8 (§7).
3. **What may and may not be claimed when it finishes** — the claim ladder (§10), which is the part
   a reviewer attacks first.

The study compares **who writes the words** around a diagnosis that is already fixed. Diagnosis,
verdict, severity and confidence are code outputs in every condition. No condition in this plan may
change them, and no result from it may be reported as a change in diagnostic accuracy.

---

## 2. Cross-check: what is actually implemented

Verified on disk. "Built" means the code exists *and* an artefact from it exists; "partial" means
one of the two is missing.

### 2.1 Surfaces

| # | surface | state | evidence | what is missing |
|---|---|---|---|---|
| 1 | **detector arms** | **partial** | `scripts/fit_operating_point_v2.py` (3 target FARs × 4 persistence settings, selection half 1–250, reporting half 251–500); `outputs/evaluation/detector_v2/operating_point_sweep.json` (12 rows); `configs/detector_v2_operating_point.json` frozen | DPCA is scored and cached but **never fused**; `fused_series` is hard-coded to `max(LSTM, CVA Q)`; **no AND mode** (`mode: or \| weighted` only); no per-arm contrasts; `per_detector_far` is configured but unused by `fused_series` |
| 2 | **classifier** | **built** | `scripts/hybrid_classifier.py`, `hybrid_crossfit.py`, `classifier_vs_adapter.py`, `score_llm_20.py`; `llm/hybrid/menu_rule.py`, `calibration.py` | no single contrasts table; root cause / subsystem / severity still reported as three accuracies instead of one merged row |
| 3 | **report** | **A built, B generated, C/M/D/E partial** | A: `scripts/fallback_v31.py`, 0 flagged rows of 40, anchor 40/40. B: `outputs/benchmark_v2/eval/prose_pilot/raw_pilot_{kb,shuffled-1,shuffled-2}.jsonl`, **520 rows each = 1,560 stored generations, 0 GPU left to spend** | C and M have no code; D's validator exists (`llm/hybrid/generate.py`) but its retry needs adapter weights that are absent here; E exists only as `scripts/prose_crossfit_train.py` |
| 4 | **dialogue** | **T live, U generated, F specified** | `prose_eval_followups.jsonl` 97 turns, `_179` 179 turns, `raw_pilot_followups.jsonl` 179; `llm/followup/*`; `docs/TRAINING_SPEC_FOLLOWUP_V1.md` | F never trained; no T/U/F comparison table |
| 5 | **integration** | **nothing** | — | the selection rule, its receipt, and the table that reports the tuple |

### 2.2 Instrument layer

| check | state | evidence |
|---|---|---|
| 40-case calibration | **passes** | `outputs/analysis/phase3a/fit/pilot_rechecks.json`: contradiction 14, tag_role 12, anchor 8, numeric 6 — precision 1.00, recall 1.00 on all four |
| numeric clause binding | **defective** | `scripts/pilot_rechecks.py` `numeric_flags` evaluates `above\|below` sentence-wide, so a sentence naming two detectors with opposite true directions flags both. 47 of 62 adapter numeric rows are this family; 27 of 62 have it as first cause |
| contradiction | **defective, and duplicated** | `scripts/prose_pilot_prompts.contradiction` returns *every* printed `FAULT_nn` ≠ gold with no negation logic. It fires on **462 of 520 rows (88.85%)**, 382 of them on plain rejections ("this pattern contradicts FAULT_01 …"). The calibrated implementation is `scripts/pilot_rechecks.contradiction_flags` — two implementations, one calibrated, one driving the gate |
| tag_role | **two readers disagree 7×** | `pilot_rechecks.json` 57 rows (10.96%) vs `pilot_eval_summary.json` 396 rows (76.15%) on the same 520 stored generations |
| preference | **instrument built, study not** | `scripts/make_blind_sheet.py`; 40-row pilot at **0.560**; no κ implementation anywhere |
| judge | **absent** | no `judge_validate.py` |
| mechanism validity / order invariance / sufficiency | **absent** | no code; order invariance is free (three orders already stored) |

### 2.3 The 520-row readings already on file

| reading | adapter (kb) | A (fallback v3.1) |
|---|---|---|
| numeric-defect rows | **62 / 520 = 11.92%** | 19 / 520 = 3.65% |
| grounding rows | 33 = 6.35% | 0 |
| tag_role rows | 57 = 10.96% *(pilot_rechecks)* / 396 = 76.15% *(prose_pilot_score)* | 0 |
| contradiction rows | **0** *(calibrated)* / **462 = 88.85%** *(uncalibrated)* | 0 |
| anchor exact, top \|z\| ≥ 1 | 170 / 329 = 51.67% | 329 / 329 = 100% |

The ordering is already visible: **A is ahead on every automated reading**. Paper-level work does not
change who wins; it changes whether the margin is measured, intervaled and defensible. The near-tied
cells — C, M, E — are where the study actually has something to decide.

### 2.4 Environment facts that dictate the schedule

| fact | value | consequence |
|---|---|---|
| CUDA on *this device* | `torch.cuda.is_available() == False`, no NVIDIA driver | no training, no generation here; all GPU work is *GPU device* only |
| GPU device of record | peak 16.218 GiB of 23.988, 12.83 s/step, 288 steps projected 1.03 h | one job at a time; two seeds cannot run concurrently |
| adapter weights | **absent** — no `.safetensors` in the tree; `outputs/tep_rca_v2_prose_pilot_seed42/` holds only `preflight_gpu.json` | any *new* adapter generation needs a retrain first; C and M must be text-level seams over stored B |
| base model | **absent** at the configured path `configs/config_prose_pilot_seed42.yaml:165` | E, F and D's retry are unrunnable until it is staged |
| teacher | **absent** — Ollama answers on `localhost:11434` but holds only `llama3.1:latest` (4.6 GiB); `qwen3:30b` not installed | E's corpus rebuild needs the teacher restored; measured 7.8–9.2 s/call, 44/50 accepted under the final filter |
| stored data | 520 items × 3 orders, 1,560 generations, 179 + 97 dialogue turns | the report and dialogue surfaces are **measurement, not generation** |

---

## 3. The five surfaces

| # | surface | unit | arms | decides | may not |
|---|---|---|---|---|---|
| 1 | detector | run × window | 6 arms × 3 FAR conditions | which monitor configuration ships | speak about prose |
| 2 | classifier | fault (20) / item | menu rule · hybrid · adapter label | which diagnosis rule ships | speak about prose |
| 3 | report | item (520), paired | A · C · M · B · D · E | which report writer ships | move the diagnosis |
| 4 | dialogue | turn (97) | T · U · F | which follow-up writer ships | stand in for a report gate |
| 5 | integration | configuration | winners of 1–4 | the shipped tuple, described not ranked | rank across surfaces |

Nothing crosses surfaces: the report is *context* for a dialogue turn; a detector FDR never enters
the report-writer rule; the 10-row routing stratum stays descriptive. Five rules, five outputs, one
integration table that reports them side by side without ranking them.

---

## 4. The A–F grid, completed

The grid is a 2×2 over who writes which part of a report, plus two toggles:

|  | **numbers by code** | **numbers by adapter** |
|---|---|---|
| **mechanism by code** | **A** — fallback v3.1 | **M** — A's mechanism, adapter's numbers |
| **mechanism by adapter** | **C** — hybrid seam | **B** — full adapter |

| label | what it is | how it is built **here** | cost |
|---|---|---|---|
| **A** | deterministic assembler; template fills every number from the store | live (`scripts/fallback_v31.py`) | 0 GPU |
| **B** | adapter authors the whole report | **stored**, 520 × 3 orders | 0 GPU |
| **C** | code writes every numeric sentence, adapter writes mechanism only | text-level seam over stored B: split B at sentence level, keep B's mechanism sentences, replace every numeric sentence with A's. Verified by the instrument, not by reading | ~1 h CPU |
| **M** | adapter's numbers, code's mechanism grammar | text-level seam: A's mechanism sentences with the adapter's numeric values substituted | ~1 h CPU |
| **D** | check–retry–fallback wrapper on B, prose otherwise identical | `llm/hybrid/generate.py` applied to stored B; **retries need the adapter**, so on this checkout D degenerates to "B with fallback on invalid rows" and must be labelled that way | 0 GPU here; minutes on GPU device |
| **E** | adapter retrained on report prose, 3 seeds | `scripts/prose_crossfit_train.py`; corpus `outputs/benchmark_v2/sft/train_prose_v1.jsonl` (1,523 rows: 562 report, 961 dialogue, seed 7) | **GPU device, 3.0–5.2 h** |
| **F** | adapter retrained on follow-ups | 961 dialogue rows; `docs/TRAINING_SPEC_FOLLOWUP_V1.md` | **38–65 min** |

Rules the grid imposes:

- **closed**: fixed before the first run; nothing added or dropped afterwards.
- **paired**: every condition scored on every item — twelve disjoint cells over 520 items is 43 items
  per cell and carries no claim.
- **C and M are seams, not generations.** With no adapter weights on the checkout, a fresh
  mechanism-only generation pass is impossible without a 1.0–3.4 h retrain first. The seam is lawful
  only if the instrument — not a reader — certifies that no numeric sentence in C comes from the
  adapter and no mechanism sentence in M comes from the template.
- **D is not a writer axis.** It is a delivery wrapper on B and is reported as an ablation row.
- **E is not a writer axis either.** It is a checkpoint; it can only confirm or refute "training
  closes the numeric gap".

---

## 5. Metrics, by the job they do

Every metric must name a consumer: a rule rung, a table cell, or a limitation sentence. One that
names nothing is a liability with a compute bill.

### 5.1 Decide (9)

| metric | surface | mechanism |
|---|---|---|
| run-level detection, paired McNemar | detector | primary |
| detection delay | detector | declared selector — smallest mean delay among arms inside budget |
| exact root-cause accuracy | classifier | primary |
| numeric-defect rows | report | gate **and** rung 1 |
| blind preference | report | rung 2 |
| validated mechanism-claim count | report | rung 3 |
| order invariance | report | rung 4 |
| cost per 1,000 reports | report | rung 5 |
| unsupported-claim rate | dialogue | primary across T/U/F |

### 5.2 Gate or licence (14, after the cuts)

| metric | what it protects |
|---|---|
| per-check calibration, precision/recall | whether any defect count is quotable |
| realised FAR on runs 251–500 | proves the budget was spent, not nominal |
| FDR at matched FAR | the comparability axis |
| abstention precision/recall, false-abstention rate | a real budget: 4.74% of a 5% budget at threshold 0.9792 |
| ECE / Brier (OOF) | **selects the calibrator**: isotonic 0.00596 vs temperature 0.00638 |
| grounding | gate |
| anchor | gate (near-saturated — low information, kept for continuity) |
| fallback rate, empty narrative, schema validity | feasibility |
| inter-rater κ, judge κ | licences the preference claim |
| MDE per contrast | licences "no difference" |
| receipts, hashes, per-order stability | licences the confirmatory pass |

Demoted from gate to descriptive in this plan: **tag_role** (two readers disagree 7×; it cannot gate
until they agree) and **contradiction** (until one canonical implementation passes calibration).

### 5.3 Explain only (7–9)

subsystem and severity accuracy (merged into one report row), top-2 containment, per-stratum
breakdowns, mean prose length, token counts, per-order variation, mechanism-claim examples. These
change the sentence you write, never the pick.

---

## 6. The judgement design: a ladder with a veto, not a vote

Three instruments, never averaged into one score.

| tier | instrument | role | may it overrule? |
|---|---|---|---|
| gate | numeric-defect non-inferiority vs A, paired | hard gate | **yes — veto** |
| primary | blind paired human preference | decides among gate-passers | decides the ship cell |
| corroboration | judge, validated on the human sample (κ reported) | scales the result, tests strata | no — disagreement is reported |
| objective | mechanism validity, order invariance, sufficiency, abstention | explains why a cell won or lost | no — explanatory only |

**The six-rung rule, declared before results.**

1. **Gate.** Numeric-defect rows ≤ A + `m` on the fit set, paired, exact McNemar. Recommended
   `m` = **1.0 percentage point** (~5 items of 520) — far below the 4.9-point MDE, so the gate can
   only veto a large, visible gap (B at 11.92% vs A at 3.65% is 8.3 points), never adjudicate noise.
2. **Preference.** Higher blind paired preference among gate-passers; exact binomial.
3. **Mechanism validity.** Higher count of KB-validated mechanism claims per item.
4. **Order invariance.** Higher agreement across the three prompt permutations.
5. **Cost.** Lower $ per 1,000 reports.
6. **A.** The deterministic default wins any remaining tie.

**Output is a tuple, not a ranking:** `report = ⟨…⟩ · dialogue = ⟨…⟩ · detector = ⟨…⟩ at FAR ⟨…⟩ ·
classifier = ⟨…⟩ · cost = ⟨$/1,000⟩ · reason = ⟨the rung that decided⟩`. The reason line is a result:
"A shipped because C tied it on the gate, tied on preference, and cost more" is one; "A was already
there" is not.

Why the rule matters, quantitatively — selection bias on paired, correlated conditions, n = 520,
4,000 replications:

| candidate set | optimistic bias on the selected cell | P(selecting a non-best cell) |
|---|---|---|
| 2 conditions, 8-point true gap | 0.03 pts | 0% |
| 4 cells (A, B, C, M) | 0.54 pts | 52% |
| 6 conditions | 0.70 pts | 49% |
| 12 conditions | 1.24 pts | 89% |

One condition's standard error is 0.86 points at a 4% rate and 1.42 points at 12%. The inflation is
the same order as the gaps between near-tied cells, which is why the rule is declared and the
confirmatory pass is paired.

---

## 7. Remaining phases

Ordered by dependency. "CPU" is this device; "GPU" is the A5000 device. Acceptance is the command
that proves the phase is done.

### P0 — Instrument (blocking; 1–2 days CPU, 0 GPU)

| step | files | change |
|---|---|---|
| P0.1 | `scripts/prose_pilot_prompts.py`, `scripts/pilot_rechecks.py` | make **one** implementation canonical; point the other at it. `contradiction`: negate-aware, so "contradicts X", "X is ruled out", "unlike X" are not contradictions |
| P0.2 | `scripts/pilot_rechecks.py` | clause-bind `numeric_flags`: split on `;`, `, and`, `, while`, `, whereas`, sentence end; evaluate the direction word only inside the clause naming that detector; attribute `threshold_misquoted` to the clause's detector |
| P0.3 | `scripts/recheck_cases.py` | 40 → **46** cases: +mixed-direction sentence (v5-05 verbatim), +stored-value restatement (`-2.48%`, `z=-0.24`), +tag inside a hypothetical, +`XMV` as valve position, +two contradiction-rejection shapes (needed because the current 14 cases pass while the check fires on 89% of rows) |
| P0.4 | new `scripts/scorer_parity.py --prose` | fail unless the two prose readers produce **identical counts** on the 520 stored rows for every check |
| P0.5 | — | re-score the 520 rows into a **new dated artefact**; the v5 reading stays untouched |

Commands: `python -m scripts.recheck_cases` (must print 46/46, precision 1.00, recall 1.00);
then the re-score entry point.

Acceptance: 46/46 on the enlarged set; parity test green; two dated readings coexist and the diff is
written up as `docs/judging/instrument_v5_vs_v5b.md`.

Stop rule: if a case fails, fix the check, never the case. After signature the 46 cases are frozen.

**Do not skip P0.2 before P0.3's contradiction cases.** With the current uncalibrated reader the
contradiction gate (threshold 0.0) fails 462 of 520 rows, so an unattended run would eliminate every
cell but A — the instrument's preference published as a finding.

### P1 — Detector arms (1 day CPU, 0 GPU)

| step | file | change |
|---|---|---|
| P1.1 | `scripts/fit_operating_point_v2.py` | replace the hard-coded `fused_series` with an **arm table**: `lstm`, `cva`, `dpca`, `or_lstm_cva` (deployed), `and_lstm_cva`, `or_all3` |
| P1.2 | same | calibrate each arm's thresholds so the **fused** FAR meets the target: allocate the budget across components (`per_detector_far` = target/2 for two-way OR, target for AND) instead of thresholding each detector at the full target — the current mismatch is why the dynamic detectors sit near 0% while the LSTM sits at ~1.7% |
| P1.3 | `anomaly_detection/dynamic/runner.py` + `scripts/train_dynamic_detector.py` | add `mode: and` (the runtime supports `or` and `weighted` only) |
| P1.4 | new `scripts/detector_arm_contrasts.py` | paired run-level McNemar across arms at matched FAR; paired bootstrap on the delay difference |

Command: `python scripts/fit_operating_point_v2.py --rebuild-cache` (~5–6 min from CSV; seconds from
cache — note `op_point_cache.npz` is **not** in this checkout, so the first run pays the rebuild).

Acceptance: every arm's realised run FAR on 251–500 sits inside its budget; else the refit is wrong
and nothing is reported. Artefact: `outputs/evaluation/detector_v2/arms_far_matched.md`, reported as
a frontier with delay on the second axis.

### P2 — Classifier (0.5–1 day CPU, 0 GPU)

New `scripts/classifier_contrasts.py`: menu rule vs hybrid vs the adapter's own label, on identical
faults, paired McNemar. Per the cut list, root cause / subsystem / severity are emitted as **one
merged row** with the per-output numbers in an appendix. Gates: abstention precision/recall and
false-abstention rate inside the 5% budget. ECE is reported as a licence, not a result.

### P3 — Report surface (2 days CPU, 0 GPU; +1.0–3.4 h GPU only if a fresh generation is wanted)

| step | how |
|---|---|
| P3.1 | build C and M as deterministic seams over stored B and regenerated A (§4); declare the split rule in a doc before running |
| P3.2 | build D as the validator applied to stored B; label it "B + fallback, no retries" when the adapter is absent |
| P3.3 | wire the 520-item export through `scripts/prose_pilot_score.py` so all six cells are scored on all 520 items with the v5b instrument |
| P3.4 | new `scripts/select_condition.py`: implements the six rungs, writes a **signed decision receipt** naming the rung that decided, and refuses to run if `m` is unset or the rule is not committed |
| P3.5 | new `scripts/prose_flags_520.py`: per-item tidy flags (`item_id, order, condition, check, flag, sentence`) — without this, nothing paired can be computed, because the current artefacts keep only aggregate counts and five examples |

Acceptance: 520 rows per order per cell; row totals reconcile to the summary counts; schema validity
520/520; the 46/46 calibration rerun against the new readings.

### P4 — Preference (2 days human, 0 GPU; independent of P3's result)

`scripts/make_preference_sheet.py` generalised from 2 sides to N conditions with Latin-square side
assignment; `scripts/preference_agreement.py` for Cohen's/Fleiss κ (no κ code exists today). 120
items × 3 pairings × 2 raters, adjudicator, κ ≥ 0.60. Judge: one instruction model **outside** the
teacher's family, same rubric, validated on the rated subset only (`scripts/judge_validate.py`), κ
reported per stratum. On this checkout `llama3.1:latest` is the only non-Qwen model installed and it
is CPU-only here.

If P4 does not happen: the 40-row pilot (0.560) is reported as exploratory, the preference tier is
**removed from the rule**, and the ship cell is decided by rungs 1, 3, 4, 5, 6.

### P5 — Content instruments (1 day CPU, 0 GPU except the probe)

| metric | implementation | acceptance |
|---|---|---|
| mechanism validity | `scripts/mechanism_validity.py`: extract "X causes Y" / "A leads to B" claims from narrative sentences, check each as a path in `llm/benchmark/kb_causes.py` + `llm/benchmark/menu.py:lookalikes` | 30 hand-checked sentences agree |
| order invariance | `scripts/order_invariance.py`: same fault named, same top sensor, same direction across the three stored permutations — **free, data already on disk** | deterministic, no model |
| sufficiency | `scripts/sufficiency_probe.py`, **shrunk**: 5 questions × 60 items × 2 cells (B vs A) = 600 reader calls, exact match only, never judged | answers matched, not interpreted |

### P6 — Dialogue (0.5 day CPU; +38–65 min GPU if F trains)

`scripts/followup_rescore.py` on the 97 turns for T (template), U (adapter as-is), F (retrained):
refusal on false premises, unsupported claims, completeness per intent, over-refusal. Refusal and
over-refusal are always reported as a pair.

### P7 — Confirmatory pass (one shot; 0.5 day)

`scripts/confirmatory_pass.py`: write the receipt **first** — config hash, item hash, model revision,
timestamp — then run **winner + A only**, on the test export, paired, exact McNemar plus paired
bootstrap CI. One pass. A second one is a new dated section, not a re-run.

**Blocked here:** `data/llm/` holds only `split_metadata.json` — no `val.jsonl`, no test export — so
the fallback side cannot be re-derived from `Sources`. This phase runs on the machine that has the
full dataset (D9).

### P8 — Release set (1 day)

Sheets, keys, prompts, the 520 texts, instrument hash, configs, the selection receipt:
`scripts/package_prose_release.py`. `outputs/` is gitignored, so nothing is releasable today; without
this the paper cites artefacts a reviewer cannot see.

### Dependency graph

```
P0 ──► P3 ──► P7 ──► P8
 │       ▲
 ├► P5 ──┘
 ├► P1  (independent)
 ├► P2  (independent)
 └► P4  (human, parallel from day 1)
 GPU: E and/or F, overnight, ONE job at a time
```

P1, P2 and P4 run in parallel with P0 and P3 and touch nothing they produce. The critical path is
P0 → P3 → P4 → P7; everything else has slack.

### Two-night shape

- **Night 1 (CPU):** P0, P1, P2, P3, P5 (minus the hand check), P6 scoring, P8 packaging. ~4–6 h. Stop
  before P7.
- **45 minutes of a human:** sign the 46 cases, read the six-cell table, commit `m`, decide E.
- **Night 2 (GPU device):** E (3 seeds sequential, 3.0–5.2 h) and/or F (38–65 min), then the sealed
  confirmatory pass.

### Referee conditions for any unattended run

A chain of scripts is not safe; a supervisor with refusals is:

1. staged output dirs (`outputs/analysis/phase3a/p0_<date>/`), never overwriting v5;
2. calibration below 1.00/1.00 ⇒ write `BLOCKED.md`, exit non-zero, stop the night;
3. every artefact records its denominator and its expected denominator; a mismatch is a failure, not
   a smaller table;
4. `docs/judging/confirm/signed.json` (item hash + config hash) gates P7, so a queue can never spend
   the one shot;
5. `select_condition.py` refuses to emit a receipt while `m` is unset or the rule is uncommitted;
6. one morning report listing every artefact, gate value and refusal.

---

## 8. Statistics

- paired per-item flags → **exact McNemar** per check;
- risk difference with a **paired bootstrap** CI (10,000 resamples, seeded);
- one primary, secondaries under **Holm**;
- **MDE stated before the run**: at n = 520 with 15% discordance, 80% power at a **4.9-point**
  difference (4 points → 63%, 6 points → 94%). At n = 200 even 6 points reaches 53%;
- per-stratum results are **counts**, never tests — 10-row cells cannot carry a claim;
- three seeds for whichever condition ships, reported as mean and range;
- delivered text is what is measured, never first attempts.

---

## 9. Runs already paid for

| artefact | count | reuse |
|---|---|---|
| `raw_pilot_{kb,shuffled-1,shuffled-2}.jsonl` | 520 × 3 = 1,560 | B is free; order invariance is free |
| `prose_eval_followups.jsonl` / `_179` | 97 / 179 | U is free |
| `prose_eval_val_*` prompt exports | 520 × 3 | C, M, D seams read from here |
| A regeneration | 520 | deterministic, no GPU |
| `outputs/benchmark_v2/sft/train_prose_v1.jsonl` | 1,523 rows | E's and F's corpus already exists |

Cross-checked consequence: **the report and dialogue surfaces are measurement, not generation.** The
only GPU items left in the whole plan are E (3.0–5.2 h), F (38–65 min), optional fresh adapter text
(1.0–3.4 h, needs weights), and the teacher rebuild if E runs (needs `qwen3:30b`, 7.8–9.2 s/call).

---

## 10. What the paper can claim

### Version A — human tier included

> We report a pre-registered comparison of who should author the natural-language diagnosis that
> follows an alarm in a simulated chemical plant, once the diagnosis itself is fixed. Six report
> writers, three dialogue writers, six detector arms and three classifier rules are compared on 520
> paired held-out events. Contributions: (i) a validity audit of distilled diagnostic prose against
> the plant's own evidence store, separating numeric fidelity, grounding and abstention correctness,
> showing a model-authored report achieves no reduction in numeric defects relative to a
> deterministic assembler (Δ = ⟨x⟩ pp, 95% CI ⟨lo, hi⟩, exact McNemar p = ⟨p⟩); (ii) a content
> instrument counting mechanism claims validated against the knowledge base's lookalike graph, plus
> an order-invariance measure from three prompt permutations of the same event; (iii) a blind paired
> preference study of ⟨N⟩ pairs in which ⟨C⟩ was preferred in ⟨k⟩/⟨N⟩ (⟨p̂⟩, 95% CI ⟨lo, hi⟩, Cohen's
> κ = ⟨κ⟩); (iv) a selection rule declared before results, with the rung that decided each choice
> reported; (v) a one-shot confirmatory pass on the test export.

### Version B — no human tier

> The same validity audit and content instruments, reported as the negative result: a 2B model
> trained on distilled prose does not reduce numeric defects relative to a deterministic writer, and
> retraining three more times does not change that. The 40-row preference pilot (0.560) is reported
> as exploratory, **explicitly not confirmatory**, and the preference claim is absent from the
> contributions list rather than weakened.

### Tables

1. system: classification, abstention, calibration — separate track, not this study;
2. prose source: cells × {numeric, grounding, anchor, fallback rate, latency, $/1,000}, CI + pairwise
   exact p, primary named per contrast;
3. preference: blind pairwise, κ, adjudication count, exact binomial;
4. ablations: D vs B, E vs B as their own rows;
5. instrument: 46-case calibration, the false-positive families, the bracketed sensitivity reading;
6. follow-up: T vs U vs F on 97 turns;
7. detector frontier: arms × FAR with delay, report vs selection half separated.

### The one thing never to write

Never that the metrics "agree with" or "confirm" the preference result. The content metrics measure a
different quantity and the judge is anchored to the humans, so neither is independent evidence about
preference. Write "consistent with", report the disagreement rate, show the tier structure.

Without humans the other metrics are not "already there" for the preference claim — they support a
different claim. No combination of code metrics can establish that a reader prefers a text.

---

## 11. Decisions to sign before any run

| id | decision | recommendation | if declined | blocks |
|---|---|---|---|---|
| D1 | stage-1 n | **520 items × 3 orders**, reusing stored generations | 200 items; MDE widens to 15–20 points | P3, P5 |
| D2 | primary contrast | **B vs A on numeric-defect rate**, paired | C vs A; same machinery, different decision | P3 |
| D3 | selection rule | **the six rungs of §6, with m = 1.0 pt** | any undeclared selection reimports the curse | P3.4 |
| D4 | confirmatory pass | winner + A, one pass, paired, receipted | a point estimate with no in-run baseline | P7 |
| D5 | judge model | an instruction model **outside** the SFT teacher's family | teacher only, labelled a reference ceiling | P4 |
| D6 | instrument treatment | fix clause binding **and** contradiction, extend to 46, bracket the residual | the primary outcome stays artefact-dominated and is not quotable | P0, P3 |
| D7 | E inclusion | only if the training ablation is wanted (3.0–5.2 h) | E is skipped; it cannot change what ships | GPU night 2 |
| D8 | preference volume | 120 items × 3 pairings, two raters, adjudicator, κ ≥ 0.6 | preference stays a 40-row pilot, tier removed | P4 |
| D9 | missing `val.jsonl` | run P7 in the full-dataset checkout | sheets marked *fallback-unverified*; no confirmatory claim | P7 |
| D10 | release set | sheets, keys, prompts, 520 texts, instrument hash, configs | nothing is releasable | P8 |
| D11 | C/M implementation | **text-level seams** (adapter weights absent) | fresh generation: +1.0–3.4 h retrain first | P3.1 |

D3 and D6 change existing recommendations; the rest confirm or extend them.

---

## 12. Prohibitions

- No condition runs before D1–D4 and D6 are signed and dated.
- No cell, primary outcome, stage size or selection rule changes after any result is visible.
- No condition is measured on the test set except the single pass of P7.
- Post-hoc repair and stratified routing are never introduced as conditions after the fact.
- E does not proceed without explicit approval.
- No condition may change the diagnosis, verdict, severity or confidence.
- A superseding result is a **new dated section**, never an edit of §29.9 or §30.
- No metric without a named consumer.
- No composite score across instruments.

---

## 13. Handover checklist for the receiving device

**Must be present** (verify, do not assume):

- [ ] `data/raw/` TEP CSVs and `data/processed/` including `manifests/detector_split.json`;
- [ ] `data/llm/val.jsonl` and the test export (absent in this checkout — D9);
- [ ] `outputs/tep_rca_v2_prose_pilot_seed42/` **with weights** (absent here: only `preflight_gpu.json`);
- [ ] the base model at `configs/config_prose_pilot_seed42.yaml:165`;
- [ ] `qwen3:30b` in Ollama if E runs;
- [ ] `outputs/benchmark_v2/eval/prose_pilot/` (the 1,560 stored generations);
- [ ] `outputs/analysis/phase3a/fit/` (the v5 readings that must not be overwritten).

**First commands on arrival:**

```
python -m scripts.recheck_cases          # expect 40/40 today; 46/46 after P0.3
python -c "import torch; print(torch.cuda.is_available())"
python scripts/fit_operating_point_v2.py --no-freeze   # confirms the cache rebuilds
```

**Definition of done for the whole plan:** six report cells and three dialogue arms scored on their
full denominators with the v5b instrument; the detector frontier with matched FAR; the classifier
merged row; a signed selection receipt naming its deciding rung; a 46-case calibration table; a
preference table with κ or an explicit statement that none exists; and a release set that makes every
number recomputable.

---

## 14. Provenance

| source | contribution |
|---|---|
| `docs/PROTOCOL.md` §29 (3B-17) | the prose-source decision, the invariant, the v5 readings |
| `docs/PROTOCOL.md` §30 | the factorial, the clarification set, cost, prohibitions |
| `docs/PAPER_LEVEL_BENCHMARK_PLAN_V1.md` | the design, the instrument defect, the winner's-curse simulation |
| `docs/PROSE_SOURCE_EXECUTION_PLANS_V1.md` | the six plans, their gates and failure modes |
| `docs/FOLLOWUP_SPEC.md`, `docs/TRAINING_SPEC_FOLLOWUP_V1.md` | the dialogue surface and F's gates |
| `outputs/analysis/phase3a/fit/pilot_rechecks.json`, `pilot_eval_summary.json` | the 40-case calibration and the 520-row readings |
| `docs/judging/pilot_v5/` | the 520-row readings, the join check and the 40-row preference pilot at 0.560 |
| working discussion 2026-11-04/05 | the tier ladder, the six rungs, the metric-role cuts, the contradiction and tag_role divergences, the concurrency constraints and the two-night shape |

No model was loaded, no teacher was called, nothing was trained, and no stored generation, sheet, key
or checkpoint was modified in writing this document.
