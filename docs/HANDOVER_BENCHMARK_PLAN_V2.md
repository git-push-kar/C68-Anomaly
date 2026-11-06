# Benchmark implementation — handover plan V2

Status: **plan and cross-check record.** Supersedes `docs/PAPER_LEVEL_BENCHMARK_PLAN_V1.md` as the
authoritative plan; V1 is retained as the discussion record. Dated `docs/PROTOCOL.md` sections still
win where they disagree with this document, and every place this document proposes to change one is
listed in §11 as a decision, never as a decision already taken.

**Revised on the receiving machine, 2026-11-05 00:25 IST — see §15.** The receiving machine and the
*GPU device* of V2 are one machine: this checkout holds the adapter weights, the base model, the
teacher corpus and the A5000 that produced `outputs/tep_rca_v2_prose_pilot_seed42/preflight_gpu.json`.
Every "GPU device only" row in §§4 and 7 is therefore local work. §15 lists every fact that changed,
with the command that established it; facts marked *this device* below were re-verified after revision.

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
| numeric clause binding | **half fixed** | `numeric_flags` now splits on `;` and sentence end and attributes the quoted number to the detector named in that segment, but not on `, and` / `, while` / `, whereas`, and `above`-`below` is still searched across the whole segment, so a segment naming two detectors with opposite true directions still flags both. All 62 adapter numeric rows stay one family until the clause split is finished |
| contradiction | **calibrated reader exists; the wrong one still drives the gate** | `pilot_rechecks.contradiction_flags` is now negate-aware (`CAUSE`, `IS_FAULT`, `REJECT`, `VERDICT`, `NORMAL_CLAIM`) and reads **0 of 520**; `prose_pilot_prompts.contradiction` is still the naive id scan, reads **462 of 520 (88.85%)**, and is the one `read_gates` calls. Two implementations, one calibrated — P0.1 has become a de-duplication, not a design |
| tag_role | **two readers disagree 7×** | `pilot_rechecks.json` 57 rows (10.96%) vs `pilot_eval_summary.json` 396 rows (76.15%) on the same 520 stored generations |
| preference | **instrument built, study not** | `scripts/make_blind_sheet.py`; 40-row pilot at **0.560**; no κ implementation anywhere |
| judge | **absent, and now harder to satisfy** | no `judge_validate.py`, and `qwen3:30b` is the only model installed while D5 requires one outside its family |
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
| CUDA on *this device* | `torch 2.11.0+cu128`, `cuda True`; RTX A5000, 24,564 MiB total, 1,112 MiB in use at check | **the GPU device and this device are the same machine**; no GPU phase needs a handover |
| GPU of record | peak 16.218 GiB of 23.988, 12.83 s/step, 288 steps projected 1.03 h | unchanged: one job at a time, seeds sequential |
| adapter weights | **present** — `outputs/tep_rca_v2_prose_pilot_seed42/tep_rca_v2_prose_pilot_seed42/adapter_model.safetensors` plus checkpoints 48/96/144/192/240/288; 33 `.safetensors` in the tree | D's retry is real and a fresh generation pass is possible; C and M may be generated rather than seamed |
| base model | **present** at `C:/Users/Admin/Desktop/LY_C68/deepfake_project/models/InternVL3-2B`; `models/internvl3_2b_lr_fused` also present | E, F and D's retry are runnable here |
| teacher | **present** — `qwen3:30b` Q4_K_M, 18.6 GB, on `localhost:11434`; **`llama3.1` is no longer installed** | E's corpus rebuild is unblocked; D5's judge model must be pulled, because nothing outside the Qwen family is present |
| interpreters | `python` 3.11.5 carries `datasets 5.0.1`, `transformers 5.15.0`, **no `peft`**; `venv_train/Scripts/python.exe` carries `peft 0.20.0`, `torch 2.13.0+cu130`, `transformers 4.57.6`, **no `datasets`** | adapter loading and GPU generation need `venv_train`; dataset prep needs `python`; an unattended chain must name its interpreter per step |
| stored data | 520 items × 3 orders, 1,560 generations, 179 + 179 + 97 dialogue turns | the report and dialogue surfaces are **measurement, not generation** |

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
| **D** | check–retry–fallback wrapper on B, prose otherwise identical | `llm/hybrid/generate.py` applied to stored B, **with real retries now that the adapter is present**; the retry budget must be declared before the run | minutes on the local GPU |
| **E** | adapter retrained on report prose, 3 seeds | `scripts/train_tep_adapter.py --config <cfg>` → `llm/train_adapter.py`, reading `llm.training.train_file` (`train_prose_v1.jsonl`, 1,523 rows); 3 epochs, 288 steps, `save_steps` 48, 12.83 s/step = 1.03 h per seed. The pilot config authorises **one** run and requires a new PROTOCOL 24.1 approval for a second, so E needs three new configs (seed, `adapter_name`, `output_dir`) plus that approval. `prose_crossfit_train.py` is the cross-fitted rule, **not** a trainer | **local GPU, 3.1–5.2 h** |
| **F** | adapter retrained on follow-ups | the same trainer with `train_file: followup_train.jsonl` (2,405 rows on disk; the 961 in the corpus is the dialogue share of `train_prose_v1.jsonl`); gates in `docs/TRAINING_SPEC_FOLLOWUP_V1.md` | **local GPU, 38–65 min** |

Rules the grid imposes:

- **closed**: fixed before the first run; nothing added or dropped afterwards.
- **paired**: every condition scored on every item — twelve disjoint cells over 520 items is 43 items
  per cell and carries no claim.
- **C and M are seams, not generations.** Adapter weights are present here, so a fresh mechanism-only generation pass is possible at 1.0–3.4 h on the local GPU; the seam stays the default because it is cheaper, not because generation is impossible. The seam is still lawful only if the instrument, not a reader, certifies the split. The seam is lawful
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
and nothing is reported. Measured baseline from the stored sweep: only **6 of 12** rows carry `passes_run_far: true` — persistence 8 and 12 at all three targets, persistence 3 and 5 at none of 0.005 and 0.01 — so the budget today is met only by heavy persistence, which is why P1.2 allocates it per component. Artefact: `outputs/evaluation/detector_v2/arms_far_matched.md`, reported as
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

**No longer blocked:** `outputs/benchmark_v2/dataset/val.jsonl` (4.3 MB) and `test.jsonl` (16.4 MB)
are present, and `data/llm/{train,val,test}` hold the arrow splits (2376/396/396), so the fallback
side can be re-derived here and **D9 closes**. The one-shot rule is unchanged: the signed receipt
gates the run.

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

- **The night (this device, CPU then GPU):** P0, P1, P2, P3, P5 (minus the hand check), P6 scoring,
  P8 packaging — ~4–6 h; then D's retries on the real adapter and, only on approval, F and E. Stop
  before P7 (§15.3 gives the slot table).
- **45 minutes of a human:** sign the 46 cases, read the six-cell table, commit `m`, approve or
  decline E, and restore or re-score `audit_ab_100_v5_scored_rater1.csv`.
- **After signature:** the sealed confirmatory pass, once, on this machine.

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

**Verified present on this device, 2026-11-05:**

- [x] `data/raw/` TEP CSVs and `data/processed/` including `manifests/detector_split.json`;
- [x] `outputs/benchmark_v2/dataset/val.jsonl` and `test.jsonl`, plus `data/llm/{train,val,test}`
      arrow splits (2376/396/396) — **D9 closes**;
- [x] `outputs/tep_rca_v2_prose_pilot_seed42/` **with weights** (checkpoints 48–288 and the final adapter);
- [x] the base model at `configs/config_prose_pilot_seed42.yaml:165`;
- [x] `qwen3:30b` in Ollama (18.6 GB) — but nothing outside its family, so D5's judge must be pulled;
- [x] `outputs/benchmark_v2/eval/prose_pilot/` (the 1,560 stored generations, 520 × 3);
- [x] `outputs/analysis/phase3a/fit/` (the v5 readings that must not be overwritten);
- [ ] `outputs/analysis/phase3a/fit/audit_ab_100_v5_scored_rater1.csv` — **absent**, and its two tests
      fail without it (`python -m pytest tests -q` → 631 passed, 2 failed).

**First commands on arrival:**

```
python -m scripts.recheck_cases            # 40/40 until P0.3; calibration 14/12/8/6 at 1.00/1.00
python -c "import torch; print(torch.cuda.is_available())"      # True: RTX A5000
python scripts/fit_operating_point_v2.py --no-freeze            # cache present: seconds
venv_train/Scripts/python.exe -c "import peft, torch; print(peft.__version__, torch.cuda.is_available())"
python -m pytest tests -q                                       # 631 pass, 2 fail on the missing scored sheet
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

---

## 15. Machine delta, 2026-11-05 — the receiving device is the GPU device

Re-verified on `DESKTOP-6EOC710` at 00:25 IST. Each row was a V2 assumption; the third column is what
follows for the schedule.

| V2 recorded | actually true | consequence |
|---|---|---|
| CUDA unavailable, no NVIDIA driver | `torch 2.11.0+cu128`, `cuda True`, RTX A5000 24,564 MiB | every GPU phase runs here; §7's GPU column is local |
| adapter weights absent | final adapter plus checkpoints 48–288 present | D retries for real; a fresh generation pass is possible; C/M need not be seams |
| base model absent | present at the configured path | E, F and D's retry unblocked |
| teacher absent, only `llama3.1` | `qwen3:30b` installed; `llama3.1` gone | E's corpus rebuild unblocked; D5's judge must be pulled |
| `op_point_cache.npz` absent | present | P1's first run is seconds, not ~6 min |
| `data/llm/` only `split_metadata.json`; P7 blocked | `val.jsonl`, `test.jsonl` and arrow splits present | D9 closes; P7 can run here after signature |
| 97 + 179 stored dialogue turns | plus `raw_run1_followups.jsonl` (179) | a second stored arm for T/U/F |

### 15.1 What the tree already contains that V2 listed as new work

| V2 step | actual state |
|---|---|
| P0.1 one canonical contradiction reader | the calibrated, negate-aware reader exists (`pilot_rechecks.contradiction_flags`, 0/520); the naive one still drives `read_gates` (462/520). De-duplication only |
| P0.2 clause binding | segmented split done; clause conjunctions and segment-wide direction words outstanding |
| P0.3 46 cases | still 40 (`CASES`: 14 contradiction, 12 tag_role, 8 anchor, 6 numeric; all 1.00/1.00) |
| P0.4 prose parity | `scripts/scorer_parity.py` exists but compares the **label scorer with the decoder**; P0.4 is a `--prose` subcommand, not a new file |
| P6 scorer | `scripts/followup_rescore.py` exists |
| P1.4, P2, P3.4, P3.5, P4, P5, P7, P8 scripts | absent, confirmed by listing `scripts/` |

### 15.2 Three defects found by running, not reading

1. `python -m pytest tests -q` reports **631 passed, 2 failed**, both in
   `tests/test_v5_scoring_decision.py` for the missing
   `outputs/analysis/phase3a/fit/audit_ab_100_v5_scored_rater1.csv` (the sheet, key and provenance
   *are* present). The 0.560 preference reading therefore cannot be reproduced from the sheet here,
   and `outputs/` is gitignored: it exists only if restored or re-scored. Until then P4's starting
   value is a report, not an artefact.
2. **Interpreter split** — see §2.4. `peft` lives only in `venv_train`, `datasets` only in `python`.
   An unattended chain fails at the first `import peft` unless each step names its interpreter.
3. **Disk** — 56 GB free of 885 GB (94% used). Adapters are small; three merged seed checkpoints are
   not. Budget the disk before E starts, or train adapters only and merge afterwards.

### 15.3 What can run overnight here, unattended

CPU first, GPU last, one job at a time, under the §7 supervisor (staged directories, `BLOCKED.md` on
a calibration miss, denominators recorded, refusals rather than silent shrinkage).

| slot | work | interpreter | expected | gate that must be green |
|---|---|---|---|---|
| N1 | P0.1 de-duplicate, P0.2 finish the clause split, P0.3 to 46 cases, P0.4 `--prose`, P0.5 dated re-score | `python` | 2–3 h | 46/46; parity identical on 520 rows; v5 untouched |
| N2 | P1 arm table, DPCA fused, `and` mode, per-component budget, `detector_arm_contrasts.py` | `python` | 1–2 h | every arm inside budget on runs 251–500 |
| N3 | P2 `classifier_contrasts.py` on stored labels | `python` | 1 h | one merged row, paired McNemar, abstention gates |
| N4 | P3.1–P3.3 seams C and M, D on stored B, six cells scored; P3.5 per-item flags | `python` | 2–3 h | 520 rows per cell per order; totals reconcile |
| N5 | P5 order invariance (free) and `mechanism_validity.py`; P6 `followup_rescore.py` on 97 turns | `python` | 1–2 h | 30 hand-checked sentences are the morning check |
| N6 | P8 `package_prose_release.py` — without it no number is citable | `python` | 1 h | manifest hashes recompute |
| G1 | D's retries on the real adapter; optional fresh B generation | `venv_train` | 1.0–3.4 h | schema validity, retry rate declared |
| G2 | F (38–65 min), then E (3.0–5.2 h, three seeds) **only on D7 approval** | `venv_train` | 4–6 h | three seeds reported as mean and range |
| A–F coverage | see the grid table below | — | — | five cells by default; E and F conditional |

**Signatures gate the table, not the night.** §12 forbids running any *condition* before D1–D4 and D6 are signed and dated. N1, N2, N3 and N5 are instrument and surface measurements; **N4 is the first slot that scores real conditions**, so it waits for that signature — roughly 15 minutes of human decisions at about T+4 h — or it moves to after the morning signature, and the night ends with the instrument, the arms, the classifier and the content measures complete, but no cell table.

**A–F coverage of the night.** The report surface has six cells (A, B, C, M, D, E) and the dialogue surface has F; only two of the seven labels need a GPU.

| label | produced by | slot | GPU | default |
|---|---|---|---|---|
| **A** | `fallback_v31` regenerated deterministically | N4 | none | yes |
| **B** | the stored 520 × 3 generations, re-scored | N4 | none | yes |
| **C** | numeric-by-code seam over stored B | N4 | none | yes |
| **M** | mechanism-by-code seam over stored B | N4 | none | yes |
| **D** | validator over stored B; retries only in G1 | N4, G1 | optional | yes, labelled degraded without G1 |
| **E** | three retrained checkpoints | G2 | 3.1–5.2 h | **no — D7** |
| **F** | the follow-up checkpoint | G2 | 38–65 min | **no — same approval line** |

So the night covers A, B, C, M and D completely and leaves the report table at five of six cells unless E is approved. That is a claim consequence, not a table cosmetic: §10's Version B — "retraining three more times does not change that" — presupposes E, so declining E moves the paper to a version that makes no training-robustness claim at all. E cannot change which cell ships; it can only confirm or refute that training closes the numeric gap.

**Measured compute on this checkout, not estimated:** `build_orders` 0.1 s for 520 items × 3 prompts; one instrument pass (520 rows, all four checks) **1.2 s**; the 40-case calibration 0.0 s; all six cells × three orders ≈ **22 s**. The night's hours are therefore authoring the seams, contrasts and receipts — the compute inside them is minutes. The long jobs are training only: E **1.03 h per seed** (288 steps × 12.83 s), F **38 min** at the 961-row share or **1.6 h** at the 2,405 rows on disk, and a fresh generation pass would be ~**24 min per 520-row order** (the spacing of the stored `raw_pilot_*` files).

**Not overnight, by design:** P3.4 `select_condition.py` must refuse while `m` is unsigned; P4 needs two humans and κ; P7 is one sealed shot behind a signed receipt. Those are the morning's 45 minutes.

---

## 16. Machine delta, 2026-11-06 — D29 and D30 ran, and both fail their gate

Dated result section, appended per the rule that a superseding result is a new section and never an
edit. It supersedes two §2.2 rows (preference: "study not … no κ implementation anywhere") and the
§15.2 item 1 status of the v5 sheet. Nothing above is modified.

### 16.1 D30 — the scored sheet arrived and the join runs

Inputs: `outputs/analysis/phase3a/fit/audit_ab_100_v5_scored_rater1.csv`, 40 rows, **320/320 score
cells** filled, preference 20 A / 20 B / 0 TIE. The sheet's `text_A` and `text_B` are byte-identical
to `audit_ab_100_v5.csv` on 40/40 rows, so there is no column swap; the key is balanced 20 adapter /
20 fallback and agrees with `audit_ab_100_v5_provenance.json` on every row inspected.

Commands: `scripts/audit_v5_sheet.py join --scored <csv> --out v5_join_report.json`, then `reveal
--scored <csv> --out audit_v5_reveal.csv`. Both return 0.

| reading | adapter | fallback |
|---|---|---|
| correct | 19 / 40 | 40 / 40 |
| grounded | **0 / 40** | 40 / 40 |
| consistent | 33 / 40 | 40 / 40 |
| useful | 12 / 40 | 40 / 40 |
| **preferred** | **0** | **40** |

Pooled: adapter 0, fallback 40, tie 0, share **0.0000**, decision **`fallback_v31`**. Per stratum:
`removed` 0/10, `open_set` 0/10, `normal` 0/20. The preference column agrees with the higher-scored
side on 40 of 40 rows, so it is internally coherent — this is not a sign inversion.

Two facts must be settled by the owner before this is frozen:

1. The fallback is `1` on **all 160 cells** while the adapter is `0` on grounded in every row. A
   strict reading of "every number appears in the panel" does produce this: the v5 template writes
   derived multiples (`reconstruction error 333.60x`) that are not panel values, and the fallback
   prose does not. But 160/160 with zero variance is also what a default-filled column looks like.
2. `flagged_cells` records 47 adapter grounding/consistency failures and 0 fallback failures.

Test consequence: `pytest tests/test_v5_scoring_decision.py` → **3 passed, 1 failed**.
`test_decision_reading_reproduces_from_the_stored_scoring` now reads 0/40/0 against the
pre-registered 14/11/15, share 0.56. The assertion encodes the pre-registered reading and is **not**
edited; the disagreement is the finding.

### 16.2 D29 — κ 0.2556, below the 0.60 floor

Inputs: `sheet_rater1_human.csv` (59 left / 58 right / 63 tie) and `sheet_rater2.csv` (88 left / 70
right / 22 tie), 180 shared pairs, identical side order on all 180. Rater 1's filename said `_llm`;
the owner confirmed it is a human read, so the sheet is stored as `_human`.

`scripts/preference_agreement.py --rater1 … --rater2 … --out d29/kappa.json` returns: agree 90/180,
po 0.5000, pe 0.3283, **κ 0.2556**, `meets_0_60: false`.

Second reading, excluding 17 pairs whose two texts are byte-identical: 163 pairs, po 0.4479,
**κ 0.1415**. Both raters said `tie` on all 17, so those pairs are free agreements, not noise.

Defect recorded, not fixed: `scripts/preference_sheet.py` does not reject a pair whose two texts are
identical, so **17/180 (9.4 %)** of the instrument is uninformative. The guard belongs to the next
generation — regenerating now would void both human reads.

What makes the failure informative: rater 1 chooses `tie` almost three times as often (63 vs 22), and
on the 163 discriminable pairs the two agree 44.8 % of the time.

### 16.3 What each gate is now blocked by

| gate | state | blocker |
|---|---|---|
| D29 | **failed**, κ 0.2556 | third read over the 90 disagreed pairs (human, ~20 min), then re-κ; or accept instrument failure |
| D30 | measured, **not frozen** | owner confirms the 160/160 fallback column or re-scores; the 0/40 result is otherwise recorded as-is |
| D31 | unblocked to write, not run | `judge_validate.py` absent; needs a frozen sheet and a judge outside the Qwen family |
| D32 | blocked | D29 gives rung 2 no preference input; the rung-3 rule choice (rate-with-n vs abstain vs raw counts) is still unsigned |
| D33 | blocked | D32 must leave the receipt non-provisional |

### 16.4 Carried forward: the F arm ran

`scripts/prose_pilot_run.py --config configs/config_followup_f_seed42.yaml --adapter
outputs/tep_rca_v2_followup_seed42 --mode followups` generated **179/179** turns in 267 s
(1.52 s/item) into `outputs/analysis/phase3a/fit/farm_2026-11-06/`. Rescored through the new
`--answers` path of `scripts/followup_rescore.py`: agreement 0.7318 vs the stored baseline 0.7374,
refusals 57 vs 64, over-refusal 26 vs 29, under-refusal 22 vs 18. F trades over-refusal for
under-refusal on `causal_why` (over 0.7778 → 0.5556, under 0.3333 → 1.0). Rescoring the stored
baseline reproduces `outputs/analysis/phase3a/followup_rescore.json` exactly, so the edit changed no
existing number.

## 17. D29 third read, 2026-11-06

The 90 disputed pairs were read a third time, blind to both earlier sheets and to the condition
key. Sheet `docs/judging/confirm/d29_sheets/third_read_sheet.csv`, sha256 `6ae78d8dd29ac6f37cc88d46539bf8a04d0de9e3bdd164752cb08188cde5e477`, 90 rows, 0 blank,
0 invalid, texts byte-identical to the rater-1 sheet. The third reader chose left 32,
right 38, tie 20.

| comparison | result |
|---|---|
| third read against rater 1, 90 pairs | 34 agree, kappa 0.0968 |
| third read against rater 2, 90 pairs | 28 agree, kappa -0.1124 |
| third read agrees with neither | 28 |
| three-reader majority | 152/180 pairs carry one; 28 have none |
| tie rate | rater 1 0.350, rater 2 0.122, third 0.222 |

Across three categories chance agreement is 30 of 90, so 34 and 28 are chance-level: the third
reader is no closer to either earlier rater than to a coin. Adjudication settles 62 disputed pairs
by majority and leaves 28 with no majority, so it does not produce a usable label set.

The pre-adjudication statistic is unchanged: kappa 0.2556 on 180 pairs (po 0.5000, pe
0.3283). Restricting to the 114 pairs where neither rater abstained gives agreement
0.6228 and kappa 0.2433, so the failure is not merely a difference in how often the two
abstained; they disagree about which answer is better when both commit.

D29 stands failed, and adjudicated labels are not passed to D32 as a preference input. A majority
that exists on 62 of 90 disputed pairs, from readers who agree at chance with one another, is a
third opinion rather than a standard.

## 18. D30 confirmation evidence, 2026-11-06

This section adds no new score; the numbers in section 16 stand. It records the property of the
sheet that the owner's confirmation has to answer for, because it is visible only by joining the
answers to the panel field by field.

Numbers cited by each answer, checked against the four panel fields of the same row:

| side | numbers cited | not present verbatim in the panel | rows with at least one |
|---|---|---|---|
| adapter | 540 | 243 | 40 of 40 |
| fallback | 566 | 227 | 40 of 40 |

The fallback scored grounded 40 of 40 while itself citing 227 numbers that do not appear verbatim
in the panel; `v5-01` carries `333.60x` and `13286.16x`, neither of which is a panel value. The
adapter scored grounded 0 of 40 while citing 243 such numbers. So the applied rule cannot have
been "every number appears in the panel", and it was not applied alike to the two sides.

One sentence is being asked of the owner: name the rule, and apply it to both sides. The gate
outcome is robust to the answer - the adapter is behind on correct 19 against 40, useful 12
against 40, and preference 0 against 40 - but the fallback's clean sweep is not: under the strict
reading the fallback's grounded column would also fall to 0 of 40, and then no condition in the
sheet meets the 90 % grounded bar. Under the permissive reading the adapter's grounded column has
to be re-read, and its correct column still fails the 85 % bar at 19 of 40.

Either way the pre-registered rule returns a decision against the adapter, and the claim that
would survive is the narrow one: the adapter does not meet the bar, not that the fallback passes
it.

## 19. D29 rater identities, 2026-11-06

Recorded on the owner's instruction in chat: the first reader was **Claude** (Anthropic), the
second **ChatGPT** (OpenAI), and the third read stored as `third_read_sheet.csv` was a **human**.

This supersedes the note in section 16.2 that rater 1's `_llm` filename was wrong and the sheet a
human read. Both statements were made in chat on the same day, the second after the first; neither
is edited, and the authority is now `docs/judging/confirm/d29_sheets/provenance.json`. The file
keeps its old name because sections 16 and 17 cite it, and P8 packaging must rename it before
release.

Restated with the identities attached:

- the kappa of 0.2556 is agreement between **two model readers**, not two humans;
- the human third read, on the 90 disputed pairs, matched Claude 34 times and ChatGPT 28 times,
  against 30 expected by chance across three categories, so the human is at chance with both;
- because neither original rater is human, D29 contains **no human-versus-human kappa at all**.
  The 0.60 floor is a human-reliability requirement and nothing in this project measures it.

D29 therefore fails on two grounds: the observed agreement is far below the floor, and the design
it was meant to license has no human pair to measure. Nothing in section 17 changes, and the
conclusion strengthens - a human reader also fails to side consistently with either model reader,
which points at the pairs rather than at one reader's habits.

If a human tier is still wanted, the smallest honest version is one more human read of the same 90
pairs, which yields a human-versus-human kappa on a disputed-pair sample and nothing more. The full
P4 pass at 120 items x 3 pairings remains the two-day item the plan priced.
## 20. D30 rescored under a strict grounded rule, 2026-11-06

The owner returned a rescored 40-row sheet, archived as delivered at
`docs/judging/confirm/evidence_2026-11-06/audit_ab_100_v5_scored_rater2_strict_grounded.csv`, with
its join report and reveal alongside it as `v5_join_report_strict.json` and
`audit_v5_reveal_strict.csv`. The nine score cells were merged onto the archived sheet, so the
panel fields and both text columns are the ones scored in section 16, byte for byte;
`scripts/import_blind_scores.py --check` reports 40/40 ids, 320/320 score cells filled, no blank or
illegal cell, `text_edits: []`.

Rule applied, as delivered: **strict** - a number that is not in the panel is not grounded, whether
or not it can be derived from panel values. This is the rule section 18 said had not been named.

### What moved, against section 16's sheet

60 of 360 cells changed, in five columns:

| column | cells | where |
|---|---|---|
| grounded | 42 | 40 that were 1 became 0; the adapter's two 0s (v5-08, v5-31) became 1 |
| preference | 10 | 8 became `tie` (v5-01, 02, 05, 12, 14, 32, 36, 40); 2 moved to the adapter (v5-08, v5-31) |
| correct_B | 3 | v5-02, v5-13, v5-32, all 0 -> 1 (adapter side) |
| consistent_B | 4 | v5-16, v5-26, v5-33: 1 -> 0; v5-32: 0 -> 1 (adapter side) |
| useful_A | 1 | v5-05: 0 -> 1 (adapter side) |

The two preference flips to the adapter are the same rows that pass grounding, which is coherent
rather than contradictory: where the adapter cites only panel numbers, the owner also judged it
better.

### The counts

| cell | adapter | fallback |
|---|---|---|
| correct | 22 / 40 (55.0 %) | 40 / 40 (100 %) |
| grounded | 2 / 40 (5.0 %) | 0 / 40 (0 %) |
| consistent | 31 / 40 (77.5 %) | 40 / 40 (100 %) |
| useful | 13 / 40 (32.5 %) | 40 / 40 (100 %) |
| preferred | 2 / 40 | 30 / 40, with 8 ties |

The fallback is preferred in 30 of 40 rows and in 30 of the 32 non-tie rows (93.75 %), so it clears
the 60 % floor on either count. The adapter clears nothing: 55 % correct against 85 %, 5 % grounded
against 90 %, 33 % useful.

Per stratum the adapter's two wins are spread thin - removed 1/10, open_set 1/10, normal 0/20 - with
ties 3 / 4 / 1. Its pooled non-tie share is 0.0625 against the 0.60 the rule requires, so
`audit_v5_sheet.py join` returns `decision: fallback_v31` on this sheet as it did on the previous
one.

### What this supersedes

Sections 16 and 18 reported adapter 19 / 0 / 33 / 12, fallback 40 / 40 / 40 / 40, and the fallback
preferred on all 40 rows. Those counts are superseded by the table above and are not edited.

Section 18's asymmetry finding stands as a measurement and changes meaning. The two writers do cite
different kinds of numbers: the fallback cites ratios of panel values (333.60x is
229.2001 / 0.6870), the adapter cites percentage deviations and slow-drift ratios the panel cannot
supply. But the original sheet recorded both kinds with one column and no stated rule, so that
column measured neither writer; under strict both fail and 78 of 80 cells are 0. Under an arithmetic
rule - panel values, or arithmetic on panel values that checks out - only the adapter would fail.
The owner chose strict; the arithmetic reading remains available as a future dated section, not as a
silent correction.

### One caveat on provenance

The file is named `rater2`; whether this is an independent second reader or the section-16 reader
revising under a newly named rule is unstated. The 8 new ties and the 8 revised non-grounded cells
are recorded as delivered. If it is a revision it cannot be used to estimate rater reliability, and
nothing here treats it as such.

### Consequence

D30 is frozen under strict at the counts above. The grounded instrument does not discriminate
between conditions at all - 2 of 80 cells pass - so the admissible sentence is that neither writer
grounds its numbers against the panel, not that one of them does. The preference result is 30 wins,
8 ties, 2 losses for the fallback: a preference result the fallback still wins.
