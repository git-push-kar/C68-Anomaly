# Prose authoring benchmark for the TEP diagnosis system — final report

**Date:** 2026-11-16 · **Branch:** `benchmark-upgrade` (fast-forwarded to `master`) · **Rules:** `docs/PROTOCOL.md`, `docs/HANDOVER_BENCHMARK_PLAN_V2.md` §10–§29

**Status in one paragraph.** The cell table, the classifier track, the instrument audit and the selection
ladder are complete and measured. The preference tier ran and failed its pre-registered reliability floor,
so it is reported as secondary and descriptive. The one-shot confirmatory pass on the test export is
**prepared but not run**: the pass needs 3,168 generations, the batched decode path is **not
output-equivalent** to single-item decode (0 of 4 texts matched), and at the only admissible speed the job
is running at the batch size the scored cells were generated with (16), at a measured 4.47 s per item on the
longer test prompts, i.e. ≈3.9 h for the 3,168 rows. The shot is spent; the result is not yet in. E's adapters are trained but unscored.

---

## 1. What this paper claims

The claims below are the ones the evidence supports. The preference claim is **absent, not weakened**.

1. A distilled 2B prose writer does **not** reduce numeric defects relative to a deterministic assembler:
   62 of 520 rows against 19, paired exact McNemar 60 vs 17, p < 1e-06 (fit/validation set).
2. It is **worse** on grounding (33 vs 0) and on instrument-role correctness (57 vs 0).
3. It is **less order-invariant**: 368 of 520 items keep an identical defect profile across three prompt
   permutations (0.7077).
4. The mechanism instrument **does not favour** the seam-written condition over the adapter
   (12/19 = 0.632 against 14/26 = 0.538, heavy interval overlap), so no content advantage can be claimed
   for model-authored mechanism prose.
5. The classifier track separates the three production systems sharply, and a plain LightGBM on the same
   evidence features matches the full hybrid exactly (0.9940, discordant 2 vs 2, p = 1).
6. Under the declared selection rule the shipped writer is **A**, the deterministic assembler, chosen on
   measured cost after rung 5 produced a tie with D that the declared tie-break resolved.

No preference claim, no severity claim, no subsystem claim, no composite score.

## 2. Systems and data

| cell | what it is |
|---|---|
| **A** | deterministic assembler: template prose built from the stored evidence, no model call |
| **B** | the trained 2B adapter (InternVL3-2B, LoRA, seed 42), free generation at cap 512 |
| **C** | numeric seam: A's stored numbers spliced into B's prose |
| **M** | mechanism seam: B's prose with mechanism claims replaced from the knowledge base |
| **D** | A plus the validator: retries on its own output, falls back to A when retries run out |

**Fit set:** 520 held-out validation events, three menu orders each (kb, shuffled-1, shuffled-2).
**Test export:** 3,168 events, one order (kb) for the confirmatory pass.
**Instrument:** four checks — numeric fidelity against the stored evidence, grounding, instrument-role
(tag-role) correctness, and self-contradiction — plus an anchor check on quoted numbers.

## 3. The cell table (520 rows per cell)

| cell | numeric defect rows | rate | grounding | tag-role | contradiction | anchor exact / 520 |
|---|---|---|---|---|---|---|
| A | **19** | 0.0365 | 0 | 0 | 0 | **520** |
| B | 62 | 0.1192 | 33 | 57 | 0 | 188 |
| C | **19** | 0.0365 | 1 | 17 | 0 | **520** |
| M | 62 | 0.1192 | 32 | 43 | 0 | 191 |
| D | 7 | 0.0135 | 0 | 0 | 0 | 286 |

Source: `outputs/overnight_2026-11-05/n4_cells.json` (`summary`), reproduced flag-for-flag in §25.

**Seam certification** (`n4_cells.seam_certification`): C leaks 0 of B's numeric values, M leaks 0 of A's;
C leaves 88 donor sentences unused, M leaves 584. The seams are doing what they claim.

**D's ceiling.** D fell back to A on **142 of 520 rows (27.3 %)** against a declared 25 % ceiling. It is
therefore reported as **"A with extra steps"**: its numeric count of 7 is published with the fallback rate
beside it, and no claim is made that D reduces defects on the rows it wrote itself. Signed acceptance:
`docs/judging/confirm/d_ceiling_decision_2026-11-16.json`. No re-cut, no threshold moved.

## 4. Primary contrast

**B vs A, numeric-defect rows, paired.** B has 62, A has 19; discordant pairs 60 (B only) and 17 (A only);
exact McNemar p < 1e-06. This is the paper's main claim and it is measured on the fit set only — the test
confirmation is prepared and unspent (§9).

## 5. Classifier track (P2 merged row)

Items scored by all five systems: **497**. Abstention is the label `UNCLASSIFIED_OPEN_SET`; 125 gold
abstentions among them.

| system | root-cause accuracy (95 % CI) | abstain precision | abstain recall | false abstention | gate (5 %) |
|---|---|---|---|---|---|
| menu rule | **0.9980** (0.9887–0.9996) | 1.0000 | 1.0000 | 0.0000 | pass |
| hybrid | 0.9940 (0.9824–0.9979) | 0.9766 | 1.0000 | 0.0081 | pass |
| **LightGBM (gbdt)** | 0.9940 (0.9824–0.9979) | 1.0000 | 1.0000 | 0.0000 | pass |
| MLP | 0.9698 (0.9508–0.9816) | 0.9918 | 0.9680 | 0.0027 | pass |
| adapter (2B) | **0.8330** (0.7977–0.8632) | 0.8723 | 0.6560 | 0.0323 | pass |

Paired exact McNemar on correctness:

| contrast | discordant | exact |
|---|---|---|
| menu rule vs hybrid | 0 / 2 | p = 0.5 |
| menu rule vs gbdt | 1 / 3 | p = 0.625 |
| menu rule vs mlp | 0 / 14 | p = 0.000122 |
| hybrid vs gbdt | 2 / 2 | **p = 1** |
| hybrid vs mlp | 1 / 13 | p = 0.001831 |
| adapter vs gbdt | 81 / 1 | p < 1e-06 |
| adapter vs mlp | 75 / 7 | p < 1e-06 |
| gbdt vs mlp | 1 / 13 | p = 0.001831 |

**Reading.** The hybrid's threshold, menu and no-fault rules add **nothing** over a plain LightGBM on the
same features — identical accuracy and a 2-vs-2 discordance. The 2B adapter is 16 points behind all three
non-neural systems on root cause and its abstention recall is 0.656 against 0.97–1.00.

**Subsystem and severity are not reported.** Both are knowledge-base functions of the predicted fault id
and are emitted independently by none of these systems, so a separate accuracy would restate the
root-cause number — the composite score §12 prohibits.

Source: `docs/judging/confirm/p2_classifier_contrasts.json` (rebuilt 2026-11-16 with the LightGBM and MLP
rows). Note: xgboost is **not** installed or used anywhere in this codebase; the gradient-boosted row is
LightGBM 4.7.0 via `scripts/evidence_classifier.py`.

## 6. Instrument

- **Calibration set:** 46 cases, 46/46 at 1.00 on the contradiction check.
- **Self-contradiction** across all cells and all 1,560 cell-rows: **0** after the v5 → v5b corrections.
- **Tag-role:** the correction reduced spurious flags from 396 to 57 on B; the human read of 30 rows
  (`n5_examples_audited.csv`) found 20 supported claims, 10 not.
- **Parity:** 1,390 disagreements between the direct and wrapper scoring paths remain **unresolved**
  (`stages.json:N1.parity_disagreements`). The gate passed on contradiction only; parity is reported as an
  open instrument limitation, not as a pass.
- **Anchors:** counts exist in `n4_cells.summary` (§3) but are **not** derivable from the stored per-item
  flags, so they are reported from the stage summary and flagged as such in §25.

## 7. The selection rule, rung by rung

| rung | input | what it returned |
|---|---|---|
| 1 numeric gate | cell table | passers **A, C, D** (B 0.1192 and M 0.1192 above the 0.0365 ceiling) |
| 2 preference | 360 blind pairs, two readers | **failed** — κ 0.5776 < 0.60 (§8) |
| 3 mechanism validity | audited sheet, floors 15 marked / 10 items | **structurally unavailable**: only C among the passers has any tag-role claims (A 0, D 0) |
| 4 order invariance | measured for B only | no data for a passer → skipped |
| 5 cost | measured generation seconds | separates B/C/M (19,671.6 s per pass) from A/D (0 s), then **ties A with D** |
| 6 tie-break | declared rule | **A** |

**Final receipt:** `docs/judging/confirm/selection_receipt_2026-11-16_final.json` — `chosen A`,
`decided_by_rung 5`, `measured_rungs_with_input [5]`, `provisional false`, tie-break recorded verbatim,
cost block signed with `unit: seconds_per_520_row_pass`.

Rung 3's structural finding is recorded in `docs/judging/confirm/rung3_structural_status_2026-11-16.json`:
with the floors applied to passers, two eligible cells are unreachable by construction, because the two
deterministic assemblers emit stored numbers and no mechanism prose.

## 8. Preference — secondary and descriptive

360 blind pairs (120 each of C–A, C–B, M–B), two readers, one byte-identical pair.

**Reliability, reported first:** observed agreement 0.7778 (280/360), expected 0.4739,
**Cohen's κ = 0.5776** against a pre-registered floor of 0.60 signed on 2026-11-05 (D8). **The floor was
not met**, by 0.0224, roughly one standard error at n = 360.

| pairing | rater 1 | rater 2 | ties |
|---|---|---|---|
| A vs C | A 98 / C 20 (A 0.831) | A 70 / C 48 (A 0.593) | 2 / 2 |
| B vs C | C 108 / B 10 (B 0.085) | C 64 / B 53 (B 0.453) | 2 / 3 |
| B vs M | M 114 / B 0 (B 0.000) | M 112 / B 2 (B 0.018) | 6 / 6 |

The readers agree that A beats C and M beats B, and disagree on B against C — the contrast a preference
claim would rest on. Declared and signed: `docs/judging/confirm/d8_deviation_2026-11-16.json`; the paper
text is `docs/judging/preference_secondary_2026-11-16.md`.

## 9. The one-shot confirmatory pass — prepared, not run

**Prepared.**
- Test export built: `outputs/benchmark_v2/dataset/test.jsonl`, **3,168 items**, sha256 `6500235f…`.
- Test prose rows built by `scripts/build_test_orders.py`:
  `outputs/benchmark_v2/eval/prose_test/rows_test.jsonl`, 3,168 rows, all three orders, order-only
  invariant on all 3,168, 1,814 gold-abstain cases.
- Runner accepts them: `scripts/prose_pilot_run.py --rows … --orders kb --batch 1`.
- Prepass receipt: `outputs/overnight_2026-11-05/confirmatory_prepass.json`, `status
  authorised-to-launch`, receipt sha `3f969df8…`.
- Guard: `scripts/confirmatory_pass.py` refuses any receipt without a measured rung (verified: exit 2 on
  both provisional receipts, exit 0 on the final one).

**What the batch-size measurements settle.** Three comparisons, all on stored prompts:

| comparison | identical |
|---|---|
| batch 1 vs batch 4, same 4 test items | **0 / 4** |
| batch 1 vs **padding-free** length-grouped batch 8, same 8 test items | **3 / 8** |
| stored probes of the same 12 validation prompts at batch 16 vs batch 8 | **1 / 12** |

Padding is therefore not the cause: grouping prompts by exact token length and batching only inside a group
removes every pad token and still changes 5 of 8 texts. Generation is **not batch-invariant**, and no
masking or grouping trick recovers the batch-1 text — I implemented the padding-free version, measured it,
and reverted it.

That result does not block the pass, because **batch 1 was never the reference.** Every scored cell comes
from texts generated at **batch 16, cap 512, 2.47 s/item** (`prose_pilot/raw_pilot_run.json`). A batch-1 pass
would be a *different generator*, not a canonical one; the admissible configuration is the one the fit set
used. **The 2.47 s/item figure was measured on validation prompts of ~1,030 tokens; test prompts are ~3,270
tokens**, so the test pass runs slower: the runner's own checkpoint at 160/3,168 reports **4.47 s/item**,
i.e. 3,168 × 4.47 s ≈ **3.9 h**. The 33.3 h quoted
earlier applied a single-item decode time of 37.83 s to a batched pipeline and is superseded.

**Caveat on reproduction of stored texts.** Re-running eight stored validation prompts at batch 16 did not
reproduce the stored texts, and the fresh output used a different JSON schema (`fault_id` first), so the
failure is at least partly a prompt-build difference and cannot be read as non-determinism of the same call.
An exact reproduction check would need the original prompt builder; until then this pass is comparable to the
fit set by configuration, not by replayed text.

**State.** The one shot is **spent and running**: launched 16:56 on the 3,168 test rows at batch 16, writing
`outputs/benchmark_v2/eval/prose_test/run_kb16/raw_testkb_kb.jsonl`, with a measured 4.47 s/item and an ETA
of about 3.7 h from the 160-row checkpoint. D33's B-versus-A contrast cannot be reported until that file reaches 3,168 rows and the
fallback texts are generated for the same rows.

## 10. E — trained, unscored, and one wording correction

Both E adapters exist and completed: `outputs/tep_rca_v2_prose_e_seed43` and `…_seed44`, each with
`global_step 288`, `epochs_completed 3.0`, 1,523 training examples, checkpoints 48–288. They are **not
scored into the cell table**: scoring needs 2 × 520 generations, i.e. about 0.7 h at the batch-16 speed the
fit set was generated with (2.47 s/item).

**Integrity check outstanding.** The 2026-11-05 incident moved a seed-43 run into seed-42's checkpoint
directory. Before these adapters are scored, their `files_sha256` blocks and telemetry must be compared
against seed 42 to prove the runs are distinct.

**Wording correction.** The shipped adapter is seed 42; E adds 43 and 44 — **two further seeds**. The
Version B sentence "retraining three more times" is unsupported as written and must read "two further
seeds", or a fourth seed must be trained.

## 11. Limitations

1. **The preference tier failed its own reliability gate** (κ 0.5776 < 0.60). Its rates are descriptive.
2. **Rung 3 cannot be decisive** among passers: only C carries mechanism claims, by construction.
3. **The mechanism comparison that was possible favours neither side**: C 12/19 vs B 14/26, intervals
   overlapping.
4. **D breaches its declared ceiling** (27.3 % vs 25 %) and is reported as A with extra steps.
5. **Parity between scoring paths shows 1,390 disagreements** and is unresolved.
6. **Generation is not batch-invariant** (1 of 12 identical between batch 16 and batch 8, 3 of 8 with
   padding removed). The pass must therefore run at the fit set's batch size, and on the longer test prompts
   that costs 4.47 s/item (~3.9 h), not the 2.47 s/item measured on validation prompts.
7. **Subsystem and severity are not measured**; reporting them would duplicate the root-cause number.
8. **The earlier summary** `docs/reports/BENCHMARK_ACHIEVED_METRICS_REPORT.md` carries figures
   (99.64 % diagnosis accuracy, ECE 0.0009, FAR 2.0 %) that **no artefact on this checkout produces**. It
   should not be circulated until each figure is sourced; this report supersedes it.

## 12. Reproduction

```
python scripts/classifier_contrasts.py                  # §5 tables
python scripts/select_condition.py --cells outputs/overnight_2026-11-05/n4_cells.json --m 0.01 \
  --rule-committed --mechanism outputs/overnight_2026-11-05/rung3_input.json \
  --cost docs/judging/confirm/cost_basis_seconds_2026-11-16.json \
  --out docs/judging/confirm/selection_receipt_2026-11-16_final.json     # §7
python scripts/preference_agreement.py --rater1 outputs/preference_2026-11-16/sheet_rater1.csv \
  --rater2 outputs/preference_2026-11-16/sheet_rater2.csv \
  --out outputs/preference_2026-11-16/kappa.json         # §8
python scripts/mechanism_sheet.py --from-sheet outputs/overnight_2026-11-05/n5_examples_audited.csv
python scripts/build_test_orders.py                      # §9 rows
# venv_train interpreter required for generation:
#   prose_pilot_run.py --rows outputs/benchmark_v2/eval/prose_test/rows_test.jsonl --orders kb --batch 16
```

## 13. Table index for the paper

| paper table | source |
|---|---|
| 1 classifier track | §5, `p2_classifier_contrasts.json` |
| 2 prose cells | §3, `n4_cells.json` |
| 3 preference (secondary) | §8, `preference_2026-11-16_result.json` |
| 4 ablations (D vs B; E vs B) | §3 for D; **E pending** |
| 5 instrument | §6, `cases46_*`, `instrument_v5_vs_v5b.md` |
| 6 follow-up (U vs F) | `followup_sanity.json`; T deferred, so the T row is absent |
| 7 detector frontier | **not built** (P1 deferred on the per-arm threshold path) |
