# Training Run 1 — Protocol A, seed 42

**Repo:** `C:\Users\Admin\Desktop\anomaly` · **Branch:** `benchmark-upgrade` · **Environment:** `venv_train` (GPU), app env (CPU)
**Date:** 2026-10-31 · **Scope authorised:** one training run on `train_mixed.jsonl`, validation-only selection, no test scoring, no other seeds, no folds.

---

## 0. Gate table

| # | Gate item | Status | Evidence |
|---|---|---|---|
| 1 | GPU idle before launch | **PASS** | 581 MiB used, 36 °C, no torch process |
| 2 | Base-model SHA-256 matches | **PASS** | `18B4CDBC…58E7` verified in `venv_train`, see §1 |
| 3 | `train_mixed.jsonl` is the §12.1b mix, hash as mandated | **PASS** | `0f16b82b3fe18728d67476fc842160b4eeaa142031e09105ae03513d980a20d2` |
| 4 | Training completes 540/540 with no hard stop | **PASS** | 3 h 20 m 02 s, final loss 0.2466, no NaN/OOM, max 80 °C |
| 5 | `training_summary.json` + SHA-256 of every adapter file | **PASS** | 12,723 B summary, `files_sha256` for all 6 checkpoints + final |
| 6 | Scoring-path parity (fast path authorised) | **PASS** | 40/40 argmax, max \|Δp\| = 0.0 (< 0.01) |
| 7 | Checkpoint table by label scoring, selection by macro-F1 | **PASS** | 6 checkpoints × 300 items, step 400 selected (§4) |
| 8 | Selected checkpoint on full Protocol A validation | **PASS** | 822 observable + 12 unobservable rows (§5) |
| 9 | Generation at 512 cap, ≤ schema completion rule | **PASS** | 50/50 complete → no 768 re-run needed (§6) |
| 10 | Merged-adapter decode speed + generation/scoring agreement | **PASS** | §7 — adapter 36.51 s/item, merged 22.87 s/item (1.60×); §6 agreement 43/50 |
| 11 | Follow-up sanity incl. in-scope controls | **PASS** | §7 |
| 12 | `evaluate_v2.py` + synthetic tests | **PASS** | 18 tests, full suite 512 passed |
| 13 | No test component opened | **PASS** | only `outputs/benchmark_v2/dataset/val.jsonl` and follow-up `val` read |
| 14 | Split overlap / leak check / manifest hashes | **PASS** | §8 |

---

## 1. Preconditions

| Check | Value |
|---|---|
| `nvidia-smi` before launch | 581 MiB used, 36 °C, 0 % util — Ollama stopped, GPU idle |
| Base model directory | `C:/Users/Admin/Desktop/LY_C68/deepfake_project/models/InternVL3-2B` |
| `model.safetensors` SHA-256 | `18B4CDBC8EAB790F447FEAC717F4D0CECBB85CAF8DF198B3B2401C22506858E7` (unchanged; verified in `venv_train`) |
| `train_mixed.jsonl` SHA-256 | `0f16b82b3fe18728d67476fc842160b4eeaa142031e09105ae03513d980a20d2` (= the mandated hash) |
| Training rows | 2,880 = 1,919 report rows + 961 dialogue rows |
| Manifest hashes | `prompt_template fe4b8e0d…`, `kb b914fd94…`, `menu r2 / 77699552…`, `filter a3da3b5b…` |
| Test components opened | none |

## 2. Configuration (`configs/config_v2_protocolA_seed42.yaml`)

Diff against `config_v2.yaml` is exactly four things: adapter name/dir (`tep_rca_v2_protocolA_seed42`, `outputs/tep_rca_v2_protocolA_seed42`), `train_file: train_mixed.jsonl`, `save_steps: 200 → 100`, `save_total_limit: 2 → 0` (keep every checkpoint), plus `val_file: null` so the eval-loss pass is off. Hyperparameters unchanged: LoRA r=16, α=32, dropout 0.05, 7 projection targets; bf16, batch 4 × grad-accum 4 = effective 16; lr 2e-4 cosine with 3 % warmup; `max_sequence_length 4096`; 3 epochs; `logging_steps 10`; telemetry on.

The header comment still says "14 of them, ~35 MB each"; the run produced **6 checkpoints at 217 MB each** (74 MB adapter + 143 MB optimizer state). Cosmetic, unchanged in code — noted so the comment is not read as a measurement.

## 3. Part A — training

| Metric | Value |
|---|---|
| Steps | 540 / 540 (3 epochs × 180, effective batch 16) |
| Wall clock | **12,003.5 s = 3 h 20 m 02 s** (21.71 s/step mean) |
| Loss, step 10 → 540 | 1.2126 → 0.2466 (mean over run 0.3494) |
| Epoch-boundary losses | 1.21 (0.056) · 0.81 (0.111) · 0.29 (1.78) · 0.2497 (2.11) · 0.2331 (2.78) · 0.2466 (3.0) |
| Grad norm | 0.47 – 0.72, no spike |
| Peak GPU memory | 24,092 MiB (checkpointing on) |
| GPU temperature | 75–80 °C, max 80 °C (limit 85 °C for 2 min) |
| GPU clock / power | 1,785–1,830 MHz, 190–207 W |
| Hard stops hit | none — no NaN, no OOM, no step above 45 s (max 29.6 s at step 10), no thermal trip, no test access |
| Adapter | `adapter_model.safetensors` 73,916,992 B, SHA-256 in `training_summary.json` |
| Disk | 6 checkpoints × 217 MB ≈ 1.3 GB, all ignored by git (`.gitignore` keeps the summary/telemetry JSON tracked instead) |

Loss falls fast and then plateaus from ~step 380 (0.2497 → 0.2466 at step 540). The tail is flat, which is consistent with the checkpoint table below: later steps do not improve validation.

The step time is stable (20.3–29.6 s, mean 21.7 s). The 29.6 s at step 10 is the CUDA-graph/allocator warm-up, not a trend.

## 4. Part B — scoring path and checkpoint selection

**Parity first.** The fast path (`model.language_model` with the vision tower bypassed) was compared against the wrapper path on 40 validation items at batch 16 before any selection: **argmax agreement 40/40, max |Δp| = 0.0** (threshold 0.01), 0.413 s/item vs 2.164 s/item (5.2× faster). Recorded in `scoring_parity.json`; selection then ran on the fast path, as PROTOCOL requires it to be authorised.

**Checkpoint table** — label scoring, batch 16, 300-item stratified validation subsample (seen faults, both menu variants, open-set and NORMAL_TRANSIENT included):

| Step | Accuracy | Top-3 | Macro-F1 | Abstention rate | Abstain P / R | s/item |
|---|---|---|---|---|---|---|
| 100 | 0.2000 | 0.3533 | 0.2727 | 0.000 | 0.00 / 0.00 | 0.758 |
| 200 | 0.3800 | 0.6200 | 0.5605 | 0.047 | 1.00 / 0.08 | 0.753 |
| 300 | 0.4633 | 0.8033 | 0.5948 | 0.100 | 1.00 / 0.18 | 0.737 |
| **400** | **0.8333** | **0.9367** | **0.7826** | 0.480 | 0.99 / 0.86 | 0.739 |
| 500 | 0.7600 | 0.8833 | 0.7659 | 0.380 | 1.00 / 0.69 | 0.740 |
| 540 (final) | 0.7733 | 0.8867 | 0.7721 | 0.393 | 1.00 / 0.71 | 0.764 |
| 540 (checkpoint, same weights) | 0.7733 | 0.8867 | 0.7721 | 0.393 | 1.00 / 0.71 | 0.775 |

**Selected: step 400** by the pre-registered rule (macro-F1, accuracy as tie-break, later step as second tie-break). The duplicate 540 row is not an error: the final saved adapter and `checkpoint-540` hold identical weights and score identically — useful as a determinism check on the scoring path (±0.0 accuracy, ±0.000 F1 across two independent loads).

The interesting shape is between 300 and 400: accuracy jumps 0.46 → 0.83 while abstention recall goes 0.18 → 0.86. That is the model learning *when to abstain* (the open-set label), not just the fault families. After step 400 it partially unlearns it (500: abstention rate 0.48 → 0.38), which is why selecting on the final checkpoint would have cost ~2 points of accuracy and ~1.5 of macro-F1.

## 5. Selected checkpoint on full Protocol A validation (834 rows)

Selection is on validation only, as required. **822 observable rows** (faults 1,2,4,5,6,7,8,10,11,12,13,14,15 + NORMAL_TRANSIENT + UNCLASSIFIED_OPEN_SET):

| Metric | Value |
|---|---|
| Accuracy | **0.8406** |
| Top-3 | 0.9331 |
| Macro-F1 | **0.7888** |
| Abstention rate | 0.4781 (450/822 gold rows are open-set) |
| Abstention precision / recall | **0.9975 / 0.8711** |
| False-abstention / missed-abstention | 0.0027 / 0.1289 |
| ECE / Brier | 0.0364 / 0.1171 |
| Scoring time | 834.7 s (0.98 s/item at batch 16) |

Per-fault:

| Class | P | R | F1 | n |
|---|---|---|---|---|
| FAULT_01 | 0.962 | 1.000 | 0.980 | 25 |
| FAULT_02 | 1.000 | 1.000 | 1.000 | 25 |
| FAULT_04 | 1.000 | 1.000 | 1.000 | 25 |
| FAULT_05 | 0.923 | 0.960 | 0.941 | 25 |
| FAULT_06 | 0.917 | 0.880 | 0.898 | 25 |
| FAULT_07 | 1.000 | 0.960 | 0.980 | 25 |
| FAULT_08 | 0.688 | 0.880 | 0.772 | 25 |
| FAULT_10 | 0.321 | 1.000 | 0.485 | 25 |
| FAULT_11 | 0.694 | 1.000 | 0.820 | 25 |
| FAULT_12 | 0.962 | 1.000 | 0.980 | 25 |
| FAULT_13 | 0.727 | 0.960 | 0.828 | 25 |
| FAULT_14 | 1.000 | 0.800 | 0.889 | 25 |
| FAULT_15 | **0.241** | 0.520 | **0.329** | 25 |
| NORMAL_TRANSIENT | **0.000** | **0.000** | **0.000** | 47 |
| UNCLASSIFIED_OPEN_SET | 0.998 | 0.871 | 0.930 | 450 |

**Unobservable rows (faults 3 and 9), reported separately — 12 rows:** accuracy 0.500, macro-F1 0.333, abstention precision/recall 1.00/1.00, ECE 0.350. The model abstains on 6/6 open-set rows and lands on 1 of 4 FAULT_09 rows and 0 of 2 FAULT_03 rows, always as a neighbouring class (all 4 FAULT_09 misses → FAULT_15, 2 FAULT_03 misses → FAULT_10/FAULT_15). With n = 12 (clusters = 6) the CI is 0.50–0.50 on accuracy and 0.33–0.50 on macro-F1; nothing should be concluded from it beyond "faults it never trained on are not recovered".

### The two real errors

Read from `preds_val_full_400.jsonl`, not from the summary:

1. **NORMAL_TRANSIENT is absorbed by FAULT_15.** 47 normal-transient rows: **39 → FAULT_15**, 6 → FAULT_10, 1 → FAULT_13, 1 → FAULT_11, and the mean confidence on those predictions is 0.773. Since NORMAL_TRANSIENT recall is 0.000 while FAULT_15 recall is 0.520 with precision 0.241, the two classes are effectively one decision boundary: FAULT_15 is predicted 25 gold rows + 39 normal rows + others = ~104 times, of which 25 are right. The abstention precision is unaffected (0.9975) because the model does not abstain on these — it commits, and commits wrongly. This is the single largest defect in the run and is what a NORMAL_TRANSIENT/FAULT_15 audit should look at first.
2. **Open-set abstention misses are small and systematic:** 398/456 abstained; 36 of the 58 misses went to FAULT_10 (the most common confusion) and 8 to FAULT_08. Abstention recall 0.871 is therefore concentrated, not diffuse.

## 6. Generation on validation (50 items, cap 512)

| Metric | Value |
|---|---|
| Schema completion | **50/50 = 1.000** → no 768 re-run (PROTOCOL §12.2 rule not triggered) |
| Schema-valid JSON | 50/50 |
| Enum-valid (fault_id and ranked_hypotheses all in the menu) | 46/50 |
| Grounding violations | 2/50 rows (48 clean) — an unlisted tag or invented number in the prose |
| Absolute-wording hits | 0/50 |
| Mean new tokens | 310.7 (cap 512) |
| **Seconds per item** | **37.83** |
| Agreement with label scoring | **43/50** (86 %) |
| Generated accuracy | **0.98** (49/50) |

Generation decodes through the KV-cached path with the LoRA adapter applied (the adapter is loaded on a copy; nothing is merged in place). The 4 enum failures are `fault_id` values that are in the menu but for which the model also ranked a non-menu string; the prose itself stayed grounded in 48/50.

Agreement is 86 %, not 100 %, because the label scorer scores the *first field* while generation writes a whole document; the two paths share 43 decisions on 50 items. Where they disagree the generated answer is the more accurate of the two on this sample (0.98 vs 0.84 on the same items), so the label-scoring path remains the conservative one — which is what a pre-registered closed-set metric wants.

**Degenerate-output check:** the generated prose in `generation_512.json` reads as evidence-grounded RCA (see §6.1), not as a template. No row repeats its question, no row leaks an event id, and no row emits severity or confidence (the schema has no field for them).

### 6.1 One full generated answer (verbatim, item `EVT-val-f01-r321::menu`)

> The evidence shows a simultaneous decrease in A_Feed_Stream1 (XMEAS_1) and A_Feed_Flow (XMV_44) at onset, with a 219.91% and 219.75% deviation respectively. The Stripper_Steam_Valve (XMV_50) increases while Stripper_Steam_Flow (XMEAS_19) rises, consistent with a reduced A/C feed ratio where the A feed is insufficient to maintain stripper operation. The onset order matches FAULT_01 (A/C feed ratio step change) as the A_Feed_Flow decrease precedes the Stripper_Steam_Valve increase, ruling out FAULT_06 (A feed loss) which would show A_Feed_Flow decreasing first without the subsequent valve action, and FAULT_07 (C header pressure loss) which would not explain the simultaneous A feed decrease and stripper steam valve increase. The A/C feed ratio is decreasing, causing a drop in A feed flow and a rise in stripper steam valve position. Check the A feed stream valve position and flow measurement for blockage or control issues first.

It states the evidence, ranks by onset order, rules out the two nearest menu candidates by name, and keeps valve position (`XMV_44`) distinct from flow (`XMEAS_1`) — the valve/flow labelling discipline from Phase 2B survived training.

## 7. Merged-adapter decode speed (20 validation items, cap 512)

Same 20 validation items, same 512-token cap, same KV-cached loop, only the weights differ. The merged run applies `merge_and_unload()` to a **copy** of the base; nothing in `outputs/tep_rca_v2_protocolA_seed42` is touched, and the adapter path remains the audit path.

| Path | Items | Cap | Wall | s/item | tok/s |
|---|---|---|---|---|---|
| Wrapper + LoRA adapter (as shipped) | 20 | 512 | 730.19 s | **36.51** | **8.42** |
| Merged weights (LoRA folded in) | 20 | 512 | 457.36 s | **22.87** | **13.54** |

Merging is worth **1.60×** (37 % less wall clock) — the per-token LoRA gather/broadcast on 7 projection targets is not free at decode time. The adapter figure agrees with the independent 50-item generation run (§6, 37.83 s/item, +3.6 %, different sample and cache state), which is the check that the two measurements are measuring the same thing.

Practical consequence for the eval budget: at 22.87 s/item a full stratified generation pass over a 300-item test subsample is ~1.9 h per config/seed, versus ~3.0 h unmerged. Label scoring stays far cheaper (0.74–0.98 s/item, §4/§5) and is what the pre-registered selection and closed-set metrics use; merging only matters for prose generation.

> Fix applied during this step: the speed command called the model without detaching the PEFT wrapper that generation had already hit, so the first attempt produced no usable timing. One line (`model = bare(model)`, `scripts/eval_protocolA.py:459`) makes the speed path run the bare chat model through `language_model` exactly like the scoring path. It changes no numbers elsewhere; the full suite was re-run after it (512 passed).

## 8. Follow-up sanity (40 stratified + 20 in-scope controls)

The 40 stratified turns are 4 per case type (10 case types); the 20 controls are the same voice and shape but ask about a sensor the event actually stores, so the difference between the two refusal numbers is the signal. No tuning was done on these results.

| Metric | Value |
|---|---|
| Refusal on `out_of_scope` + `false_premise` | **1.00** (8/8) — the intended behaviour |
| **Over-refusal on in-scope controls** | **0.05** (1/20) — the one that matters for usability |
| Refusal on the other in-scope cases | 0.1875 (6/32) |
| Grounding clean | 0.95 (57/60) |
| Repetition of the question | 0.000 |
| Seconds per item | 8.35 |

The control set is the important number: a model that refused everything would score 1.00 on out-of-scope and 1.00 on over-refusal, and this one does not — it refuses 8/8 of the unanswerable turns and 1/20 of the answerable controls. That asymmetry is what the Stage B refusal rule was added for.

The 0.1875 on the other cases is concentrated, not diffuse: `repeated_question` 3/4, `causal_why` 1/4, `mechanism` 1/4, `valve_actuator` 1/4. The three `repeated_question` refusals are the ones to look at, because that case type is supposed to *add detail on a re-ask* rather than decline; the three grounding violations are all `number_not_in_store` (3.92/0.58, 6.96, 3.89) — the model inventing a figure, which is the failure mode the grounding check exists to catch and which is clean at 95 %.

A caveat from the sample sizes: 4 turns per case type is a smoke check, not a rate. Nothing here should be quoted as a per-case-type refusal rate until a 100+ turn subsample is scored.

## 9. Part C — `llm/benchmark/evaluate_v2.py`

CPU-only, written while the GPU trained, on synthetic predictions, no model calls, no test data. It covers: closed-set accuracy, top-3 from the probability vector (not the argmax), per-class precision/recall/support, macro-F1 that does not count an absent class as zero, confusion indexed gold × prediction, abstention precision/recall plus the two distinct errors (false abstention vs missed abstention), ECE and Brier, grounding checks (stored tags and converted numbers accepted, unlisted sensor and invented number rejected), run-clustered bootstrap CIs (1,000 resamples, seeded) and an exact McNemar between two prediction files.

`tests/test_evaluate_v2.py` — 18 tests, all passing, including hand-computed ECE/Brier on a small case, a wide bootstrap interval when runs disagree, and a round trip through a predictions file.

Full suite: **512 passed** in the app environment.

## 10. Provenance and files

| Artifact | Path |
|---|---|
| Config | `configs/config_v2_protocolA_seed42.yaml` |
| Adapter | `outputs/tep_rca_v2_protocolA_seed42/tep_rca_v2_protocolA_seed42/` |
| Summary (tracked) | `outputs/tep_rca_v2_protocolA_seed42/training_summary.json` |
| Telemetry (tracked) | `.../train_telemetry.jsonl`, `.../train_telemetry.log`, `.../train_console.log` |
| Checkpoints (ignored) | `.../checkpoint-{100,200,300,400,500,540}` — 217 MB each |
| Parity | `outputs/benchmark_v2/eval/scoring_parity.json` |
| Checkpoint table | `outputs/benchmark_v2/eval/checkpoint_table.json` |
| Selection | `outputs/benchmark_v2/eval/selected_checkpoint.json` |
| Full-val predictions | `outputs/benchmark_v2/eval/preds_val_full_400.jsonl` |
| Generation | `outputs/benchmark_v2/eval/generation_512.json` |

## 11. Pending / not done

- Nothing in this run is still pending: training (§3), selection (§4), full validation (§5), generation (§6), merged decode (§7) and follow-up sanity (§8) all have measured numbers above, and the adapter was re-verified on a fresh pristine base (`outputs/benchmark_v2/eval/test_adapter.log`, "ADAPTER VERIFICATION PASSED").
- No test component was scored. No other seed, no fold, no calibration.
- The run is provisional under the user's discard rule: if the human audit of the report or follow-up sheet later fails its gate, this run is discarded and its results are not used for decisions about the mix or filters.
