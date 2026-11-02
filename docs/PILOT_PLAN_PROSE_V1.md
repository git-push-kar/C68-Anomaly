# Prose pilot v1 — plan (not run)

**Status: nothing here has been executed.** No adapter training, no test component, no teacher call
and no GPU work happened in 3B-11. This file is the runbook for the next round, written against the
dataset that now exists (`outputs/benchmark_v2/sft/train_prose_v1.jsonl`).

## 1. What is being tested

PROTOCOL §23.2 froze the architecture: **fallback v3.1 for classified faults, the adapter for
abstention / `NORMAL_TRANSIENT` / unclassified rows**, follow-ups from the adapter, deterministic
`fallback_v3.1` as the retry. §23.1 chose that split on a 40-row sheet whose abstention column was
measured against a **broken** comparator. The pilot therefore tests one thing:

> Can an adapter trained on abstention-only prose beat the repaired deterministic explainer on the
> stratum the adapter already wins — and do it without the gates in §2 slipping?

Ties go to the deterministic source (§21.2), so the adapter has to *win* the blind sheet, not draw.

## 2. Gates (PROTOCOL §23.4, fixed before the run; validation only)

| gate | threshold |
|---|---|
| schema valid | ≥ 98 % |
| completion at 512 tokens | ≥ 95 % |
| empty narrative | 0 % |
| genuine grounding flags | ≤ 2 % of rows |
| numeric-comparison flags | ≤ 2 % of abstention/NORMAL rows |
| tag-role mismatches | ≤ 2 % |
| contradiction with the supplied diagnosis | 0 |
| over-refusal on in-scope controls | ≤ 10 % |
| refusal on `out_of_scope` + `false_premise` | ≥ 90 % |
| shuffled-menu sensitivity | reported, no threshold |
| adapter preferred over v3.1 on a fresh **blind** sheet, non-ties | ≥ 60 % |

## 3. Data

| | rows | supervised tokens (512 cap) |
|---|---|---|
| report (abstention stratum) | 562 | 223,631 total, both components |
| dialogue | 961 | — |
| **total** | **1,523** | **223,631** |

Built by `scripts/build_prose_dataset.py`; `fault_id`, `verdict`, `ranked_hypotheses` and the
classifier's top-2 are printed in the prompt and supervised nowhere. Menu order is shuffled per item
(seed 7, zero identity orders). One of 563 report candidates was dropped because the §17 cross-fitted
rule would not have produced its gold label.

SHA-256 `9d0ecdfb98edcc953a6ea1e93333b4c533c4791a67437eb7823ee770e6888e62`.
Max prompt 1,382 tokens, mean 950.2, zero rows over the 4,096 cap.

## 4. Command

Copy `configs/config_v2.yaml` to `configs/config_prose_pilot_v1.yaml` and change exactly two things:
the **training file** (`outputs/benchmark_v2/sft/train_prose_v1.jsonl`) and the **output directory**
(`outputs/prose_pilot_v1`). Hyperparameters stay at run 1's values — batch 4, gradient accumulation
4 (effective 16), 3 epochs, seed 42, `max_sequence_length` 4096 — because the comparison is against
run 1's measured 21.71 s/step.

```
./venv_train/Scripts/python.exe -m llm.train_adapter --config configs/config_prose_pilot_v1.yaml
```

**Expected cost:** 96 steps/epoch × 3 = **288 steps**, ≈ **6,252 s = 1 h 44 m** at run 1's mean. This
is an estimate, not a measurement: re-measure the first 20 steps and say so if it moves.

## 5. Checkpoints and logging

* `save_steps: 48` — six checkpoints plus the final one; keep all six until the gates are scored,
  because the last epoch is not automatically the best one.
* Log `max_memory_allocated` every 48 steps to `outputs/prose_pilot_v1/memory.jsonl`, so the
  headroom claim is a number and not a hope.
* **GPU idle definition** (for the sheet and the evaluation pass): no compute process on the card
  *and* no compute-process memory above **2 GB**. `nvidia-smi` showing a small residual allocation
  with no process is not idle, and this round skipped a step on exactly that reading.

## 6. Evaluation scripts

| step | script | what it gives |
|---|---|---|
| decode the three orders | `scripts/menu_order_decode.py` | shuffled-order sensitivity, on the exports in `outputs/benchmark_v2/eval/prose_pilot/` |
| grounding / schema | `scripts/adapter_checks_3b9.py`, `scripts/fallback_v31.py` guards | the grounding, tag-role and numeric-comparison gates |
| follow-ups | `scripts/hybrid_faithfulness.py` | over-refusal and refusal rates on the 97 evaluation turns |
| blind sheet | `scripts/make_ab_sheet_v3.py` (new seed) + `scripts/make_blind_sheet.py` | the preference gate against v3.1 |

The validation exports are already built and hashed: 520 items in KB order and two shuffles
(seeds 1, 2), plus 97 follow-up turns from 42 dialogues. The 179 quoted in the round brief did not
reproduce; 608 follow-up turns exist across the whole validation split.

## 7. Order of work

1. Train one seed (≈ 1 h 44 m), checkpoint every 48 steps.
2. Score the §2 gates on validation only. **Do not open the test component.**
3. Only if every gate passes: build a fresh blind sheet against v3.1 and run the preference gate.
4. Only if the preference gate passes: `fold_1`, `fold_2`, then further seeds (§20.3).

A failure at step 3 is a real result: it would mean the frozen architecture's abstention column was
an artefact of comparing against a comparator that could not close a sentence.
