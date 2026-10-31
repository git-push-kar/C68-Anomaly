# Mix floor: per-case dialogue targets

**Repo:** `C:\Users\Admin\Desktop\anomaly`
**Branch:** `benchmark-upgrade` (HEAD `a4e23f0`)
**Date:** 2026-10-31
**CPU only.** No teacher calls, no training, no model loaded.

---

## 1. What changed

`llm/benchmark/to_sft_followup.py::mix` no longer fills the dialogue half by tier.
Each case is filled to its own row target over the 959 slots; the report half is
untouched at 1,919 rows, so the ratio stays 2:1. Selection rules:

* **whole dialogues only** — a dialogue is never split across a boundary;
* **seeded order inside each case** (`sha256(seed|dialogue_id)`), so the sample is
  not file order;
* **targets are asserted against the corpus** before selection: a target larger
  than a case can supply raises `ValueError` instead of writing a quietly short
  file;
* **the total is balanced** back to the slot count by removing from — or adding to
  — the cases furthest from their own target, never a case already at it, with a
  ±3-row tolerance reported as `target_deviation`.

`CASE_PRIORITY` and `_priority_tier` are gone; `CASE_TARGETS` and `TARGET_ROWS`
replace them.

## 2. Rows per case (exact)

| Case | Target | **Rows taken** | Available | Δ |
|---|---|---|---|---|
| out_of_scope | 114 | **114** | 114 | 0 |
| false_premise | 120 | **120** | 178 | 0 |
| clarification_reply | 80 | **81** | 112 | +1 |
| repeated_question | 80 | **80** | 137 | 0 |
| ambiguous_question | 100 | **101** | 156 | +1 |
| hypothesis_comparison | 120 | **120** | 245 | 0 |
| valve_actuator | 140 | **140** | 565 | 0 |
| causal_why | 100 | **100** | 500 | 0 |
| mechanism | 80 | **81** | 320 | +1 |
| template_sample | 25 | **24** | 78 | −1 |
| **total** | **959** | **961** | 2,405 | **+2** |

The three +1s and one −1 are whole-dialogue rounding: 447 dialogues supply 961
rows, and a dialogue carries 1-4 turns. Deviation **+2** is inside the ±3
tolerance. `out_of_scope` is filled to its entire available supply (114 of 114) —
the case cannot grow.

Rows in the file: 1,919 report + 961 dialogue = **2,880**; achieved ratio
**1.997 : 1**.

## 3. Supervised tokens per case (real tokenizer)

| Component | Rows | Prompt p50 | Sup p50 | Sup mean | Sup total | Share |
|---|---|---|---|---|---|---|
| report | 1,919 | 1,400 | 297 | 299.6 | 574,980 | **88.9%** |
| **dialogue (all)** | **961** | 749 | 71 | 75.0 | **72,028** | **11.1%** |
| — valve_actuator | 140 | 705 | 66 | 68.2 | **9,542** | |
| — hypothesis_comparison | 120 | 781 | 72 | 74.6 | 8,951 | |
| — ambiguous_question | 101 | 762 | 87 | 86.4 | 8,724 | |
| — false_premise | 120 | 746 | 67 | 68.6 | 8,232 | |
| — out_of_scope | 114 | 754 | 66 | 70.4 | 8,020 | |
| — clarification_reply | 81 | 773 | 91 | 93.5 | 7,571 | |
| — causal_why | 100 | 752 | 70 | 72.7 | 7,265 | |
| — repeated_question | 80 | 748 | 75 | 74.8 | 5,981 | |
| — mechanism | 81 | 751 | 70 | 72.1 | 5,844 | |
| — template_sample | 24 | 757 | 63 | 79.1 | 1,898 | |

Truncated answers: 0. Prompts at the 4,096 cap: 0. Steps: **180 / epoch**, **540
for 3 epochs** at effective batch 16.

**Under the tier fill (before):** dialogue 959 rows / 73,949 supervised tokens
(11.4%); valve_actuator **2 rows, 95 tokens**. **Now:** valve_actuator 140 rows,
9,542 tokens. The dialogue share of supervised tokens barely moved (11.4% →
11.1%) because the total is fixed by the 2:1 ratio; what changed is *which* cases
hold those tokens, which was the whole point.

## 4. PROTOCOL §12.1b (dated 2026-10-31)

`docs/PROTOCOL.md` §12.1b states that it **supersedes** §12.1's tier fill, gives
the reason (tier 3 received 17 of 1,463 rows), lists the ten targets and the 959
total, records the whole-dialogue/rounding/tolerance rule, the assertion on
availability, and the measured result above. Written before any adapter exists —
no model result has been produced at any point in this step.

## 5. Hashes

| Artefact | SHA-256 |
|---|---|
| `sft/train_mixed.jsonl` (**new**) | `0f16b82b3fe18728d67476fc842160b4eeaa142031e09105ae03513d980a20d2` |
| `sft/train_mixed.jsonl` (tier fill, immediately before) | `6796ecce8fe351ab58db2e74a511ac9a68a6e99d017d332c96ec2fc4ddffbf34` |
| `sft/train.jsonl` (report rows, unchanged) | `c033f5e1335ed811e668467776df59251cc8861a3fa144b1e38728f360d9a958` |
| `sft/followup_train.jsonl` (dialogue rows, unchanged) | `f78964d309854d01cd412c9ee0f0e6128e2176c842cf988330a550cbfdb0211e` |

All four are recorded in `outputs/benchmark_v2/sft/mix_summary.json`
(`train_mixed_sha256`, `report_train_sha256`, `dialogue_rows_file_sha256`) next to
the summary of the mixture they belong to.

## 6. Checks

| Check | Result |
|---|---|
| Leak check on every mixed prompt (961 dialogue rows) | **0 violations**; 0 rows mention `severity` or `confidence` |
| Answer shape per component | report rows: the slim five fields (1,919); dialogue rows: `{"answer": …}` (961); 2 distinct system prompts |
| Split overlap (`check_splits_v2.py`) | **PASS** — unique keys, no train/val/test overlap, fault-disjoint where required, recorded sha256 matches |
| Open-set scan | **0** of 481 accepted abstention answers name the removed fault |
| Eval-export row counts | val **2,408** (`matches_dataset: true`), test **10,019** (`matches_dataset: true`) |
| Full test suite, app env | **494 passed** |
| Full test suite, `venv_train` | **494 passed** |

---

## Gate

| Gate item | Status | Evidence |
|---|---|---|
| Exact rows per case | **PASS** | 114 / 120 / 81 / 80 / 101 / 120 / 140 / 100 / 81 / 24 = 961, deviation +2 (§2) |
| Token shares | **PASS** | dialogue 11.1% of 647,008 supervised tokens, per case in §3 |
| PROTOCOL §12.1b dated | **PASS** | 2026-10-31, supersedes §12.1, before any model result |
| Hashes recorded | **PASS** | new and previous `train_mixed`, report rows, dialogue rows, all in `mix_summary.json` |
| Checks clean | **PASS** | leak 0, split overlap PASS, open-set 0, eval exports match |
| Tests pass | **PASS** | 494 in both environments |
| No training, no teacher calls | **PASS** | CPU only |
| Pushed | **PASS** | `a4e23f0` on `origin/benchmark-upgrade` |

---

## 7. What is still open

* **`out_of_scope` is capped by its corpus.** Its 114-row target consumes all 114
  rows that exist. Raising that case, or any future re-balancing, needs more
  accepted out-of-scope turns, which the replay could only recover from prose that
  already exists.
* **The human audits remain unscored** (0 of 1,200 cells across the two 100-row
  sheets), and **the training run is still unapproved**.
