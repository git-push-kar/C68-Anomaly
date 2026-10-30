# Phase 2B wrap-up 2 and Phase 3A environment + smoke run

Repo: `C:\Users\Admin\Desktop\anomaly`, branch `benchmark-upgrade` (base `c1ad87e`).
Total teacher calls: **150 of the 150 allowed** (81 in pass 1, 69 in pass 2).
No full training. Installs went **only** into `venv_train`.

---

## 1. Summary

| question | answer |
|---|---|
| Scaffolding in accepted prose | **90 answers** said "decided diagnosis", 7 "candidate menu", 15 "affected area", 1 "KB"; all are template phrases, now rejected |
| Direction rule | scoped to XMEAS; the teacher menu reports a valve's *stored trend* instead of a flow direction |
| Fault 13 | KB says `decrease`, evidence says `increasing` in **102/102** train events; 79 of 119 accepted answers assert a direction and **4 contradict** the evidence |
| Re-calls | 117 items called in 150 calls, 107 recovered, 10 dropped; **62 items had no prose and were excluded** |
| Final train set | **1,868** items: 1,328 classified / 442 unclassified / 98 normal; unclassified share **24.97%** |
| Checks | 5,870 prompts, 0 leaks; splits PASS; open-set scan 0/482; audit v3 100 rows |
| Environment | `venv_train`: torch 2.13.0+cu130, transformers 4.57.6, peft 0.20.0, timm 1.0.30; **18,464,768** trainable params |
| Smoke run | 20 steps, **17.01 s/step**, loss 1.8253 → 0.7355, peak **17.92 GB**; hand CE matches the model loss to 5e-4 |
| Generation speed | **4.0 tokens/s**, 24.0 s/item (forward-only decode, 96-token cap) — the single most important number in this report |

---

## 2. Task A1 — scaffolding scan

### How the phrase list was derived

[`llm/benchmark/scaffolding.py`](llm/benchmark/scaffolding.py) builds the same
teacher prompt the builder builds, keeps the wrapper (the teacher system prompt
plus everything after the evidence and menu), and reads the phrases out of it:

* **headings** — a label line that is not a `- field:` bullet and has more than one
  word: `decided diagnosis` (from `DECIDED DIAGNOSIS (do not change):`), and
  `return exactly`. These match anywhere, because they cannot occur in plant prose.
* **labels** — inline field labels: `the disturbance`, `affected area`,
  `considered and rejected`. These match only when used *as* labels (followed by a
  colon or by "given/listed/above"), because "the disturbance is a sticking valve"
  is ordinary process prose.
* **identifiers** — quoted output keys from the template's own `keys` line:
  `physical_reasoning`, `narrative`. A quoted English word (`"flow"`, from the
  system prompt's valve example) is **not** taken as a field name.
* **menu references** — bigrams around the menu word: `candidate menu`,
  `listed candidate`.

Exemption: an abstention is *required* to name the nearest listed candidates it
rules out, so `listed candidate` is not a rejection on an abstention item.

### What it finds in the 1,849 accepted answers (before the fix)

| phrase | answers | where |
|---|---|---|
| `decided diagnosis` | **90** | f14 23, f15 15, f4 14, f10 12, f12 6, f13 6, f5 6, f7 3, f2 2, f11 2, f8 1 |
| `candidate menu` / `menu` | 7 | f16 1, f20 1, 5 normal excursions |
| `affected area` | 15 | f10 5, f15 4, f7 2, and singles |
| `listed candidate` | 47 | all abstention items (exempt by contract) |
| `knowledge base` / `KB` | 1 | f15 |

Only the first two groups are rejected: 102 rule firings over **98 items** (some
answers quote two phrases). The `affected area` and `menu` occurrences did not
match the label/menu rules' usage test and were left alone; `listed candidate` is
exempt.

### Ten examples (raw teacher text, now rejected)

- `EVT-train-f04-r049` — "The evidence shows a step change in Reactor_Cooling_Water_Flow [XMV_51] … **the decided diagnosis** FAULT_04 …"
- `EVT-train-f14-r003` — "… trending upward with a residual share of 0.300 … **the decided diagnosis** FAULT_14 …"
- `EVT-train-f14-r010` — "… trending increasing (z=0.08), but **the decided diagnosis** … implies the valve should be stuck closed …"
- `EVT-train-f15-r058` — "… despite the valve position increasing … This contradicts FAULT_05 … **the decided diagnosis** …"
- `EVT-train-f10-*` (12 answers) — "**the decided diagnosis** FAULT_10 is consistent with a C feed temperature variation …"
- `EVT-train-f16-*`, `EVT-train-f20-*`, 5 normal items — "**the candidate menu** does not list a matching family …"
- `EVT-train-f15-*` — "… the **knowledge base** describes a valve sticking …"
- `EVT-train-f04-*` — "**affected area:** feed system, with …"
- `EVT-train-f13-*` (6 answers) — "**the decided diagnosis** FAULT_13 (reaction kinetics slow drift) …"
- `EVT-train-f12-*` (6 answers) — "**the decided diagnosis** FAULT_12 …"

Replayed against the cache: **98 accepted answers become invalid** and were queued
for re-call.

---

## 3. Task A2 — direction

**(a) Scope.** `filters._direction_conflict` now returns `None` for any XMV tag:
for faults 1, 4, 5, 6, 7, 11, 12, 14 and 15 the KB's `initiating.sensor` is a valve
*position* while its `direction` describes the *flow*, and the stored trend
disagreed with the KB in every fault 7 (120/120), 14 (120/120) and 15 (26/26)
event. Measurements keep the rule.

**(b) Teacher prompt.** `teacher.teacher_menu` rewrites only the teacher's copy of
the menu. For an XMV-initiating entry the signature becomes the position's stored
trend, or says the tag is absent:

```
- FAULT_07: C header pressure loss (reduced availability) — valve position XMV_45,
  trend in this event: increasing; a valve position is not a flow
```

The published menu is untouched, so a student never sees the rewritten line
(`test_teacher_menu_is_teacher_only`). The teacher system prompt gains one sentence
saying a position trend is not a flow direction.

**(c) `direction_inconsistent` over every accepted answer.** Before the re-calls
(1,849 answers): **12 flags in 12 answers** — f13 4, f17 3, f2/f10/f12/f15/f20 1
each; **all measurements**; all basis `trend`; no valve tag (they are skipped by
design). After the re-calls (1,868 answers): **8 flags in 8 answers** — f17 3,
f2/f10/f12/f15/f20 1 each, still all measurements and all `trend`. Examples:

| fault | sensor | stored | sentence |
|---|---|---|---|
| 2 | XMEAS_24 | increase | "…the onset order confirms Component_B_Stream6 deviation precedes…" |
| 10 | XMEAS_19 | increase | "…causing a gradual decrease in stripper steam flow…" |
| 12 | XMEAS_11 | increase | "…XMEAS_22 … as a dominant residual contributor…" |
| 15 | XMEAS_28 | increase | "…XMEAS_28 and XMEAS_6 as dominant variables…" |
| 17 | XMEAS_21 | increase | "…reactor cooling water outlet temperature is dropping sharply…" (×3) |
| 20 | XMEAS_20 | increase | "…Compressor_Work is decreasing (-1.23%)…" |

These are advisory. Three of the f17 flags are the same sentence pattern, and one
names a *consequence* rather than the variable itself — the precision problem the
flag has always had.

**Fault 13 specifically.** Of 119 accepted fault-13 answers, **79 assert a
direction for XMEAS_7**; **4 say it decreases**, contradicting the stored trend
(increasing) and matching the KB's wrong sign. The other 75 say increasing, which
is what the evidence shows. Before the re-calls: 81 asserted, 5 contradicted.

**(d) Proposed KB sign (not applied).** Across the train split, `Reactor_Pressure`
(XMEAS_7) is `increasing` in **102 of 102** events where it is ranked. The KB's
`initiating.direction` for fault 13 should therefore be **`increase`**, not
`decrease`. The perturbation is a slow drift whose *magnitude* grows; the evidence
records the resulting pressure as rising. This is a KB edit, which moves the KB
hash, the menu text and every prompt, so it awaits your decision. Note that
`_direction_conflict`'s vocabulary does not currently include the gerund
"increasing", so extending it *before* the KB fix would reject the 75 correct
answers — the two changes must land together.

---

## 4. Task A3 — re-calls

Budget: 150 teacher calls. The plan needed 165 items (98 scaffolding + 67 with no
prose), which at the measured retry rate is ≈190 calls, so the budget was spent in
priority order: items with no prose first, then scaffolding, and the run stopped
when the cap was reached.

| pass | items requested | items called | API calls | retried | dropped after retry | seconds | s/call |
|---|---|---|---|---|---|---|---|
| 1 (no prose first, + 4 fault-13) | 71 | 60 | 81 | 21 | 9 | 801 | 13.4 |
| 2 (scaffolding + remainder) | 118 | 57 | 69 | 12 | 1 | 615 | 8.9 |
| **total** | **165** | **117** | **150** | **33** | **10** | **1,416** | **11.9** |

Acceptance of the called items: **84 accepted on the first attempt (71.8%)**, 33
retried, **23 recovered**, 10 dropped → **107 of 117 = 91.5%**. Remaining drops are
10 rejected + **52 items the budget never reached**.

Consequence: **62 items had no valid prose and were excluded** from the final train
set (`--exclude-rejected` now tests the prose, not the status). They are
concentrated where scaffolding was most common:

| family | items before | final | retained | family | items before | final | retained |
|---|---|---|---|---|---|---|---|
| FAULT 14 | 101 | 83 | **0.822** | FAULT 10 | 106 | 105 | 0.991 |
| FAULT 15 | 103 | 86 | **0.835** | FAULT 12 | 101 | 101 | 1.000 |
| FAULT 05 | 106 | 105 | 0.991 | FAULT 13 | 106 | 105 | 0.991 |
| FAULT 04 | 109 | 108 | 0.991 | FAULT 07 | 101 | 101 | 1.000 |
| FAULT 02 | 108 | 108 | 1.000 | FAULT 08 | 110 | 110 | 1.000 |
| FAULT 01/06/11 | — | — | 1.000/1.000/1.000 | FAULT 16-20 | 52-53 | 49-51 | 0.94-0.98 |
| OPEN_SET | 193 | 191 | 0.990 | NORMAL | 117 | 98 | **0.838** |

Every family stays above the 0.70 floor; the lowest are fault 14 (0.822), fault 15
(0.835) and NORMAL (0.838).

---

## 5. Task A4 — final train set

| class | items |
|---|---|
| classified | 1,328 |
| UNCLASSIFIED_OPEN_SET | 442 |
| NORMAL_TRANSIENT | 98 |
| **total** | **1,868** |

- Unclassified share of faulty items: **442 / 1,770 = 24.97%** (cap 25%; the cap
  re-applied after exclusion dropped 9 more items, all fault 16-20).
- Manifest rebuilt; `files_sha256.train.jsonl = feea8b065d69703304d8112c12fe2a74750e4b0a68a3ccc3107a6f86c190a3a2`.
- **Leak check**: 5,870 prompts across the three splits, **0 violations** of the
  forbidden fragments or the split/fault patterns.
- **Split overlap**: `check_splits_v2.py` → PASS (unique keys, no train/val/test
  overlap, fault-disjoint folds, recorded SHA-256 match).
- **Open-set scan**: 487 abstention records, 482 accepted, **0 naming the removed
  fault**.
- **Audit sheet v3**: 100 rows, 5 per family including open-set and
  normal_transient, with teacher text on every row and a `scaffolding` column that
  is **blank on all 100** (the dataset no longer contains the defect). Not scored.

---

## 6. Task A5 — SFT export

[`llm/benchmark/to_sft.py`](llm/benchmark/to_sft.py) writes
`{item_id, system, question, answer}`:

- `system` is `prompt_v2.SYSTEM_PROMPT_V2` — never `llm.dataset.SYSTEM_PROMPT`,
  which still carries the fault→severity rubric.
- `question` is the rendered student prompt.
- `answer` is the report body as JSON: `fault_id, verdict, abstain,
  primary_subsystem, ranked_hypotheses, mitigation, table_selection,
  suggested_followups, clarifying_question, physical_reasoning, narrative`.
  **No `severity`, no `confidence`.**

Guards (each with a test): empty system raises; `severity`, `affected subsystem` or
`candidate affected subsystem` anywhere in system+question raises; an abstain label
other than `UNCLASSIFIED_OPEN_SET`/`NORMAL_TRANSIENT` raises; a row without prose
is not exported; a missing prose field raises. Evaluation splits are **skipped**
rather than written with blank prose.

One full row (fault 01, abridged in the middle):

```json
{"item_id": "EVT-train-f01-r001::menu",
 "system": "You are a control-room diagnostic assistant for the Tennessee Eastman Process (TEP). ...",
 "question": "A new anomaly has been detected in the Tennessee Eastman Process.\n\nEvent: EV-b05f8f4520a1\nOverall anomaly score: 258.770\n\nDETECTORS\n- LSTM autoencoder reconstruction error 21.0718 (threshold 0.6870, ratio 30.67x, alarmed True)\n- CVA Q-statistic 11092.8 (threshold 194.6, ratio 56.99x, alarmed True)\n- Fused trigger: both (mixed change)\n\nTOP DEVIATING SENSORS\n- A_Feed_Flow [XMV_44, valve position, manipulated variable]: z=17.97, deviation 221.58%, trend increasing\n...\nCANDIDATE MENU (answer with one of these fault_id values):\n- FAULT_01: A/C feed ratio step change ...",
 "answer": "{\"fault_id\": \"FAULT_01\", \"verdict\": \"classified\", \"abstain\": false, \"primary_subsystem\": \"feed_system\", \"ranked_hypotheses\": [\"FAULT_01\", \"FAULT_06\", \"FAULT_07\"], \"mitigation\": \"Inspect the Stream 4 A/C ratio controller setpoint in the DCS. ...\", \"table_selection\": \"evidence\", \"suggested_followups\": [\"What are the sensor values?\", \"What is the recommended action?\", \"What is the evidence this is an anomaly?\"], \"clarifying_question\": null, \"physical_reasoning\": \"The evidence shows a step change in the A/C feed ratio ...\", \"narrative\": \"...\"}"}
```

1,868 rows written to `outputs/benchmark_v2/sft/train.jsonl`.

---

## 7. Tasks B6-B7 — `venv_train`

Created with `python -m venv venv_train`; nothing outside it was installed, and the
app environment still passes its suite (below).

| package | version | note |
|---|---|---|
| torch | **2.13.0+cu130** | the requested pin exists for cu130; `cu128` tops out at 2.11.0 |
| transformers | **4.57.6** | |
| peft | **0.20.0** | |
| timm | **1.0.30** | the InternVL runtime's `requirements.txt` is absent and its README never names a version, so this is the latest that works — recorded, not assumed |
| accelerate | 1.15.0 | |
| pandas 3.1.0, scikit-learn 1.9.1, pydantic 2.14.2, numpy 2.4.6 | | pulled in by the repo's own import chain (`preprocessing.scaler`, `llm.response.schemas`), not by the trainer |
| einops | 0.8.2 | required by `modeling_intern_vit.py` |

`pip freeze` → [configs/env_train_freeze.txt](configs/env_train_freeze.txt), 45
lines, SHA-256 `317508b6b60cda3db51a0ab97f9a72e983afc9bb99fabddef49ae1bd095e8377`.

**Driver note.** `nvidia-smi` reports driver 596.71 with 24,564 MiB. The cu130 wheel
is installed and `torch.cuda.is_available()` is True, so no deviation from the
requested torch pin was needed.

**Verification** ([scripts/verify_training_env.py](scripts/verify_training_env.py)):

| claim | result |
|---|---|
| base model directory | exists |
| `model.safetensors` SHA-256 | `18B4CDBC8EAB790F447FEAC717F4D0CECBB85CAF8DF198B3B2401C22506858E7` — **exact match** |
| loads with `trust_remote_code` | `InternVLChatModel`, 2,088,957,440 base parameters |
| one forward pass | logits `[1, 13, 151674]` |
| LoRA trainable parameters | **18,464,768** — exact match with the earlier run |
| app environment unchanged | `python -m pytest -q` → **290 passed** on the app interpreter |

---

## 8. Task B8 — `configs/config_v2.yaml`

Copy of `configs/config.yaml`; hyperparameters unchanged (`max_sequence_length`
stays 4096, LoRA r/alpha 16/32, batch 4 × accum 4, 3 epochs, lr 2e-4, bf16,
gradient checkpointing). Changed: `paths.llm_dataset_path →
outputs/benchmark_v2/sft`, `adapter_name → tep_rca_v2_seed42`, `adapter_dir →
outputs/tep_rca_v2_seed42`, plus a header explaining the loader. The trainer's code
path is untouched; it already accepts `{question, answer, system}`.

---

## 9. Task B9 — smoke run

`venv_train/Scripts/python scripts/smoke_train.py` — 20 optimizer steps on the
report task, then 20 validation generations.

**Loss check (same batch, same logits).**

| quantity | value |
|---|---|
| model-reported loss | 1.7890625 |
| hand-computed masked CE | 1.7885442 |
| absolute difference | **0.0005** (bf16 rounding) |
| supervised tokens / masked prompt tokens in the batch | 1,821 / 5,807 |
| supervised fraction | **23.87%** |
| answer tokens per example | **455.2** |
| prompt tokens per example | **1,451.8** |

The 0.0005 gap is the bf16 accumulation difference, so the trainer is computing the
masked assistant-token cross-entropy and nothing else.

**Training.**

| quantity | value |
|---|---|
| steps | 20 |
| wall time | **340.2 s** |
| time per optimizer step | **17.01 s** (effective batch 16) |
| peak GPU memory | **17.92 GB** of 24.6 GB |
| loss, step 1 → step 20 | **1.8253 → 0.7355** |
| mean reported loss | 1.0908 |

An earlier identical run took 480.8 s (24.0 s/step) with the last steps drifting to
36 s/step; the difference is thermal/clock variation, so the step time is
17-24 s and the projection uses the measured 17.01 s.

**Projection (3 epochs).** 1,868 rows / 16 effective = **117 steps per epoch**;
351 steps total → **1.66 h per configuration**, **≈14.9 h** for 3 configurations ×
3 seeds (range 14.9-21.1 h across the two measured step times). Excludes generation
and evaluation overhead.

**Generation** (greedy, unbatched, unmerged adapter, forward-only decode as
`llm/inference.py` does, 96-token cap):

| quantity | value |
|---|---|
| items | 20 |
| seconds per item | **23.98 s** |
| tokens per second | **4.0** |
| mean new tokens | 96.0 (every item hit the cap) |

**This is the most important number in the report.** Four tokens per second means a
400-token answer takes ~100 s, and evaluating 24,000 generated items for one seed
would take **~160 hours**. Two causes, both fixable and neither measured yet:
the decode loop carries no KV cache, and the adapter is not merged. Either a
KV-cached decode or a merged model should be measured before any evaluation budget
is committed; until then, Phase 4 evaluation is not affordable.

---

## 10. Changes

| commit | contents |
|---|---|
| `6e1649e` | scaffolding module + rule, XMV direction scope, teacher-only menu rewrite, 13 tests |
| `8797410` | `to_sft`, `verify_training_env`, `plan_teacher_recalls`, `config_v2.yaml`, freeze, `.gitignore` |
| `0bd388d` | export guard fixes for abstentions/normal items, `--exclude-rejected`, `smoke_train` |
| `4adae6b` | exclusion by prose rather than status; audit sheet v3 with `scaffolding` |
| this commit | this report |

Data artifacts rebuilt in place: `train.jsonl` (1,868), `manifest.json`,
`audit_v3.csv`, `sft/train.jsonl` (1,868).

---

## 11. Tests

| suite | result |
|---|---|
| app environment, full | **290 passed** in 9.1 s (268 before; 22 added) |
| `tests/test_benchmark_scaffolding.py` | 13 passed |
| `tests/test_to_sft.py` | 9 passed |

---

## 12. Gate

| gate item | status | evidence |
|---|---|---|
| Scaffolding phrases derived from the template, not hand-written | **PASS** | §2; `derived_phrases()` reads the built prompt |
| Counts by fault + 10 examples | **PASS** | §2 (90 / 7 / 15 / 47 / 1; 10 examples) |
| `teacher_scaffolding` rule + replay count | **PASS** | 98 accepted answers invalidated |
| Direction scoped to XMEAS | **PASS** | `_direction_conflict` returns early for XMV; test both ways |
| Teacher prompt changed only for XMV | **PASS** | §3(b); `test_teacher_menu_is_teacher_only` |
| `direction_inconsistent` over all accepted, by fault/kind/basis + 10 examples | **PASS** | 12 → 8 flags, all measurements, all `trend`, §3(c) |
| Fault 13 assertion/contradiction counts | **PASS** | 79 assert, 4 contradict of 119 |
| Fault-13 KB sign proposed from train trends, not applied | **PASS** | 102/102 increasing → propose `increase`, §3(d) |
| Re-call acceptance by fault, remaining drops | **PASS** | 117 called, 107 recovered, 10 dropped, 52 unreached, §4 |
| Final class counts ≤ 25% unclassified | **PASS** | 442 / 1,770 = **24.97%** |
| Manifest + hashes rebuilt | **PASS** | §5 |
| Leak check / split overlap / open-set scan | **PASS** | 5,870 prompts 0 violations; checker PASS; 0 of 482 |
| Audit sheet v3 with `scaffolding` | **PASS** | 100 rows, 0 with scaffolding text |
| SFT export + guards, no severity/confidence | **PASS** | §6, 9 tests |
| `venv_train` created, nothing else touched | **PASS** | §7; app suite 290 passed |
| Base model hash, load, forward, 18,464,768 params | **PASS** | §7 |
| `configs/config_v2.yaml` | **PASS** | §8 |
| Smoke: memory, s/step, loss 1 and 20, hand CE, token counts, projection | **PASS** | §9 |
| Measured seconds per item / tokens per second | **PASS** | 23.98 s, 4.0 tok/s |
| Full test suite in the app environment | **PASS** | 290 passed |
| Full teacher job beyond 150 calls | **STOPPED** | 52 items unreached; excluded from train |
| Full training | **NOT RUN** | by instruction |

---

## 13. Deviations

- **D1 — 52 of the 165 planned re-calls were not made.** The budget was 150 calls
  and the plan needed ~190; they were spent on the items with no prose first. The
  unreached items are excluded rather than left as label-only rows.
- **D2 — `--exclude-rejected` excludes on prose, not status.** An unreached item is
  "not_called", not "rejected", and both are equally unusable as a prose row.
- **D3 — `timm==1.0.30` is not a pinned target.** The InternVL3 directory records no
  version; it is the latest that loads, and it is recorded in the freeze.
- **D4 — three extra packages in the venv** (pandas, scikit-learn, pydantic) are
  needed because importing `llm.benchmark` reaches the response layer and the
  preprocessing scaler. They were not needed by the trainer itself.
- **D5 — generation uses a forward-only loop, not `model.generate`.** InternVL's own
  `generate` asserts on image-context tokens and a text-only prompt has none; the
  application already decodes this way. The measured 4 tok/s reflects that path and
  is therefore a worst case, not a general statement about the model.
- **D6 — the 4 fault-13 items flagged by the advisory check were re-called** even
  though they were already valid, because their prose argued with a KB entry that
  is wrong. After the re-call the final set still has 4 answers that contradict the
  stored trend: the KB is still wrong, so this cannot be fixed by re-calling.

---

## 14. Open questions

1. **Fault 13's KB sign** — approve `decrease → increase`, and extend
   `_direction_conflict`'s vocabulary to the gerund at the same time? Doing either
   alone is worse than doing neither.
2. **The 52 unreached items** — a further ~60 calls (~12 min) would recover most of
   them; they are concentrated in faults 14, 15 and NORMAL, the three families with
   the weakest retention.
3. **Generation budget** — 4 tok/s makes evaluation unaffordable. Approve a
   KV-cache/merged-adapter benchmark before committing Phase 4, or plan on a
   subset of the evaluation set.
4. **Window size** — 2,560 tokens would fit every row (max 2,065) and cut attention
   cost; it also moves the run off "matches the original", so it is your call.
