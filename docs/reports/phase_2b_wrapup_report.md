# Phase 2B wrap-up and Phase 3A preparation

Repo: `C:\Users\Admin\Desktop\anomaly`, branch `benchmark-upgrade` (base `8f07392`).
CPU only. **No teacher calls, no training, no installs.** The only model touched
was the tokenizer (9 s, CPU) and one base-model load attempt that failed before
allocating weights (missing `timm`).

---

## 1. Summary

| question | answer |
|---|---|
| Do the two rejection tables reconcile? | **No, and not for the reason assumed.** No open-set double counting: the tables printed *rule firings* under an `items` heading. First pass is 289 items / 300 codes; final is 90 items / 95 retry codes. |
| Is the direction rule wrong for 14, 07, 15? | **Yes for all three, and for every valve-position initiating variable.** The KB names a *flow* and the tag is a *valve position*; the stored trend is opposite in 120/120 of fault 14's events, 120/120 of fault 7's, 26/26 of fault 15's. |
| Are dropped items random? | **No — they are the weak-evidence tail.** Median fused ratio 11.7 vs 27.9, median top-\|z\| 0.39 vs 3.97. |
| Train set excluding the 90 | 1,849 items: 1,307 classified, 442 unclassified, 100 normal. Unclassified share of faulty is **25.27%**, 7 items over the 25% cap. |
| Does the environment match the training run? | **No.** `peft` and `timm` missing, `transformers` 5.15.0 vs 4.57.6, `torch` 2.11.0+cu128 vs 2.13.0+cu130. Base model hash verified exact. |
| Can a smoke run start? | **Not yet** — the base model cannot be constructed without `timm`, and the adapter cannot be wrapped without `peft`. Both are installs, which need approval. |

---

## 2. Task A1 — the two tables, reconciled

Definitions now enforced in [`scripts/analyze_teacher_rejections.py`](scripts/analyze_teacher_rejections.py)
and asserted at the end of `reconcile()`: one item, one family (from the cache
record, so an open-set episode is counted under `OPEN_SET` and **not** also under
its event's fault); `items` counts items; `codes` counts rule firings and may
exceed items; `primary` counts one item under its first code only.

**Totals.** 1,939 items = 1,849 accepted + 90 dropped. First pass: **289 items**
rejected across **300 code occurrences** (11 items failed two rules). Retried:
289 items, 199 recovered, 90 dropped. Final: **90 items**, **95 retry-code
occurrences** (5 items failed two rules on the retry).

**Where 288 came from.** The run log's `retried: 288` is correct for the run: the
cache holds 289 records with `attempts == 2`, and one of them is a line carried
over from an earlier run, not retried in this one. `1,914 + 288 = 2,202` API
calls, exactly the run statistic.

**Where 300 and 95 came from.** 300 is the number of *first-attempt* rule
firings (143 `number_not_in_store`, 100 `direction_conflict`, 24
`sensor_not_grounded`, 20 `unknown_tag`, 13 `reasoning_length`), and 95 is the
number of *retry*-attempt firings (47, 28, 12, 6, 2 in the same order). Both were
printed in a column headed `items`. The open-set hypothesis is disproved: the
`OPEN_SET` family appears once, at 20 first-pass rejections and 9 drops.

### First pass — reissued

| family | items | rejected 1st pass | retried | recovered | dropped | codes (occurrences) |
|---|---|---|---|---|---|---|
| FAULT 14 | 101 | 53 | 53 | 39 | 14 | direction_conflict 53 |
| FAULT 15 | 103 | 40 | 40 | 25 | 15 | number_not_in_store 30, direction_conflict 11, unknown_tag 5 |
| NORMAL | 117 | 36 | 36 | 19 | 17 | number_not_in_store 30, sensor_not_grounded 4, unknown_tag 2 |
| FAULT 07 | 101 | 26 | 26 | 16 | 10 | direction_conflict 22, number_not_in_store 4 |
| OPEN_SET | 193 | 20 | 20 | 11 | 9 | number_not_in_store 13, unknown_tag 5, sensor_not_grounded 3 |
| FAULT 05 | 106 | 18 | 18 | 14 | 4 | number_not_in_store 12, direction_conflict 3, reasoning_length 2, sensor_not_grounded 1, unknown_tag 1 |
| FAULT 01 | 101 | 15 | 15 | 14 | 1 | number_not_in_store 12, unknown_tag 3, sensor_not_grounded 1 |
| FAULT 08 | 110 | 14 | 14 | 10 | 4 | number_not_in_store 12, reasoning_length 2 |
| FAULT 10 | 106 | 13 | 13 | 9 | 4 | number_not_in_store 9, sensor_not_grounded 5 |
| FAULT 13 | 106 | 12 | 12 | 10 | 2 | direction_conflict 5, number_not_in_store 4, sensor_not_grounded 3 |
| FAULT 12 | 101 | 10 | 10 | 9 | 1 | reasoning_length 7, direction_conflict 1, number_not_in_store 1, sensor_not_grounded 1 |
| FAULT 06 | 107 | 7 | 7 | 5 | 2 | number_not_in_store 6, direction_conflict 1 |
| FAULT 04 | 109 | 5 | 5 | 3 | 2 | direction_conflict 4, sensor_not_grounded 1 |
| FAULT 16 | 52 | 5 | 5 | 4 | 1 | number_not_in_store 4, sensor_not_grounded 1 |
| FAULT 17 | 52 | 4 | 4 | 2 | 2 | number_not_in_store 2, sensor_not_grounded 1, unknown_tag 1 |
| FAULT 20 | 53 | 4 | 4 | 4 | 0 | number_not_in_store 3, unknown_tag 1 |
| FAULT 02 | 108 | 3 | 3 | 2 | 1 | sensor_not_grounded 3, unknown_tag 1 |
| FAULT 11 | 108 | 2 | 2 | 2 | 0 | reasoning_length 2 |
| FAULT 19 | 53 | 2 | 2 | 1 | 1 | number_not_in_store 1, unknown_tag 1 |
| FAULT 18 | 52 | 0 | 0 | 0 | 0 | — |
| **total** | **1,939** | **289** | **289** | **199** | **90** | **300** |

### Final attempt — reissued

| family | dropped items | retry codes (occurrences) |
|---|---|---|
| NORMAL | 17 | number_not_in_store 10, unknown_tag 4, sensor_not_grounded 3 |
| FAULT 15 | 15 | number_not_in_store 13, unknown_tag 2, direction_conflict 1, reasoning_length 1 |
| FAULT 14 | 14 | direction_conflict 14 |
| FAULT 07 | 10 | direction_conflict 7, number_not_in_store 3 |
| OPEN_SET | 9 | unknown_tag 4, number_not_in_store 4, sensor_not_grounded 1 |
| FAULT 05 | 4 | number_not_in_store 2, direction_conflict 2 |
| FAULT 08 | 4 | number_not_in_store 4 |
| FAULT 10 | 4 | number_not_in_store 4 |
| FAULT 04 | 2 | direction_conflict 2 |
| FAULT 06 | 2 | number_not_in_store 1, direction_conflict 1 |
| FAULT 13 | 2 | direction_conflict 1, number_not_in_store 1 |
| FAULT 17 | 2 | number_not_in_store 2 |
| FAULT 01 | 1 | number_not_in_store 1 |
| FAULT 02 | 1 | sensor_not_grounded 1 |
| FAULT 12 | 1 | reasoning_length 1 |
| FAULT 16 | 1 | number_not_in_store 1 |
| FAULT 19 | 1 | unknown_tag 1 |
| **total** | **90** | **95** |

Reproduce: `python scripts/analyze_teacher_rejections.py --section reconcile`.

---

## 3. Task A2 — direction conflicts on 14, 07, 15

### The rule being enforced

`llm/benchmark/filters.py::_direction_conflict` takes the KB's `initiating.sensor`
and `initiating.direction`, resolves the sensor to a tag, and rejects the answer
if the 60 characters after the tag contain an increase word while the KB says
decrease (or the reverse). Per fault:

| fault | KB initiating variable | tag | KB direction | tag kind | agrees / observed | tag absent |
|---|---|---|---|---|---|---|
| 1 | A_C_Feed_Flow | XMV_45 | decrease | valve position | — (0) | 120 |
| 2 | Component_B_Stream6 | XMEAS_30 | increase | measurement | 33/33 | 87 |
| 4 | Reactor_Cooling_Water_Flow | XMV_51 | increase | valve position | 120/120 | 0 |
| 5 | Condenser_Cooling_Water_Flow | XMV_52 | increase | valve position | 2/2 | 118 |
| 6 | A_Feed_Flow | XMV_44 | decrease | valve position | 0/1 | 119 |
| **7** | A_C_Feed_Flow | **XMV_45** | **decrease** | valve position | **0/120** | 0 |
| 8 | Component_A_Stream6 | XMEAS_29 | increase | measurement | 38/38 | 82 |
| 10 | Reactor_Temperature | XMEAS_9 | increase | measurement | 10/10 | 110 |
| 11 | Reactor_Cooling_Water_Flow | XMV_51 | increase | valve position | 118/118 | 0 |
| 12 | Condenser_Cooling_Water_Flow | XMV_52 | increase | valve position | — (0) | 120 |
| 13 | Reactor_Pressure | XMEAS_7 | decrease | measurement | **0/102** | 18 |
| **14** | Reactor_Cooling_Water_Flow | **XMV_51** | **decrease** | valve position | **0/120** | 0 |
| **15** | Condenser_Cooling_Water_Flow | **XMV_52** | **decrease** | valve position | **0/26** | 94 |
| 16-20 | see KB | — | increase | measurement | 1/1 (18 only) | 51-53 each |

`agrees / observed` counts only events where the initiating tag was ranked in the
evidence at all; `tag absent` is the rest.

**Finding.** For faults 7, 14, 15 — and for fault 13, which is a measurement —
the stored trend contradicts the KB direction in *every* event where it can be
checked. This is not noise, and it is not the teacher being wrong.

### Five first-attempt rejections each

**Fault 7** — `XMV_45` (A_C_Feed_Flow), KB **decrease**, stored **increasing** in
120/120.

- `EVT-train-f07-r001::menu` — "…a slow drift in the A_C_Feed_Flow (XMV_45) with a 24.64% increase…"
- `EVT-train-f07-r004::menu` — "…trending increasing (z=12.00, deviation 24.34%), while the disturbance is C header pressure loss … leading to a decrease in A_C_Feed_Flow. However, the observed trend is increasing…"
- `EVT-train-f07-r016::menu` — "…z=12.33 and a deviation of 25.02% with a trend increasing, but the decided diagnosis … would cause A_C_Feed_Flow to decrease."
- `EVT-train-f07-r034::menu` — "…trending increasing (z=12.30, deviation 24.95%), which contradicts FAULT_01's requirement…"
- `EVT-train-f07-r038::menu` — "…simultaneous increase in A_C_Feed_Flow [XMV_45] (z=12.28, +24.92%)…"

**Fault 14** — `XMV_51` (Reactor_Cooling_Water_Flow), KB **decrease**, stored
**increasing** in 120/120.

- `EVT-train-f14-r003::menu` — "…trending upward with a residual share of 0.300…"
- `EVT-train-f14-r005::menu` — "…trending increasing (z=0.05, deviation 0.06%), but the decided diagnosis FAULT_14 specifies a valve sticking that would cause the flow to decrease."
- `EVT-train-f14-r007::menu` — "…trending upward (z=0.13, deviation 0.17%)…"
- `EVT-train-f14-r010::menu` — "…trending increasing (z=0.08), but FAULT_14 … implies the valve should be stuck closed…"
- `EVT-train-f14-r014::menu` — "…trending increasing (z=0.15, deviation 0.19%), but the decided diagnosis … specifies a valve sticking that would cause the valve to close…"

**Fault 15** — `XMV_52` (Condenser_Cooling_Water_Flow), KB **decrease**, stored
**increasing** (26 observed) or absent (94).

- `EVT-train-f15-r003::menu` — "…trending increasing … but the decided diagnosis is 'Condenser cooling water valve sticking' which implies the valve should be stuck closed" (also `number_not_in_store:-5.0`)
- `EVT-train-f15-r020::menu` — tag not ranked; the answer still names XMV_52 and the rule fires on the name (also `-6.0`)
- `EVT-train-f15-r046::menu` — "…the dynamic residual attribution points to XMV_52 (valve position) as dominant…"
- `EVT-train-f15-r056::menu` — "…The Condenser_Cooling_Water_Flow [XMV_52] is a valve position (manipulated variable) showing a de…" (also `unknown_tag:Condenser_Underflow_Stream11`)
- `EVT-train-f15-r058::menu` — "…trending downward (-1.50% deviation) despite the valve position increasing…"

**Why they conflict.** In each case the teacher is describing the *measured valve
position* correctly and adding that the KB implies the opposite *flow*. The KB's
`initiating.sensor` for these faults is an XMV tag, which is a valve position, not
the flow the KB's wording names. The rule compares a position trend with a flow
direction. The teacher's own sentences ("but the KB says decreasing") are the
contradiction being punished.

### Proposal (not applied)

Two candidate changes, both replayable with no model calls:

1. **Scope the rejection rule to measurements.** Apply `direction_conflict` only
   when the KB initiating tag is an XMEAS measurement; for an XMV valve position
   the direction is undefined, exactly as `direction_inconsistent` already treats
   valve tags for the advisory flag (close-out 3 decision).
2. **Fix the KB rather than the filter.** Replace the XMV `initiating.sensor`
   with the measurement the fault actually moves, and correct fault 13's sign
   (0/102 agreement). This is the cleaner long-term fix but edits the knowledge
   base, which changes the KB hash, the menu text and every prompt.

Replay of option 1, measured on the cached retry answers:

| scope | recovered | invalidated | flipping the KB sign instead would invalidate | recovered by fault |
|---|---|---|---|---|
| faults 14, 15 only | 15 | 0 | 6 | f14 14, f15 1 |
| faults 7, 14, 15 | 21 | 0 | 10 | f14 14, f07 6, f15 1 |
| every valve-position initiator | 25 | 0 | 84 | f14 14, f07 6, f04 2, f05 1, f06 1, f15 1 |
| no direction rule at all | 26 | 0 | 105 | + f13 1 |

Relaxation cannot invalidate an accepted answer, so `invalidated = 0` there by
construction; the column that matters is the third — flipping the KB sign keeps
the rule and would throw away 84 accepted answers. **Recommendation: option 1,
scoped to every XMV initiator (25 recovered, 0 lost), with fault 13's KB entry
flagged for separate review.** The last row shows the rule's remaining cost after
that change is exactly 1 item, which is the fault-13 bug, not a prose problem.

---

## 4. Task A3 — drop bias

Accepted vs dropped train items, per family (medians):

| family | accepted n | dropped n | ratio acc | ratio drop | delay acc | delay drop | \|z\| acc | \|z\| drop |
|---|---|---|---|---|---|---|---|---|
| ALL | 1,849 | 90 | **27.86** | **11.74** | 64 | 69 | **3.97** | **0.39** |
| FAULT 14 | 87 | 14 | 14.41 | 14.40 | 59 | 59 | 0.19 | 0.19 |
| FAULT 15 | 88 | 15 | 1.13 | 1.11 | 84 | 84 | 0.21 | 0.24 |
| FAULT 07 | 91 | 10 | 29.01 | 32.22 | 59 | 59 | 12.17 | 12.22 |
| NORMAL | 100 | 17 | 1.007 | 1.006 | 279 | 254 | 0.34 | 0.22 |
| OPEN_SET | 184 | 9 | 26.50 | 21.43 | 64 | 64 | 3.95 | 6.93 |

**Dropped items skew weaker, not harder.** Across all families the median fused
ratio is 2.4× lower and the median top-|z| is 10× lower in the dropped set. The
effect is concentrated where the physics is subtle: faults 14 and 15 sit at
ratio ≈1.1 and |z| ≈0.2 whether accepted or dropped — a sticking valve barely
moves the 20-window snapshot — and NORMAL items are dropped at a *lower* |z| than
they are accepted. Faults 7 and open-set are the exceptions (drops are slightly
stronger).

Consequence: the dropped 4.6% is not a random sample of difficulty. It is
disproportionately the low-signal events, which are exactly the ones a deployed
system sees most often. Training therefore sees a slightly easier distribution
than validation and test, where those events remain. The per-fault floor (all ≥ 0.70)
is not the binding constraint here; the composition is.

---

## 5. Task A4 — training-set policy (computed, nothing changed)

Excluding the 90 dropped items from `train.jsonl`:

| class | items |
|---|---|
| classified | 1,307 |
| UNCLASSIFIED_OPEN_SET | 442 |
| NORMAL_TRANSIENT | 100 |
| **total** | **1,849** |

- Unclassified share of faulty items: **442 / 1,749 = 25.27%**
- Unclassified share of all items: 442 / 1,849 = 23.90%

The cap (`≤ 25%` of faulty train items) is **breached by 0.27 pp, i.e. 7 items**,
because the drops removed 64 classified items but only 9 unclassified ones. Two
ways to stay compliant, both one line of code and neither applied:

1. **Keep the 90 as label-only examples** (the current file already does this):
   share is 455 / 2,160 = 21.06%, no cap change, and a label-only loss can use
   them. This is the option the full-run report recommended and it is still the
   simplest.
2. **Exclude the 90 and drop 7 more unclassified items** by the same seeded
   subsample: 435 / 1,742 = 24.97%.

The 25% cap was chosen to stop unclassified from dominating; a 0.27 pp overshoot
is immaterial to that purpose, but it is a stated rule, so it should be either
honoured or re-stated.

---

## 6. Task B5 — environment

| item | installed | target (post-training report) | verdict |
|---|---|---|---|
| interpreter | Python 3.11.5, `C:\Users\Admin\AppData\Local\Programs\Python\Python311\python.exe` | — | same interpreter the app uses |
| torch | **2.11.0+cu128** | 2.13.0+cu130 | upgrade |
| transformers | **5.15.0** | 4.57.6 | **downgrade** (major) |
| peft | **missing** | 0.20.0 | install |
| accelerate | 1.14.0 | — | keep |
| datasets | 5.0.1 | — | not used on the jsonl path |
| tokenizers / safetensors | 0.22.2 / 0.8.0 | — | keep |
| bitsandbytes | missing | — | **not needed** (`use_4bit: false`) |
| trl | missing | — | **not needed** (plain `Trainer`) |
| **timm** | **missing** | — | **required**: InternVL's remote `modeling_internvl_chat.py` imports it |
| CUDA | available, 12.8, RTX A5000 (sm_86, arch list includes sm_120) | — | ok |

**Base model.** `C:/Users/Admin/Desktop/LY_C68/deepfake_project/models/InternVL3-2B`
exists; `model.safetensors` is 4,177,999,192 bytes and its SHA-256 is
`18B4CDBC8EAB790F447FEAC717F4D0CECBB85CAF8DF198B3B2401C22506858E7` — **exact match**
with the expected value (computed in 9 s).

**Load attempt.** `llm.model.load_base_model(cfg, eval_mode=True)` fails with
`ModuleNotFoundError: No module named 'timm'` after a `torch_dtype is deprecated`
warning (non-fatal under 5.15). It fails before allocating weights, so nothing was
left resident. Two blockers, both installs:

1. `timm` — needed to construct the model at all.
2. `peft` — needed by `wrap_peft` for the adapter.

**Not installed, pending approval.** The pinned set to match the original run is
`torch==2.13.0+cu130`, `transformers==4.57.6`, `peft==0.20.0`, plus `timm` at
whatever version the InternVL3 runtime pins (the post-training report does not
record it; it must be read from the model's `requirements`/README rather than
guessed). Because transformers must move *down* a major version, this belongs in a
dedicated venv: downgrading the app's environment would need `llm/inference.py`
and the response layer re-verified, and the 268 tests currently pass on 5.15.

---

## 7. Task B6 — trainer inspection and token lengths

### How examples are built (`llm/train_adapter.py`, `llm/dataset.py`)

- `_load_tokenized_datasets` reads `paths.llm_dataset_path` as JSONL rows of
  `{question, answer, system?}`, or an HF `load_from_disk` split. **It does not
  read `prompt`/`target`/`prose`**, so the benchmark dataset needs a small adapter
  (Phase 3A work item), not just a config change.
- `TEPInstructionDataset.__getitem__` builds
  `<|im_start|>system\n{system}<|im_end|><|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n`
  via `build_chat_prompt`, then appends `answer + EOS` where `EOS = "<|im_end|>"`.
- **Masking:** `labels = [-100] * len(prompt_ids) + answer_ids`. Everything up to
  and including the assistant header is masked; only answer tokens and the closing
  `<|im_end|>` are learned. Standard SFT masking, correct.
- **Truncation:** prompt is tokenized with `truncation=True, max_length=4096`
  *silently*; then `budget = max_length - len(prompt_ids)` and the answer is
  sliced to it. If the prompt fills the window the answer is empty and the sample
  contributes zero loss — no warning, no skip. This is why the lengths below matter.
- **Collator:** right-pads to the batch's longest sequence, pads labels with
  `-100` and attention with 0; carries a dummy zero `pixel_values` (1×3×448×448)
  and `image_flags=[0]`, the official InternVL text-only pattern. Note the dummy
  image is a real 448² activation per sample — it costs VRAM, not tokens.
- **Two stale constants matter for dataset v2:**
  - `SYSTEM_PROMPT` ([llm/dataset.py:31-49](llm/dataset.py#L31-L49)) still contains
    the **fault→severity rubric** ("critical … Faults 14, 15", "low: … Fault 13")
    and the "affected subsystems" framing. It is applied whenever a sample has no
    `system`, which would re-inject exactly the leak dataset v2 removed.
  - `ANSWER_KEYS` / `ANSWER_SCHEMA_HINT` describe the **v1** answer
    (`reasoning_steps`, `severity`, `confidence`), not the benchmark schema.
  Both are reasons the benchmark items must carry their own `system` and `answer`,
  and why the trainer should not fall back to these constants for v2.

### Token lengths (measured with the real tokenizer, 4,096 window)

| split / system | rows | prompt p50 | prompt p95 | prompt max | answer p50 | answer p95 | answer max | total max | truncated | zero budget |
|---|---|---|---|---|---|---|---|---|---|---|
| train, current SYSTEM_PROMPT | 1,939 | 1,497 | 1,523 | 1,536 | 402 | 476 | 562 | 2,065 | 0 | 0 |
| train, no system | 1,939 | 1,128 | 1,154 | 1,167 | 402 | 476 | 562 | 1,696 | 0 | 0 |
| val, current SYSTEM_PROMPT | 834 | 1,478 | 1,521 | 1,533 | 169 | 186 | 188 | 1,710 | 0 | 0 |
| test, current SYSTEM_PROMPT | 3,168 | 1,477 | 1,521 | 1,538 | 169 | 186 | 188 | 1,706 | 0 | 0 |

**Nothing truncates.** The longest training example is 2,065 tokens, half the
window; the prompt alone never exceeds 1,536. Two consequences:

- A smoke run cannot silently lose the evidence tail — the failure mode above
  does not occur at this size.
- `max_sequence_length: 4096` is paying for roughly 2× the tokens needed. 2,560
  fits every row (max 2,065) and would roughly halve attention cost per step.
  Not changed here, since it alters the training configuration.

---

## 8. Cost (before any job over 30 minutes)

Measured anchor stays `8,877 s / 2,376 examples / 3 epochs` ≈ 19.9 s per
optimizer step at effective batch 16.

| job | estimate | status |
|---|---|---|
| 20-step smoke run | 20 × ~20 s + ~2-3 min load ≈ **8-10 min** | **not run** — blocked on `timm` + `peft` |
| Full training, per configuration | 1,849 items × 3 epochs / 16 ≈ 347 steps ≈ **1.9 h** | not approved |
| 3 configurations × 3 seeds | **≈ 17 h** GPU | not approved |
| Evaluation generation (24,000 items over 3 seeds) | **assumption** 2-4 s/item → 13-27 h | must be measured |

The per-configuration number is below the Phase 0 estimate because the real train
set (1,849) is smaller than the budgeted 2,000-3,000.

---

## 9. Changes

| commit | contents |
|---|---|
| `5ac2575` | `scripts/analyze_teacher_rejections.py` (reconcile / direction / replay / bias / policy), `scripts/measure_token_lengths.py`, `tests/test_teacher_rejection_analysis.py` |
| this commit | this report, the errata in the full-run report, the PROTOCOL amendment |

---

## 10. Tests

`python -m pytest -q` → **268 passed** in 7.18 s (263 before, 5 added).

New tests pin the accounting rules: one item with two codes is one rejected item;
an open-set episode is counted once, under its own family; a dropped record with
no retry code is reported separately instead of breaking the primary-attribution
invariant; accepted + dropped = items; the policy excludes dropped items and
reports both shares.

---

## 11. Gate

| gate item | status | evidence |
|---|---|---|
| A1 rejection tables reconciled, non-overlapping, reissued | **PASS** | 289 items / 300 first codes; 90 items / 95 retry codes; open-set double counting disproved; §2 |
| A2 per-fault KB variable + enforced direction | **PASS** | §3 table, 18 classifiable faults |
| A2 five rejected examples each for 14, 07, 15 with stored trend | **PASS** | §3, 15 examples |
| A2 agreement fraction over all train events | **PASS** | 0/120, 0/120, 0/26; also 0/102 on fault 13 |
| A2 proposal + replay recovery / invalidation | **PASS** | 25 recovered, 0 invalidated, 84 would be lost by the sign flip |
| A3 drop bias medians, dropped vs accepted | **PASS** | §4, per family and overall |
| A4 train policy counts and share | **PASS** | §5, 25.27% — 7 items over the cap, reported |
| B5 interpreter, versions, CUDA | **PASS** | §6 |
| B5 base model path + SHA-256 | **PASS** | `18B4…58E7` exact match |
| B5 install list, nothing installed | **PASS** | `peft`, `timm` missing; versions differ; no install performed |
| B6 trainer inspection (build, mask, truncate, collator) | **PASS** | §7, includes the two stale constants |
| B6 token lengths reported | **PASS** | §7, 0 truncations, max 2,065 |
| 20-step smoke run | **NOT RUN** | blocked: base model cannot load without `timm`; awaiting approval to install |
| Full test suite | **PASS** | 268 passed |
| Teacher calls / training | **NONE** | by instruction |

---

## 12. Deviations

- **D1 — the report's numbers are corrected, not deleted.** The full-run report
  keeps its tables and gains an errata block pointing here, so the failed
  arithmetic stays visible rather than being quietly overwritten.
- **D2 — the reconciliation script asserts its own invariants.** A future edit that
  reintroduces the code/item confusion fails the run instead of producing a table.
- **D3 — "invalidated = 0" in the replay is structural.** A relaxation cannot
  reject an accepted answer; the informative comparison is the third column (what
  a KB sign flip would cost), which is why it is reported.
- **D4 — the direction proposal is deliberately not applied.** It changes what the
  teacher accepts, which changes the dataset; that is an owner decision, and the
  replay above is the evidence for it.

---

## 13. Open questions

1. **Direction rule scope** — apply option 1 to every XMV initiator (my
   recommendation), to 7/14/15 only, or leave as is and accept a 90-item drop that
   is concentrated on valve faults? A KB fix for fault 13 is needed either way.
2. **The 7-item cap breach** — keep the 90 as label-only (21.06%), or exclude them
   and re-subsample 7 unclassified items (24.97%)?
3. **`timm` pin** — must come from the InternVL3 runtime, not from memory. If it
   pulls a newer torch, the whole pin set moves.
4. **Smoke-run scope** — 20 steps on the current stack is meaningless until the
   pinned venv exists; after that, the run should use one fold's train file and
   assert loss is finite and the first-step loss matches a hand-computed
   cross-entropy on one batch.
5. **Window size** — 2,560 instead of 4,096 is a free ~2× on attention; it is a
   config change and it moves the run off the "matches the original run" footing.
