# Pre-run step: follow-up corpus rebuild, mix composition, amendments

**Repo:** `C:\Users\Admin\Desktop\anomaly`
**Branch:** `benchmark-upgrade` (HEAD `1a8bc21`)
**Environments:** app env `python` (3.11), training env `venv_train`
**Date:** 2026-10-31
**Teacher calls used:** **0 of the 330 allowed.** No training of any kind.

---

## 1. Follow-up train corpus rebuilt under the fixed refusal rule

### 1.1 What the plan run found first

A dry `--plan` run (no calls) was made before spending anything. It reported
**393 missing primary prompts and 786 estimated calls** — the cap is 330, so the
run was stopped before it started and the gap was traced instead.

**Root cause, a bug in the builder.** `fill_answer` reassigned `prompt` to the
retry prompt and then wrote the cache under it at the end of the turn:

```python
prompt = retry_prompt
...
if cache is not None and not used_cache:
    cache.put(prompt, text, model, seconds)   # the retry's key, twice
```

So a turn that was rejected once and accepted on the retry kept *only* the retry
entry; its first attempt was bought, paid for, and thrown away. The miss set was
exactly the retried turns — every one of the 306 turns with
`answer_attempts == 2`, plus 87 turns downstream of them — which is why the
estimate was suspiciously round at 393 ≈ 306 + 87. Fixed in `cf51026`: the first
attempt is cached under its own prompt, and the retry is cached under its own.

### 1.2 Why nothing was re-bought

Every sentence a rebuild needs is already in the repository: `train.jsonl` stores
the accepted text of every accepted turn and the last rejected text of every
dropped one, and all 185 rejected texts are stored **untruncated** (max 412 chars,
the 600-char cap never reached). Re-buying a prompt does not produce new
information about the rule; it produces a new sentence and pays for it.

So the rebuild is a **replay** (`--replay-from`): each turn's stored prose is fed
back through the current filters, with no cache read and no teacher, and
`prose_replayed` + `replayed_under_filter_sha256` are stamped on every turn it
touches. Two fidelity details were needed:

* the stored text is the *extracted* answer, so it is re-wrapped in the
  `{"answer": …}` envelope before parsing — the parser and the rules run on the
  same shape they run on live;
* a turn that ended accepted after a retry has its **second** attempt stored
  (the first was lost to the bug above). That answer was written against a prompt
  carrying the rejection note, and the note's offending tokens count as grounded
  numbers. Judged against the bare prompt it loses a number it was entitled to
  quote — 52 turns were dropped exactly this way on the first attempt at a
  replay. Those turns are now judged against the retry prompt rebuilt from the
  codes the text itself raises (`retry_note_reconstructed`, 37 turns), which is
  the two-attempt rule replayed rather than re-bought.

### 1.3 Acceptance before and after, by case type

Corpus: 1,139 dialogues, 2,562 teacher turns, exactly the same dialogue ids and
turn counts as the pre-fix build (turn-count alignment: 797×2, 332×3, 10×4 —
zero mismatches), same `--seed 20261030`, identical contexts.

| Case type | Before | After | Δ |
|---|---|---|---|
| out_of_scope | 89 / 132 (67.4%) | **124 / 132 (93.9%)** | **+35** |
| false_premise | 181 / 193 (93.8%) | **187 / 193 (96.9%)** | **+6** |
| ambiguous_question | 156 / 159 (98.1%) | 156 / 159 (98.1%) | 0 |
| causal_why | 508 / 520 (97.7%) | 508 / 520 (97.7%) | 0 |
| clarification_reply | 113 / 115 (98.3%) | 113 / 115 (98.3%) | 0 |
| hypothesis_comparison | 245 / 245 (100%) | 245 / 245 (100%) | 0 |
| mechanism | 323 / 330 (97.9%) | 323 / 330 (97.9%) | 0 |
| repeated_question | 166 / 199 (83.4%) | 166 / 199 (83.4%) | 0 |
| template_sample | 10 / 11 (90.9%) | 10 / 11 (90.9%) | 0 |
| valve_actuator | 586 / 658 (89.1%) | 586 / 658 (89.1%) | 0 |
| **total** | **2,377 / 2,562 (92.78%)** | **2,418 / 2,562 (94.38%)** | **+41** |

The rule change recovers 41 turns and loses **none**. The 41 are all plain-English
declines ("the incident context does not provide …", "… is not mentioned") that
the old vocabulary could not read — the 50-item failure this fix existed for, now
landing where it should.

### 1.4 Remaining drops, by code (144 turns)

| Code | Turns | What it means |
|---|---|---|
| `near_duplicate` | 119 | a repeated-question answer that reuses the first answer's words |
| `refusal_no_alternative` | 14 | declines without pointing at anything the record can answer |
| `number_not_in_store` | 11 | quotes a figure that is in neither the record nor the conversation |

Attribution matters and was wrong on the first pass: a dropped replayed turn has
no second attempt to fail, so its final code read `invalid_json: empty response`
and the real rule disappeared. `drop_reason()` now reports the code that rejected
the prose, marked `(no retry text)`.

**Budget accounting.** 0 calls were needed, against a 330 cap. Had the cache been
the only source, the honest estimate was 393 first attempts plus roughly 15%
retries, i.e. **≈450 calls (≈1.4 h at the measured 11 s/call)** — over the cap, and
still not the 786 the plan printed, because the plan assumed every fresh call
would return an empty answer and trigger a second purchase.

---

## 2. Mix composition: 2:1 kept, filled by priority

`llm/benchmark/to_sft_followup.py::mix` now fills dialogue slots by tier, whole
dialogues only, seeded hash inside a tier (`CASE_PRIORITY`). The report rows are
untouched, so the ratio is still two report rows per dialogue row.

| | |
|---|---|
| report rows | 1,919 |
| dialogue budget (`rows // 2`) | 959 |
| dialogue rows taken | 959 (from 430 dialogues) |
| total rows | **2,878** |
| achieved ratio | **2.001 : 1** |

| Tier | Case | Rows taken | Rows available |
|---|---|---|---|
| 1 | out_of_scope | 114 | 114 |
| 1 | false_premise | 178 | 178 |
| 1 | clarification_reply | 112 | 112 |
| 1 | repeated_question | 137 | 137 |
| 2 | ambiguous_question | 156 | 156 |
| 2 | hypothesis_comparison | 245 | 245 |
| 3 | causal_why | **11** | 500 |
| 3 | valve_actuator | **2** | 565 |
| 3 | mechanism | **2** | 320 |
| 3 | template_sample | **2** | 78 |

**The consequence, stated plainly.** Tiers 1 and 2 hold 942 of the 959 rows
(98.2%), so tier 3 receives **17 rows out of 1,463 available**. 1,446 accepted
dialogue rows are left unused, all of them in tier 3. The mixed file is therefore
a refusal-and-comparison corpus with the ordinary explanatory turns effectively
absent. That is the literal reading of the instruction, it is recorded in
PROTOCOL §12.1 before any training result exists, and §5 below proposes a floor
rather than applying one.

### Supervised tokens and optimizer steps (real tokenizer, `token_share.json`)

| Component | Rows | Prompt p50 | Supervised p50 | Supervised mean | Supervised total | Share |
|---|---|---|---|---|---|---|
| report | 1,919 | 1,400 | 297 | 299.6 | 574,980 | **88.6%** |
| dialogue (all) | 959 | 761 | 74 | 77.1 | 73,949 | **11.4%** |
| — ambiguous_question | 156 | 774 | 88 | 87.4 | 13,639 | |
| — hypothesis_comparison | 245 | 777 | 72 | 74.5 | 18,249 | |
| — false_premise | 178 | 746 | 65 | 68.3 | 12,161 | |
| — repeated_question | 137 | 749 | 74 | 74.4 | 10,196 | |
| — out_of_scope | 114 | 754 | 66 | 70.4 | 8,020 | |
| — clarification_reply | 112 | 770 | 94 | 94.0 | 10,530 | |
| — causal_why | 11 | 767 | 70 | 69.1 | 760 | |
| — valve_actuator | 2 | 656 | 46 | 47.5 | 95 | |
| — mechanism | 2 | 691 | 79 | 86.5 | 173 | |
| — template_sample | 2 | 686 | 63 | 63.0 | 126 | |

Truncated answers: 0. Prompts at the 4,096 cap: 0. Steps: **180 / epoch**, **540
for 3 epochs** at effective batch 16 (gradient accumulation 4).

---

## 3. PROTOCOL amendments, both dated before any adapter exists

`docs/PROTOCOL.md` §12 (dated 2026-10-31, "before any training result"):

* **§12.1 How the 2:1 mix is filled** — tier order, whole dialogues, seeded inside
  a tier, no reserved floor, and the measured consequence that tier 3 gets 17 rows.
* **§12.2 Generation cap and the completion rule** — cap **512**; if the trained
  adapter's 50-item validation schema check completes below **95%**, repeat at
  **768** and report both; the reported cap is the smallest that clears 95%, and
  both numbers are reported either way. Basis: 5 of 8 sampled validation answers
  finished a complete JSON object at 512 in §11.1's decode comparison.

---

## 4. Rebuild, hashes and checks

### Artefacts and SHA-256

| Artefact | SHA-256 |
|---|---|
| `outputs/benchmark_v2/followup/train.jsonl` (**new**, replayed corpus) | `2a602e8fffa04f34da06f208485235d6b34f300f13c70f8f65c2f5cd5d5fabae` |
| `outputs/benchmark_v2/followup_archive/train_pre_refusal_fix.jsonl` (old) | `61bcf0091fcb60955fb811cec45aa9452db26e8022a0d0c4a6eb0e49e1ff4432` |
| `outputs/benchmark_v2/followup/val.jsonl` (unchanged, structural) | `7bb4f4f50b176c4a61dd89b0328437a4a2b33ce59ce67fa443b36f2c2f179448` |
| `outputs/benchmark_v2/followup/test.jsonl` (unchanged, structural) | `317ba0e49b1db7ec5fdfec2e228c7e220927ed5fbfb85c39ab8865955af8a591` |
| `outputs/benchmark_v2/sft/train_mixed.jsonl` (**new**) | `6796ecce8fe351ab58db2e74a511ac9a68a6e99d017d332c96ec2fc4ddffbf34` |
| `outputs/benchmark_v2/sft/train.jsonl` (report rows, unchanged) | `c033f5e1335ed811e668467776df59251cc8861a3fa144b1e38728f360d9a958` |
| `outputs/benchmark_v2/sft/followup_train.jsonl` (export) | `f78964d309854d01cd412c9ee0f0e6128e2176c842cf988330a550cbfdb0211e` |

**The old `train_mixed.jsonl` hash was never recorded** — the file is untracked
build output and no hash of it was written anywhere in the repo, so it cannot be
quoted honestly. That gap is closed for all future builds: `mix_summary.json` now
carries `train_mixed_sha256`, `report_train_sha256` and
`dialogue_rows_file_sha256` next to the mixture it describes.

The follow-up manifest records `replay` (before/after totals, flips, drops by
code, candidate shapes, corpus path, filter hash), the merged per-split counts,
and `files_sha256` for all five directory files.

### Evaluation exports (rebuilt)

| Component | Rows |
|---|---|
| A/test | 3,168 |
| fold_1/test | 957 |
| fold_1/test_seen | 2,528 |
| fold_2/test | 638 |
| fold_2/test_seen | 2,728 |
| **test total** | **10,019** (`matches_dataset: true`) |
| A/val | 834 |
| fold_1/val | 637 |
| fold_1/test | 150 |
| fold_2/val | 687 |
| fold_2/test | 100 |
| **val total** | **2,408** (`matches_dataset: true`) |

### Audit sheets (regenerated, unscored)

* follow-up: `outputs/benchmark_v2/followup/audit_sample.csv`, 100 rows, stratified
  by case type, columns `correct / grounded / refusal_appropriate / not_repetitive /
  useful / notes` left blank.
* report: `outputs/benchmark_v2/dataset/audit_v3.csv`, 100 rows, 5 per fault family
  including `open_set` and `normal_transient`, blank scoring columns.

### Checks

| Check | Result |
|---|---|
| Prompt leak check (each context and question, in-build) | 1,139 dialogues, 0 violations |
| Split overlap (`check_event_overlap`, `zero_overlap`) | true, 0 pairwise comparisons failing |
| Paraphrase hold-out disjointness | disjoint on every label |
| Open-set scan (`scan_open_set_leaks.py`) | 481 accepted abstention answers, **0** name the removed fault |
| Turn-count alignment vs pre-fix corpus | 1,139 / 1,139 dialogues identical |
| Full test suite, app env | **491 passed** |
| Full test suite, `venv_train` | **491 passed** |

---

## Gate

| Gate item | Status | Evidence |
|---|---|---|
| Acceptance before/after by case type | **PASS** | 92.78% → 94.38%; +35 out_of_scope, +6 false_premise, 0 lost (§1.3) |
| Teacher-call budget respected | **PASS** | 0 of 330 calls; the 786-call plan was stopped and diagnosed instead of run |
| Mix composition and token shares | **PASS** | 2,878 rows, 959 dialogue rows, ratio 2.001:1; dialogue 11.4% of supervised tokens (§2) |
| PROTOCOL amendments dated | **PASS** | §12.1 and §12.2, both dated 2026-10-31, written before any adapter exists |
| Hashes recorded | **PASS**, one gap | new corpus/mixed/report/export hashes above; the *old* mixed hash was never recorded (called out) |
| Checks clean | **PASS** | leak, split overlap, paraphrase, open-set scan (§4) |
| Tests pass | **PASS** | 491 in the app env and 491 in `venv_train` |
| No training performed | **PASS** | no optimizer step was run |

---

## 5. What needs your decision before training

**D1 — the mix has no floor for the ordinary turns.** Tier 3 gets 17 of 1,463
available rows; `valve_actuator`, the largest case in the corpus, contributes 2
rows. As built, the adapter would learn refusals and comparisons and almost
nothing about explaining a valve excursion. Proposed (not applied): reserve a
floor for tier 3, e.g. 40% of the 959 slots (≈384 rows split across the four
cases), and let tiers 1-2 take 575. That is a one-line change in `CASE_PRIORITY`
handling plus a re-run of the mix (CPU, seconds). The alternative — a larger
dialogue budget, i.e. 1,439 rows at 2:1 over the 2,878 *report* rows — would also
make room, but it changes the 2,878-row scale you fixed.

**D2 — sell the 393 lost first attempts, or leave them.** With the cache bug
fixed, a real run would now buy 393 primary prompts again (≈450 calls with
retries, ≈1.4 h). It buys *new wording*, not new rule coverage: the corpus already
holds accepted prose for every one of those turns, and the replay has applied the
current rule to it. Spend it only if you want the corpus to be first-pass rather
than retry-pass prose.

**D3 — the training run itself** remains unapproved and unscheduled; nothing in
this step touched the trainer, the configs, or the adapters.
