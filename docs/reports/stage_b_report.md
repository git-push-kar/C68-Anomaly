# Stage B — follow-up prose (train only) — report

**Repo:** `C:\Users\Admin\Desktop\anomaly`, branch `benchmark-upgrade`.
No training was started. No model other than the teacher ran.

Teacher calls in this stage: **191** pilot (fresh) + **2,384** full-train (fresh) +
**≈870** on the two corrective rebuilds (mostly cache re-judgements; see §9) =
**≈3,450**, all `qwen3:30b` at temperature 0.2, seed 20261030, concurrency 1.

---

## 1. Summary

The follow-up corpus is built from the **final Stage A events**, with the
teacher's prose accepted at **92.8%** and three filter faults fixed on the way —
two of which were rejecting correct answers, one of which was leaking the answer
into an open-set prompt.

* **Pilot: 188/191 accepted (98.4%)** against an 80% stop gate, then **replayed**
  after the filter fixes: **184/191 (96.3%)** of *fresh* calls plus 4 deterministic
  template turns.
* **Full run: 2,562 teacher turns, 2,377 accepted (92.8%), 185 dropped**, in
  **7,361 s (2 h 03 m)** — 2.869 s/call, against a 10.9 s/call projection. The
  corpus is therefore ~3.8× cheaper than budgeted.
* **Combined SFT: 2,878 rows** — 1,919 report + 959 dialogue rows, achieved ratio
  **2.001** report rows per dialogue row (target 2.000), 443 of 1,139 dialogues
  used, whole dialogues only.
* **Open-set scan: 0 violations and 0 operator echoes**, after the prompt-side
  leak fix (three fault-15 valve questions quoting the removed fault's KB name).
* Leak check on every context and question: **0**. Split overlap: **0**.
  Paraphrase hold-out: disjoint for every label. Tests: **480 passed** in the app
  environment and **480 passed** in `venv_train`.

## 2. Changes (commit hashes)

| commit | contents |
|---|---|
| `5532b7f` | refusal vocabulary gets the absence family; `unknown_tag_like` stops reading schema identifiers as sensors; tests both ways; `scripts/sync_cache_provenance.py` tracked |
| *(this report)* | removed-fault exclusions in the false-premise claim pool and in the paraphrase picker, tests, corpus rebuild, SFT mix, audit sheet, manifest |

The 1,919 report answers were re-stamped by `scripts/sync_cache_provenance.py`
(**0** model calls): all replay valid under the new filter hash, so no report
answer was re-called for a follow-up filter change.

## 3. Filter fixes found by the pilot

| failure | before | cause | after |
|---|---|---|---|
| `refusal_missing` | 9 / 191 | the decline list held only the *inability* family ("cannot", "not stored"); a refusal worded as an absence ("XMEAS_2 is not listed among the variables that moved") read as a non-refusal | 1 / 191 |
| `unknown_tag` | 4 / 191 | `unknown_tag_like` treats every underscore token as a display-name claim, so an answer naming its own schema (`overview_fallback`, `ranked_hypotheses`) was rejected as a fabricated sensor | 0 / 191 |
| `sensor_not_grounded` | 1 / 191 | same root cause on the tag path | 0 / 191 |
| `near_duplicate` | 8 / 191 | model behaviour, not a filter fault | 6 / 191 |

The vocabulary is **derived**, not typed out: intent and severity and table enums,
every pydantic field name, and the slim export's `ANSWER_KEYS` /
`DERIVED_AT_RENDER` are read from their modules, so a new enum member cannot
silently become a "fabricated tag". A test asserts the exact pilot sentence now
passes and that a real fabricated tag (`XMEAS_99`, `Condenser_Loop_Pressure_Value`)
still fails.

## 4. Acceptance by case type (full train run, final filters)

| case type | teacher turns | accepted | rate | dropped |
|---|---|---|---|---|
| valve_actuator | 658 | 586 | 89.1% | 72 |
| causal_why | 520 | 508 | 97.7% | 12 |
| mechanism | 330 | 323 | 97.9% | 7 |
| hypothesis_comparison | 245 | 245 | **100.0%** | 0 |
| repeated_question | 199 | 166 | 83.4% | 33 |
| false_premise | 193 | 181 | 93.8% | 12 |
| ambiguous_question | 159 | 156 | 98.1% | 3 |
| out_of_scope | 132 | 89 | **67.4%** | 43 |
| clarification_reply | 115 | 113 | 98.3% | 2 |
| template_sample | 11 | 10 | 90.9% | 1 |
| **total** | **2,562** | **2,377** | **92.8%** | **185** |

Rejection codes, attributed once per turn (a turn can carry more than one code;
the 185 drops produced 194 codes):

| code | n | where it concentrates |
|---|---|---|
| `near_duplicate` | 119 | valve_actuator 71, repeated_question 31, causal_why 8 |
| `refusal_missing` | 50 | out_of_scope 42, false_premise 8 |
| `refusal_no_alternative` | 14 | out_of_scope 8, false_premise 6 |
| `number_not_in_store` | 11 | causal_why 4, clarification_reply 2 |

**`out_of_scope` is the weak family at 67.4%** and the 42 `refusal_missing`
drops are the single largest fixable gap: the model answers the off-domain
question instead of declining, even after the retry that quotes the reason. That
is a behaviour the training data should teach — 89 accepted refusals remain, which
is enough to learn from, but the family is thinner than its 5% mix share implies.
`repeated_question` at 83.4% is the near-duplicate rule doing its job: the case
asks the same question twice and the model must not answer it in the same words.

## 5. Open-set leak, found and fixed at the cause

The scan separates *violations* (an answer that names the removed fault) from
*operator echoes* (the operator's own question naming it). Both matter: on an
open-set item the target is `UNCLASSIFIED_OPEN_SET`, so a question that says
"is the condenser cooling water valve sticking?" hands the model fault 15's KB
cause in the prompt.

| run | violations | operator echoes | fix |
|---|---|---|---|
| first full run | 0 | 3 | — |
| + false-premise claim pool excludes the removed fault | 0 | 3 | not the cause: the questions came from the paraphrase bank |
| + paraphrase picker filters with `names_removed_fault` | **0** | **0** | the case-type paraphrase bank is written per case, and some wording carries one fault's cause |

The knob is the same signature rule that already rejects an *answer*: the bank is
filtered with `names_removed_fault(question, {fault_id: UNCLASSIFIED,
truth_fault_number: removed})`. Two tests guard it — one on the question, one on
the claim pool — so a future bank edit cannot re-open the hole.

## 6. SFT export and mix

`llm/benchmark/to_sft_followup.py` writes one row per **accepted assistant turn**,
with the earlier turns rendered into the question as a transcript: the existing
trainer contract (mask up to the assistant header, learn the rest) is untouched,
at the cost of re-sending prior turns as prompt tokens.

| item | value |
|---|---|
| dialogue rows available | 2,310 (from 2,377 accepted turns, 68 template turns excluded) |
| rows used by the 2:1 mix | 959 (443 dialogues, whole dialogues) |
| report rows | 1,919 |
| combined rows | 2,878 |
| achieved ratio | 2.001 report rows per dialogue row |
| combined file | `outputs/benchmark_v2/sft/train_mixed.jsonl` |

Roughly 55% of the teacher prose built here is **not** used by the 2:1 mix. That
is the requested ratio, not a loss: the surplus exists so a later 1:1 experiment
needs no new teacher calls.

## 7. Audit sheet

`outputs/benchmark_v2/followup/audit_sample.csv` — **100 rows, 20 columns**,
stratified by case type (exactly 10 per case, 10 cases), columns
`dialogue_id, case_type, split, turn, question, answer, answer_source, accepted,
intent, scoring_path, table, citations, clarifying_question, refusal_reason,
correct, grounded, refusal_appropriate, not_repetitive, useful, notes`. The five
scoring columns are blank: a number filled in by the builder would be a claim
nothing measured.

`scoring_path` separates the two populations — **36** rows are intent-format
turns and **64** are free-form turns with no intent, which cannot be scored on
intent agreement and are judged on grounding alone.

## 8. Checks

| check | result |
|---|---|
| leak check, every context and question | **0** findings (manifest `leak_check`) |
| split overlap train/val/test | **0** overlapping `(source_file, fault, run)` keys; 2,305 train events confirmed against `data/splits_v2.json` |
| paraphrase hold-out | disjoint for every label (train 40 / heldout 10 per label) |
| open-set scan | 0 violations, 0 operator echoes, 2,630 turns scanned |
| faults 3 and 9 in train | 15 dialogues skipped (`fault_3_9_excluded_from_train`) |
| provenance per record | `filter_sha256`, `menu_revision r2`, `menu_sha256`, `kb_sha256` |
| tests, app environment | 480 passed |
| tests, `venv_train` | 480 passed |

## 9. Cost, measured against projected

| phase | projected | measured |
|---|---|---|
| pilot (191 calls) | 10.9 s/call → 35 min | **3.283 s/call → 628 s (10.5 min)** |
| full train (2,562 calls) | 7.76 h serial (8.69 h with retries) | **2.869 s/call → 7,361 s (2 h 03 m)** |
| corrective rebuilds | not projected | 1,246 s + 1,036 s, of which ~600 were fresh calls (the rest replay cached prose under the new rules) |

The 2.9 s/call figure is 3.8× better than the 10.9 s/call assumption recorded in
the Phase 2C estimate; the difference is answer length (1–4 sentences, not a full
report). Two side effects to note: the manifest's `cost_estimate.seconds_per_call`
still prints the **projection constant** (10.9) next to the rebuild's
`measured_seconds_total` (1,026 s), so the block reads as if the run were 7.8 h;
and `mean_seconds_per_call` in `acceptance` is the *rebuild's* 0.401 s because
2,244 of 2,562 prompts were cache hits. The honest figures are in the table above.

## 10. Gate

| gate item | status | evidence |
|---|---|---|
| 2C follow-up code merged in this repo | **PASS** | `git log` shows the 2C merge; `llm/followup/*` present and passing |
| follow-up cache provenance carries `filter_sha256`, `menu_sha256`, `kb_sha256`, `menu_revision` | **PASS** | manifest `provenance` block |
| dialogues rebuilt from the FINAL Stage A events, fake teacher | **PASS** | val 261 / test 927 dialogues built earlier from the same events; train rebuilt here |
| `to_sft_followup.py` writes multi-turn rows with loss on assistant turns only | **PASS** | one row per accepted assistant turn, earlier turns in the question; trainer untouched |
| mixer produces 2 report : 1 dialogue | **PASS** | achieved 2.001; 1,919 + 959 = 2,878 rows |
| pilot acceptance ≥ 80% | **PASS** | 96.3% (184/191) after the fixes; 98.4% before |
| full run acceptance and drops by case type | **PASS** | §4; 92.8% overall, worst family 67.4% |
| ratio achieved | **PASS** | 2.001 vs 2.000 |
| leak / split / open-set clean | **PASS** | 0 / 0 / 0 |
| audit sheet written, not scored | **PASS** | 100 rows, 5 blank scoring columns |
| tests pass, app and `venv_train` | **PASS** | 480 + 480 |

## 11. Limitations and open items

1. **`out_of_scope` refusals are the weakest case** (67.4%). The corpus has 89
   accepted refusals; the target distribution is met, but the failure mode is
   systematic, not random, and a training run should be judged on whether this
   family improves.
2. **`near_duplicate` is a rule, not an error** (119 drops): the model reuses its
   previous wording. If a later analysis shows these are good answers, the
   overlap threshold is the knob to revisit — after the training run, not now.
3. **Surplus prose.** 55% of dialogue rows are unused by the 2:1 mix. Any change to
   the ratio is free until that budget is exhausted.
4. **CRLF and the filter hash.** `core.autocrlf=true` in this checkout means a
   re-checkout can rewrite line endings and change `filter_sha256` without a
   semantic change, which marks cached prose stale. The 1,919 report answers
   survived this only because the replay stamp exists; follow-up answers have no
   such stamp. Normalising the hashed content (strip `\r`) is the fix and is not
   applied here.
5. **Follow-up items exist for train only.** Val and test follow-up dialogue files
   were built with the fake teacher to size the split; no teacher prose exists for
   them, and none is needed (evaluation is not trained on).
6. **Not started:** training. The 20-step smoke run remains the only training-like
   job in this project so far.
