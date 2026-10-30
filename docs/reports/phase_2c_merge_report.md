# Phase 2C merge + real build — report

**Repo:** `C:\Users\Admin\Desktop\anomaly`, branch `benchmark-upgrade`.
**Scope:** merge `origin/benchmark-upgrade-2c`, re-verify the 2C numbers on *this*
checkout, put the follow-up filters under the current Phase 2B rules, and build
the follow-up dataset from the **real** Phase 2B events. **No real teacher call
was made**: every build used `--teacher fake`, which is a pure-Python function.

## 1. Summary

The merge was clean, but not for the reason the brief expected: the 2C work was
already merged into this branch earlier, so `llm/response/router.py` was already
at the `_hypotheses` fix and there was nothing to resolve. Only one commit was
outstanding — the repo-relative path fix in `data/splits_v2.json` — and it
touched three files none of which had changed on this side. `git merge` reported
no conflict, and the final `router.py` diff for this merge is **empty** because
the file is unchanged.

Re-verification produced one **CORRECTION** of the 2C report and two
confirmations: the routing family shares reproduce exactly, the detector
artefacts verify against `configs/benchmark_inputs.json`, and the Phase 2B event
set exists here — but `data/llm` **does** hold 1,120 `grounded_qa` rows on this
checkout, which the 2C report said it did not. The 2C report's stated blocker
(`model.pt` at 32/16 against a 128/64 config) does not reproduce here either.

The build then ran on the real events: **2,327 dialogues / 5,392 turns / 5,272
teacher turns**, zero rejections, zero leak-check violations, zero split
overlap, paraphrases disjoint on 17/17 labels. Real-teacher cost for the train
split at the requested 2:1 pairing is **2,182 calls ≈ 6.6 h serial** (7.4 h with
one retry per rejection) — half the 1:1 figure the earlier report quoted.

Three defects surfaced while building and are fixed in this pass: a hard crash
on faults 16-20 for one case type, a cost projection that ignored the pairing,
and a stale train-item constant.

## 2. Changes

| commit | what |
|---|---|
| `caf5e32` | merge of `origin/benchmark-upgrade-2c` (3 files, no conflicts) |
| `4b4c31c` | `llm/followup/provenance.py`; the follow-up filter calls the 2B direction, scaffolding and valve rules; the follow-up context labels every tag with its kind; `TeacherCache` and the manifest carry `filter_sha256` + menu provenance; `tests/test_followup_provenance.py` (17 tests) |
| `5dcbc82` | `--report-items-per-dialogue` with per-family seeded thinning; `condition_case` substitution for unclassified targets; pairing-aware cost projection; train-item count read from the Phase 2B manifest; `audit_sample.csv` gains `scoring_path`; `scripts/scan_followup_open_set.py`; 5 pairing/audit tests; regenerated routing artefacts |

## 3. Step 1 — merge

```
$ git fetch origin --prune
$ git merge-base HEAD origin/benchmark-upgrade-2c
e467ce01adb693175b0d409d1bf31aacfa5ed8df          # already an ancestor of HEAD
$ git log --oneline HEAD..origin/benchmark-upgrade-2c
01f617a Record the splits document repo-relative so it matches in any checkout
$ git merge --no-ff origin/benchmark-upgrade-2c
Merge made by the 'ort' strategy.
 data/splits_v2.json             |  2 +-
 docs/reports/phase_2c_report.md | 22 +++++++++++----------
 scripts/build_splits_v2.py      |  8 +++++++-
 3 files changed, 20 insertions(+), 12 deletions(-)
```

**Conflict resolutions: none — there were no conflicts.** The brief anticipated
`llm/response/router.py` and the docs; both were already merged here at
`75b94e0`, and `router.py`'s last change was `84cab4f` ("Treat an unmeasured
hypothesis confidence as unmeasured, not as a tie"), which is the 2C fix.

Final `router.py` for this merge:

```
$ git diff HEAD~1 HEAD -- llm/response/router.py
(no output — the file is unchanged by the merge)
$ git log --oneline -2 -- llm/response/router.py
84cab4f Treat an unmeasured hypothesis confidence as unmeasured, not as a tie
7659a80 Add the response-layer package: contract, store access, severity, render, route
```

The rule that makes clarifications possible is present and does what 2C needed:

```python
if item.get("confidence") is None:
    continue        # a missing confidence is not a 0.0-vs-0.0 tie
```

Full suite after the merge: **374 passed in 8.34s**.

## 4. Step 2 — re-verification on this checkout

### (a) Routing shares — **VERIFIED**

```
$ python scripts/inspect_followup_routing.py
live_template    n=550  template=545  model=5    model_share=0.0091
live_free_form   n=150  template=4    model=146  model_share=0.9733
benchmark_only   n=150  template=17   model=133  model_share=0.8867
```

Byte-identical to the 2C report's `0.0091 / 0.9733 / 0.8867`, and the per-label
table matches line for line (e.g. `overview` 46/4, `action` 49/1, `mechanism`
4/46). No difference to list.

### (b) `grounded_qa` rows — **CORRECTED**

Both `data/llm` and `data/llm_1` are HuggingFace arrow directories, not JSONL.

```
$ python -c "load_from_disk('data/llm')['train']['kind']"     # Counter
data/llm     train 2016 rows  report 144, followup 576, grounded_qa 1008,
                              severity 144, discrimination 144
             val     96 rows  grounded_qa 56
             test    96 rows  grounded_qa 56
data/llm_1   identical to data/llm (2016 / 96 / 96)
data/llm_v1  train 1944 rows  report 216, grounded_qa 1728
             val     36 rows  grounded_qa 32
             test    36 rows  grounded_qa 32
```

| claim in the 2C report | measured here | verdict |
|---|---|---|
| `data/llm` holds 720/40/40 rows and **no** grounded_qa rows | 2,016/96/96 rows with **1,008/56/56** grounded_qa | **CORRECTED** |
| `data/llm_1` holds 1,120 grounded_qa rows (1,008 + 56 + 56) | identical in `data/llm` **and** `data/llm_1` | VERIFIED |
| routed as the operator ask: "6 intents, 144 each" | **7** intents × 144 (`detector_scores` also appears) | **CORRECTED** |
| routed verbatim: `evidence_why` 864, `sensor_values` 144 | 864 / 144 | VERIFIED |
| the brief's 2,112 grounded_qa rows in v0 | not reproducible: v0 = 1,120, `llm_1` = 1,120, `llm_v1` = 1,792, total 4,032 | **CORRECTED** |

Consequence for the argument, not the conclusion: the preamble hijack
(85.7 % of `grounded_qa` rows collapsing into `evidence_why` when routed
verbatim) is real, but v0 is not an empty corpus — it duplicates `llm_1`
exactly, so quoting both as separate evidence would double-count 1,120 rows.

### (c) Events and detector artefacts — **VERIFIED**

```
outputs/benchmark_v2/events/  faulty/{train 2188, val 456, test 1814} event files
                              misses.jsonl, normal.jsonl, oracle.jsonl
```

```
model.pt         sha256 MATCH  bytes MATCH  d0ec3f386f1119c4…
cva_model.npz    sha256 MATCH  bytes MATCH  97f27ae9ab10fd87…
dpca_model.npz   sha256 MATCH  bytes MATCH  a1b29a5b086ac891…
scaler.pkl       sha256 MATCH  bytes MATCH  6b67a6a18e747377…
ALL VERIFIED
```

against `configs/benchmark_inputs.json`. The 2C report's stated blocker
(`model.pt` hidden 32 / latent 16 against a 128/64 config) does not apply here:

```
$ torch.load("outputs/anomaly_detector/model.pt")["model_state_dict"]
encoder.lstm.weight_ih_l0 (512, 52)   encoder.project.weight (64, 128)
decoder.latent_proj.weight (256, 64)  decoder.out_proj.weight (52, 128)
num_features 52  sequence_length 60  val_mse 0.6192
```

512 = 4 × hidden 128, 64 = latent, so the checkpoint is the 128/64 model
`config.json` declares. The mismatch was on the other checkout.

## 5. Step 3 — the follow-up path now runs the current 2B rules

`llm/followup/filters.py` imported the 2B pattern lists but re-implemented the
valve rule and applied neither direction nor scaffolding. It now calls:

| rule | how |
|---|---|
| direction | `_direction_conflict(body, ctx.kb_fault)` — measurement-scoped, valve positions skipped, `kb_fault=None` on open-set items |
| scaffolding | `scaffolding_hits(body, abstain=...)` — phrases derived from the prompt template, not a hand-written list |
| valve/flow | `_valve_mislabel` now uses the shared `_tag_kind` and checks **both** directions (an XMV called a pressure reading, an XMEAS called a valve position), skipping negations |

The context was the other half of the gap: `cases._sensor_lines` printed
`A_Feed_Flow (XMV_45)`, which is exactly the confusion 2B removed from the
report prompt. It now prints the kind:

```
- A_Feed_Stream1 [XMEAS_1, measurement]: observed 0.8054, baseline 0.2505, …
```

Provenance is one hash over both halves plus the menu fields, written into every
cache record and the manifest:

```
filter_sha256  4d7d22815524cda60e4e9d0fa207f57f04a971029c1e2bf6a70c07ba46381a48
menu_revision  r1
menu_sha256    1d63c8bd1babc11c810671deb63ec053a72384b2ad520173f848f5b6815ca9c8
kb_sha256      f01236ffe01b7b4f4191ac6deefeef62d407a9d13fe987101e225c5e5dcac2fa
```

A record now reads `{"key", "model", "seed", "temperature", "filter_sha256",
"menu_revision", "menu_sha256", "kb_sha256", "text", "seconds"}`. Tests cover
both directions of every rule and the cache/manifest fields (17 tests, §8).

## 6. Step 4 — the real-event build

```
$ python scripts/build_followup_dataset.py \
    --events outputs/benchmark_v2/events --out outputs/benchmark_v2/followup \
    --teacher fake --report-items-per-dialogue 2 --audit 100 --seed 20261030
real 0m18.4s
```

| split | events available | dialogues | turns | teacher turns | rejected |
|---|---:|---:|---:|---:|---:|
| train | 2,305 | **1,139** | 2,630 | 2,562 | 0 |
| val | 503 | **261** | 601 | 591 | 0 |
| test | 1,854 | **927** | 2,161 | 2,119 | 0 |
| total | 4,662 | **2,327** | 5,392 | 5,272 | 0 |

Cases per split (manifest `counts`):

| case | train | val | test |
|---|---:|---:|---:|
| `valve_actuator` | 292 | 74 | 253 |
| `causal_why` | 231 | 44 | 166 |
| `mechanism` | 146 | 38 | 111 |
| `hypothesis_comparison` | 108 | 27 | 98 |
| `ambiguous_question` | 70 | 16 | 56 |
| `repeated_question` | 63 | 16 | 57 |
| `out_of_scope` | 58 | 15 | 56 |
| `clarification_reply` | 51 | 14 | 43 |
| `false_premise` | 86 | 12 | 66 |
| `template_sample` | 34 | 5 | 21 |
| `case_substituted` | 45 | 14 | 30 |
| open-set dialogues | 97 | 23 | 87 |
| normal-excursion dialogues | 59 | 24 | 20 |

All ten case types appear in every split. Train skipped faults 3 and 9 (the
2B exclusion) and counts them rather than dropping them silently.

**Real-teacher cost** at the measured 10.9 s/call:

| quantity | value |
|---|---|
| teacher turns per dialogue (measured, fake teacher) | 2.27 |
| full train at exactly 2:1 — 1,925 report items → 963 dialogues | **2,182 calls** |
| serial hours | **6.61 h** |
| with one retry per 12 % rejection (2,400 calls) | **7.40 h** |
| every split as built (5,272 calls) | 15.96 h / 17.88 h with retries |

**Proposed training mix: 2 report items : 1 dialogue.** Each dialogue carries
2.27 model answers, so the mix puts ≈ 1,925 report answers against ≈ 2,180
follow-up answers per epoch — the report task stays the headline metric, and the
follow-up branch still gets enough volume for refusal and clarification
behaviour. Template turns are 68 of 2,630 (2.6 % of turns) and cost no calls.

## 7. Steps 5-6 — checks and the audit sheet

| check | result |
|---|---|
| leak check (every context and question, in-build) | **0 violations**; the build aborts on a hit |
| split overlap | **zero**, 3 pairwise comparisons, 0 duplicate keys, all 2,305/503/1,854 keys confirmed by `data/splits_v2.json` |
| paraphrase hold-out | **17/17 labels disjoint** (40 train vs 10 held-out) |
| open-set scan (`scripts/scan_followup_open_set.py`) | **822 open-set dialogues** (207 removed-variant, 615 native faults 16-20), **0 violations**, 3 operator echoes |
| audit sheet | 100 rows, **10 per case type**, `scoring_path` 72 `free_form_grounding` / 28 `intent_format`, **0 scored cells** |

The open-set scan checks contexts, questions and answers for the true fault's
number and KB name. Its first version also flagged the KB *initiating variable*
and reported 118 violations — all of them the fault's own variable appearing in
the evidence block, which is what the model is meant to reason from. That needle
was wrong, not the data; it is now a separate opt-in diagnostic
(`--include-initiating`).

The 3 remaining echoes are fault 15's plain physical description inside the
**operator's question** ("is the condenser cooling water valve sticking?"). That
is the documented exception from close-out 3: an operator can ask exactly that,
and the rule that matters is that the answer must not confirm it — which the
`names_removed_fault` filter enforces, and which the scan confirms (0 answer
hits).

The audit sheet now carries a `scoring_path` column, so a free-form turn (intent
`null`: `valve_actuator`, `causal_why`, `mechanism`) is scored on grounding
alone and never pooled with intent-agreement scores. A test asserts both paths
appear and that the judgement columns stay empty.

## 8. Tests

| suite | result |
|---|---|
| `tests/test_followup_provenance.py` (new) | **17 passed** |
| `tests/test_followup_dataset.py` (+5 pairing/audit) | **54 passed** |
| `tests/test_followup_filters.py` | **67 passed** (with the dataset file) |
| full suite, app environment | **396 passed in 8.30s** |

New tests, in both directions where a rule has two: the 2B file list cannot
drift from the builder's; the hash changes when either half changes; the correct
direction passes and the opposite is rejected; a valve position is not held to a
flow direction; an open-set item has no KB direction to contradict; a prompt
heading in the prose is rejected while plain prose passes; an XMV called a
measurement and an XMEAS called a valve position are both rejected while the
correct negation passes; cache records carry all four provenance fields; pairing
halves each family and is reproducible; the audit sheet separates the two
scoring paths.

One test pins a **known gap rather than a behaviour**: the direction vocabulary
holds `increase` but not `increasing`, so a gerund escapes
`_direction_conflict`. Widening it belongs to the paused fault-13 patch, so the
test records today's behaviour and is the one to flip when that patch lands.

## 9. Gate table

| gate | result |
|---|---|
| merge clean | **PASS** — `caf5e32`, no conflicts (none were possible: 2C was already merged here) |
| tests pass | **PASS** — 374 after the merge, 396 after the build changes |
| (a) routing shares re-verified here | **PASS / VERIFIED** — 0.0091 / 0.9733 / 0.8867 identical |
| (b) grounded_qa counts re-verified, differences listed | **PASS / CORRECTED** — v0 holds 1,120 grounded_qa rows, not zero; routed asks span 7 intents, not 6; 2,112 not reproducible |
| (c) events exist and artefacts verify | **PASS / VERIFIED** — 4,458 event files; 4/4 hashes match |
| follow-up filters use the current 2B rules | **PASS** — `4b4c31c`; direction, scaffolding, valve/flow; 17 tests |
| follow-up cache provenance includes the new filter hash | **PASS** — `filter_sha256` + r1 menu fields in every record and the manifest |
| real-event build with counts | **PASS** — 2,327 dialogues / 5,392 turns / 5,272 teacher turns |
| leak / split / paraphrase checks clean | **PASS** — 0 / 0 / 17-of-17 |
| open-set scan on follow-up contexts | **PASS** — 0 violations, 3 documented operator echoes |
| audit sheet design, unscored | **PASS** — 100 rows, 10 per case type, 0 scored |
| cost estimate | **PASS** — 2,182 calls, 6.6 h (7.4 h with retries) for the 2:1 train mix |
| no real teacher call | **PASS** — `--teacher fake` throughout |

## 10. Deviations

1. **No conflict resolutions to show.** The merge was trivial because the 2C
   branch had been merged into this branch before; only the repo-relative path
   fix was outstanding. `router.py` is unchanged by the merge.
2. **The achieved pairing is 1.69:1, not exactly 2:1.** The builder's unit is
   the event and two report items can share one, so 2,305 train events thinned
   by 2 give 1,139 dialogues against 1,925 items. The exact 2:1 figure (963
   dialogues) is what the cost projection uses. The overshoot means slightly
   more follow-up weight than requested, not a correctness problem.
3. **The follow-up context format changed** (tag kinds), so follow-up prompts
   differ from the fixture build. There was no real follow-up cache to
   invalidate; the fake-teacher cache is keyed by prompt and simply missed.
4. **The built dataset is not committed** — `outputs/*` is gitignored (only
   `outputs/evaluation/*` is tracked). The numbers above come from
   `outputs/benchmark_v2/followup/manifest.json`, which records its own file
   hashes.
5. **Zero rejections from the fake teacher.** The new direction and scaffolding
   rules never fire on it because it writes from the stored trends; the unit
   tests prove they fire on prose that disagrees. Real-teacher behaviour on
   these rules is unmeasured — the 12 % rejection rate in the cost estimate is
   the Phase 2B measurement, not a follow-up measurement.

## 11. Open questions

1. **Which pairing?** 2:1 as built, exactly 2:1 (963 dialogues), or 1:1
   (13.3 h, 4,393 calls)? The number only needs settling before the real run.
2. **Does the follow-up branch need teacher prose at all for val/test?** The
   cost is quoted for train; val and test would add 2,710 more calls (8.2 h) if
   they are generated the same way.
3. **The gerund gap** in `_direction_conflict` (see §8) is still open and is
   shared with the paused fault-13 patch. Whoever lands one should land both.
