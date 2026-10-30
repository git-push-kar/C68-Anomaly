# Pending decisions and stage plan — status report

**Repo:** `C:\Users\Admin\Desktop\anomaly`
**Branch:** `benchmark-upgrade` @ `69c85f6`, working tree clean.
**Date:** 2026-10-31.
**Scope of this document:** no work was performed. It records where the build
stands, what is blocked on a decision, and the ordered actions to reach the next
stage. Every number is either quoted from a committed report or re-measured on
this checkout in this pass; the two are labelled.

---

## 1. Where the build stands

| phase | state | evidence |
|---|---|---|
| 0 — protocol, splits, KB enum, tags | **done** | `docs/PROTOCOL.md`, `data/splits_v2.json`, `scripts/check_splits_v2.py` |
| 1 — detector evaluation, baselines | **done** | `outputs/evaluation/detector_v2/`, `docs/reports/phase_1_report.md` |
| 2A — response layer | **done** | `llm/response/`, 8 modules + tests |
| 2B — events, teacher, report dataset | **done except fault-13** | `outputs/benchmark_v2/dataset/` |
| 2C — follow-up dataset (fake teacher) | **done** | `outputs/benchmark_v2/followup/` |
| 3A — training environment | **done** | `venv_train` matches the pinned target |
| 3B — full training | **not started** | only a 20-step smoke run exists |
| 3C — benchmark evaluator | **does not exist** | no script scores the report task |
| 4 — evaluation and RQ1-RQ5 | **not started** | criteria are still placeholders |

### Verified artifact inventory (this checkout, this pass)

Dataset (`outputs/benchmark_v2/dataset/manifest.json`):

| split | items | open-set | `UNCLASSIFIED_OPEN_SET` | `NORMAL_TRANSIENT` | teacher prose |
|---|---:|---:|---:|---:|---|
| train | **1,925** | 193 | 453 | 117 | 1,925 accepted, 0 rejected |
| val | **834** | 331 | 456 | 47 | **none (by design)** |
| test | **3,168** | 1,314 | 1,814 | 40 | **none (by design)** |

SFT export (`outputs/benchmark_v2/sft/`): `train.jsonl` **1,925 rows**;
`val.jsonl` and `test.jsonl` are **0 bytes**.

Follow-up (`outputs/benchmark_v2/followup/`, fake teacher): **1,139 / 261 / 927**
dialogues, 5,392 turns, 5,272 teacher turns, 0 rejections.

Environment (`venv_train`): torch **2.13.0+cu130**, transformers **4.57.6**,
peft **0.20.0**, accelerate **1.15.0**, CUDA available. Base model present at the
configured path, `model.safetensors` SHA-256 **`18B4CDBC8EAB790F…`** (matches the
pinned value), 4.18 GB.

Tests: **396 passed** (app environment, last committed run).

---

## 2. What actually blocks the next stage

Three things, in this order:

1. **The fault-13 edit is stopped at its own ceiling.** The measured AFFECTED set
   is **493** against a 120-call cap, so nothing was applied and no file was
   rebuilt. The dataset is internally consistent but carries a menu line the KB
   says the opposite of, contradicted by **102/102** train events.
2. **`val.jsonl` is empty.** `llm/train_adapter.py:33-45` loads `val.jsonl`; an
   empty file yields an empty dataset, and the checkpoint rule in `PROTOCOL.md`
   §3 selects on **validation macro-F1 measured by generation**. Training cannot
   start without this export.
3. **Generation runs at 4.0 tokens/s** (23.98 s per item, 96-token cap, no KV
   cache, adapter unmerged). At that speed validation costs 5.6 h per pass and
   the full test split 21.1 h per run — **9 runs would need ~240 h**, which is
   not an evaluation budget. This is a hard blocker on Phase 4, not a preference.

---

## 3. Pending decisions

Each entry: what is being decided, the measured cost of each option, the
recommendation, and what it unblocks.

### D1 — Fault-13 KB sign: which scope? (blocks Stage A)

The `FAULT_13` menu line says the level **decreases**; the stored `trend` of
`XMEAS_7` is **increasing in 102/102** train events (deviation sign is mixed
61/41). Flipping it re-renders the menu line into **5,731 of 5,870** prompts.

| option | items re-called | calls (×1.12) | model time |
|---|---:|---:|---:|
| A. what the rules can see (21 gerund + 5 contradicting) | 26 | 29 | 5 min |
| B. answers naming `FAULT_13` | 68 | 76 | 14 min |
| C. every fault-13 answer that mentions the variable | 113 | 127 | 23 min |
| D. every accepted fault-13 answer | 119 | 133 | 24 min |
| E. the measured union | 493 | 552 | 99 min |

**Recommendation: D (119).** The prompt's claim changed for every fault-13 item,
so every fault-13 answer was written from a line that is now known to be wrong;
A and B leave answers standing that quote the old claim. E is dominated by prose
that mentions reactor pressure as *another* fault's cascade (425 of 493) and buys
nothing. Cost: 24 min of calls, ~50 min wall including the rebuild.

Sub-decision, must be recorded either way: five answers state "decreasing trend",
and one of them (`EVT-train-f13-r030`) quotes a printed **negative z**. The flip
is justified on the *trend* field; if a reader takes the KB line to mean *below
baseline*, that answer is not wrong. Recommendation: flip on the trend field,
re-call all 119, and record the deviation/trend distinction in PROTOCOL.md.

### D2 — Replay-validation stamp? (blocks any future teacher run)

The cache holds four different `filter_sha256` values, so the strict reuse rule
marks **1,770** already-cleared answers stale (≈5.3 h to re-call). Every one of
them was re-judged by `replay_cache` under the current filters, so the prose is
valid; only the provenance field is missing.

**Recommendation: yes, implement.** Stamp `validated_filter_sha256` when
`replay_cache` accepts an answer under the current hash. Reasoning to record in
the docstring: re-judgement is evidence and should count. ~1-2 h CPU, no calls.
Without it, D1's 119 calls would silently become 1,889.

### D3 — Should `_direction_conflict` read display names? (blocks D1 completeness)

Of 212 accepted answers with a direction word near a mention of the affected
variable, **80 use only the display name** (`Reactor_Pressure`, "reactor
pressure") and **34 only the tag**; 98 use both. The rule searches the literal
canonical tag, so it checks roughly half the prose it claims to check. 29 of
fault 13's 60 direction-bearing answers are in the blind half.

**Recommendation: implement as a flag first, price it, then promote.** Widening
the matcher to display names will invalidate an unknown number of accepted
answers; measure on the cache (no model calls) before committing to re-calls.

### D4 — Direction rule vs XMV-initiated faults (blocks nothing; needs a record)

For faults 1, 4, 5, 6, 7, 11, 12, 14, 15 the KB `initiating.sensor` is an **XMV
valve position** while the KB wording names a *flow*, so the rule compares a
position with a flow direction. The stored trend contradicts the KB in
**120/120** of fault 14's train events, **120/120** of fault 7's, **26/26** of
fault 15's. These are wording mismatches, not model errors.

**Recommendation: scope the rule to XMEAS measurements (as the teacher prompt
already is) and record the KB wording as a known limitation.** Do not re-call
~266 items to satisfy a rule that is comparing the wrong pair of quantities.

### D5 — Follow-up teacher scope (blocks Stage B)

| option | calls | serial | with 12 % retry |
|---|---:|---:|---:|
| train only, 2:1 pairing | **2,182** | 6.6 h | 7.3 h |
| train only, exactly 2:1 (963 dialogues) | 2,182 | 6.6 h | 7.3 h |
| all splits as built | 5,272 | 16.0 h | 17.9 h |
| train + val/test prose | 4,892 | 14.7 h | 16.5 h |

**Recommendation: train only.** Val/test carry structured targets with no prose
by design; prose there buys auditability, not training signal, and the metrics do
not read it. Revisit only if the audit sheet needs realistic val answers.

### D6 — Training mix — **recommendation: 2 report items : 1 dialogue**, as built.

Each dialogue carries 2.27 answers, so the mix gives ≈1,925 report answers against
≈2,180 follow-up answers per epoch. The build achieved 1.69:1 (1,139 dialogues
against 1,925 items) because two report items can share one event; that overshoot
is in the safe direction.

### D7 — Seed coverage for training (blocks Stage D)

| option | runs | GPU time (1.7 h/run, measured) |
|---|---:|---:|
| 3 configs × 3 seeds | 9 | **15.3 h** (range 15.3-21.6 h at 17-24 s/step) |
| Protocol A × 3 seeds, each fold × 1 seed | 5 | **8.6 h** (range 8.6-12.0 h) |

**Recommendation: 5 runs first**, because RQ5 only needs the seed spread on
Protocol A (RQ1), and the folds are reported per fold without a spread claim. If
the fold result is close to its criterion, add the two extra seeds (≈3.5 h).

### D8 — Generation performance — **hard blocker, must be measured first**

Measured today: **4.0 tok/s**, 23.98 s per 96-token item, no KV cache, adapter not
merged. Consequences for one run: val 834 items = **5.6 h**; test 3,168 =
**21.1 h**. The protocol asks for ≤6 s per answer (RQ5).

**Recommendation: measure KV-cached, merged, batched decode on 20 items before
any other Phase 3 work** (30 min). The expected gain is one to two orders of
magnitude, but this must be measured, not assumed; the whole Phase 4 budget moves
with it. If batching gets to ~3 s/item, a full evaluation pass is 3.3 h per run.

### D9 — `max_sequence_length` 4096 → 2560?

Measured max row is **2,065** tokens (assistant tokens are masked and tokenize in
the same range), so 4096 pays for roughly 2× the tokens needed. Halving it roughly
halves attention cost per step. It is a hyperparameter change from the documented
configuration.

**Recommendation: yes, with the deviation recorded**, after the first full run is
confirmed healthy. Not before — it should not be confounded with the baseline.

### D10 — Success criteria — **already decided: set at Phase 4, relative, on validation only.**

Still open in substance: the actual thresholds. They cannot be set before there is
a validation number, so this is a Phase 4 item, not a Stage A blocker.

### D11 — Confidence calibrator (RQ: none; needed before the report is believable)

The contract is implemented and documented (`PROTOCOL.md` §10.8): confidence is
code-computed from the `fault_id` token probability, calibrated on validation with
isotonic regression, and renders "uncalibrated (no calibrator fitted)" until one
exists. No calibrator is fitted.

**Recommendation: implement after the first training run**, when there are
validation logits to fit on. ~1 h CPU. Cheap, and the alternative is shipping a
benchmark that cannot state a confidence.

### D12 — Val/test SFT export — **required, not optional**

`val.jsonl` is 0 bytes and `train_adapter.py` reads it. Export `val` (834 rows)
and `test` (3,168) as `{system, question, answer}` with the **deterministic
target JSON as the reference answer** (no teacher prose exists and none is
needed). ~10 min CPU. Without val there is no checkpoint selection at all.

### D13 — Stale v0 fault lists in `config_v2.yaml`

`llm.training.train_faults / val_faults [4,14] / test_faults [15,21]` are the v0
split, which contradicts `data/splits_v2.json`. They are read only by
`scripts/benchmark_tep_adapter.py` and the legacy dataset generators — **not** by
`llm/train_adapter.py` — but they are a live trap for anyone re-running the old
benchmark script against a v2 adapter.

**Recommendation: neutralise in `config_v2.yaml`** (comment them out and point at
`splits_v2.json`), one line, no behaviour change.

---

## 4. Stage-wise plan

Machine time is the GPU/Ollama/CPU cost; wall time adds the rebuilds and checks.
All teacher figures use the measured 10.8 s/call; all training figures the
measured 17.01 s/step at effective batch 16.

### Stage A — close out the dataset (owner decision D1, then CPU + 24 min of calls)

| step | action | time |
|---|---|---|
| A1 | apply fault-13: KB `decrease→increase`, gerund vocabulary, `MENU_REVISION = "r2"`, tests both ways — one commit | 0.5 h |
| A2 | implement the replay stamp (D2) and the display-name flag (D3) | 1.5 h |
| A3 | re-call the 119 affected fault-13 answers + any D3-flagged items | 24 min + 0.5 h |
| A4 | rebuild dataset, SFT export (**including val/test**, D12), manifest + hashes, leak check on 5,870 prompts, split overlap, open-set scan, audit sheet v4 | 0.5 h |
| A5 | neutralise the v0 fault lists (D13); full suite in both environments | 20 min |

**Stage A total: ≈3.5 h wall.** Exit: `val.jsonl` non-empty, old menu line absent
from every prompt, fault-13 contradictions 0 or listed.

### Stage B — follow-up prose (decision D5, then 6.6 h of calls)

| step | action | time |
|---|---|---|
| B1 | write `to_sft_followup` and the 2:1 mixer with guards (no severity, no confidence, system prompt from `prompt_v2`) | 2-3 h |
| B2 | real teacher run over 2,182 train calls, resume on, concurrency 1 | **6.6 h** (7.3 h with retries) |
| B3 | rebuild the follow-up set, re-run leak/split/open-set checks, audit sheet v4 | 0.5 h |

**Stage B total: ≈9.5-10 h wall**, of which 6.6 h is unattended model time.
Optional val/test prose: +9.1 h — recommended against (D5).

B cannot overlap Stage D: the teacher and the trainer both want the GPU.

### Stage C — make evaluation affordable and possible (runs during B)

| step | action | time |
|---|---|---|
| C1 | measure KV-cached + merged + batched decode on 20 items; report s/item | 0.5 h |
| C2 | write the benchmark evaluator: schema parse, macro-F1 over the observable set, abstention on `test` / false abstention on `test_seen`, grounding checker (rewritten, not `evaluate_grounding`), intent agreement, table fidelity | 4-6 h |
| C3 | tests on synthetic outputs, one per metric | 1 h |

**Stage C total: ≈6-8 h**, CPU plus one short GPU measurement. Exit: the numbers
in RQ1-RQ4 have a script that produces them, and the evaluation budget is known.

### Stage D — training (decision D7)

| step | action | time |
|---|---|---|
| D1 | 5 runs (Protocol A × 3 seeds, both folds × 1 seed) at `max_sequence_length` 4096 | **8.6 h** (12.0 h worst) |
| D2 | validation generation per checkpoint for checkpoint selection | 0.7 h per pass at 3 s/item; **5.6 h per pass at today's speed** |
| D3 | optional: +2 fold seeds if a fold lands near its criterion | +3.5 h |

**Stage D total: ≈10-11 h** if C1 lands a usable decode, otherwise not viable.

### Stage E — evaluation, criteria, report

| step | action | time |
|---|---|---|
| E1 | generate val (9 runs) and test (5 runs) | 3-20 h depending on C1 |
| E2 | score: RQ1-RQ5, seed spread, latency, VRAM | 1 h |
| E3 | fit the isotonic calibrator (D11) | 1 h |
| E4 | set the success criteria relative to validation, then compare | 0.5 h |
| E5 | write the Phase 4 report | 2 h |

**Stage E total: 8-25 h**, dominated by generation.

### Owner actions (not agent work)

| # | action | why |
|---|---|---|
| O1 | score `audit_v3.csv` (100 report rows) | the only human check on grounding and usefulness |
| O2 | score the follow-up audit sheet (100 rows) | same, and it is the only check on refusal/clarification behaviour |
| O3 | approve the four decisions in §3 that cost model time (D1, D2/D3, D5, D7) | keeps the call budget explicit |

---

## 5. Critical path and total

Serial, with the recommendations above:

```
A (3.5 h) → B (10 h, unattended) → C (runs inside B) → D (10 h) → E (8-25 h)
                                                           ≈ 32-49 h wall
```

With today's 4 tok/s decode the same plan is **~240 h of generation** and is not
worth starting; that single measurement is what turns Phase 4 from infeasible
into a day and a half.

GPU contention is the reason the order matters: Stage B (Ollama, 30 B) and Stage D
(LoRA training) cannot run together, while Stage C is CPU work and fits inside B.

---

## 6. Risks

| risk | exposure | mitigation |
|---|---|---|
| decode stays at 4 tok/s | Phase 4 infeasible | C1 first; do not commit an evaluation budget before it |
| fault-13 flip invalidates more prose than counted | 5 known contradictions; display-name blind half is unmeasured | D3 prices it on the cache before any call |
| fold results miss the RQ2 criterion | 2 extra seeds, +3.5 h | add seeds only if the first fold is near the line |
| teacher and trainer contend for the GPU | a 6.6 h stage serialises behind a 10 h one | schedule B before D, never beside it |
| stale v0 split lists leak into an old script's numbers | wrong bench numbers quoted later | D13, one line |
