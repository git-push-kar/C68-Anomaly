# TRAINING SPEC — FOLLOW-UP-ONLY ADAPTER V1 (spec only; nothing trained)

Written 2026-11-03, after the v5 decision in `docs/PROTOCOL.md` §29.9. Branch 1 of §29.3 applies:
fallback v3.1 writes report prose for the three open strata, and the adapter is kept for
follow-ups only. This document is the specification that §29.3 requires to exist **before** that
run is launched. It authorises nothing. The run needs explicit approval, the training device, and
`venv_train`; it does not run on a CPU checkout.

## 0. What changes and what does not

| | Status after §29.9 |
|---|---|
| Classified-fault report prose | frozen to fallback v3.1, unchanged |
| Open-set / menu-removed / `NORMAL_TRANSIENT` report prose | fallback v3.1, decided |
| 11 live template intents (`FOLLOWUP_SPEC.md` §2) | deterministic lookup, never a model, unchanged |
| Free-form, refusal, clarification, comparison turns | **the adapter**, and this run is only about these |
| Detector, classifier, menu rule, severity, confidence | untouched, all code |

The follow-up adapter is the only remaining generative component. Its job is narrow: take a
question, a supplied diagnosis, the evidence store and the dialogue state, and answer in grounded
prose — or refuse, or ask a clarifying question.

## 1. Data

`outputs/benchmark_v2/sft/train_prose_v1.jsonl`, rows with `kind == "dialogue"`: **961 rows** of
the 1,523 that the pilot used (562 report rows are dropped, not regenerated). No teacher call is
needed: every dialogue prompt stays exactly as built for the pilot, so the answers, the leak
checks and the split membership are already in place and unchanged.

What must be re-verified before the first step, not assumed:

- the 961 rows are all `split == "train"` under `data/splits_v2.json`;
- no held-out item appears in a dialogue prompt (the pilot's leak checks were run on the 1,523;
  the subset inherits them, and the check is re-run, not cited);
- the loss mask still covers answer tokens only, with the supplied diagnosis, verdict, menu and
  store masked, as in `TRAINING_SPEC_PROSE_V1.md` §3.

## 2. Hyperparameters

Unchanged from the pilot, so that a difference in outcome is attributable to the data subset and
not to a second change: InternVL3-2B, text-only, LoRA r=16, alpha=32, bf16, effective batch 16,
the pilot's learning rate and schedule, 3 epochs.

Steps: 961 rows ÷ 16 = 60 steps per epoch, **180 steps** for three epochs (the pilot's 1,523 rows
gave 96 per epoch and 288 total on the same arithmetic).

## 3. Cost estimate (estimate, to be replaced by the measured value)

Two rates are on record and both are quoted rather than one:

- at the pilot's measured 12.5 s/step (288 steps in 1.0 h): **≈ 38 min**;
- at run 1's 21.71 s/step: **≈ 65 min**.

Memory is expected to fit inside the pilot's envelope (peak 24,159 of 24,564 MiB, 1.6 % headroom,
max 81 °C) because the sequence lengths are the follow-up prompts, not the longer reports; the run
records its own peak rather than inheriting that claim.

## 4. Gates, fixed before the run

Measured on `outputs/benchmark_v2/eval/prose_pilot/prose_eval_followups.jsonl` (97 rows: 47
`valve_actuator`, 22 `mechanism`, 15 `causal_why`, 5 `ambiguous_question`, 3 `out_of_scope`, 3
`clarification_reply`, 1 `hypothesis_comparison`, 1 `repeat`) plus the 40 in-scope and 28
out-of-scope items of the pilot gate table, scored by the same code that scored the pilot so the
two readings are comparable.

| Gate | Threshold |
|---|---|
| completion (non-empty, schema-valid answer) | 100 % |
| empty answer | 0 % |
| refusal on the 28 out-of-scope items | 100 % |
| over-refusal on the 40 in-scope items | at most 10 % |
| grounding (numbers and tags absent from the store) | at most 2 %, by the §28 checks |
| numeric comparison (direction, threshold quoted, value stored) | at most 2 % |
| false premise contradicted with the stored value | every `false_premise` case, not a paraphrase of the premise |
| repeat answered identically to the first answer | 0 (the second answer must differ in wording) |
| ambiguous question answered instead of clarified | 0 |

The three case types with one or three rows are reported as **counted, not estimated**: a single
failure fails that gate, and the small n is stated in the report rather than smoothed.

## 5. What result ships

Ship the follow-up-only adapter only if every gate above passes **and** no gate is worse than the
pilot adapter measured on the identical items with identical code. Otherwise the pilot adapter
stays, and the outcome is recorded as a measured result in the protocol, not explained away.

## 6. Prohibitions

No report prose in this run, no 3B-17, no v6 sheet, no fold-1 or fold-2 adapter, no seeds 123 or
999, no teacher call, no test component, no edit to any stored generation, sheet, key or
checkpoint, no change to the pilot's prompt layout. If the launch happens, the session ends at
the launch and the result is read in a later session.

## Appendix — the 3B-17 instruction set, deferred

§29.3 makes 3B-17 conditional on an adapter-preferred v5. The reading is 0.560, so it is **not
run**. If a human rescore reverses that, this is the whole instruction set, to be pasted into a
session on the training device:

> Run 3B-17 on the 40 items of `outputs/analysis/phase3a/fit/audit_ab_100_v5_key.csv`. For each
> item: take the pilot adapter's stored generation for that item as the first attempt (no new
> sampling); validate it against the prompt and the supplied classifier values with the four
> re-specified checks of PROTOCOL §28 (grounding, numeric comparison, tag role, anchor); if any
> check flags, regenerate once with a corrective suffix that names only the specific failures and
> changes nothing else in the prompt; if the retry still flags, deliver fallback v3.1 for that
> item. Report the fallback rate over the 40 items, and recompute every gate of §29.5 on the
> delivered text, not on the first attempt. Build `audit_ab_100_v6.csv` to the rules of §26.3 —
> delivered text against fallback v3.1, identical layout, 50/50 key per stratum, blank scoring
> cells — with a fresh seed and a provenance file. Do not edit v5, its key or its provenance. Do
> not train. Report the measured numbers and stop.
