# TRAINING SPEC — PROSE V1 (docs only; nothing built)

**Repo:** `C:\Users\Admin\Desktop\anomaly` · **Branch:** `benchmark-upgrade` · **Date:** 2026-11-01
**Status:** specification for **outcome A** and **outcome B** of PROTOCOL §20.3. **No adapter was
trained, no training file was modified, no teacher call was made, and no new data file was written
to produce this document.** Every count below is re-read from artefacts that already exist.

**Why two outcomes and not one.** §19.2/§20.3 decide the report path on the A/B sheet. Until that
sheet is scored, both branches have to be specified, because they train different targets:

| | **Outcome A** (adapter prose passes §19.2) | **Outcome B** (it fails; v2 ships prose) |
|---|---|---|
| report path prose | the adapter | **fallback v2** (deterministic) |
| what the adapter is trained for | `ranked_hypotheses`, `physical_reasoning`, `narrative` **+ follow-ups** | **follow-ups and refusals only** |
| report rows in training | prose supervised | prose **masked out**; only follow-up turns are supervised |
| pilot purpose | does prose-only clear the §20.4 gates | do refusals hold at ≥ 90 % without prose drag |

Outcome **C** (tie on preference, both ≥ 85 % correct and ≥ 90 % grounded) uses **B's** training and
**v2** for reports, so it needs no third spec.

---

## 1. Prompt layout

Fixed for both outcomes, so the two branches differ only in the loss mask:

| element | supplied in the prompt | notes |
|---|---|---|
| event page | detectors, top deviating sensors with z/deviation/trend, CVA residual shares, onset order | unchanged from v2's rendering |
| **`fault_id`** | **supplied** (classifier + menu rule) | the adapter must not restate it differently |
| **`verdict`** | **supplied** | derived from the diagnosis, never independently |
| **classifier top-2 with calibrated probabilities** | **supplied**, printed on the page | new in this contract: v2's differential quotes these, and §20.6's grounding count shows why they must be *on the page* — a probability the reader cannot see is a side channel (see §4 below) |
| candidate menu | **shuffled per item** (seeded), **or removed** with the diagnosis's own KB entry supplied | see below |

**No fixed KB menu order, anywhere.** This is the §20.3 requirement and it follows from the 3B-7
order result: permuting the sixteen menu lines cost the adapter 28 points of accuracy and turned a
2 % false-abstention rate into ~44 % (§3b8_corrections.md §a). Two admissible layouts, both fixed
per row by a seeded hash of `(seed, item_id)` so a run is reproducible:

1. **shuffled menu** — all sixteen entries, permuted; the diagnosis is somewhere in the list;
2. **menu removed, diagnosis's KB entry supplied** — the menu block is dropped and the single KB
   entry for the supplied diagnosis is printed in its place, so the model still sees the mechanism
   it is writing about without seeing a list whose order could become a feature.

Both are used in training **and** in evaluation; a row trained in one layout is evaluated in both.

**Data-dependent caution, stated as a risk rather than a solution.** Layout 2 removes the menu
order dependency by construction but changes what the model can do: on a menu-removed row the
adapter cannot name an alternative that is not the supplied entry, so `ranked_hypotheses` loses
its second and third entries. That is acceptable for a follow-up-only adapter (outcome B) and is a
real loss for outcome A, whose §20.4 gate includes `ranked_hypotheses`. **Recommendation: none
here** — this is one of the two decisions §20.3 leaves to the pilot.

---

## 2. Which label is supplied in training: gold or cross-fitted prediction

Both are listed. **No recommendation is made**, because the deciding number (§4) is small enough
that the choice turns on what the pilot is for, not on the counts alone.

| | **gold label** | **cross-fitted prediction** |
|---|---|---|
| source | `target.fault_id` | out-of-fold classifier + menu rule on the train split |
| leakage | none in the training target, but it is **not** the deployment condition: at inference the label comes from the classifier | none — it is exactly the deployment condition, by construction |
| teaches the model to | write prose for the **true** fault | write prose for **the fault it will actually be handed** |
| disagreement with gold | — | **8 of 1,919 events differ (0.42 %)**: **2** name a wrong fault, **63** abstain where gold is a named fault |
| **pool of events where the supplied label is not the truth** | 0 | **65 / 1,919 = 3.39 %** (§20.10 basis; the 254 figure is withdrawn — see 3b8_corrections.md §c) |
| what the model learns from a disagreement | nothing (the label is right) | either a rationalisation of a wrong fault (2 events) or confident prose around an abstention where a fault exists (63 events) |

`65` is the count that matters for either choice: under the cross-fitted supply, **65 events in
1,919** hand the model a prefix that contradicts the evidence, and at inference the same 65 will be
the rows where the deployment prefix is already wrong. Training on gold hides that; training on
cross-fitted bakes it in. The pilot's §20.4 gates apply to both, and outcome A's correctness gate
(≥ 85 %) is measured *against the supplied diagnosis*, which is why the distinction has to be
declared per run rather than absorbed.

---

## 3. Loss masking

| masked (context, no gradient) | supervised |
|---|---|
| `fault_id` value and key | `ranked_hypotheses` |
| `verdict` value and key | `physical_reasoning` |
| the opening `{` and the separators around them | `narrative` |
| — | **every follow-up turn's answer** (unchanged) |

Measured cost of masking the two fields, on `train_mixed.jsonl` with the trained model's tokenizer:

| quantity | tokens |
|---|---|
| supervised today (report 573,061 + dialogue 71,067) | **644,128** |
| with `fault_id` + `verdict` masked | **609,697** |
| removed on a value + terminator basis | **22,917** |
| removed in total, keys/quotes/brace/separators included, net of BPE merges | **32,512** |
| **change in step time** | **≈ 0** — the sequence length is unchanged, so forward and backward do the same work |

For **outcome B**, the report rows' prose is masked as well, so a report row contributes **no**
supervision and only the 961 dialogue rows do. That is a 5.2× reduction in supervised rows and it
is the main risk of outcome B: the adapter would be trained on dialogue alone. The mitigation that
does **not** change the token count is to keep report rows in the batch as context-only rows, at
≈ 0 change in step time.

---

## 4. What the v2 measurement adds to this spec (finding, not a request)

Fallback v2's differential prints `p = 0.xx` from the classifier. Counted against the prompt alone
— the rule that measured v1 — v2 has **666 such number tokens over 429 of 834 rows**, and **0**
number or tag violations of any other kind. Under §19.2 criterion 2 as literally written ("every
number it cites appears in that item's evidence block") that is a failure; under the reading "the
supplied classifier values are evidence for this item" it is not.

**Consequence for this spec, stated as a requirement:** if the report path ships v2, the supplied
top-2 with calibrated probabilities must be **printed on the page**, so the differential quotes a
number the reader can see. Otherwise the deterministic explainer would be the one source that
cites something outside the prompt. This is a prompt-layout change, not a model change.

---

## 5. Unchanged hyperparameters

No hyperparameter is proposed to change, and this document proposes nothing:

| | value |
|---|---|
| LoRA | r = 16, α = 32, dropout 0.05, **7 projection targets** |
| precision | bf16 |
| batch | 4 × grad-accum 4 = **effective 16** |
| optimiser | lr 2e-4, cosine, 3 % warmup |
| sequence | `max_sequence_length` 4096; `max_new_tokens` 512 |
| epochs | 3 |
| logging | `logging_steps` 10, telemetry on |
| eval-loss | off (`val_file: null`) |

---

## 6. Data rebuild steps, and which need teacher calls

| # | step | teacher calls |
|---|---|---|
| 1 | supply `fault_id`, `verdict`, and the classifier's top-2 with calibrated probabilities in each report row's prompt (all from the existing cross-fitted artefact) | **none — programmatic** |
| 2 | print or shuffle the menu per row (seeded), or drop it and print the diagnosis's KB entry | **none — programmatic** |
| 3 | mask `fault_id` and `verdict` (and, for outcome B, the prose) in the answer text | **none — programmatic** |
| 4 | rebuild the manifest and hashes; re-run the split / leak / open-set checks | **none** |
| 5 | regenerate dialogue answers **only if** step 1 or 2 changes a dialogue prompt (dialogue rows carry the event, so they may not) | **teacher calls**, ≈ 961 rows, ≈ 250–330 calls at the observed 3–4 rows per call |
| 6 | train, then evaluate on validation | **none** — GPU |
| 7 | pilot gates, then `fold_1`, `fold_2`, then seeds (§20.3) | **none** — GPU |

Steps 1–4 and 6–7 are programmatic. **Only step 5 can spend teacher calls**, and only
conditionally — which is why no teacher call was made to write this spec.

---

## 7. Pilot gates (validation only, §20.4 verbatim)

| gate | threshold |
|---|---|
| schema valid | **≥ 98 %** |
| completion at 512 tokens | **≥ 95 %** |
| **genuine** grounding flags (§16 adjudicated reading: rounding, substring and inside-tag matches are not flags) | **≤ 2 % of rows** |
| empty narrative | **0 %** |
| follow-up over-refusal on in-scope controls | **≤ 10 %** |
| refusal on `out_of_scope` + `false_premise` | **≥ 90 %** |
| shuffled-order sensitivity of any adapter-written field | **reported**, no threshold |

Folds (`fold_1`, `fold_2`) are required for any prose claim under every outcome. No test component
is opened at any point; every number is validation (§15.2, §20.5).

---

## 8. Time estimate per run, per outcome

Grounded in run-1's measurement, not in a fresh run: **540 steps** (3 epochs × 180 steps at
effective batch 16), **21.71 s/step mean**, **12,003.5 s = 3 h 20 m 02 s** wall clock.

| branch | runs | per run | branch total |
|---|---|---|---|
| **A pilot** (prose-only, seed 42) | 1 | ≈ 3 h 20 m | 3 h 20 m |
| **A + folds** (`fold_1`, `fold_2`) | 3 | ≈ 3 h 20 m | ≈ 10 h 00 m |
| **A + folds + 2 further seeds** | 9 | ≈ 3 h 20 m | ≈ 30 h 00 m |
| **B pilot** (follow-ups/refusals only) | 1 | ≈ 3 h 20 m | 3 h 20 m |
| **B + folds + seeds** | 9 | ≈ 3 h 20 m | ≈ 30 h 00 m |

**Why the step count does not change between outcomes:** masking changes which positions contribute
to the cross-entropy, not the sequence length, so forward and backward do identical work. The only
thing that would move the estimate is *shortening* the answer or dropping rows from the batch —
outcome B drops supervision, not rows, if report rows are kept as context-only rows. A run is
estimated at **≈ 3 h 20 m** (the measured mean above) and should be re-measured, not assumed, on the
first run of either branch.

**Token accounting behind the estimate**, per epoch: 609,697 supervised tokens after masking,
2,880 rows, ~299.6 supervised tokens per report row (median 297), 180 steps at effective batch 16.
