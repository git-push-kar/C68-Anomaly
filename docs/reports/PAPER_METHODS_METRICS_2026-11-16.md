# Paper methods, metrics and claims — presentable without further experiments

**Date:** 2026-11-16
**Purpose:** everything the paper needs to describe the approach, define its metrics and state its results,
written against the artefacts that exist now. No outstanding experiment is required for any statement here;
the four items that would *strengthen* the paper are listed in §7 and are all optional.

---

## 1. System under evaluation

An industrial anomaly-detection and root-cause pipeline in which **diagnosis is deterministic**: the verdict,
fault identity, severity and confidence are computed by code and calibrated classifiers, and the language
model only writes prose and answers follow-up dialogue. This separation is the architectural claim (C8) and is
enforced by construction, not measured.

The classifier is wired into the live report path behind `hybrid.enabled` (default false) against the retrained
`vocabfix` artefacts. When it is on, `report["diagnosis"]` carries the menu-legal verdict and confidence, the
prose fields are untouched, and events with no detector evidence are refused with `evidence_incomplete` rather
than labelled. The wiring adds no metric: the numbers below are this same pipeline run offline on the stored
splits.

Two report writers are compared on identical evidence:

| label | writer | mechanism |
|---|---|---|
| **A** | fallback v3.1 | deterministic template filled from the evidence store |
| **B** | InternVL3-2B + LoRA adapter (seed 42) | generative, greedy, cap 512, batch 16 |
| C | seam | A's stored numbers spliced into B's prose |
| M | seam | B's prose with mechanism claims taken from the knowledge base |
| D | wrapper | B, with substitution by A on rows where B is flagged |

Seams C and M are controls: they isolate *quantity* from *mechanism* while keeping both alive.

## 2. Data, splits and composition

| | validation (fit set) | test (confirmatory) |
|---|---|---|
| events | 520 | 3,168 |
| orders | kb, shuffled-1, shuffled-2 | kb only |
| open-set share | 87.3 % | 57.3 % |
| normal-transient share | **9.0 %** | **1.3 %** |
| classified share | 3.7 % | 41.5 % |

**Composition must be stated wherever a rate is quoted.** The two splits have very different class mixes, and
the only class where the writer fails is normal-transient. That single fact explains the whole apparent
difference between the splits (§5.3).

## 3. Metric definitions, exactly as implemented

All checks run on the parsed prose body, after schema parsing. Thresholds are the pre-registered gates.

**3.1 Numeric fidelity (gate ≤ 2 %).** A sentence is a defect when it names a detector and quotes a number
that the page does not supply, or when its inequality points the wrong way for that detector's value and
threshold. Three rules make this exact, and each was added after a measured artefact:

1. A number is attributed to **every detector named in the sentence**, not only the first.
2. A number quoted **in a role** must match a value of that role: `z=17.99` must equal a sensor's z-score,
   `deviation 221.8` a deviation, `lag 4` the dominant lag. A number with no role marker passes if it appears
   anywhere on the page.
3. **Identifiers are not quantities.** Tokens that begin with a letter and contain a digit (`FAULT_08`,
   `Component_D_Stream9`, `XMEAS_17`, `r481`) are masked before numbers are read; unit-suffixed numbers
   (`24.12x`, `221.80%`) survive.

**3.2 Grounding (gate ≤ 2 %).** Prose that references a sensor tag not supplied in the prompt, or a value not
printed on the page.

**3.3 Instrument-role (tag-role) claims (gate ≤ 2 %).** Prose that describes a sensor as an actuator or a
detector as a measurement — a binding error, not a numeric one.

**3.4 Self-contradiction (gate ≤ 2 %).** Negation-aware: prose that asserts an alarm the evidence contradicts
and vice versa. The checker is calibrated on 46 hand-written cases at 46/46.

**3.5 Anchor exactness (gate ≥ 90 %).** Whether the report states the top anomalous sensor's z-score (or
names the top sensor), computed over rows that have a top sensor.

**3.6 Primary contrast.** Paired McNemar on numeric defects, exact two-sided.

## 4. Instrument validation (why these metrics can be trusted)

The same function is used for every condition and every split. Two reproductions anchor it:

- rebuilding A on validation gives **0/520** numeric defects — the writer copies from the page, so zero is
  the only defensible answer;
- the identical code reproduces **834/834** stored validation probability vectors and the adapter's test
  probabilities, so the test contrast is not a different construction.

The checker's three revisions and their effect (< 19 → 13 → 0 for A) are documented in §8. All numbers in
this document come from the final version.

## 5. Results

### 5.1 Cell table, 520 validation rows per cell

| cell | numeric | rate | grounding | tag-role | contradiction | anchor exact |
|---|---|---|---|---|---|---|
| A | **0** | 0.00 % | 0 | 0 | 0 | **520/520** |
| B | 30 | 5.77 % | 33 | 57 | 0 | 188/520 |
| C | **0** | 0.00 % | 1 | 17 | 0 | **520/520** |
| M | 30 | 5.77 % | 32 | 43 | 0 | 191/520 |
| D | 0 | 0.00 % | 0 | 0 | 0 | 262/520 |

McNemar, B against A: **30 vs 0**, exact p ≈ 1.9 × 10⁻⁹. Seam certification: C leaks 0 of B's numeric values,
M 0 of A's. D substitutes A on **114/520 rows (21.9 %)**, inside its pre-registered 25 % ceiling.

### 5.2 Confirmatory pass, 3,168 test rows

| check | A | B | gate |
|---|---|---|---|
| numeric | **0 (0.00 %)** | **40 (1.26 %)** | ≤ 2 % — both pass |
| grounding | 250 (7.89 %) | 325 (10.26 %) | ≤ 2 % — both fail |
| contradiction | 0 | 5 (0.16 %) | ≤ 2 % — both pass |
| tag-role | 0 | 458 (14.46 %) | ≤ 2 % — B fails |
| anchor exact | **3,168 (100 %)** | **1,170 (36.93 %)** | ≥ 90 % — B fails |

McNemar: 40 vs 0, exact p ≈ 2 × 10⁻¹². B's anchor failures: z omitted 46.06 %, contradicting value 6.41 %,
sensor absent 10.61 %.

### 5.3 Where the adapter actually fails

| gold class | validation | test |
|---|---|---|
| normal-transient | **29/47 = 61.7 %** | **29/40 = 72.5 %** |
| open-set | 1/454 = 0.2 % | 6/1,814 = 0.3 % |
| classified | 0/19 = 0 % | 5/1,314 = 0.4 % |

This is the paper's sharpest descriptive result: **the adapter is numerically reliable almost everywhere
(≈0.3 %) and fails on no-alarm narratives (62–73 %)**, consistently across splits. The overall rates
(5.77 % vs 1.26 %) differ only because normal-transient rows are 9.0 % of validation and 1.3 % of test.

### 5.4 Diagnosis channel

497 shared items: menu rule **1.0000**, LightGBM **0.9980**, hybrid **0.9960** (2 vs 1 discordant, p = 1), MLP
0.9799, adapter alone **0.7646** (0.7253–0.7998; the pre-vocabulary-fix 0.8330 is superseded). The knowledge-base
layer earns its place in *explanation*, not accuracy.

*Re-measured 2026-11-16.* These replace the earlier hybrid 0.9940 / LightGBM 0.9940 / MLP 0.9698 / menu rule
0.9980. The cause was a vocabulary defect, not a modelling choice: `TAGS` named the manipulated variables
`XMV_1..11` while the page prints `XMV_42..52`, so 22.4 % of listed sensors had no feature column. Retraining
every classifier-side model on the aligned vocabulary also moves the cross-fitted observable accuracy from
0.9781 to 0.9878 (macro-F1 0.9708 → 0.9843) and the out-of-fold threshold from 0.9792 to 1.0. The adapter's row
is unchanged because it never read those features.

### 5.5 Preference

360 pairs, two readers, κ = **0.5776** (observed 0.7778, expected 0.4739) — below the declared 0.60 floor,
so the preference tier is reported descriptively and carries no claim.

### 5.6 Selection

Six rungs: rung 1 vetoes B (5.77 % against 1.00 % = A + 1 pp); rungs 2–4 unavailable (preference below its
floor; no eligible passer carries a mechanism claim); rung 5 compares seconds per 520-row pass on a signed
basis and leaves A and D tied, resolved by the declared tie-break. Receipt: **chosen A,
`decided_by_rung 5`**, no dollar figure claimed.

## 6. What may and may not appear in the RP

**May appear, as written above:** the architecture and its invariant; all metric definitions in §3; the cell
table; the test contrast; the stratified failure profile; the classifier and preference results; the selection
record; the batch-invariance measurement (1 of 12 texts identical between batch 16 and batch 8, so batch size
is a measurement parameter).

**Must not appear:** any claim that the adapter is more numerically faithful than the deterministic writer;
any bare rate comparison between validation and test; "neither writer meets the numeric gate"; a D pass; a
preference ordering; a mechanism-quality claim; any latency figure for components other than the generator.

## 7. Declarations required if the remaining experiments are not run

1. **Prompt-template drift.** None of the 520 stored validation prompts matches the current builder (0/520
   byte-identical; the menu header reads "answer with one of these fault_id values" in the stored set and
   "order varies by event" now). Validation and test are therefore different prompt builds: report both
   numbers, compare them only within class (§5.3), and say so in the limitations.
2. **Composition.** State the class mix of each split next to every rate.
3. **Instrument history.** Include §8, or a citation to it, so a reader can see why A moved 19 → 13 → 0.
4. **D's ceiling.** The signed decision records a 27.3 % breach; the corrected table shows 21.9 %. Either
   keep the conservative "A with extra steps" wording (recommended: D contributes nothing of its own on the
   rows it hands back) or re-sign the decision — do not silently claim a pass.
5. **Single configuration.** One adapter seed, one order (kb) on test, E unscored, parity's 1,390
   disagreements unresolved, no detector-frontier table, no follow-up T condition.

## 8. Instrument changelog

| version | rule added | validation A / B | test A / B |
|---|---|---|---|
| v1 | numbers attributed to the first named detector only | 19/520 · 62/520 | 619 · 115 (role fix included) |
| v2 | role matching (`z=` against sensor z) | 13/520 · 50/520 | 619 · 111 |
| **v3 (final)** | identifiers masked (`FAULT_08`, `XMEAS_17`) | **0/520 · 30/520** | **0 · 40** |

Earlier rate claims that moved (3.65 % → 2.50 % → 0.00 % for A) are superseded; the pre-revision cell table
is kept as `n4_cells_preinstrument_2026-11-16.json`.

A second changelog belongs here because it moved classifier-side numbers only: the feature vocabulary was
corrected from `XMV_1..11` to `XMV_42..52` (§5.4). Everything trained on those features was retrained into
`outputs/analysis/phase3a/hybrid/vocabfix/`, beside the published artefacts, and the text quotes the post-fix
numbers. The writer-side checks (numeric, grounding, role, anchor) are untouched by it: they read prose against
the page, not the classifier's columns.

**Closed 2026-11-16: the diagnosis figures are rebuilt** (`fig3`, `fig6`, `fig8`, `approach_data.json`;
`fig7` unchanged because nothing in it moved). The leave-one-fault-out surrogate was retrained across all 13
faults into `vocabfix/` (~100 s per shard, deterministic), and `fig8` is now rendered from that artefact rather
than from a stale comparison block. Cross-fitted intervals come from a cluster bootstrap over the 497 validation
events: accuracy 0.9878 (0.9783–0.9962), macro-F1 0.9843 (0.9711–0.9947), reproducing the quoted point estimates
exactly.

## 9. Optional work, ranked by what it would buy

1. Regenerate the 520 validation texts under the current template (≈35 min) — removes declaration 1.
2. Two extra orders on the 520 validation rows (≈60 min) — bounds order sensitivity without 6.3 h.
3. E seed 43 (≈45 min, after a minutes-long integrity check) — turns "at seed 42" into a seed-robust range.
4. Parity on the numeric check only — provenance for the one number the claims lean on.
