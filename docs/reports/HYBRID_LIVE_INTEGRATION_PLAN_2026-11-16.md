# Wiring the classifier + menu rule into the live `TEPApp` report path — plan only

**Date:** 2026-11-16 · **Status:** proposal, nothing implemented. No file in the live path has been modified.
**Goal:** make the shipped report path decide the fault with the classifier (as measured), instead of letting the
adapter's generation decide it.

---

## 1. What exists today, and the exact seam

| piece | where | state |
|---|---|---|
| single decision point for a report | `main.py:384` `TEPApp._generate_report(event)` | `report, raw = self.rca.generate_report(evidence)` if the adapter loaded, else `_fallback_report(event)` |
| report generation | `llm/inference.py:327` `RCAInference.generate_report(evidence)` | builds `format_evidence_question(evidence)` + `build_chat_prompt`, decodes free-form JSON, `sanitize_answer` |
| classifier + rule | `outputs/analysis/phase3a/hybrid/protocolA/{classifier.txt, calibration.json, threshold.json, metadata.json}` | proven, but only called from `scripts/` |
| rule | `llm/hybrid/menu_rule.py:28` `menu_rule(label, confidence, menu_ids, threshold)` | pure function, no I/O — reusable as is |
| feature parser | `scripts/evidence_classifier.py:76` `prompt_features(prompt)` → 323 floats | parses the **rendered page**, not an evidence dict |
| forced-prefix generation | `llm/hybrid/generate.py` (`continuation_prompt`, `parse_slim`, `validate_slim`, `contradictions`, `valve_flow_mislabels`, `fallback_answer`, `retry_question`) | written for the hybrid, never called from the live path |
| app assembly | `main.py:135` `TEPApp.__init__` | constructs detector, scaler, event store, then optionally `RCAInference` |
| HTTP surface | `api/server.py` | returns whatever `process_sensor_stream` / report dict contains; no diagnosis-specific code |

So the integration has exactly **one** seam to change (`_generate_report`) plus one constructor branch
(`__init__`), and it can reuse every measured component as-is.

---

## 2. The one real obstacle: feature parity, and the menu

**2.1 Features come from the page, not from the event.** The classifier was trained on
`prompt_features(prompt)`, where `prompt` is the rendered evidence page (`DETECTORS`, `TOP DEVIATING SENSORS`,
`DYNAMIC RESIDUAL ATTRIBUTION`, `ONSET ORDER`, `CANDIDATE MENU`). The live app has an `evidence` dict, so it must
render the *same* page. `llm/dataset.py:128 format_evidence_question` produces that page today — but the
benchmark pages carry `CANDIDATE MENU (order varies by event)` while the live renderer emits
`CANDIDATE MENU (answer with one of these fault_id values)`. That difference is precisely the template drift
already documented between the fit and test splits, and it is a 323-feature input, so **parity must be tested,
not assumed**. This is the gating item of the whole plan: a correct classifier fed a differently-rendered page is
worse than the current fallback, because it fails silently.

**2.2 The menu does the abstention work, not the threshold.** On validation, 92.8 % of calibrated maxima sit at or
above the threshold; the abstention arm fires 22 times in 834, while the menu rule fires 322. In deployment:

- if the menu is the **full 15-fault list**, `not_on_menu` can never trigger, so open-set faults (16–20) get named
  as some known fault — the failure mode the classifier was adopted to remove;
- if the menu is a **KB shortlist** derived from the evidence (the benchmark's `::menu` construction), open-set
  protection survives, but the shortlist generator becomes part of the live contract and needs its own test;
- a **novelty gate** (distance/margin signal from §17) would be the principled fix, but the signals separate
  unseen faults at AUROC ≈ 0.88 and the lookalike problem (`FAULT_05 → FAULT_12`, 25/25) remains, so this is a
  research change, not a wiring change.

**Recommendation for the first wiring:** KB shortlist menu (preserves measured behaviour), full-list only if the
product requirement is "always name a fault". Record the choice in the report payload so a reader can tell which
regime produced a label.

---

## 3. Minimal change

**3.1 New module — `llm/hybrid/live.py` (~120 lines).** Loads the three artefacts once, exposes one method:

```python
class HybridDiagnoser:
    def __init__(self, artefact_dir: Path, threshold: float | None = None) -> None: ...
    def page(self, evidence: dict) -> str:          # delegates to the canonical renderer, no reimplementation
    def diagnose(self, evidence: dict, menu_ids: Sequence[str]) -> dict:
        # returns {label, calibrated_confidence, top3, menu_ids, reason, verdict,
        #          threshold, variant, classifier_sha256}
    def features(self, evidence: dict) -> np.ndarray  # exposed so the parity test can hash it
```

Load-time validation, fail-fast: missing booster, calibration whose `labels` length ≠ booster output width, or a
threshold outside (0, 1] raises at construction — the app must not start half-wired.

**3.2 `main.py`, two edits (~25 lines total).**

- `__init__`: after the adapter block, `self.diagnoser = HybridDiagnoser(...)` when
  `config["hybrid"]["enabled"]`, wrapped so any failure logs and leaves `self.diagnoser = None`.
- `_generate_report`:

```python
def _generate_report(self, event):
    evidence = event.evidence.to_dict()
    if self.diagnoser is not None:
        decision = self.diagnoser.diagnose(evidence, self.menu_provider(evidence))
        if self.rca is not None and self.config["hybrid"]["prose"] == "adapter":
            report, raw = self.rca.generate_report_with_diagnosis(evidence, decision)
        else:
            report = fallback_answer(decision["label"], decision["verdict"], decision["menu_ids"], ...)
        return stamp_diagnosis(report, decision)          # provenance fields, §3.5
    return self.rca.generate_report(evidence)[0] if self.rca else _fallback_report(event)
```

**3.3 `llm/inference.py`, one method (~25 lines).** `generate_report_with_diagnosis(evidence, decision)`: build the
prompt with `continuation_prompt(page, decision["label"], decision["verdict"])`, decode with the existing
`_generate`/`decode_text`, then `parse_slim` → `validate_slim` → `contradictions` → `valve_flow_mislabels`.
Reject-and-replace, never repair: any contradiction, any invalid schema, or a label differing from
`decision["label"]` returns `fallback_answer(...)` and sets `report["prose_fallback"] = reason`. This mirrors what
the measurement showed: forced-prefix prose contradicted its prefix 0 times in 242, and the fallback is the only
prose path that cannot contradict a label it does not choose.

**3.4 `configs/config.yaml`, new block (~10 lines), default off.**

```yaml
hybrid:
  enabled: false                 # feature flag; analysis path unaffected
  artefact_dir: outputs/analysis/phase3a/hybrid/protocolA
  threshold_override: null       # null = use threshold.json
  menu: kb_shortlist             # kb_shortlist | full_list
  prose: deterministic           # deterministic | adapter
  on_missing_artefacts: fallback # fail-closed to _fallback_report
```

**3.5 Report contract additions** (fields, not prose): `diagnosis_source` (`classifier` | `fallback`),
`classifier_variant`, `classifier_sha256`, `calibrated_confidence`, `rule_reason`, `threshold`, `menu_ids`,
`prose_source` (`adapter` | `deterministic`), `prose_fallback` (null or reason). Without these the new path is
unauditable and indistinguishable from the old one in a stored event.

**3.6 `api/server.py`: no change required.** `process_sensor_stream` returns the event dict, so the new fields
surface automatically. `/api/status` should gain `"hybrid": {"enabled": ..., "variant": ...}` — two lines, useful
for the smoke test.

---

## 4. Files touched

| file | change | size |
|---|---|---|
| `llm/hybrid/live.py` | **new** — artefact loading, page rendering, predict + calibrate + rule | ~120 lines |
| `main.py` | `__init__` branch + `_generate_report` branch | ~25 lines |
| `llm/inference.py` | `generate_report_with_diagnosis` | ~25 lines |
| `configs/config.yaml` | `hybrid:` block | ~10 lines |
| `api/server.py` | status fields only | ~2 lines |
| `llm/dataset.py` | **only if parity fails** — extract one canonical page renderer and call it from both sides | ~30 lines refactor |
| `tests/test_hybrid_live.py` | **new** — unit tests | ~150 lines |
| `scripts/replay_hybrid.py` | **new** — replay harness (reuses `run_stream_from_file`) | ~80 lines |
| `docs/reports/HYBRID_LIVE_INTEGRATION_PLAN_2026-11-16.md` | this document | — |

No detector, scaler, event-store, schema, prompt or dataset file changes. The adapter and its artefacts are
untouched.

---

## 5. How it would be tested

Ordered so the cheapest failure stops the work.

1. **Feature-parity gate (blocks everything).** For 50 validation events, build the page through the live
   renderer, compute `prompt_features`, and assert equality (hash) with the features from the stored benchmark
   prompt for the same event. Pass criterion: 50/50 identical. If they differ, stop and fix the renderer — no
   accuracy number from a mismatched page is meaningful.
2. **Replay equivalence.** `scripts/replay_hybrid.py` drives `TEPApp.run_stream_from_file` over the validation CSVs
   with the flag on, captures `diagnosis` per event, and compares against `comparison.json` predictions for the
   same events. Pass criterion: 100 % agreement with the analysis pipeline (same booster, same features, so any
   disagreement is an integration bug, not a model result).
3. **Rule unit tests** (extend `tests/test_hybrid.py`): on-menu above threshold → `classified`; on-menu below →
   `below_threshold`; off-menu → `not_on_menu`; normal → `no_fault`; `None` label → `no_prediction`. Already
   covered; re-run unchanged.
4. **Prose invariant with a stub model.** Inject a fake `RCAInference` that returns (a) valid JSON matching the
   prefix, (b) valid JSON naming a different fault, (c) garbage, (d) prose with a contradiction, (e) prose with a
   valve/flow mislabel. Assert: (a) passes through; (b)–(e) produce `fallback_answer` and a non-null
   `prose_fallback`. Assert every report's `fault_id` equals `decision["label"]` or is
   `UNCLASSIFIED_OPEN_SET`/`NORMAL_TRANSIENT` — the invariant that makes the wiring worth doing.
5. **Failure modes.** Missing `classifier.txt` → app starts with `diagnosis_source: "fallback"` and
   `_fallback_report`; mismatched calibration width → construction raises and the app refuses the hybrid flag;
   corrupted threshold → same.
6. **Determinism.** Same event twice in one process and across two processes → identical label, confidence and
   report bytes except timestamps.
7. **Regression.** `python -m pytest tests -q` (currently 558 passing) and `./venv_train/Scripts/python.exe -m
   pytest tests -q` (555 + 3 skipped); no existing test may be relaxed. `scripts/package_release.py --check` must
   stay clean, with the new classifier hash recorded.
8. **Latency and cost.** Measure per-event cost with the flag on: classifier + calibration should be single-digit
   milliseconds on CPU; prose mode `adapter` adds ~3.5 s/item (measured on 3,168 rows), mode `deterministic` adds
   none. Record both; do not claim the new path is free.

**Shadow mode before enabling.** Run 1–2 days with `enabled: true, prose: deterministic` but keep the old report
in the API response alongside `shadow_diagnosis`. If the classifier's label disagrees with the adapter's on real
traffic, that is a finding worth having before it is customer-visible.

---

## 6. Effort and risks

| item | effort |
|---|---|
| `live.py` + two seams + config | 0.5 day |
| parity harness + renderer fix if needed | 0.25–0.5 day |
| tests (unit, stub-model, failure modes) | 0.5 day |
| replay equivalence + shadow run | 0.5 day + observation time |

**Risks, ranked.**

1. **Page-parity drift** (§2.1) — silent accuracy loss. Mitigated by test 1 as a hard gate.
2. **Menu semantics** (§2.2) — a full-list menu silently disables open-set protection; the threshold will not save
   it at 92.8 % saturation. This is a product decision, not a coding one.
3. **The unobservable stratum** — 12 validation rows where every classifier says `NORMAL_TRANSIENT` at confidence
   1.00. Wiring the classifier makes that error live; the current adapter-only path abstains on half of them.
   Accept it explicitly in the release note, or add the "no trace" signal first (research, not wiring).
4. **Prose cost** — `adapter` mode keeps the 3.5 s/item generation; `deterministic` is the shipped, cheaper,
   structurally safer mode (commit `a835ba3`, §7 of `RP_FINAL_REPORT`).
5. **Two labels in one system** — follow-up answers and the stored event must both read the classifier's label, or
   the app will explain one fault and store another. `stamp_diagnosis` plus a test on `answer_followup` closes it.

---

## 7. Addendum, 2026-11-16 — the menu policy is decided, and the parity gate **fails**

Run: `python scripts/check_menu_and_parity.py` (artefacts: `crossfit_val_rows.jsonl`, `events.db`).

### 7.1 Menu policy: **full 15-fault list plus an explicit `UNCLASSIFIED_OPEN_SET` option**

Simulated on the stored per-item probabilities of the 834 validation rows at the cross-fitted threshold 0.9792,
restricted to the 503 rows a deployment can actually present (`variant = menu`, i.e. the exhaustive menu; the 331
`removed` rows are constructed open-set cases whose gold assumes the true fault was withheld, and are not
deployable):

| policy | accuracy | abstentions | mechanism |
|---|---|---|---|
| **A: 15 faults + abstain option** | **0.9583 (482/503)** | **22** | `below_threshold` 22, `on_menu` 427, `no_fault` 54 |
| B: 15 faults, no abstain option | 0.9583 (482/503) | 137 | `below_threshold` 12, `not_on_menu` 125, `no_fault` 54 |

Reproduction check with each row's own menu: 0.9640 over 834 = **0.9781 over the 822 observable**, matching the
recorded cross-fitted accuracy exactly.

**Decision: A.** The two policies are decision-equivalent (482/503 either way), so accuracy does not choose
between them; the difference is accounting. Under A, "nothing listed" is an answer the classifier gives with its
own confidence (125 rows), kept separate from "evidence too thin" (22 rows). Under B both collapse into one
137-item abstention bucket reached by menu inference, and the monitoring signal that distinguishes an open-set
event from a low-confidence known fault is lost. A is also the regime every reported number was measured in.

**Shortlist rejected.** Every benchmark menu is exhaustive (15 faults + the abstain option); no measurement
exists for a shorter one. The only shortlist-shaped construction in the data is the `removed` variant, where
withholding the true fault converts **331 of 331** predictions into forced abstentions — i.e. a shortlist that
omits the true fault destroys a correct answer rather than protecting it. Keeping a fault out of the menu is
safe only if you already know the answer.

Unaffected by this decision: the 12 unobservable rows (faults 3, 9), which are predicted `NORMAL_TRANSIENT` at
confidence 1.00 under every menu policy.

### 7.2 Renderer parity: gate **fails**, on two layers

Same evidence dicts from `events.db` (25 events), both renderers, features compared element-wise:

| renderer | identical feature vectors | populated features |
|---|---|---|
| live `format_evidence_question` | **0 / 25** | ~1 of 323 |
| benchmark `prompt_v2.render_prompt` | n/a (reference) | 18.9 of 323 |
| stored benchmark page | n/a | 25–27 of 323 |

The live renderer is unusable for the classifier:

- section labels differ (`Top sensor deviations:` vs `TOP DEVIATING SENSORS`, `Event ID:` vs `Event:`);
- it emits **no `deviation …%` field**, so the sensor regex matches nothing — that is why only one feature is
  populated;
- it has no `Fused trigger` line, so the three fusion flags are always zero.

Swapping in the benchmark renderer fixes those but is **not sufficient**. Section-by-section on live evidence:
sensors 5/5, onset 6/5, residual **0/5**, LSTM and CVA scalars present. Two concrete gaps remain:

1. **`detector_evidence.lag_attribution` does not exist in live evidence.** The benchmark renderer reads it for the
   CVA residual block; live evidence carries `detector_evidence.dynamic.cva.sensor_contributions` instead. Mapping
   the contributions to `top_variables` (+ `dominant_lag`, `slow_drift_ratio`) lifts populated features from 18.9
   to 24.8 of 25–27, i.e. it closes the residual gap.
2. **Onset lists are one entry too long** (6 vs 5 in every stored page): cap `temporal_sequence` to the same five
   entries the benchmark used.

**Revised gate for the integration.** The parity test becomes a contract test on the mapping layer plus a replay
test, not a swap-and-hope:

1. build a live page with `prompt_v2.render_prompt(evidence, build_menu())`;
2. assert the rendered page reproduces the stored benchmark page for a replayed event;
3. assert the populated-feature count and per-section counts match the stored distribution (sensors 5, onset 5,
   residual 5, scalars present);
4. only then compare `prompt_features` hashes.

Until those pass, the classifier must not be wired in: a classifier fed the current live page sees almost no
evidence and would be worse than the deterministic fallback while looking confident.

### 7.3 Correction, 2026-11-16 — the onset gap was not one; the mapping layer is built and tested

§7.2 item 2 ("onset lists are one entry too long") is **withdrawn**. It was inferred from five pages. Measured
over all 834 stored validation pages, onset entries per page are `{1: 83, 2: 90, 3: 77, 4: 62, 5: 54, 6: 468}`
— six is the mode. A five-entry cap reproduced 8 of 20 stored pages and destroyed the other 12. The list is now
passed through unchanged.

Built, not yet wired into `TEPApp`:

- `llm/benchmark/live_page.py` — `derive_lag_attribution` (top five shares of the 52 CVA contributions, plus
  `dominant_lag` and `slow_drift_ratio`), `normalise_evidence` (fills `lag_attribution` when absent, hoists a
  nested `dynamic.cva` score so the CVA line renders, leaves onsets alone) and `render_live_page` (benchmark
  renderer, with the teacher instruction inserted ahead of the schema line as stored pages have it).
- `tests/test_live_page_contract.py` — 34 tests, all passing. Twenty stored pages are parsed back into evidence
  and re-rendered: the text is byte-identical apart from the event-id hash and the two ratio strings, and the
  feature vectors are identical apart from the two ratio features, which agree to better than 1 % (they divide
  unrounded operands, so the printed values cannot reproduce them exactly).

Live effect, 25 events from `events.db`: populated features **24.8** mean (18–30), against **25.5** (25–27) for
stored pages; sensors, onsets, five residual shares, CVA line, fused trigger and instruction present in 25/25.

Open: the minimum of 18 sits below the stored floor of 25 — events with short sensor or onset lists produce
fewer features than any stored page. Not yet characterised per fault, and not a reason to wire the classifier in
before it is.

### 7.4 The classifier on the stored live events, 2026-11-16

Run: `python scripts/report_live_classifier.py --limit 489` (artefact:
`outputs/analysis/phase3a/hybrid/live_events_report.json`). 489 events from `events.db`, Protocol A booster,
isotonic calibration, deployment menu A. There is **no ground truth** for these events, so nothing below is an
accuracy: it is what the model answers and how the rule treats it.

| threshold | abstentions | classified | `no_fault` | argmax = rule, non-abstained |
|---|---|---|---|---|
| shipped artefact **1.0** | 387 / 489 | 86 | 16 | 102 / 102 |
| paper cross-fitted **0.9792** | 263 / 489 | 210 | 16 | 226 / 226 |

The agreement column is 100 % by construction — `menu_rule` only ever abstains, it never renames a prediction —
so the meaningful column is the abstention count. **The shipped artefact carries threshold 1.0, not the paper's
0.9792**, which alone accounts for 124 of the abstentions. Answers: `FAULT_15` 167, `FAULT_11` 106,
`FAULT_01` 73, `FAULT_08` 33, `FAULT_02` 26, `UNCLASSIFIED_OPEN_SET` 26, `NORMAL_TRANSIENT` 16, remainder ≤ 9.
Calibrated confidence: mean 0.743, min 0.350, max 1.000.

**Page population splits by cohort, not by renderer.** Detector-bearing events (156) produce 25.1 features on
average (15–33), close to the stored mean of 28.4 (19–39). Events with no `detector_evidence` (333) produce
13.5 (8–22). The 32 events at 18–20 features are 24 legacy records and 8 vocabulary mismatches.

**The gap is not evidence sparsity.** It has three causes, two of which are contract bugs:

1. **Legacy records (333)** carry no `detector_evidence` at all — schema v1 fields (`severity`,
   `candidate_subsystem`) and no LSTM or CVA score. Nothing to render: the record is missing the evidence, and
   the classifier was never trained on a page shaped like this. This is sparsity of the *record*.
2. **Tag vocabulary mismatch.** `evidence_classifier.TAGS` names `XMV_1..11`; the page prints `XMV_42..52`, so
   those sensors' z, deviation, trend and presence are silently dropped. Measured: **22.4 %** of listed sensors
   on stored pages (932/4,170) and **28.8 %** on live pages (703/2,445) have no column. This one is not a live
   problem — it weakens the benchmark features too, which is why stored `present` averages 3.88 of 5.
3. **Zero encoding.** `trend = stable` becomes 0.0 and an onset at `+0.0 min` becomes 0.0, indistinguishable
   from absent. Live trends are mostly `stable` (mean nonzero trend 1.1 against 3.65 stored), so a live page
   looks thinner than a stored one even when nothing is missing.

**Gate changes before wiring.** (a) decide what happens to the 333 legacy records — refuse to diagnose them, or
backfill detector evidence; a page without detector rows is out of distribution. (b) fix `TAGS` to
`XMV_42..52` and retrain: this changes the benchmark features and therefore every reported number, so it cannot
be smuggled in with the wiring commit. (c) the live default must use the cross-fitted 0.9792, not the artefact's
1.0. (d) re-run this report afterwards; the cohort split is the acceptance test, not the mean.

### 7.5 Fixing the vocabulary and re-measuring, 2026-11-16

`evidence_classifier.TAGS` now names `XMV_42..52`, matching the rendered tags. Retrained Protocol A and rebuilt
the cross-fitted calibration/threshold into a **separate** artefact directory
(`outputs/analysis/phase3a/hybrid/vocabfix/`, gitignored), so the published artefacts are untouched and the two
can be compared. Feature audit still clean; 323 features.

| cross-fitted validation (822 observable) | before | after |
|---|---|---|
| accuracy | 0.9781 | **0.9878** |
| macro F1 | 0.9708 | **0.9843** |
| abstention rate | 55.84 % | 55.23 % |
| ECE | 0.0200 | 0.0114 |
| out-of-fold threshold | 0.9792 | 1.0 (achieved false abstention 2.72 %, budget 5 %) |
| saturation share | 92.81 % | 98.32 % |

Policy A over the 503 menu-present rows: **0.9583 → 0.9722** (482 → 489 correct), open-set verdicts 344 → 331.
No fault got worse; six improved: F1 0.962 → 1.0 (04), 0.889 → 0.936 (08), 0.936 → 0.980 (10), 0.913 → 0.958
(12), 0.980 → 1.0 (13), 0.984 → 0.991 (open set).

Feature population: stored pages now fill all five `present` columns (3.88 → 5.00). Live pages go from 17.2 to
23.0 populated features overall — legacy records 13.5 → 18.9, detector-bearing 25.1 → 31.8 — because the
manipulated variables stopped being dropped.

**Consequence: every reported number that quotes 0.9781 or 0.9583 is now stale.** The manuscript, the methods
metrics file and the RP report were **not** edited here; they need a deliberate re-audit, not a find-and-replace,
because the threshold also changed regime (0.9792 → 1.0) and the saturation share with it. The test split was
not re-run: retraining on train and re-measuring validation is within the pre-registered split, while opening
the test component again is the owner's decision.

### 7.6 What the live path should do with the 333 events without `detector_evidence`

They are pre-instrumentation records (schema v1: `severity`, `candidate_subsystem`, no LSTM or CVA score). Even
after the vocabulary fix they populate 18.9 features against ~33 for a detector-bearing page, so they remain out
of distribution.

Recommendation, in order:

1. **Gate, do not classify.** The seam should test for a usable detector block (`lstm_ae.score` and
   `dynamic.score`, or a derived equivalent) and, when it is absent, skip the classifier. Emitting
   `UNCLASSIFIED_OPEN_SET` would claim a model judgement that never happened; the honest verdict is
   *"diagnosis unavailable — evidence incomplete"*, with the deterministic summary still shown.
2. **Backfill only from telemetry.** For stored runs whose raw windows still exist, recompute the detector block
   in a batch job and classify afterwards. Never synthesise scores to make a page look complete.
3. **Segment the metrics.** Report accuracy and rule reasons per cohort (`detector_bearing` vs `legacy`), so a
   mixed average cannot hide the fact that a third of the traffic is unclassifiable.

### 7.7 Wired, behind a flag, 2026-11-16

This section supersedes the "not yet wired" statements in §7.3 and §7.5.

| piece | where |
|---|---|
| `HybridDiagnosis` | `llm/hybrid/live_classifier.py` — page → 323 features → Protocol A booster → isotonic calibration → `menu_rule` with deployment menu A |
| flag | `hybrid.enabled` in `configs/config.yaml`, **false by default**, with `hybrid.artifact_dir` (default `outputs/analysis/phase3a/hybrid/vocabfix`) and optional `threshold` override |
| seam | `TEPApp._generate_report`; `TEPApp._diagnose` holds the guard and its refusal |
| guard | `render_classifier_page` raises `IncompleteEvidence`; `_diagnose` turns that into a verdict, not an exception |

The adapter is independent of the flag: with `enable_llm=False` the report is the deterministic fallback and
still carries `report["diagnosis"]`. The verdict is a **separate field**, never written into `root_cause` or the
summary, so a template sentence can never look like a classifier decision.

Report contract when the flag is on:

| field | meaning |
|---|---|
| `fault_id` | menu-legal answer: a fault id, `UNCLASSIFIED_OPEN_SET`, `NORMAL_TRANSIENT`, or `None` when refused |
| `predicted_class` | what the classifier ranked first, kept so an abstention can be explained |
| `confidence` | calibrated confidence of the winning class; `None` when refused |
| `verdict` / `reason` | `classified` / `unclassified_open_set` / `no_fault` / `unavailable`, and the reason (`on_menu_above_threshold`, `below_threshold`, `not_on_menu`, `open_set_predicted`, `no_fault_predicted`, `evidence_incomplete`, `degenerate_scores`, `diagnosis_error`) |
| `threshold` / `source` / `artifacts` | the threshold in force, `classifier+menu_rule`, and the artefact directory |

When the flag is off, no `diagnosis` key is added: the report is byte-for-byte the previous one.

**Failure containment.** `_diagnose` catches the guard's `IncompleteEvidence` and then *any* other exception —
a missing booster, a feature-shape mismatch, a scoring error — logs it with a traceback, and returns
`verdict="unavailable", reason="diagnosis_error"` with the exception in `detail`. Report generation continues in
both cases, and `_unavailable_diagnosis` reads `threshold`/`artifacts` defensively, so even a half-initialised
hybrid yields a well-formed block. A diagnosis outage must not become a monitoring outage.

**First real run (2026-11-16, `scripts/run_live_event.py`).** `ANOM-0489` (the newest event carrying detector
evidence) through `TEPApp(enable_llm=False)` with the flag on: `classified`, `FAULT_15`, calibrated confidence
1.0 at the stored threshold 1.0, rule reason `on_menu_above_threshold`, and the fallback prose intact. The
fallback's sensor heuristic called it "B composition step change or purge variation" while the classifier said
`FAULT_15` (condenser cooling water). Nothing was reconciled: the two live in different fields by design, and
the difference is exactly why the verdict is not written into `root_cause`. `ANOM-0333` (schema-v1) was refused
with `evidence_incomplete` and still produced a full report.

Tests (`tests/test_hybrid_report_path.py`, 5 passing): a real `TEPApp` with `enable_llm=False` builds with the
hybrid on; a report keeps its prose and gains a verdict; a pre-instrumentation record is refused with
`evidence_incomplete` and the report is still produced; the flag-off path adds nothing.

The wiring adds **no new metric**. Every number the documents quote comes from the same pipeline executed
offline on the stored splits, so enabling the flag changes which code path runs, not what has been measured.

### 7.8 Live audit over every detector-bearing event, 2026-11-16

`scripts/audit_live_diagnosis.py` runs the wired path over all 156 stored events that carry detector evidence.
It is read-only and writes its row-level detail to `outputs/analysis/phase3a/hybrid/live_diagnosis_audit.json`
(gitignored).

| verdict | n | share |
|---|---|---|
| `classified` | 74 | 47.4 % |
| `unclassified_open_set` | 60 | 38.5 % (43 below threshold, 17 the model's own open-set label) |
| `no_fault` (`NORMAL_TRANSIENT`) | 20 | 12.8 % |
| `unavailable` (degenerate scores) | 2 | 1.3 % |

Named faults: `FAULT_15` 26, `FAULT_04` 9, `FAULT_14` 9, `FAULT_08` 5, and 4 each of 02/05/07/10/11/12, `FAULT_01` 1.
`FAULT_06` and `FAULT_13` were never named. Calibrated confidence is median 1.0, p25 0.84, p10 0.44, with
**110 of the 154 scored events (71.4 %) at exactly 1.0** (against 99.6 % on the benchmark validation split) —
live traffic is less saturated than the benchmark but still near-binary. Two events carry all-zero calibrated
probabilities (isotonic degeneracy) and are now refused instead of being given the argmax of a zero vector.

**Disagreement with the fallback prose.** Mapping both sides to subsystem groups (the prose names a group, not a
fault id), the classifier's argmax agrees with the prose heuristic on 43 of 117 events (36.8 %); among the 74
named faults, 29 (39.2 %). Confusions: condenser→feed 33, condenser→reactor 9, feed→stripper 8, reactor→feed 5,
plus seven smaller cells. The prose is a top-three-sensor heuristic and the live events are unlabelled, so this
measures divergence, not error. It is consistent with the `FAULT_15` skew (26 of 74), the class whose surrogate
check already shows symmetric confusion with `FAULT_05`.

**Threshold.** The cross-fitted threshold of *this* generation is 1.0 (`crossfit.json`
`threshold.threshold`, 2.7 % achieved false abstention on 1,729 out-of-fold items; the validation artefact says
1.0 at 0.4 % of 497). 0.9792 belongs to the superseded `XMV_1..11` vocabulary and has nothing to retrain against
here. Operationally it would change nothing: live confidences have no mass in [0.9792, 1.0), so 0.9792 and 1.0
give identical verdicts on all 156 events. 0.95 and 0.90 each name four more events (78 vs 74 classified) with an
unchanged 36.8 % group agreement.

**Decision: keep the artefact threshold (1.0); do not import 0.9792.** The threshold is not the lever. With 70.5 %
of events pinned at the ceiling the decision has collapsed to "does the argmax name a fault", so the useful
follow-ups are a non-saturated score (recalibrate on held-out live-like data, or use a margin/entropy rule) and
the two semantic wrinkles the audit exposed: a menu that contains the abstain label lets a model open-set
prediction be reported as `classified`, and all-zero calibrated rows still get an argmax. Neither is changed
here, because both alter decision semantics the measured numbers were produced under.

### 7.9 Why `FAULT_15` dominates the live verdicts, 2026-11-16

`scripts/diagnose_fault15_skew.py` tests three explanations against the records.

**Windowing inflates every count.** The 50 `FAULT_15` argmax events fall into **8 episodes** at a 10-minute gap
(sizes 6, 13, 2, **24**, 1, 1, 1, 2); the 24-window episode runs from sample 2305 to 69845. So "50 predictions"
is roughly eight incidents.

**The trigger mix, not the fault mix, drives the label.** Every one of the **32 dynamics-only** live pages is
labelled `FAULT_15`, and all 67 other named faults come from `both/mixed` pages. That is not an extrapolation:
benchmark `FAULT_15` pages are themselves overwhelmingly dynamics-only (115 of 118 in train, 42 of 50 in val),
while dynamics-only pages elsewhere in the corpus are `NORMAL_TRANSIENT` (47) and `FAULT_10` (26). The classifier
has learned "CVA-only page ⇒ sticking fault", which is what its training data says.

**Sensor evidence is not consulted for that page type.** Only 14 of 50 `FAULT_15` predictions carry a
condenser-cooling sensor in the top five, against 28 of 50 carrying a feed sensor. The prose heuristic, which
reads sensors only, therefore names feed/reactor — hence the 33 `condenser→feed` cells in §7.8.

**Neither a broken scorer nor a novel regime, then.** The label prior is a symptom: the live detector emits 32
dynamics-only pages (21 % of scored events) against 10 % in training, and windowing multiplies them. The
mechanism is a page-type shortcut that is faithful to the training labels. Fixing it needs training pages where
dynamics-only episodes carry other labels (or the sensor block wired into the decision), not a threshold change.
This remains unverifiable against ground truth: the live events have no labels.

### 7.10 Closure checklist for the RP

| item | state |
|---|---|
| classifier and menu rule wired, flagged, guarded | done (§7.7) |
| failure containment and degenerate-score refusal | done, 10 seam tests |
| live audit over all detector-bearing events | done (§7.8) |
| verdict taxonomy consistent with the diagnosis field | done 2026-11-16; benchmark reasons restated 319/323/61/6 + 125, accuracy unchanged |
| `FAULT_15` live behaviour explained | done (§7.9) |
| surrogates and CIs | **done 2026-11-16**: cross-fitted cluster bootstrap 0.9783–0.9962 (accuracy) / 0.9711–0.9947 (macro-F1), 497 event clusters, seed 42 |
| 13-fault novelty/absorption table | **done 2026-11-16**: all 13 faults retrained into `vocabfix/surrogate_wide.json` (2 shards, ~100 s each, deterministic); `fig8` is rendered from that file |
| live accuracy | **not estimable**: 20 of 489 events carry a label and only 6 of those have detector evidence. All six are `FAULT_15` and all six are predicted `FAULT_15` at 1.0 — a sanity check, not an accuracy figure |
| older adapter predictions | **not reproducible** from stored decode sources; carried over unchanged |

The RP can close on the benchmark and system claims once the four bold rows are either run or explicitly
declared as out of scope. Nothing in the wiring or the live audit blocks it.

**Cost of the four, measured 2026-11-16.** The widened surrogate is cheap: one fault retrains and scores in
**20 s** (`--faults 1`), so all 13 are ~4.5 min serial. It still reads the published feature vocabulary, not
`vocabfix` (its shard threshold came out 0.9971 against the retrained 1.0), so the work is the override, not the
compute. The cross-fitted CIs need no new data: per-fold out-of-fold rows already exist and a clustered bootstrap
(`accuracy_ci`) already exists. The adapter preds are the expensive item: 36.5 s/item at the 512-token cap means
~5 h for the 497-item set and ~4 h for the 400-item one, conditional on a GPU and the pristine adapter weights.

## 8. What this plan does not do

It does not change the detector, the adapter, the prompts, the datasets or the measured numbers; it does not
enable the flag; it does not choose the menu policy for you. The one decision that needs you before coding is
§2.2 — shortlist menu (keeps measured abstention behaviour) or full list (always names a fault, loses open-set
protection).
