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

## 7. What this plan does not do

It does not change the detector, the adapter, the prompts, the datasets or the measured numbers; it does not
enable the flag; it does not choose the menu policy for you. The one decision that needs you before coding is
§2.2 — shortlist menu (keeps measured abstention behaviour) or full list (always names a fault, loses open-set
protection).
