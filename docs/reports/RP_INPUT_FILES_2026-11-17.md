# Files needed to produce the RP

**Date:** 2026-11-17
**Scope:** the complete, minimal input set for assembling the journal research paper (`TEP_FP_JOURNAL`). Every path
below exists in this repository at commit `064ae7e` or later; nothing here is a placeholder.

---

## 1. Manuscript sources (the paper itself)

| file | role | notes |
|---|---|---|
| `docs/reports/TEP_FP_JOURNAL.md` | journal paper source of truth | built by `scripts/build_paper_latex.py --paper journal` |
| `docs/reports/TEP_FP_JOURNAL.tex` | generated LaTeX | do not hand-edit; regenerate from the Markdown |
| `docs/reports/TEP_FP_JOURNAL.pdf` | submission PDF | single column, 10 pt, 1-inch margins |
| `docs/reports/PAPER_MANUSCRIPT_2026-11-16.md` | Version-B manuscript | the signed framing the journal paper follows |
| `docs/reports/PAPER_CLAIMS_2026-11-16.md` | claims registry C1–C8 | what may and may not be claimed |
| `docs/reports/PAPER_METHODS_METRICS_2026-11-16.md` | metrics definitions | gates, check definitions, changelog |
| `docs/reports/references.bib` | bibliography | 11 entries, 10 cited |
| `docs/reports/paper_preamble.tex` | table/figure typography | longtable, booktabs, figure width |

## 2. Figures and their inputs

| file | figure | source data |
|---|---|---|
| `docs/reports/figures/fig1_cells.png` | numeric-defect rate by condition | `n4_cells.json` |
| `docs/reports/figures/fig2_stratified.png` | defects by gold class, both splits | `figure_data.json` |
| `docs/reports/figures/fig3_classifier.png` | diagnosis accuracy with 95 % CIs | `classifier_contrasts.json` |
| `docs/reports/figures/fig4_anchor.png` | anchor exactness against the 90 % gate | `n4_cells.json`, `test_contrast.json` |
| `docs/reports/figures/figure_data.json` | strata/cell numbers behind figures 1–2 | — |
| `docs/reports/figures/approach_data.json` | approach comparison inputs | — |

## 3. Evidence bundle (committed, hashed)

| file | claim | SHA-256 (first 16) |
|---|---|---|
| `docs/reports/evidence/index.json` | claim → file map, full hashes | — |
| `docs/reports/evidence/README.md` | human-readable version of the map | — |
| `docs/reports/evidence/files/n4_cells.json` | C1, C2, C6 | `5a9392ff5399785f` |
| `docs/reports/evidence/files/test_contrast.json` | C1, C2 | `9de4c06656d57fbf` |
| `docs/reports/evidence/files/classifier_contrasts.json` | C3 | `2c91b6f626abec16` |
| `docs/reports/evidence/files/order_invariance.json` | C1 | `40da08c93c9c01c8` |
| `docs/reports/evidence/files/preference_result.json` | C4 | `0f1f463e373b0690` |
| `docs/reports/evidence/files/preference_kappa.json` | C4 | `bdb8792bf49abcc1` |
| `docs/reports/evidence/files/selection_receipt.json` | C5 | `0bd8ed9c0f39839f` |
| `docs/reports/evidence/files/d_ceiling_decision.json` | C6 | `d57cdcfd7c92bb1b` |
| `docs/reports/evidence/files/cases46_audit.csv` | instrument calibration | `fbb0044d8ac0f939` |
| `docs/reports/evidence/files/adapter_summary.json` | C3 / adapter row | `ba6e142d335324d0` |
| `docs/reports/evidence/files/preds_val_full_300.jsonl` | adapter baseline predictions | `785c657f38a3c619` |

## 4. Governance and human evidence

| file | role |
|---|---|
| `docs/judging/confirm/paper_version_decision_2026-11-17.json` | signed Version-B framing |
| `docs/judging/confirm/d_ceiling_decision_2026-11-16.json` | signed D decision ("A with extra steps") |
| `docs/judging/confirm/selection_receipt_2026-11-16_final.json` | rung-5 selection receipt |
| `docs/judging/confirm/evidence_2026-11-06/kappa.json` | rater agreement |
| `docs/judging/confirm/evidence_2026-11-06/d30_rater_agreement.json` | agreement statistics |
| `docs/judging/confirm/evidence_2026-11-06/provenance.json` | sheet provenance |
| `docs/judging/confirm/evidence_2026-11-06/audit_ab_100_v5_scored_rater1.csv` | rater 1 sheet |
| `docs/judging/confirm/evidence_2026-11-06/audit_ab_100_v5_scored_rater2_strict_grounded.csv` | rater 2 sheet |
| `docs/judging/confirm/d29_sheets/` | third read, sheets, key |

## 5. Build and audit

| script | purpose |
|---|---|
| `scripts/build_paper_latex.py` | Markdown → `.tex` → PDF (`--paper journal`) |
| `scripts/verify_paper_numbers.py` | re-derives every manuscript table, inspects the PDF (104 checks) |
| `scripts/audit_claim_sheets.py` | recomputes claims and resolves cited paths, for the manuscript and the journal paper (91 checks) |
| `scripts/bundle_evidence.py` | refreshes the evidence bundle; `--check` verifies hashes (11 entries) |
| `tools/pandoc`, `tools/tectonic` | toolchain, gitignored; download on a fresh clone |

## 6. What is deliberately not an input

- `outputs/**` working artefacts: the bundle above replaces them for citation; the originals stay untracked.
- Superseded drafts, the internal `RP_FINAL_REPORT*`, the live-integration plan, the handover plan, the quarantine
  tree: background, not paper material.
- The free-decode partial run (`preds_adapter_val497_app_prompt_partial.jsonl`): a different instrument, must not be
  scored.

## 7. Submission-time fields that are not files

Author names, affiliations, funding, competing-interest statement and the target journal's class file are completed
at submission; the paper leaves them as placeholders rather than inventing them.
