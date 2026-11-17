# Evidence bundle — every number source behind the paper

Committed copies of the artefacts the paper cites. The originals live under `outputs/`, which is
gitignored, so these copies are what a reader can actually fetch. Each file is byte-identical to its
origin and is listed with its SHA-256 in `index.json`; `python scripts/bundle_evidence.py --check`
re-verifies them.

| claim | file | what it proves | SHA-256 |
|---|---|---|---|
| C1, C2, C6 | `files/n4_cells.json` | Fit-set cell table: numeric/grounding/tag-role/anchor per cell, seam certification, D fallback rows, McNemar | `5a9392ff5399785fff0ecd74a874696fa32cb14e5447b242ccd719570e318213` |
| C1, C2 | `files/test_contrast.json` | 3,168-row confirmatory counts, rates and anchor decomposition | `9de4c06656d57fbf4c276de2ef5d3d5eb84268cdbe75fcf1ba2e4b83dbf05195` |
| C3 | `files/classifier_contrasts.json` | Five-system diagnosis accuracy with 95 % CIs and every pairwise discordant count | `2c91b6f626abec16c9aced995808e641dc1608dd6709e2ab078485bfea5678e7` |
| C1 | `files/order_invariance.json` | 368/520 identical defect profiles across prompt orders (0.7077) | `40da08c93c9c01c8daa8ee001fc3d17da26e8aa8b6ccf990cef2860169ab17d2` |
| C4 | `files/preference_result.json` | 360 pairs, three orderings, both rater sheets, exact binomial p per pairing | `0f1f463e373b0690e8db5ae45bd5cced42f1c1d3ca9adb26bcf1e2c8e3ed7219` |
| C4 | `files/preference_kappa.json` | Cohen's kappa 0.5776, observed 0.7778, expected 0.4739, 80 disagreements listed | `bdb8792bf49abcc1d42724d963727649c98aeaa070d85c5be907d70988c849cd` |
| C5 | `files/selection_receipt.json` | Rung-by-rung selector receipt: chosen A, decided_by_rung 5 | `0bd8ed9c0f39839ffaaebf88b93b0131806802cd44536daf24a57e95b2b9ee00` |
| C6 | `files/d_ceiling_decision.json` | Signed D decision: 'A with extra steps', 27.3 % pre-correction rate retained | `d57cdcfd7c92bb1b476d9e88cc038e4bf63e029194a7ae42fc6d0f495dcbbee4` |
| instrument | `files/cases46_audit.csv` | 46 hand-written contradictions; the checker agrees on all 46 | `fbb0044d8ac0f9399eb86c5fa10c4871c1194d7f50c5273877761e85ae4bf9fe` |
| C3, adapter baseline | `files/adapter_summary.json` | Adapter at checkpoint 300 on the 497 shared items: 414/497, abstention precision/recall | `ba6e142d335324d0d8aacfe40aa8a5421b3f161b72279e1d747e34867d7aa56f` |
| adapter baseline | `files/preds_val_full_300.jsonl` | The checkpoint-300 predictions the summary is computed from (selected by macro-F1) | `785c657f38a3c61926b0e8fa4e8938b09644024fa149955647f565bda025c035` |

## How the claims map

- **C1** numeric fidelity — `n4_cells.json` (fit set), `test_contrast.json` (test), `order_invariance.json` (kb vs shuffled orders).
- **C2** anchor exactness — `n4_cells.json` summary `anchor_exact`, `test_contrast.json` `anchor_status`.
- **C3** classifier parity — `classifier_contrasts.json`; adapter row from `adapter_summary.json`, whose predictions are `preds_val_full_300.jsonl`.
- **C4** preference — `preference_result.json`, `preference_kappa.json`.
- **C5** selection — `selection_receipt.json`.
- **C6** D wrapper — `n4_cells.json` seam certification, `d_ceiling_decision.json`.
- Instrument — `cases46_audit.csv` (46/46).

Not in this bundle because they are already committed elsewhere: `docs/reports/figures/figure_data.json`
(strata table and figure inputs), `docs/reports/figures/*.png`, `docs/reports/references.bib`.
