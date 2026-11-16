# Manuscript audit — 2026-11-16

Scope: every number in `PAPER_MANUSCRIPT_2026-11-16.md`, the four figures in `docs/reports/figures/`, and the
eleven entries in `references.bib`. Each number was read back from the artefact the manuscript cites, not from
another report. Result: **all substantive numbers match; three presentational defects and one citation error
were found and fixed.**

## 1. Numbers against artefacts

| manuscript claim | artefact | artefact value | verdict |
|---|---|---|---|
| A fit: 0 numeric, 520/520 anchor | `n4_cells.json` summary A | 0, 520/520 | pass |
| B fit: 30 (5.77 %), grounding 33, role 57, contradiction 0, anchor 188 | summary B | 30, 0.0577, 33, 57, 0, 188 | pass |
| C fit: 0, grounding 1, role 17, anchor 520 | summary C | 0, 1, 17, 520 | pass |
| M fit: 30, grounding 32, role 43, anchor 191 | summary M | 30, 32, 43, 191 | pass |
| D fit: 0 on every check except anchor 262 | summary D | 0,0,0,0, 262 | pass |
| D substitutes 114/520 (21.9 %) | `seam_certification.D_fallback_rows` | 114 | pass |
| C leaks 0 of B's values, M leaks 0 of A's | `seam_certification` | 0, 0 | pass |
| test A: numeric 0, grounding 250 (7.89 %) | `test_contrast.json` | 0, 250, 0.0789 | pass |
| test B: numeric 40 (1.26 %), grounding 325 (10.26 %), contradiction 5 (0.16 %), role 458 (14.46 %) | same | 40, 0.0126, 325, 0.1026, 5, 0.0016, 458, 0.1446 | pass |
| anchor A 3,168 (100 %), B 1,170 (36.93 %) | `anchor_status` | 3168, 1170 | pass |
| anchor failure split 46.1 / 6.4 / 10.6 % | `z_omitted` 1459, `contradicting_value` 203, `sensor_absent` 336 | 46.05 %, 6.41 %, 10.61 % | pass |
| stratified fit: 29/47, 1/454, 0/19 | recomputed (`figure_data.json`) | 0.617, 0.0022, 0.0 | pass |
| stratified test: 29/40, 6/1814, 5/1314 | same | 0.725, 0.0033, 0.0038 | pass |
| composition fit 454/47/19, test 1814/40/1314 | `rows_test.jsonl`, `build_orders()` | match; percentages 87.3/9.0/3.7 and 57.3/1.3/41.5 | pass |
| classifier 0.9980 / 0.9940 / 0.9940 / 0.9698 / 0.8330 with CIs | `classifier_contrasts.json` | exact match, 497 items | pass |
| hybrid vs LightGBM 2 vs 2, p = 1.0 | `contrasts.hybrid_vs_gbdt` | b=2, c=2, p=1.0 | pass |
| preference 360 pairs, κ 0.5776, obs 0.7778, exp 0.4739 | `outputs/preference_2026-11-16/kappa.json` | exact match | pass |
| receipt: chosen A, rung 5, ceiling 1.00 % | `selection_receipt_..._corrected2.json` | A, 5, m = 0.01 | pass |
| changelog 19/62/619/115 → 13/50/619/111 → 0/30/0/40 | `D33_TEST_CONTRAST` §1, handover line 1513 | match | pass |
| 834/834 reproduction, 46/46 calibration, 1 of 12 batch-identical | D33 record, final report §125, §272 | match | pass |
| McNemar p = 1.9 × 10⁻⁹ and 1.8 × 10⁻¹² | stored fields are 0.0 (float underflow) | derived 2·0.5³⁰, 2·0.5⁴⁰ | **fixed** — manuscript now states the derivation |

No number required a change. One provenance gap was closed: the two p-values are stored as `0.0` in the
artefacts, so the manuscript now derives them from the discordant counts instead of quoting a figure no file
contains.

## 2. Figures

All four render and plot the intended data (verified against `figure_data.json`: cells 0 / 5.77 / 0 / 5.77 / 0,
anchor A 3,168 exact, B 1,170 with the three failure components).

| figure | defect found | fix |
|---|---|---|
| fig1 cells | the "2 % gate" label sat on the line at the panel edge | label moved to mid-axis above the line |
| fig2 stratified | the 0.2 %, 0.3 % and 0.4 % bars were labelled "0", reading as zero defects | sub-10 % values now print one decimal |
| fig3 classifier | none — bars, CIs and values correct | — |
| fig4 anchor | legend floated inside the plotting area next to the B bar | legend moved to lower centre, two columns |

## 3. Bibliography

| entry | checked against | verdict |
|---|---|---|
| `downs1993plant` | ScienceDirect record: Comput. Chem. Eng. 17(3):245–255, 1993 | pass; DOI added |
| `yin2012comparison` | ScienceDirect, OpenAlex: J. Process Control 22(9):1567–1581, 2012 | pass; DOI added |
| `zhu2025internvl3` | arXiv:2504.10479 (v3, 19 Apr 2025) | **fixed** — first author is Jinguo Zhu; entry was attributed to Zhe Chen (third author), and the key renamed from `chen2025internvl3` |
| `hu2021lora` | arXiv:2106.09685; ICLR 2022 poster | pass |
| `ke2017lightgbm` | NeurIPS proceedings, NIPS 30 | pass; page range 3149–3157 added |
| `cohen1960coefficient` | SAGE / APA: EPM 20(1):37–46, 1960 | pass; DOI added |
| `mcnemar1947note` | Springer: Psychometrika 12(2):153–157, June 1947 | pass; DOI added |
| `clopper1934use` | Oxford Academic: Biometrika 26(4):404–413, Dec 1934 | pass; DOI added |
| `guo2017calibration` | ICML 2017, PMLR 70:1321–1330 | pass (standard record); series and pages added |
| `zheng2023judging` | NeurIPS 2023 Datasets and Benchmarks; arXiv:2306.05685 | pass |
| `russell2000data` | Springer, Advances in Industrial Control series | pass |

The manuscript's in-text key for the multimodal model was updated to `zhu2025internvl3`.

## 4. What this audit does not cover

The manuscript's *prose* claims were checked for consistency with the tables, not re-reviewed for style. No
claim was found that exceeds its table. The non-claims section still holds: nothing in the draft asserts an
adapter advantage on numeric fidelity, a preference ordering, or a mechanism-quality effect.
