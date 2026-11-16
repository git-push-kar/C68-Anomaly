# Paper text — preference study reported as secondary and descriptive, 2026-11-16

All three contrasts are reported. Rung 2 remains recorded as failed in every selection receipt. Numbers
are transcribed from `docs/judging/confirm/preference_2026-11-16_result.json`; do not retype them from
this file, copy them across so the figures and the record cannot drift apart.

## Methods (as it should read in the paper)

> Preference was measured on 120 held-out events in three pairings — C–A, C–B and M–B — giving 360 pairs.
> Sides were assigned by a Latin square so that no condition occupied one side consistently, and the
> rating sheets carried only the two texts, with no condition names, gold label or key. Two readers who
> had not authored the texts each rated every pair independently, blind to the other reader's choices,
> selecting the answer they would place in front of an operator (`left`, `right`, or `tie`); `tie` was
> reserved for pairs where neither answer was better. Reliability was pre-registered at Cohen's
> κ ≥ 0.60 (decision D8, signed 2026-11-05, before data collection). Observed agreement was 0.778 against
> an expected 0.474, giving **κ = 0.578 (95 % interval ≈ 0.52–0.64, n = 360)**, below the pre-registered
> floor. The preference analyses are therefore reported as **secondary and descriptive**, and no
> confirmatory preference claim is made in this paper.

## Results subsection — "Preference (secondary, descriptive)"

> Table X reports the pairwise outcomes over decided pairs, with ties counted separately. Both readers
> preferred A to C (0.831 and 0.593 of decided pairs) and both preferred M to B almost unanimously
> (0.981 and 0.982). On B against C the readers diverged: one preferred C in 0.915 of decided pairs, the
> other in 0.547. Because the reliability floor was not met, these rates are descriptive: they show where
> two readers converged and where they did not, and they are not used to select a report writer.

| pairing | rater 1 | rater 2 | ties (rater 1 / rater 2) |
|---|---|---|---|
| A vs C | A 0.831 (98/118) | A 0.593 (70/118) | 2 / 2 |
| B vs C | C 0.915 (108/118) | C 0.547 (64/117) | 2 / 3 |
| B vs M | M 0.981 (114/116) | M 0.982 (112/114) | 6 / 6 |

Denominators are decided pairs; ties are excluded and shown separately, so neither reader can inflate a
share by abstaining.

## Limitations paragraph

> The reliability criterion for the preference study was not met (κ = 0.578 against a pre-registered 0.60;
> the gap is within one standard error at n = 360). We report the preference outcomes descriptively and
> draw no preference conclusion from them. The two readers disagreed substantially on the adapter-versus-
> seam contrast (0.915 against 0.547 in favour of C), which is the contrast a preference claim would have
> had to rest on; establishing it would require a larger, separately pre-registered study.

## Five rules for using this text

1. All three contrasts appear, in this order. Dropping B vs C is the single change that would turn this
   section into selective reporting.
2. No claim verbs — "preferred", "better received", "more readable" — in the abstract, the contributions
   list or the discussion. They belong only inside the subsection above, and only with rates attached.
3. Every rate carries its denominator and its tie count, every time it is quoted.
4. Rung 2 stays failed in `selection_receipt_*.json`. These rates never justify a cell; the shipped cell
   is decided at rung 5 on cost, and the preference table is not cited beside it.
5. κ stays 0.578. PABAK, Gwet's AC1 or a 0.55 floor may appear only as clearly labelled robustness figures
   printed next to the primary κ, never in place of it.
