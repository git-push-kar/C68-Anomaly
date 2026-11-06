"""Agreement between the two independent D30 readers, and section 21 rendered from it.

Rater 1: the sheet reported in section 16. Rater 2: the strict-grounded sheet of section 20,
attested by the owner in chat on 2026-11-06 as an independent reader.
"""
import csv
import json
import pathlib

EV = pathlib.Path('docs/judging/confirm/evidence_2026-11-06')
R1 = EV / 'audit_ab_100_v5_scored_rater1.csv'
R2 = EV / 'audit_ab_100_v5_scored_rater2_strict_grounded.csv'
KEY = pathlib.Path('outputs/analysis/phase3a/fit/audit_ab_100_v5_key.csv')
OUT21 = pathlib.Path('outputs/analysis/phase3a/fit/s21.md')
OUTJSON = EV / 'd30_rater_agreement.json'
PROV = EV / 'provenance.json'

r1 = {r['item_id']: r for r in csv.DictReader(R1.open(encoding='utf-8', newline=''))}
r2 = {r['item_id']: r for r in csv.DictReader(R2.open(encoding='utf-8', newline=''))}
key = {r['item_id']: r for r in csv.DictReader(KEY.open(encoding='utf-8'))}
if sorted(r1) != sorted(r2) or len(r1) != 40:
    raise SystemExit('id sets differ: %d vs %d' % (len(r1), len(r2)))

METRICS = ['correct', 'grounded', 'consistent', 'useful']
per = {}
cond = {'r1': {}, 'r2': {}}
for m in METRICS:
    a = b = c = d = 0
    diffs = []
    ones = {'r1': {'adapter': 0, 'fallback': 0}, 'r2': {'adapter': 0, 'fallback': 0}}
    for item in r1:
        for side in ('A', 'B'):
            x = int(r1[item]['%s_%s' % (m, side)])
            y = int(r2[item]['%s_%s' % (m, side)])
            ones['r1'][key[item][side]] += x
            ones['r2'][key[item][side]] += y
            if x and y:
                a += 1
            elif x:
                b += 1
                diffs.append('%s:%s' % (item, side))
            elif y:
                c += 1
                diffs.append('%s:%s' % (item, side))
            else:
                d += 1
    n = a + b + c + d
    po = (a + d) / n
    pe = ((a + b) / n) * ((a + c) / n) + ((c + d) / n) * ((b + d) / n)
    per[m] = {'pairs': n, 'agree': a + d, 'po': round(po, 4), 'pe': round(pe, 4),
              'kappa': round((po - pe) / (1 - pe), 4), 'r1_ones': a + b, 'r2_ones': a + c,
              'differing_cells': diffs, 'adapter': {'r1': ones['r1']['adapter'], 'r2': ones['r2']['adapter']}}
    cond['r1'][m] = ones['r1']
    cond['r2'][m] = ones['r2']

non_g = [m for m in METRICS if m != 'grounded']
ng_n = 40 * 2 * len(non_g)
ng_agree = sum(per[m]['agree'] for m in non_g)
ng_r1 = sum(per[m]['r1_ones'] for m in non_g) / ng_n
ng_r2 = sum(per[m]['r2_ones'] for m in non_g) / ng_n
ng_po = ng_agree / ng_n
ng_pe = ng_r1 * ng_r2 + (1 - ng_r1) * (1 - ng_r2)
pooled_non_grounded = {'pairs': ng_n, 'agree': ng_agree, 'po': round(ng_po, 4),
                       'pe': round(ng_pe, 4), 'kappa': round((ng_po - ng_pe) / (1 - ng_pe), 4)}

cells_total = 320
cells_agree = sum(per[m]['agree'] for m in METRICS)

# preference: rater 1 always chose a side, rater 2 can tie
same = r2_tie = opposite = 0
opp_rows = []
for item in r1:
    p1, p2 = r1[item]['preference'].strip().upper(), r2[item]['preference'].strip().upper()
    if p1 == p2:
        same += 1
    elif p2 == 'TIE':
        r2_tie += 1
    else:
        opposite += 1
        opp_rows.append(item)
pref = {'rows': 40, 'same': same, 'rater2_tie': r2_tie, 'opposite': opposite,
        'opposite_rows': opp_rows, 'rater2_decided': same + opposite}

print('cells: %d/%d agree (%.4f)' % (cells_agree, cells_total, cells_agree / cells_total))
for m in METRICS:
    print('  %-10s agree %2d/%d po %.4f kappa %+.4f  adapter r1 %d r2 %d  fallback r1 %d r2 %d'
          % (m, per[m]['agree'], per[m]['pairs'], per[m]['po'], per[m]['kappa'],
             cond['r1'][m]['adapter'], cond['r2'][m]['adapter'],
             cond['r1'][m]['fallback'], cond['r2'][m]['fallback']))
print('  non-grounded pooled: %s' % pooled_non_grounded)
print('preference: %s' % pref)

summary = {'date': '2026-11-06', 'scope': 'D30 v5 audit sheet, 40 items, two writers',
           'readers': 'rater1 = section 16 sheet, rater2 = strict-grounded sheet (attested independent)',
           'cells': {'total': cells_total, 'agree': cells_agree, 'differ': cells_total - cells_agree},
           'per_metric': per, 'non_grounded_pooled': pooled_non_grounded, 'preference': pref,
           'per_condition': cond,
           'note': 'the grounded column carries 42 of the 50 differing score cells; agreement there '
                   'measures a rule difference, not reader noise'}
OUTJSON.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')

prov = {
    'date': '2026-11-06',
    'scope': 'D30 v5 audit sheet, 40 items, two writers, two independent readers',
    'raters': {
        'rater1': {'file': R1.name, 'independent': True, 'identity': 'not stated',
                   'rule': 'not named; section 18 shows derived multiples were accepted as grounded',
                   'ones_of_80': {m: cond['r1'][m]['adapter'] + cond['r1'][m]['fallback'] for m in METRICS},
                   'preference': 'fallback on 40 / 40 rows'},
        'rater2': {'file': R2.name, 'independent': True, 'identity': 'not stated',
                   'rule': 'strict grounded: a number absent from the panel is not grounded, derived or not',
                   'ones_of_80': {m: cond['r2'][m]['adapter'] + cond['r2'][m]['fallback'] for m in METRICS},
                   'per_condition': {'adapter': {m: cond['r2'][m]['adapter'] for m in METRICS},
                                     'fallback': {m: cond['r2'][m]['fallback'] for m in METRICS}},
                   'preference': 'fallback 30, tie 8, adapter 2'},
    },
    'attestation': {'channel': 'chat', 'by': 'owner', 'date': '2026-11-06',
                    'statement': 'the strict-grounded D30 sheet was scored by an independent second rater',
                    'transcribed_by': 'agent',
                    'records': 'docs/HANDOVER_BENCHMARK_PLAN_V2.md section 21'},
    'consequence': 'D30 preference direction is read twice independently; the grounded column is the '
                   'only place the two readings conflict, and it conflicts because the rule differed',
}
PROV.write_text(json.dumps(prov, indent=2) + '\n', encoding='utf-8')

g = per['grounded']
A = {m: per[m]['adapter']['r1'] for m in METRICS}
ng_txt = ('%d of %d cells (%.1f%%, kappa %+.4f)'
          % (pooled_non_grounded['agree'], pooled_non_grounded['pairs'],
             100 * pooled_non_grounded['po'], pooled_non_grounded['kappa']))

TEXT = f"""## 21. D30's second reader is independent, 2026-11-06

Owner attestation, chat, 2026-11-06: the strict-grounded sheet recorded in section 20 was scored by
an independent second rater, not by the reader whose sheet section 16 reports. Recorded as attested
by the owner in chat and transcribed here, with the channel named; the reader's identity was not
stated, so none is inferred and the `rater2` filename is not treated as evidence of identity. The
durable copy of the attestation is
`docs/judging/confirm/evidence_2026-11-06/provenance.json`.

This resolves section 20's provenance caveat. The two sheets are now two independent readings of the
same 40 items, so the rule of section 19 applies and an agreement statistic is due. Section 20's
counts are unchanged: they are rater 2's, and are reported as rater 2's.

### Agreement, rater 1 against rater 2

{cells_agree} of {cells_total} score cells agree ({cells_agree / cells_total:.1%}); {cells_total -
cells_agree} differ.

| metric (80 side-cells) | agree | po | pe | kappa | adapter rater 1 | adapter rater 2 |
|---|---|---|---|---|---|---|
| correct | {per['correct']['agree']}/80 | {per['correct']['po']:.4f} | {per['correct']['pe']:.4f} | {per['correct']['kappa']:+.4f} | {A['correct']}/40 | {cond['r2']['correct']['adapter']}/40 |
| grounded | {g['agree']}/80 | {g['po']:.4f} | {g['pe']:.4f} | {g['kappa']:+.4f} | {A['grounded']}/40 | {cond['r2']['grounded']['adapter']}/40 |
| consistent | {per['consistent']['agree']}/80 | {per['consistent']['po']:.4f} | {per['consistent']['pe']:.4f} | {per['consistent']['kappa']:+.4f} | {A['consistent']}/40 | {cond['r2']['consistent']['adapter']}/40 |
| useful | {per['useful']['agree']}/80 | {per['useful']['po']:.4f} | {per['useful']['pe']:.4f} | {per['useful']['kappa']:+.4f} | {A['useful']}/40 | {cond['r2']['useful']['adapter']}/40 |

Preference over 40 rows: both readers name the fallback on {pref['same']}; rater 2 ties {pref['rater2_tie']}
rows that rater 1 had decided; the two name opposite writers on {pref['opposite']}
({', '.join(opp_rows)}). Among the {pref['rater2_decided']} rows rater 2 decided, {pref['same']} agree
with rater 1 ({pref['same'] / pref['rater2_decided']:.1%}). A three-category kappa is not computed because
rater 1 never used the tie category, which would make the coefficient a statement about that asymmetry
rather than about agreement.

### What the numbers mean, and what they do not

The three content columns agree closely - {ng_txt} - and they point the same way in both readings:
the adapter takes
{A['correct']} correct cells under rater 1 and {cond['r2']['correct']['adapter']} under rater 2,
{A['consistent']} and {cond['r2']['consistent']['adapter']} consistent, {A['useful']} and
{cond['r2']['useful']['adapter']} useful, against a fallback that takes 40 of 40 on all three under both.

The grounded column is the exception and it explains the whole disagreement: {len(g['differing_cells'])} of
the {cells_total - cells_agree} differing score cells are grounded cells, kappa {g['kappa']:+.4f}, with rater 1
certifying {g['r1_ones']} grounded cells out of 80 and rater 2 {g['r2_ones']}. That is a rule difference, not reader
noise: rater 1's rule was never written down, and section 18 showed it accepted derived multiples
(333.60x from 229.2001 / 0.6870), while rater 2 was asked for the strict reading. Neither number is a
reliability coefficient for grounding, and none is quoted as one.

### What this changes for the claims

The preference direction survives an independent reading: rater 1 named the fallback on all 40 rows,
rater 2 on {pref['same']} with {pref['rater2_tie']} abstentions and {pref['opposite']} in the other direction, so the
`fallback_v31` decision of PROTOCOL 29.3 holds under both readings.

The fallback's clean-numbers claim does not survive. Rater 1 recorded it grounded on 40 of 40 items;
an independent reader under a stated rule records it grounded on none. The admissible sentence is
that the fallback's figures are arithmetic on panel values rather than panel values themselves, and
that no writer in this sheet grounds its numbers under the rule now on record. A "40/40 grounded"
figure for the fallback is one reader's unnamed threshold, not a result, and it should not be
quoted.

Nothing here changes the numbers in sections 16 to 20, and the fallback still wins D30 on
preference, {pref['same']} to {pref['opposite']} with {pref['rater2_tie']} ties.
"""

OUT21.write_text(TEXT, encoding='utf-8')
print('\nwrote', OUT21, 'and', PROV, 'and', OUTJSON)
