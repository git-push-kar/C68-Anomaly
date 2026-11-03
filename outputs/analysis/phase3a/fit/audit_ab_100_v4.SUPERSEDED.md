SUPERSEDED — do not rate, cite or score this sheet.

The owner's review of `audit_ab_100_v4.csv` found it invalid (PROTOCOL 26.1):

- every row pairs two texts generated for **different** items, so `text_A` and `text_B` describe
  different events;
- the key's `adapter`/`fallback` labels do not match the text styles (both texts of a row come from
  one generator);
- seven classified-fault rows sit inside the NORMAL family;
- the NORMAL stratum holds 13 `NORMAL_TRANSIENT` rows instead of 20.

Root cause: `scripts/make_ab_sheet_v4.py:51` puts `b["text_adapter"]` on row_a next to
`a["text_adapter"]`, and line 55 puts `a["text_fallback"]` on row_b next to `b["text_fallback"]`;
`scripts/build_v4_sheet.py:45` returns `normal` for every row whose gold fault id is not
`UNCLASSIFIED_OPEN_SET`, which admits the 19 classified items.

The files are kept unchanged for the audit trail. The replacement is `audit_ab_100_v5.csv` with
`audit_ab_100_v5_key.csv`, packaged in `docs/judging/pilot_v5/`. Per-row evidence:
`ab_v4_diagnosis.csv`.
