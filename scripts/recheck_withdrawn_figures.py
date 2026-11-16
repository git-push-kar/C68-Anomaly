"""Re-check the figures the audit withdrew, against every number stored in the artefacts.

A literal search only finds a number written the way the report wrote it. A percentage stored as a fraction
(0.9416), a ratio stored as counts (424/450) and a rate stored rounded (0.94) would all have been missed.
This script harvests every numeric token from every JSON/JSONL/CSV file under outputs/ and docs/judging/ and
then asks, per withdrawn figure, whether *any* stored number equals it in any of those forms.

Matching is by rounding to a requested precision and is reported with the file and line, so a hit can be
judged rather than trusted: a rate of 0.0009 that appears inside an unrelated per-item record is not
evidence that the report's ECE was computed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

# (label, value, precision). Percentages are given here already divided by 100.
TARGETS = [
    ("adapter accuracy 94.16%", 0.9416, 4),
    ("hybrid accuracy 99.64%", 0.9964, 4),
    ("macro F1 adapter 0.9263", 0.9263, 4),
    ("macro F1 hybrid 0.9963", 0.9963, 4),
    ("abstention recall adapter 424/450", 424 / 450, 4),
    ("abstention precision adapter 99.07%", 0.9907, 4),
    ("abstention precision hybrid 99.34%", 0.9934, 4),
    ("false abstention adapter 1.08%", 0.0108, 4),
    ("false abstention hybrid 0.81%", 0.0081, 4),
    ("missed abstention adapter 5.78%", 0.0578, 4),
    ("ECE 0.0009", 0.0009, 4),
    ("Brier 0.0017", 0.0017, 4),
    ("chi-square 41.19", 41.19, 2),
    ("AUROC margin 0.8808", 0.8808, 4),
    ("AUROC entropy 0.8828", 0.8828, 4),
    ("AUROC max_prob 0.8815", 0.8815, 4),
    ("AUROC novelty 0.8456", 0.8456, 4),
    ("mean output length 287.3 tokens", 287.3, 1),
    ("routing 545/550", 545 / 550, 4),
    ("routing 146/150", 146 / 150, 4),
    ("over-refusal before 77.78%", 0.7778, 4),
    ("over-refusal after 55.56%", 0.5556, 4),
    ("adapter latency 0.65 s/report", 0.65, 2),
    ("replay 4.8 s", 4.8, 1),
    ("replay 1.5 s", 1.5, 1),
    ("anchor exactness B 51.67%", 0.5167, 4),
    ("anchor 170/329", 170 / 329, 4),
]

SKIP_SUFFIXES = {".pkl", ".png", ".pt", ".safetensors", ".bin"}


def harvest(paths):
    """value (rounded at several precisions) -> [(path, line, token)]"""
    index = defaultdict(list)
    for path in paths:
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for token in NUMBER.findall(line):
                try:
                    value = float(token)
                except ValueError:
                    continue
                for precision in (1, 2, 4):
                    index[(round(value, precision), precision)].append((str(path), lineno, token))
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "docs/judging/confirm/withdrawn_recheck_2026-11-16.json"))
    args = ap.parse_args()
    paths = []
    for base in ("outputs", "docs/judging"):
        paths.extend(p for p in (ROOT / base).rglob("*") if p.is_file())
    print("scanning %d files" % len(paths), flush=True)
    index = harvest(paths)
    print("harvested %d distinct (value, precision) keys" % len(index), flush=True)
    report = []
    for label, value, precision in TARGETS:
        hits = index.get((round(value, precision), precision), [])
        # Occam: report a bounded sample, and only files that are not raw per-item generation dumps.
        interesting = [h for h in hits if "/raw_" not in h[0] and "preds_" not in h[0]]
        report.append({"figure": label, "value": value, "precision": precision,
                       "hits": len(hits), "hits_outside_raw": len(interesting),
                       "sample": interesting[:4]})
        print("%-38s hits=%-6d outside_raw=%-5d %s" % (label, len(hits), len(interesting),
                                                       interesting[:2]), flush=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
