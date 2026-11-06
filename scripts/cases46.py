"""The 28.7 calibration extended from 40 to 46 cases (D6, approved 2026-11-05).

The six added cases are exactly the families that produced the stored false positives:
mixed-direction sentences, a stored value quoted as a threshold, a hypothetical tag, and the
XMV rule. Each carries the label a human would give it, so precision and recall are measured
on the family that broke, not only on the families that already worked.
"""
from __future__ import annotations

from scripts.recheck_cases import CASES as BASE, FAULT_STORE

MIXED = {
    "detectors": [
        {"name": "LSTM autoencoder reconstruction error", "value": "0.6971",
         "threshold": "0.6870", "ratio": "1.01"},
        {"name": "CVA Q-statistic", "value": "10960.5", "threshold": "194.6", "ratio": "56.31"},
    ],
    "sensors": [],
}

ADDED = [
    # ---- v5-05 family: one sentence, two detectors, opposite directions (must not flag)
    {"check": "numeric", "expect": False, "store": MIXED,
     "text": "The LSTM reconstruction error is 0.6971, above its threshold of 0.6870, "
             "while the CVA Q-statistic is 10960.5, above its threshold of 194.6.",
     "note": "two detectors in one sentence, both true: attributing one ratio to the other is "
             "the stored false positive"},
    # ---- v5-20 family: a stored value quoted as a threshold (must flag)
    {"check": "numeric", "expect": True, "store": MIXED,
     "text": "The CVA Q-statistic is 10960.5, above the threshold 10960.5.",
     "note": "the value is printed as the threshold"},
    # ---- clause binding: two clauses, second one true (must not flag)
    {"check": "numeric", "expect": False, "store": MIXED,
     "text": "The LSTM error is 0.6971 above 0.6870, and the CVA Q-statistic is 10960.5 "
             "against a threshold of 194.6.",
     "note": "comma-and clause boundary; a whole-sentence reader blames both detectors"},
    # ---- v5-08 family: hypothetical tag (must not flag)
    {"check": "contradiction", "expect": False, "gold_fault_id": "FAULT_01",
     "gold_verdict": "classified",
     "text": "If this were FAULT_08, the reactor cooling water flow would rise first.",
     "note": "hypothetical, not an assertion"},
    # ---- v5-37 family: the XMV rule (must not flag)
    {"check": "tag_role", "expect": False,
     "text": "The reactor cooling water flow valve (XMV_51) opens further.",
     "note": "XMV_51 is the reactor cooling water flow (the table numbers XMV tags 42-52, "
             "so XMV_10 is not a tag at all); valve and flow classes agree"},
    # ---- a real class mismatch on the same rule (must flag)
    {"check": "tag_role", "expect": True,
     "text": "The reactor pressure (XMV_51) rises.",
     "note": "XMV_51 meters cooling water flow, not reactor pressure"},
]

CASES = list(BASE) + ADDED
