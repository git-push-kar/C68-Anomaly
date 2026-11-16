"""Turn a live evidence dict into the benchmark evidence page, with the derived fields the page needs.

The classifier reads the rendered page, and two fields the benchmark renderer needs do not exist in live
evidence:

* ``detector_evidence.lag_attribution`` — the benchmark page carries a CVA residual block built from
  ``top_variables`` (variable + share), ``dominant_lag`` and ``slow_drift_ratio``. Live evidence carries the raw
  52-long ``sensor_contributions`` vector instead, so the block renders empty and five features plus two scalars
  go to zero.
* an onset list length — measured later and *not* a gap: stored benchmark pages carry four to six entries, so
  the list is passed through unchanged (an earlier five-entry cap destroyed valid pages).

This module derives the first from the second and caps the third, then renders through **the benchmark renderer
itself** (``prompt_v2.render_prompt``). Nothing here reimplements formatting: the page a live event produces is
byte-identical to the page the classifier was measured on, which is what ``tests/test_live_page_contract.py``
checks by parsing stored pages back into evidence and demanding the same text out.

Not wired into ``TEPApp`` yet, by design.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence

import numpy as np

from llm.benchmark.menu import build_menu
from llm.benchmark.prompt_v2 import render_prompt, teacher_instruction

#: The 52 canonical variables in contribution order: 41 measurements, then the 12 manipulated variables
#: tagged ``XMV_42`` .. ``XMV_52`` (see ``evidence/process_relationships.py``).
CANONICAL_ORDER: List[str] = [f"XMEAS_{i}" for i in range(1, 42)] + [f"XMV_{i}" for i in range(42, 53)]

#: The benchmark page lists five residual variables; the top five shares are what the renderer was read on.
LAG_TOP = 5

#: No onset cap. Stored pages carry four to six entries, and truncating to five reproduced 8 of 20 pages and
#: destroyed the other 12 (the sixth entry was there). ``None`` means "pass through".
ONSET_LIMIT: Optional[int] = None


def derive_lag_attribution(evidence: Dict) -> Optional[Dict]:
    """CVA residual attribution in the shape ``prompt_v2._lag_lines`` reads.

    Returns ``None`` when the evidence carries neither a precomputed ``lag_attribution`` nor a usable
    contribution vector, so a caller can tell "nothing to attribute" from "attributed nothing".
    """
    det = evidence.get("detector_evidence") or {}
    if not isinstance(det, dict):
        return None
    existing = det.get("lag_attribution")
    if isinstance(existing, dict) and existing.get("top_variables"):
        return existing

    dynamic = det.get("dynamic") or {}
    cva = (dynamic.get("cva") or {}) if isinstance(dynamic, dict) else {}
    contrib = cva.get("sensor_contributions") if isinstance(cva, dict) else None
    if not contrib:
        return None
    arr = np.asarray(contrib, dtype=np.float64).ravel()
    if arr.size != len(CANONICAL_ORDER) or not np.isfinite(arr).all() or arr.sum() <= 0:
        return None

    shares = arr / arr.sum()
    order = np.argsort(shares)[::-1][:LAG_TOP]
    return {
        "top_variables": [{"variable": CANONICAL_ORDER[int(i)], "share": float(shares[int(i)])}
                          for i in order],
        "dominant_lag": cva.get("dominant_lag"),
        "slow_drift_ratio": cva.get("slow_drift_ratio"),
    }


def _cap_sequence(sequence: object, limit: Optional[int]) -> object:
    if limit is None:
        return sequence
    if isinstance(sequence, list):
        return sequence[:limit]
    if isinstance(sequence, dict):
        capped = dict(sequence)
        seq = capped.get("sequence")
        if isinstance(seq, list):
            capped["sequence"] = seq[:limit]
        return capped
    return sequence


def normalise_evidence(evidence: Dict, onset_limit: Optional[int] = ONSET_LIMIT) -> Dict:
    """A copy of the evidence in the benchmark shape the page renderer expects.

    Three adjustments, each reversible and each tested:

    1. ``detector_evidence.lag_attribution`` is filled from the CVA contributions when absent.
    2. the onset list is left as-is unless ``onset_limit`` is given;
    3. a CVA score that lives only inside ``dynamic.cva`` is hoisted to ``dynamic`` (the renderer reads it
       there), together with its threshold and alarm flag.
    """
    out = copy.deepcopy(evidence)
    det = out.get("detector_evidence")
    if not isinstance(det, dict):
        return out

    lag = derive_lag_attribution(out)
    if lag is not None and not det.get("lag_attribution"):
        det["lag_attribution"] = lag

    dynamic = det.get("dynamic")
    if isinstance(dynamic, dict) and "score" not in dynamic:
        cva = dynamic.get("cva")
        if isinstance(cva, dict) and "score" in cva:
            dynamic["score"] = cva.get("score")
            dynamic["threshold"] = cva.get("threshold")
            if dynamic.get("alarmed") is None:
                dynamic["alarmed"] = cva.get("alarmed", cva.get("is_anomalous"))

    out["temporal_sequence"] = _cap_sequence(out.get("temporal_sequence"), onset_limit)
    return out


SCHEMA_MARKER = "Return one JSON object"


def render_live_page(evidence: Dict, menu: Optional[Sequence[Dict[str, str]]] = None,
                     event_id: Optional[str] = None, include_schema: bool = True) -> str:
    """Render one live event as the benchmark evidence page.

    ``event_id`` defaults to the evidence's own id; either way ``render_prompt`` replaces it with its opaque
    hash, so a caller cannot leak the internal id by passing it.

    The teacher instruction that names the legal answers is appended here rather than by the caller: every
    stored page carries it, so a page without it is not the page the classifier was measured on.
    """
    mapped = normalise_evidence(evidence)
    entries = list(menu) if menu is not None else build_menu()
    page = render_prompt(mapped, entries, event_id=event_id, include_schema=include_schema)
    instruction = teacher_instruction([entry["fault_id"] for entry in entries])
    # Stored pages put the instruction after the menu and the schema after that; the renderer appends the
    # schema itself, so the instruction is inserted in front of it rather than at the end.
    index = page.rfind(SCHEMA_MARKER) if include_schema else -1
    if index > 0:
        return page[:index].rstrip() + "\n\n" + instruction + "\n\n" + page[index:]
    return page.rstrip() + "\n\n" + instruction


__all__ = ["CANONICAL_ORDER", "ONSET_LIMIT", "LAG_TOP", "derive_lag_attribution",
           "normalise_evidence", "render_live_page"]
