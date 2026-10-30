"""Which teacher answers a menu edit invalidates, decided by provenance.

The filters' hash (``FILTER_CODE_FILES``) answers "do the *rules* still agree?".
It does not answer "was this prose written from *this* menu?", because the menu is
generated from the knowledge base rather than written in the filter code. Folding
``menu.py`` and ``fault_knowledge.py`` into the filter hash would be the blunt
fix, and it is the wrong one: a one-word KB edit would mark all ~1,900 cached
answers stale and re-run the whole teacher job for an edit that cannot touch most
of them.

So the menu gets its own provenance, recorded per answer:

``menu_revision``
    A short label for the menu *text* revision, bumped whenever a KB edit changes
    what a menu line says. ``r1`` is the text as first published; ``r2`` is
    reserved for the fault-13 direction correction.
``menu_sha256``
    Hash of the rendered menu (every fault line plus the abstention line), so a
    silent KB edit is visible even if nobody bumps the label.
``kb_sha256``
    Hash of the knowledge-base module the menu was generated from.

A record is reusable when its filter hash matches **and** its menu revision
matches, unless the record is explicitly marked as unaffected by the intervening
revision(s) -- ``menu_revision_unaffected`` with a reason. That is the honest
middle: an answer about fault 7 does not become wrong because fault 13's one-line
signature changed, but the claim must be recorded, not assumed.

The filter hash has the same problem in a milder form. ``replay_cache`` re-judges
cached raw text with the *current* filters and keeps what still passes; that is a
stronger statement than "the hash is old", but the cache line cannot say so. So a
passing replay writes ``validated_filter_sha256`` (plus ``validated_at`` and
``validation_method``) next to the record, and a record is reusable when *either*
its own hash or its validated hash is current. Without this stamp, one filter edit
re-prices every already-checked answer as a fresh model call, which is exactly the
cost the replay exists to avoid.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]

#: The menu text revision in force. Bump together with any edit that changes a
#: rendered menu line; ``scripts/check_menu_revision.py`` proves the hash matches
#: the label so a forgotten bump cannot pass silently.
MENU_REVISION = "r1"

#: Records written before ``menu_revision`` existed were produced from r1 text.
LEGACY_REVISION = "r1"

KB_PATH = "scripts/fault_knowledge.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def menu_text() -> str:
    """The published menu, one line per entry, in menu order."""
    from llm.benchmark.menu import build_menu

    return "\n".join(f"{entry['fault_id']}: {entry['signature']}"
                     for entry in build_menu())


def menu_sha256() -> str:
    return hashlib.sha256(menu_text().encode("utf-8")).hexdigest()


def kb_sha256() -> str:
    return sha256_file(ROOT / KB_PATH)


def menu_provenance() -> Dict[str, str]:
    """The three fields every teacher cache record and the manifest carry."""
    return {
        "menu_revision": MENU_REVISION,
        "menu_sha256": menu_sha256(),
        "kb_sha256": kb_sha256(),
    }


def unaffected_revisions(record: Dict) -> Sequence[str]:
    """Revision labels this record is documented to survive."""
    value = record.get("menu_revision_unaffected")
    if isinstance(value, dict):
        value = value.get("revisions")
    if isinstance(value, str):
        value = [value]
    return [str(item) for item in (value or [])]


def validation_stamp(filter_sha256: str, method: str = "replay",
                     at: Optional[str] = None) -> Dict[str, str]:
    """The three fields a passing replay writes next to a cached answer.

    ``validation_method`` is recorded so a future rule change can tell a machine
    re-check from any other claim of validity; ``replay`` means the raw text was
    re-judged by the current filter code and passed.
    """
    import time as _time

    return {
        "validated_filter_sha256": str(filter_sha256),
        "validated_at": at or _time.strftime("%Y-%m-%dT%H:%M:%S"),
        "validation_method": str(method),
    }


def reuse_decision(record: Dict, provenance: Dict) -> Tuple[bool, str]:
    """Can a cached answer be re-attached without a model call?

    Returns ``(reusable, reason)``. The reasons are stable strings because they
    are counted in the manifest and quoted in reports.

    The filter half accepts either the answer's own hash or a ``replay`` stamp
    made under the current code, because both mean "the current rules accepted
    this text". A stale hash with no stamp is ``filter_changed``; a stale hash
    with a current stamp is ``validated_revision_match`` (or
    ``validated_revision_unaffected``).
    """
    expected_filter = str(provenance.get("filter_sha256") or "")
    recorded_filter = str(record.get("filter_sha256") or "")
    validated_filter = str(record.get("validated_filter_sha256") or "")
    if recorded_filter != expected_filter and validated_filter != expected_filter:
        return False, "filter_changed"
    prefix = "" if recorded_filter == expected_filter else "validated_"
    current = str(provenance.get("menu_revision") or MENU_REVISION)
    recorded = str(record.get("menu_revision") or LEGACY_REVISION)
    if recorded == current:
        return True, f"{prefix}revision_match"
    if current in unaffected_revisions(record):
        return True, f"{prefix}revision_unaffected"
    return False, "menu_revision_changed"


def stale_reason(record: Dict, provenance: Dict) -> Optional[str]:
    """``None`` when reusable, else the reason -- for logging."""
    reusable, reason = reuse_decision(record, provenance)
    return None if reusable else reason


__all__ = [
    "KB_PATH",
    "LEGACY_REVISION",
    "MENU_REVISION",
    "kb_sha256",
    "menu_provenance",
    "menu_sha256",
    "menu_text",
    "reuse_decision",
    "sha256_file",
    "stale_reason",
    "unaffected_revisions",
    "validation_stamp",
]
