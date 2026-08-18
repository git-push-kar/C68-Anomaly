"""SQLite-backed anomaly event store.

Each anomaly event gets a unique id (ANOM-0001, ...) and stores the raw event
metadata, the structured evidence, the automatic report, and every follow-up
question/answer, so the UI can revisit a past event and continue the
conversation.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils import ensure_dir

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    start_time      TEXT,
    detection_time  TEXT,
    end_time        TEXT,
    max_score       REAL,
    mean_score      REAL,
    severity        TEXT,
    fault_label     INTEGER,
    event_json      TEXT,
    report_json     TEXT,
    evidence_json   TEXT,
    created_at      TEXT
);
CREATE TABLE IF NOT EXISTS followups (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    TEXT NOT NULL,
    question    TEXT,
    answer      TEXT,
    created_at  TEXT,
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);
CREATE TABLE IF NOT EXISTS app_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


@dataclass
class FollowUpEntry:
    event_id: str
    question: str
    answer: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    id: Optional[int] = None

    def to_dict(self) -> Dict:
        return {"id": self.id, "event_id": self.event_id,
                "question": self.question, "answer": self.answer,
                "created_at": self.created_at}


class EventStore:
    """Thin, dependency-free persistence around SQLite."""

    def __init__(self, db_path: str) -> None:
        self.db_path = str(db_path)
        ensure_dir(Path(self.db_path).parent)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("Event store ready at %s", self.db_path)

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    def next_event_id(self) -> str:
        row = self._conn.execute(
            "SELECT value FROM app_meta WHERE key='last_event_counter'"
        ).fetchone()
        counter = int(row["value"]) if row else 0
        counter += 1
        self._conn.execute(
            "INSERT OR REPLACE INTO app_meta(key, value) VALUES ('last_event_counter', ?)",
            (str(counter),),
        )
        self._conn.commit()
        return f"ANOM-{counter:04d}"

    def store_event(self, event: Any) -> str:
        """Persist an AnomalyEvent (or anything exposing to_dict())."""
        data = event.to_dict() if hasattr(event, "to_dict") else event
        event_id = data.get("event_id") or self.next_event_id()
        evidence = data.get("evidence") or {}
        self._conn.execute(
            """
            INSERT OR REPLACE INTO events(
                event_id, start_time, detection_time, end_time, max_score,
                mean_score, severity, fault_label, event_json, report_json,
                evidence_json, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                event_id,
                data.get("start_time"),
                data.get("detection_time"),
                data.get("end_time"),
                data.get("max_anomaly_score"),
                data.get("mean_anomaly_score"),
                evidence.get("severity"),
                data.get("fault_label"),
                json.dumps(data, ensure_ascii=False),
                json.dumps(data.get("report"), ensure_ascii=False) if data.get("report") else None,
                json.dumps(evidence, ensure_ascii=False),
                datetime.now().isoformat(),
            ),
        )
        self._conn.commit()
        return event_id

    def update_report(self, event_id: str, report: Dict) -> None:
        current = self.get_event(event_id)
        if current is None:
            raise KeyError(f"Unknown event: {event_id}")
        current["report"] = report
        current["evidence"] = current.get("evidence") or {}
        self._conn.execute(
            "UPDATE events SET report_json=?, event_json=? WHERE event_id=?",
            (
                json.dumps(report, ensure_ascii=False),
                json.dumps(current, ensure_ascii=False),
                event_id,
            ),
        )
        self._conn.commit()

    def add_followup(self, event_id: str, question: str, answer: str) -> FollowUpEntry:
        entry = FollowUpEntry(event_id=event_id, question=question, answer=answer)
        cursor = self._conn.execute(
            "INSERT INTO followups(event_id, question, answer, created_at) VALUES (?,?,?,?)",
            (event_id, question, answer, entry.created_at),
        )
        self._conn.commit()
        entry.id = int(cursor.lastrowid)
        return entry

    def get_event(self, event_id: str) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT event_json FROM events WHERE event_id=?", (event_id,)
        ).fetchone()
        return json.loads(row["event_json"]) if row else None

    def list_events(self, limit: int = 100) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT event_json FROM events ORDER BY detection_time DESC LIMIT ?", (limit,)
        ).fetchall()
        return [json.loads(r["event_json"]) for r in rows]

    def get_followups(self, event_id: str) -> List[FollowUpEntry]:
        rows = self._conn.execute(
            "SELECT id, event_id, question, answer, created_at FROM followups "
            "WHERE event_id=? ORDER BY id", (event_id,),
        ).fetchall()
        return [
            FollowUpEntry(
                id=r["id"], event_id=r["event_id"], question=r["question"],
                answer=r["answer"], created_at=r["created_at"],
            )
            for r in rows
        ]

    def conversation_history(self, event_id: str) -> List[Dict]:
        """Messages in InternVL-friendly [{role, content}] format."""
        msgs: List[Dict] = []
        for f in self.get_followups(event_id):
            msgs.append({"role": "user", "content": f.question})
            msgs.append({"role": "assistant", "content": f.answer})
        return msgs