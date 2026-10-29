"""Typed access to everything a rendered cell is allowed to show.

The renderer never formats a raw model string into a table: it asks this object
for a **citation key** and prints what comes back. A key that does not resolve
raises :class:`UnknownKeyError` — an unknown key is a bug in the caller (or an
invented citation), never a value to guess.

Key grammar (all lowercase, dot separated)::

    event.event_id | event.max_score | event.mean_score | event.n_windows
    event.start_sample | event.end_sample | event.severity_stored
    detector.lstm.score | detector.lstm.threshold | detector.lstm.ratio | detector.lstm.alarmed
    detector.dynamic.score | .threshold | .ratio | .alarmed | .method
    detector.cva.Q | detector.cva.T2 | detector.cva.Tr
    detector.dpca.SPE | detector.dpca.T2
    sensor.<tag>.observed | .baseline | .deviation | .z | .trend | .contribution | .limit | .display_name
    timeline.<i>.sensor | .display_name | .event | .minutes | .window
    subsystem.candidate | subsystem.candidate_score
    kb.fault.<n>.name | .subsystem | .severity | .mechanism | .action | .differential
    count.top_sensors | count.timeline | count.sensors_total
    severity.stored | severity.computed | severity.computed_score | severity.kb_base | severity.rule
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from scripts.fault_knowledge import FAULT_KNOWLEDGE
from scripts.tags import canonical_tag, display_name

TOTAL_SENSORS = 52


class UnknownKeyError(KeyError):
    """Raised when a citation key cannot be resolved from the event/KB."""

    def __init__(self, key: str, hint: str = "") -> None:
        self.key = key
        self.hint = hint
        super().__init__(f"unknown citation key {key!r}{(' — ' + hint) if hint else ''}")


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class StoreAccess:
    """Resolve citation keys against one stored event plus the knowledge base."""

    def __init__(self, event: Dict, report: Optional[Dict] = None) -> None:
        self.event = event if isinstance(event, dict) else {}
        self.report = report if isinstance(report, dict) else {}
        evidence = self.event.get("evidence")
        self.evidence: Dict = evidence if isinstance(evidence, dict) else {}
        det = self.evidence.get("detector_evidence")
        self.detector: Dict = det if isinstance(det, dict) else {}
        sensors = self.evidence.get("top_anomalous_sensors")
        self.sensors: List[Dict] = [s for s in sensors if isinstance(s, dict)] if isinstance(sensors, list) else []
        timeline = self.evidence.get("temporal_sequence")
        self.timeline: List[Dict] = [t for t in timeline if isinstance(t, dict)] if isinstance(timeline, list) else []
        self._by_tag: Dict[str, Dict] = {}
        for sensor in self.sensors:
            for key in ("name", "display_name"):
                tag = canonical_tag(sensor.get(key))
                if tag and tag not in self._by_tag:
                    self._by_tag[tag] = sensor

    # ------------------------------------------------------------------
    def _sensor(self, spelling: str) -> Dict:
        tag = canonical_tag(spelling) or (spelling if spelling in self._by_tag else None)
        if tag is None or tag not in self._by_tag:
            raise UnknownKeyError(
                f"sensor.{spelling}", f"sensor not present in this event; known tags: {sorted(self._by_tag)}"
            )
        return self._by_tag[tag]

    @staticmethod
    def _kb(number: int) -> Dict:
        entry = FAULT_KNOWLEDGE.get(int(number))
        if entry is None:
            raise UnknownKeyError(f"kb.fault.{number}", "knowledge base has no entry for this fault")
        return entry

    # ------------------------------------------------------------------
    def resolve(self, key: str) -> Any:
        """Return the typed value for ``key``; raise :class:`UnknownKeyError`."""
        if not isinstance(key, str) or not key.strip():
            raise UnknownKeyError(str(key), "empty key")
        parts = key.strip().lower().split(".")
        head = parts[0]

        if head == "event":
            field = parts[1] if len(parts) > 1 else ""
            mapping = {
                "event_id": lambda: str(self.event.get("event_id", "")),
                "max_score": lambda: _as_float(self.event.get("max_anomaly_score", self.evidence.get("anomaly_score"))),
                "mean_score": lambda: _as_float(self.event.get("mean_anomaly_score")),
                "n_windows": lambda: int(self.event.get("n_windows") or 0),
                "start_sample": lambda: int(self.event.get("start_sample") or 0),
                "end_sample": lambda: int(self.event.get("end_sample") or 0),
                "severity_stored": lambda: str(self.evidence.get("severity", "")),
            }
            if field in mapping:
                value = mapping[field]()
                if value is None:
                    raise UnknownKeyError(key, "event does not carry this field")
                return value
            raise UnknownKeyError(key, "event.<field> supports event_id, max_score, mean_score, n_windows, start_sample, end_sample, severity_stored")

        if head == "detector":
            if len(parts) < 2:
                raise UnknownKeyError(key, "detector.<name>")
            name = parts[1]
            if name == "lstm":
                lstm = self.detector.get("lstm_ae") if isinstance(self.detector.get("lstm_ae"), dict) else {}
                field = parts[2] if len(parts) > 2 else "score"
                score = _as_float(lstm.get("score"))
                threshold = _as_float(lstm.get("threshold"))
                if field == "score":
                    return score if score is not None else 0.0
                if field == "threshold":
                    return threshold if threshold is not None else 0.0
                if field == "ratio":
                    return float((score or 0.0) / max(threshold or 1e-9, 1e-9))
                if field == "alarmed":
                    return bool(lstm.get("alarmed") or lstm.get("is_anomalous"))
                raise UnknownKeyError(key, "lstm supports score, threshold, ratio, alarmed")
            if name == "dynamic":
                dyn = self.detector.get("dynamic") if isinstance(self.detector.get("dynamic"), dict) else {}
                field = parts[2] if len(parts) > 2 else "score"
                score = _as_float(dyn.get("score"))
                threshold = _as_float(dyn.get("threshold"))
                if field == "score":
                    return score if score is not None else 0.0
                if field == "threshold":
                    return threshold if threshold is not None else 0.0
                if field == "ratio":
                    return float((score or 0.0) / max(threshold or 1e-9, 1e-9))
                if field == "alarmed":
                    return bool(dyn.get("alarmed") or dyn.get("is_anomalous"))
                if field == "method":
                    return str(dyn.get("method", "cva"))
                raise UnknownKeyError(key, "dynamic supports score, threshold, ratio, alarmed, method")
            if name in ("cva", "dpca"):
                block = self.detector.get("dynamic") if isinstance(self.detector.get("dynamic"), dict) else {}
                sub = block.get(name) if isinstance(block.get(name), dict) else {}
                stat = parts[2] if len(parts) > 2 else None
                if stat is None:
                    raise UnknownKeyError(key, f"{name}.<stat> e.g. {name}.Q")
                stats = sub.get("stats") if isinstance(sub.get("stats"), dict) else {}
                thresholds = sub.get("thresholds") if isinstance(sub.get("thresholds"), dict) else {}
                if stat in stats:
                    return _as_float(stats[stat])
                if stat in thresholds:
                    return _as_float(thresholds[stat])
                # fall back to the top-level dynamic score/threshold when the
                # detailed block is absent (older stored events)
                if stat.lower() == "q" and name == "cva":
                    return _as_float(block.get("score"))
                if stat.lower() == "spe" and name == "dpca":
                    return _as_float(block.get("score"))
                raise UnknownKeyError(key, f"{name} stats present: {sorted(stats) }")
            raise UnknownKeyError(key, "detector.<lstm|dynamic|cva|dpca>")

        if head == "sensor":
            if len(parts) < 3:
                raise UnknownKeyError(key, "sensor.<tag>.<field>")
            spelling = ".".join(parts[1:-1])  # tags never contain dots, but be tolerant
            field = parts[-1]
            sensor = self._sensor(spelling)
            mapping = {
                "observed": ("current_value", _as_float),
                "baseline": ("baseline_value", _as_float),
                "deviation": ("deviation_percent", _as_float),
                "z": ("z_score", _as_float),
                "trend": ("trend", str),
                "contribution": ("contribution", _as_float),
                "error": ("error", _as_float),
                "limit": ("limit_context", str),
                "display_name": ("display_name", str),
            }
            if field not in mapping:
                raise UnknownKeyError(key, f"sensor.<tag>.<field> supports {sorted(mapping)}")
            attr, caster = mapping[field]
            value = sensor.get(attr, sensor.get("zScore") if attr == "z_score" else None)
            if value is None:
                raise UnknownKeyError(key, f"sensor has no {attr}")
            return caster(value) if caster is _as_float else caster(value)

        if head == "timeline":
            if len(parts) < 3:
                raise UnknownKeyError(key, "timeline.<index>.<field>")
            index = int(parts[1])
            if not 0 <= index < len(self.timeline):
                raise UnknownKeyError(key, f"timeline has {len(self.timeline)} entries")
            item = self.timeline[index]
            field = parts[2]
            mapping = {
                "sensor": "sensor",
                "display_name": "display_name",
                "event": "event",
                "minutes": "relative_time_minutes",
                "window": "onset_window",
            }
            if field not in mapping:
                raise UnknownKeyError(key, f"timeline.<i>.<field> supports {sorted(mapping)}")
            value = item.get(mapping[field])
            if value is None:
                raise UnknownKeyError(key, f"timeline entry has no {mapping[field]}")
            return value

        if head == "subsystem":
            field = parts[1] if len(parts) > 1 else "candidate"
            if field == "candidate":
                return str(self.evidence.get("candidate_subsystem") or self.report.get("affected_subsystem") or "unknown")
            if field == "candidate_score":
                return float(self.evidence.get("candidate_subsystem_score") or 0.0)
            raise UnknownKeyError(key, "subsystem.candidate | subsystem.candidate_score")

        if head == "kb":
            if len(parts) < 4 or parts[1] != "fault":
                raise UnknownKeyError(key, "kb.fault.<n>.<field>")
            entry = self._kb(int(parts[2]))
            field = parts[3]
            mapping = {
                "name": "name",
                "subsystem": "subsystem",
                "severity": "severity",
                "mechanism": "physical_mechanism",
                "action": "recommended_action",
                "differential": "differential_diagnosis",
            }
            if field not in mapping:
                raise UnknownKeyError(key, f"kb.fault.<n>.<field> supports {sorted(mapping)}")
            value = entry.get(mapping[field])
            if value is None:
                raise UnknownKeyError(key, f"KB entry has no {mapping[field]}")
            return value

        if head == "count":
            field = parts[1] if len(parts) > 1 else ""
            if field == "top_sensors":
                return len(self.sensors)
            if field == "timeline":
                return len(self.timeline)
            if field == "sensors_total":
                return TOTAL_SENSORS
            raise UnknownKeyError(key, "count.top_sensors | count.timeline | count.sensors_total")

        if head == "severity":
            field = parts[1] if len(parts) > 1 else ""
            if field == "stored":
                return str(self.evidence.get("severity", ""))
            if field in ("computed", "computed_score", "kb_base", "rule"):
                # The response layer's own code-computed assessment, which is a
                # different rule from the one that wrote severity.stored.
                from llm.response.severity import assess

                assessment = assess(self.event, report=self.report)
                return {
                    "computed": assessment.level.value,
                    "computed_score": float(assessment.score),
                    "kb_base": assessment.kb_base.value,
                    "rule": assessment.rule_version,
                }[field]
            raise UnknownKeyError(
                key, "severity.stored | severity.computed | severity.computed_score | severity.kb_base | severity.rule"
            )

        raise UnknownKeyError(key, "unknown namespace")

    # ------------------------------------------------------------------
    def resolve_many(self, keys: Iterable[str]) -> Dict[str, Any]:
        return {key: self.resolve(key) for key in keys}

    def numeric_values(self) -> List[float]:
        """Every number the store can legitimately supply for this event."""
        values: List[float] = []
        for key in (
            "event.max_score",
            "event.mean_score",
        ):
            try:
                value = self.resolve(key)
            except UnknownKeyError:
                continue
            if isinstance(value, (int, float)):
                values.append(float(value))
        for name in ("lstm", "dynamic"):
            for field in ("score", "threshold", "ratio"):
                try:
                    values.append(float(self.resolve(f"detector.{name}.{field}")))
                except UnknownKeyError:
                    pass
        try:
            values.append(float(self.resolve("severity.computed_score")))
        except UnknownKeyError:
            pass
        for tag in self._by_tag:
            for field in ("observed", "baseline", "deviation", "z", "contribution"):
                try:
                    value = self.resolve(f"sensor.{tag}.{field}")
                except UnknownKeyError:
                    continue
                if isinstance(value, (int, float)):
                    values.append(float(value))
        return values

    def known_tags(self) -> List[str]:
        return sorted(self._by_tag)

    def display(self, tag: str) -> str:
        return str(display_name(tag) or tag)
