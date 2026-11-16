"""Live diagnosis for the report path: evidence page -> features -> classifier -> calibration -> menu rule.

This is the shipped hybrid, packaged so ``TEPApp`` can call it. It reads the same artefact directory the
measurements quote and does nothing the benchmark path does not do:

1. render the evidence page with :func:`llm.benchmark.live_page.render_classifier_page`, which refuses events
   that carry no detector score;
2. parse the 323 features from that page — never from the event object, so the classifier sees what it was
   trained on;
3. score 15 classes with the Protocol A booster, apply the stored isotonic calibration;
4. let :func:`llm.hybrid.menu_rule.menu_rule` decide whether the prediction may be named, using the deployment
   menu (15 faults + an explicit abstain option).

The verdict is returned as its own object. It is deliberately **not** written into the prose: the report's
``root_cause`` remains whatever the writer produced, and ``report["diagnosis"]`` is the label a consumer should
read. Conflating the two would make a template sentence look like a classifier decision.

Usage::

    diagnosis = HybridDiagnosis("outputs/analysis/phase3a/hybrid/vocabfix")
    diagnosis.diagnose(event.evidence.to_dict())
"""
from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Optional, Sequence

import numpy as np

from llm.benchmark.live_page import render_classifier_page
from llm.benchmark.menu import build_menu
from llm.hybrid.calibration import apply_calibrator
from llm.hybrid.menu_rule import menu_rule
from scripts.evidence_classifier import prompt_features

#: The retrained generation the paper quotes (see MANUSCRIPT_AUDIT §3b).
DEFAULT_ARTIFACTS = "outputs/analysis/phase3a/hybrid/vocabfix"


class DegenerateProbabilities(RuntimeError):
    """Every calibrated probability is zero, so there is no ranking to report.

    Isotonic calibration maps a whole score interval to 0.0. For an event whose scores land there, the argmax
    is whichever class happens to come first in the label order — a position, not a prediction. Scoring it
    would manufacture a diagnosis, so the caller is told instead.
    """


class HybridDiagnosis:
    """The classifier half of the hybrid, ready to be called per event."""

    def __init__(self, artifact_dir: str = DEFAULT_ARTIFACTS, threshold: Optional[float] = None,
                 menu: Optional[Sequence[Dict[str, str]]] = None) -> None:
        import lightgbm as lgb

        self.artifacts = pathlib.Path(artifact_dir)
        protocol = self.artifacts / "protocolA"
        if not (protocol / "classifier.txt").exists():
            raise FileNotFoundError(f"no Protocol A booster under {protocol}")
        self.booster = lgb.Booster(model_file=str(protocol / "classifier.txt"))
        self.calibration = json.loads((protocol / "calibration.json").read_text(encoding="utf-8"))
        self.labels: List[str] = list(self.calibration["labels"])
        stored = float(json.loads((protocol / "threshold.json").read_text(encoding="utf-8"))["threshold"])
        self.threshold = stored if threshold is None else float(threshold)
        self.menu = list(menu) if menu is not None else build_menu()
        self.menu_ids = [entry["fault_id"] for entry in self.menu]

    def page(self, evidence: Dict) -> str:
        """The rendered page the classifier reads. Raises ``IncompleteEvidence`` for pre-v2 records."""
        return render_classifier_page(evidence, menu=self.menu)

    def probabilities(self, evidence: Dict) -> Dict[str, float]:
        page = self.page(evidence)
        features = np.asarray([prompt_features(page)], dtype=np.float64)
        raw = self.booster.predict(features)[0]
        row = {label: float(value) for label, value in zip(self.labels, raw)}
        calibrated = apply_calibrator(self.calibration, [row])[0]
        if not any(float(value) > 0.0 for value in calibrated.values()):
            raise DegenerateProbabilities(
                "all calibrated probabilities are 0.0; the argmax would be a label order, not a prediction")
        return calibrated

    def diagnose(self, evidence: Dict, menu_ids: Optional[Sequence[str]] = None) -> Dict:
        """The verdict for one event.

        ``fault_id`` is the menu-legal answer: a fault id, ``UNCLASSIFIED_OPEN_SET`` or ``NORMAL_TRANSIENT``.
        ``predicted_class`` keeps what the classifier itself ranked first, so an abstention can still be
        explained without pretending the model had no opinion.
        """
        probabilities = self.probabilities(evidence)
        predicted = max(probabilities, key=probabilities.get)
        confidence = float(probabilities[predicted])
        decision = menu_rule(predicted, confidence, menu_ids if menu_ids is not None else self.menu_ids,
                             self.threshold)
        return {
            "fault_id": decision["diagnosis"],
            "predicted_class": predicted,
            "confidence": round(float(decision["confidence"] or 0.0), 6),
            "verdict": decision["verdict"],
            "reason": decision["reason"],
            "abstained": bool(decision["abstained"]),
            "threshold": self.threshold,
            "source": "classifier+menu_rule",
            "artifacts": str(self.artifacts),
        }


__all__ = ["DEFAULT_ARTIFACTS", "DegenerateProbabilities", "HybridDiagnosis"]
