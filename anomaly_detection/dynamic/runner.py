"""Dynamic detector runtime runner for streaming and windowed inference.

Integrates CVA and DPCA dynamic detectors with the LSTM autoencoder,
providing unified scoring, EWMA smoothing, and fusion (OR / weighted).
"""
from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from anomaly_detection.dynamic.cva import CVAModel
from anomaly_detection.dynamic.dpca import DPCAModel
from anomaly_detection.dynamic.lag_builder import build_augmented_matrix, build_past_future
from anomaly_detection.dynamic.statistics import apply_cusum, apply_ewma
from anomaly_detection.dynamic.contributions import sensor_contributions

logger = logging.getLogger(__name__)


class DynamicDetectorRunner:
    """Runtime runner for CVA / DPCA dynamic anomaly detection."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        dyn_cfg = config.get("dynamic_detector", {})
        self.enabled = bool(dyn_cfg.get("enabled", False))
        self.method = dyn_cfg.get("method", "cva")
        
        # Paths
        model_dir = Path(config.get("anomaly_detector", {}).get("model_dir", "outputs/anomaly_detector"))
        dyn_dir = model_dir / "dynamic"
        
        self.cva_model: Optional[CVAModel] = None
        self.dpca_model: Optional[DPCAModel] = None
        self.thresholds: Dict[str, Any] = {}
        
        # Thresholding and smoothing configuration
        thr_cfg = dyn_cfg.get("thresholding", {})
        self.smoothing = thr_cfg.get("smoothing", "ewma")  # Default to ewma for dynamic sensitivity
        self.ewma_lambda = float(thr_cfg.get("ewma_lambda", 0.1))
        self.cusum_k = float(thr_cfg.get("cusum_k", 0.5))
        self.cusum_h = float(thr_cfg.get("cusum_h", 5.0))
        
        # Primary statistics
        self.cva_primary = dyn_cfg.get("cva", {}).get("primary_statistic", "Q")
        self.dpca_primary = dyn_cfg.get("dpca", {}).get("primary_statistic", "SPE")
        
        # Fusion configuration
        self.fusion_cfg = dyn_cfg.get("fusion", {})
        self.fusion_mode = self.fusion_cfg.get("mode", "or")
        self.fusion_weights = self.fusion_cfg.get("weights", {"lstm_ae": 0.5, "dynamic": 0.5})
        
        # Load artifacts if enabled
        if self.enabled:
            self._load_artifacts(dyn_dir)

    def _load_artifacts(self, dyn_dir: Path) -> None:
        thr_path = dyn_dir / "dynamic_thresholds.json"
        if thr_path.exists():
            try:
                with open(thr_path, "r", encoding="utf-8") as f:
                    raw_thr = json.load(f)
                self.thresholds = raw_thr.get("thresholds", raw_thr)
            except Exception as e:
                logger.warning("Failed to load dynamic thresholds from %s: %s", thr_path, e)

        cva_path = dyn_dir / "cva_model.npz"
        if self.method in ("cva", "both") and cva_path.exists():
            try:
                self.cva_model = CVAModel.load(cva_path)
                logger.info("Loaded CVA model (r=%d, n_past=%d, n_future=%d)",
                            self.cva_model.r, self.cva_model.n_past, self.cva_model.n_future)
            except Exception as e:
                logger.warning("Failed to load CVA model: %s", e)

        dpca_path = dyn_dir / "dpca_model.npz"
        if self.method in ("dpca", "both") and dpca_path.exists():
            try:
                self.dpca_model = DPCAModel.load(dpca_path)
                logger.info("Loaded DPCA model (n_comp=%d, n_lags=%d)",
                            self.dpca_model.n_components, self.dpca_model.n_lags)
            except Exception as e:
                logger.warning("Failed to load DPCA model: %s", e)

    def _smooth_series(self, x: np.ndarray) -> np.ndarray:
        if self.smoothing == "ewma":
            return apply_ewma(x, lam=self.ewma_lambda)
        elif self.smoothing == "cusum":
            return apply_cusum(x, k=self.cusum_k, h=self.cusum_h)
        return x

    def score_window(self, window_scaled: np.ndarray) -> Dict[str, Any]:
        """Score a single sliding window of scaled sensor readings [W, 52].
        
        Returns a dict with dynamic detection results.
        """
        if not self.enabled or (self.cva_model is None and self.dpca_model is None):
            return {
                "enabled": False,
                "is_anomalous": False,
                "score": 0.0,
                "threshold": 0.0,
                "warming_up": False,
                "stats": {},
            }

        W, F = window_scaled.shape
        cva_res: Dict[str, Any] = {}
        dpca_res: Dict[str, Any] = {}
        
        # 1. CVA Scoring
        if self.cva_model is not None:
            P, F_mat, t_idx = build_past_future(window_scaled, self.cva_model.n_past, self.cva_model.n_future)
            if len(P) > 0:
                raw_scores = self.cva_model.score(P)
                # Apply smoothing across the window samples
                smoothed_q = self._smooth_series(raw_scores["Q"])
                smoothed_t2 = self._smooth_series(raw_scores["T2"])
                smoothed_tr = self._smooth_series(raw_scores["Tr"])
                
                # Window score is max smoothed score in window
                q_score = float(smoothed_q[-1] if len(smoothed_q) < 5 else np.max(smoothed_q[-5:]))
                t2_score = float(smoothed_t2[-1] if len(smoothed_t2) < 5 else np.max(smoothed_t2[-5:]))
                tr_score = float(smoothed_tr[-1] if len(smoothed_tr) < 5 else np.max(smoothed_tr[-5:]))
                
                cva_thrs = self.thresholds.get("cva", {})
                q_thr = float(cva_thrs.get("Q", 239.85))
                t2_thr = float(cva_thrs.get("T2", 127.50))
                tr_thr = float(cva_thrs.get("Tr", 239.85))
                
                # Primary
                cva_score = q_score if self.cva_primary == "Q" else (t2_score if self.cva_primary == "T2" else tr_score)
                cva_thr = q_thr if self.cva_primary == "Q" else (t2_thr if self.cva_primary == "T2" else tr_thr)
                cva_anom = cva_score > cva_thr
                
                # Sensor contributions from last residual
                last_e = raw_scores["e"][-1] if "e" in raw_scores and len(raw_scores["e"]) else None
                contrib, lag_profile = (None, None)
                if last_e is not None:
                    contrib, lag_profile = sensor_contributions(last_e, self.cva_model.n_past, F)
                
                cva_res = {
                    "score": cva_score,
                    "threshold": cva_thr,
                    "is_anomalous": cva_anom,
                    "stats": {"T2": t2_score, "Q": q_score, "Tr": tr_score},
                    "thresholds": {"T2": t2_thr, "Q": q_thr, "Tr": tr_thr},
                    "sensor_contributions": contrib.tolist() if contrib is not None else None,
                    "state_order": int(self.cva_model.r),
                }
            else:
                cva_res = {"warming_up": True, "is_anomalous": False, "score": 0.0}

        # 2. DPCA Scoring
        if self.dpca_model is not None:
            Xa, t_idx = build_augmented_matrix(window_scaled, self.dpca_model.n_lags)
            if len(Xa) > 0:
                raw_dpca = self.dpca_model.score(Xa)
                smoothed_spe = self._smooth_series(raw_dpca["SPE"])
                smoothed_t2 = self._smooth_series(raw_dpca["T2"])
                
                spe_score = float(np.max(smoothed_spe[-5:]) if len(smoothed_spe) >= 5 else smoothed_spe[-1])
                t2_score = float(np.max(smoothed_t2[-5:]) if len(smoothed_t2) >= 5 else smoothed_t2[-1])
                
                dpca_thrs = self.thresholds.get("dpca", {})
                spe_thr = float(dpca_thrs.get("SPE", 38.20))
                t2_thr = float(dpca_thrs.get("T2", 124.66))
                
                dpca_score = spe_score if self.dpca_primary == "SPE" else t2_score
                dpca_thr = spe_thr if self.dpca_primary == "SPE" else t2_thr
                dpca_anom = dpca_score > dpca_thr
                
                dpca_res = {
                    "score": dpca_score,
                    "threshold": dpca_thr,
                    "is_anomalous": dpca_anom,
                    "stats": {"T2": t2_score, "SPE": spe_score},
                    "thresholds": {"T2": t2_thr, "SPE": spe_thr},
                }
            else:
                dpca_res = {"warming_up": True, "is_anomalous": False, "score": 0.0}

        # Primary dynamic detector output (CVA by default)
        primary_res = cva_res if (self.method == "cva" or self.cva_model is not None) else dpca_res
        return {
            "enabled": True,
            "method": self.method,
            "is_anomalous": bool(primary_res.get("is_anomalous", False)),
            "score": float(primary_res.get("score", 0.0)),
            "threshold": float(primary_res.get("threshold", 0.0)),
            "cva": cva_res,
            "dpca": dpca_res,
        }

    def fuse(
        self,
        lstm_score: float,
        lstm_threshold: float,
        dyn_result: Dict[str, Any],
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Fuse LSTM autoencoder score with dynamic detector score.
        
        Returns:
            (is_anomalous, combined_score, detector_evidence_dict)
        """
        lstm_anom = bool(lstm_score > lstm_threshold)
        dyn_anom = bool(dyn_result.get("is_anomalous", False))
        
        if not self.enabled or not dyn_result.get("enabled", False):
            # Dynamic not active, fall back to pure LSTM AE
            triggered_by = "lstm_ae" if lstm_anom else "none"
            change_type = "level" if lstm_anom else "none"
            evidence = {
                "triggered_by": triggered_by,
                "change_type": change_type,
                "lstm_ae": {"score": float(lstm_score), "threshold": float(lstm_threshold), "alarmed": lstm_anom},
            }
            return lstm_anom, float(lstm_score), evidence

        if self.fusion_mode == "or":
            is_anomalous = lstm_anom or dyn_anom
            # Normalized score for display
            norm_lstm = lstm_score / max(lstm_threshold, 1e-9)
            norm_dyn = dyn_result["score"] / max(dyn_result["threshold"], 1e-9)
            combined_score = float(max(norm_lstm, norm_dyn) * lstm_threshold)
        else:
            # Weighted
            w_lstm = float(self.fusion_weights.get("lstm_ae", 0.5))
            w_dyn = float(self.fusion_weights.get("dynamic", 0.5))
            norm_lstm = lstm_score / max(lstm_threshold, 1e-9)
            norm_dyn = dyn_result["score"] / max(dyn_result["threshold"], 1e-9)
            weighted_norm = w_lstm * norm_lstm + w_dyn * norm_dyn
            is_anomalous = weighted_norm > 1.0
            combined_score = float(weighted_norm * lstm_threshold)

        if lstm_anom and dyn_anom:
            triggered_by = "both"
            change_type = "mixed"
        elif dyn_anom:
            triggered_by = "dynamic"
            change_type = "dynamics"
        elif lstm_anom:
            triggered_by = "lstm_ae"
            change_type = "level"
        else:
            triggered_by = "none"
            change_type = "none"

        evidence = {
            "triggered_by": triggered_by,
            "change_type": change_type,
            "lstm_ae": {
                "score": float(lstm_score),
                "threshold": float(lstm_threshold),
                "alarmed": lstm_anom,
            },
            "dynamic": {
                "method": dyn_result.get("method"),
                "score": float(dyn_result.get("score", 0.0)),
                "threshold": float(dyn_result.get("threshold", 0.0)),
                "alarmed": dyn_anom,
                "cva": dyn_result.get("cva"),
                "dpca": dyn_result.get("dpca"),
            },
        }
        return is_anomalous, combined_score, evidence
