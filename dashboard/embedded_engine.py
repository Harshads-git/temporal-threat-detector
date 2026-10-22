"""Embedded Local Pipeline Engine for 100% Offline Standalone Operation.

Enables the Streamlit dashboard to perform full dual-tier threat detection,
sliding-window aggregation, and SHAP XAI attribution directly in-process
without requiring a separate running FastAPI server or external cloud network.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from api.schemas import ThreatAlert, WindowAggregate
from cloud.storage_sink import InMemoryAlertSink
from data.synthetic_generator import SyntheticTelemetryGenerator
from engine.explainability import (
    IncidentResponseRecommender,
    SemanticReasoningEngine,
    ShapExplainerEngine,
)
from engine.isolation_forest import HybridThreatEnsemble, IsolationForestDetector
from engine.windowing import (
    CANONICAL_FEATURE_NAMES,
    TemporalWindowAggregator,
    compute_window_features,
)
from engine.zscore_filter import StatisticalFilterEngine

logger = logging.getLogger("dashboard.embedded_engine")


class OfflineThreatEngine:
    """Standalone in-memory dual-tier detection and XAI engine for offline UI demonstration."""

    def __init__(self, seed: int = 42) -> None:
        logger.info("Initializing offline embedded detection engine...")
        self.generator = SyntheticTelemetryGenerator(seed=seed)
        self.aggregator = TemporalWindowAggregator(window_size_seconds=60.0, hop_seconds=10.0)
        self.stat_engine = StatisticalFilterEngine(z_threshold=3.0)
        self.alert_sink = InMemoryAlertSink(max_capacity=500)

        # 1. Generate diurnal baseline
        self.baseline_events = self.generator.generate_batch(count=1200, inject_attack=None)
        self.baseline_windows = self.aggregator.aggregate_batch(self.baseline_events)

        # 2. Fit Tier 2 Isolation Forest
        self.ai_detector = IsolationForestDetector(n_estimators=50, contamination=0.03, random_state=seed)
        self.ai_detector.fit(self.baseline_windows)
        self.ai_detector.calibrate_threshold(self.baseline_windows, target_fpr=0.02)

        # 3. Warm up Tier 1 statistical engine
        for w in self.baseline_windows:
            self.stat_engine.evaluate_window(w)

        # 4. Hybrid Ensemble
        self.ensemble = HybridThreatEnsemble(
            statistical_engine=self.stat_engine,
            ai_detector=self.ai_detector,
        )

        # 5. SHAP TreeExplainer
        background_matrix = self.ai_detector.scaler.inverse_transform(
            self.ai_detector.scaler.transform(
                [[w.to_feature_vector().get(c, 0.0) for c in self.ai_detector.feature_names] for w in self.baseline_windows]
            )
        )
        self.explainer = ShapExplainerEngine(
            model=self.ai_detector.model,
            scaler=self.ai_detector.scaler,
            background_data=background_matrix,
            feature_names=self.ai_detector.feature_names,
        )

        # Baseline expected feature map for UI comparison
        self.baseline_feature_means: Dict[str, float] = {
            feat: round(float(mean_val), 2)
            for feat, mean_val in zip(self.ai_detector.feature_names, self.ai_detector.baseline_means)
        }
        logger.info("Offline embedded detection engine ready.")

    def evaluate_window(self, window: WindowAggregate) -> Dict[str, Any]:
        """Evaluate a window aggregate and generate detailed output with XAI attribution."""
        alert = self.ensemble.evaluate(window)
        is_anomaly = alert is not None

        # Always generate SHAP attributions for visualization
        attributions = self.explainer.explain_window(window, top_k=6)

        if alert is None:
            # Nominal baseline condition
            raw_res = self.ai_detector.evaluate_window(window)
            return {
                "is_anomaly": False,
                "verdict": "NOMINAL WORKLOAD",
                "severity": "NORMAL",
                "confidence_score": 0.0,
                "anomaly_score": round(raw_res.anomaly_score, 2),
                "decision_threshold": round(float(raw_res.decision_threshold), 2),
                "detector_tier": "TIER1_TIER2_AGREED_NOMINAL",
                "threat_hypothesis": "Normal Operational Baseline",
                "explanation_narrative": "Observed telemetric attributes lie within nominal diurnal distribution boundaries.",
                "recommended_mitigation": "No automated mitigation required. Workload operating within standard tolerances.",
                "top_contributions": [a.model_dump() for a in attributions],
                "input_features": window.to_feature_vector(),
                "baseline_expected": self.baseline_feature_means,
            }

        # Enrich alert with SHAP attribution and semantic reasoning
        alert.top_contributions = attributions
        hyp, nar = SemanticReasoningEngine.infer_threat_semantics(attributions, window)
        alert.threat_hypothesis = hyp
        alert.explanation_narrative = nar

        playbook = IncidentResponseRecommender.get_playbook(hyp)
        if playbook:
            alert.recommended_mitigation = playbook.mitigation_summary

        self.alert_sink.save_alert(alert)

        return {
            "is_anomaly": True,
            "verdict": "THREAT DETECTED",
            "alert_id": alert.alert_id,
            "severity": alert.severity.value,
            "confidence_score": alert.confidence_score,
            "anomaly_score": alert.anomaly_score,
            "decision_threshold": alert.decision_threshold,
            "detector_tier": alert.detector_tier.value,
            "threat_hypothesis": alert.threat_hypothesis,
            "explanation_narrative": alert.explanation_narrative,
            "recommended_mitigation": alert.recommended_mitigation,
            "playbook_actions": playbook.remediation_steps if playbook else [],
            "top_contributions": [a.model_dump() for a in attributions],
            "input_features": window.to_feature_vector(),
            "baseline_expected": self.baseline_feature_means,
        }

    def inject_synthetic_attack(self, attack_type: str, count: int = 250) -> Dict[str, Any]:
        """Generate and evaluate a synthetic attack vector."""
        events = self.generator.generate_batch(count=count, inject_attack=attack_type)
        min_time = min(e.timestamp for e in events)
        max_time = max(e.timestamp for e in events)
        window = compute_window_features(events=events, window_start=min_time, window_end=max_time)
        return self.evaluate_window(window)
