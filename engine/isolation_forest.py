"""Tier-2 Multivariate Unsupervised AI (Isolation Forest) & Hybrid Ensemble.

Detects multi-variable contextual anomalies in cloud workloads using Scikit-Learn Isolation Forest.
Provides:
1. IsolationForestDetector: Scaled Isolation Forest pipeline with model persistence.
2. Baseline Training Routine: Automated warm-up training using synthetic circadian telemetry.
3. HybridThreatEnsemble: Dual-tier arbiter coordinating Tier-1 statistical filters and Tier-2 AI.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from api.schemas import (
    DetectorTier,
    FeatureAttribution,
    SeverityLevel,
    ThreatAlert,
    WindowAggregate,
)
from engine.windowing import CANONICAL_FEATURE_NAMES, TemporalWindowAggregator
from engine.zscore_filter import StatisticalFilterEngine


@dataclass
class IsolationForestResult:
    """Inference result for a single temporal window evaluated by Isolation Forest."""

    is_anomaly: bool
    anomaly_score: float  # Normalized between 0.0 (normal) and 1.0 (extreme outlier)
    raw_score: float      # Raw decision_function score (negative implies outlier)
    decision_threshold: float
    severity: SeverityLevel
    confidence: float
    top_deviations: List[FeatureAttribution]


class IsolationForestDetector:
    """Multivariate unsupervised anomaly detector utilizing Isolation Forest and feature scaling."""

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        random_state: int = 42,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize Isolation Forest pipeline.

        Args:
            n_estimators: Number of decision trees in the isolation forest (default 100).
            contamination: Expected proportion of outliers in the data (default 0.05).
            random_state: Seed for reproducible tree partitioning.
            feature_names: Canonical list of feature names (defaults to CANONICAL_FEATURE_NAMES).
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.feature_names = feature_names or list(CANONICAL_FEATURE_NAMES)

        self.model: Optional[IsolationForest] = None
        self.scaler: StandardScaler = StandardScaler()
        self.is_fitted: bool = False

        # Baseline reference statistics for feature deviation attribution
        self.baseline_means: Optional[np.ndarray] = None
        self.baseline_stds: Optional[np.ndarray] = None
        self.empirical_threshold: float = 0.0

    def fit(self, windows: List[WindowAggregate]) -> "IsolationForestDetector":
        """Fit feature scaler and train Isolation Forest ensemble on baseline window aggregates."""
        if len(windows) < 10:
            raise ValueError(f"Insufficient training samples: need at least 10 windows, got {len(windows)}")

        # Extract 2D feature matrix [N, n_features]
        rows = []
        for w in windows:
            fmap = w.to_feature_vector()
            rows.append([fmap.get(col, 0.0) for col in self.feature_names])
        X = np.array(rows, dtype=np.float64)

        # Fit standard scaler and normalize
        X_scaled = self.scaler.fit_transform(X)

        # Record baseline distributions for explainability attribution
        self.baseline_means = np.mean(X, axis=0)
        self.baseline_stds = np.std(X, axis=0)
        self.baseline_stds = np.where(self.baseline_stds < 1e-4, 1.0, self.baseline_stds)

        # Train Isolation Forest
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def evaluate_window(self, window: WindowAggregate) -> IsolationForestResult:
        """Evaluate a single temporal window and return anomaly assessment with feature deviations."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("IsolationForestDetector must be fitted before evaluating windows.")

        fmap = window.to_feature_vector()
        raw_vec = np.array([[fmap.get(col, 0.0) for col in self.feature_names]], dtype=np.float64)
        scaled_vec = self.scaler.transform(raw_vec)

        # Decision function: negative values indicate anomaly, positive indicate normal
        raw_decision = float(self.model.decision_function(scaled_vec)[0])
        pred = int(self.model.predict(scaled_vec)[0])  # -1 for anomaly, 1 for inlier
        is_anomaly = (pred == -1) or (raw_decision <= self.empirical_threshold)

        # Calculate normalized anomaly score: 0.0 to 1.0 (higher = more anomalous)
        # decision_function typically lies in [-0.5, 0.5]
        normalized_score = max(0.0, min(1.0, 0.5 - raw_decision))

        # Severity categorization based on negative depth
        if raw_decision <= -0.25:
            severity = SeverityLevel.CRITICAL
        elif raw_decision <= -0.15:
            severity = SeverityLevel.HIGH
        elif raw_decision < 0.0:
            severity = SeverityLevel.MEDIUM
        elif raw_decision < 0.08:
            severity = SeverityLevel.LOW
        else:
            severity = SeverityLevel.INFO

        confidence = min(1.0, max(0.5, normalized_score * 1.4)) if is_anomaly else 0.0

        # Calculate feature deviations against baseline expected
        deviations: List[FeatureAttribution] = []
        if self.baseline_means is not None and self.baseline_stds is not None:
            feature_z_scores = np.abs((raw_vec[0] - self.baseline_means) / self.baseline_stds)
            # Rank features by absolute deviation
            ranked_indices = np.argsort(feature_z_scores)[::-1]

            for rank, idx in enumerate(ranked_indices[:4], start=1):
                col_name = self.feature_names[idx]
                obs_val = float(raw_vec[0][idx])
                exp_val = float(self.baseline_means[idx])
                dev_z = float(feature_z_scores[idx])

                deviations.append(
                    FeatureAttribution(
                        feature_name=col_name,
                        observed_value=round(obs_val, 4),
                        baseline_expected=round(exp_val, 4),
                        shap_value=round(dev_z / 10.0, 4),  # Proxy importance until SHAP tree explainer
                        importance_rank=rank,
                    )
                )

        return IsolationForestResult(
            is_anomaly=is_anomaly,
            anomaly_score=round(normalized_score, 4),
            raw_score=round(raw_decision, 4),
            decision_threshold=0.0,
            severity=severity,
            confidence=round(confidence, 4),
            top_deviations=deviations,
        )

    def save_model(self, file_path: str) -> None:
        """Persist trained model, scaler, and baseline parameters to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "baseline_means": self.baseline_means,
            "baseline_stds": self.baseline_stds,
            "empirical_threshold": self.empirical_threshold,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        joblib.dump(payload, path)

    @classmethod
    def load_model(cls, file_path: str) -> "IsolationForestDetector":
        """Load a persisted model and scaler pipeline from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at: {file_path}")

        payload = joblib.load(path)
        detector = cls(
            n_estimators=payload.get("n_estimators", 100),
            contamination=payload.get("contamination", 0.05),
            feature_names=payload.get("feature_names"),
        )
        detector.model = payload["model"]
        detector.scaler = payload["scaler"]
        detector.baseline_means = payload["baseline_means"]
        detector.baseline_stds = payload["baseline_stds"]
        detector.empirical_threshold = payload.get("empirical_threshold", 0.0)
        detector.is_fitted = True
        return detector


    def calibrate_threshold(self, baseline_windows: List[WindowAggregate], target_fpr: float = 0.02) -> float:
        """Calibrate empirical decision threshold against clean baseline data to achieve target false positive rate."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Detector must be fitted before calibrating threshold.")

        rows = []
        for w in baseline_windows:
            fmap = w.to_feature_vector()
            rows.append([fmap.get(col, 0.0) for col in self.feature_names])
        X = np.array(rows, dtype=np.float64)
        X_scaled = self.scaler.transform(X)

        raw_scores = self.model.decision_function(X_scaled)
        # Select threshold at (1 - target_fpr) percentile of negative scores
        self.empirical_threshold = float(np.percentile(raw_scores, target_fpr * 100))
        return self.empirical_threshold


def train_baseline_model(
    output_model_path: str = "models/isolation_forest_baseline.joblib",
    n_baseline_events: int = 1500,
    window_size_seconds: float = 60.0,
    hop_seconds: float = 10.0,
    contamination: float = 0.03,
) -> Tuple[IsolationForestDetector, dict]:
    """Generate diurnal baseline telemetry and train a baseline Isolation Forest detector with validation metadata."""
    from data.synthetic_generator import SyntheticTelemetryGenerator

    generator = SyntheticTelemetryGenerator(seed=42)
    baseline_events = generator.generate_batch(count=n_baseline_events, inject_attack=None)

    aggregator = TemporalWindowAggregator(
        window_size_seconds=window_size_seconds,
        hop_seconds=hop_seconds,
    )
    baseline_windows = aggregator.aggregate_batch(baseline_events)

    detector = IsolationForestDetector(contamination=contamination)
    detector.fit(baseline_windows)
    threshold = detector.calibrate_threshold(baseline_windows, target_fpr=contamination)

    # Persist artifact
    detector.save_model(output_model_path)

    metadata = {
        "model_path": output_model_path,
        "n_baseline_events": len(baseline_events),
        "n_baseline_windows": len(baseline_windows),
        "contamination": contamination,
        "calibrated_threshold": round(threshold, 4),
        "feature_count": len(detector.feature_names),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    return detector, metadata


class HybridThreatEnsemble:
    """Coordinates Tier-1 Statistical Filter (Z-Score/EWMA) and Tier-2 AI (Isolation Forest).

    Implements dual-tier arbitration:
    - Tier-1 fast-path catches obvious volumetric and rate surges instantly.
    - Tier-2 multivariate AI catches subtle contextual anomalies that evade single-feature 3-sigma thresholds.
    """

    def __init__(
        self,
        statistical_engine: Optional[StatisticalFilterEngine] = None,
        ai_detector: Optional[IsolationForestDetector] = None,
    ) -> None:
        """Initialize hybrid threat detection ensemble."""
        self.statistical_engine = statistical_engine or StatisticalFilterEngine()
        self.ai_detector = ai_detector

    def evaluate(self, window: WindowAggregate) -> Optional[ThreatAlert]:
        """Evaluate temporal window across Tier-1 and Tier-2 engines.

        Arbitration Strategy:
        1. Run Tier-1 Statistical Filter (instant O(1) checks).
        2. Run Tier-2 Isolation Forest if trained.
        3. If BOTH flag -> Emit HYBRID_ENSEMBLE alert with elevated confidence and CRITICAL severity.
        4. If ONLY Tier-1 flags -> Emit TIER1_STATISTICAL alert.
        5. If ONLY Tier-2 flags -> Emit TIER2_UNSUPERVISED_IFOREST alert (contextual threat).
        """
        # Tier 1 Statistical evaluation
        t1_alert = self.statistical_engine.evaluate_window(window)

        # Tier 2 AI evaluation
        t2_result: Optional[IsolationForestResult] = None
        if self.ai_detector and self.ai_detector.is_fitted:
            t2_result = self.ai_detector.evaluate_window(window)

        # Case 1: Both Tier 1 and Tier 2 flagged an anomaly (Dual-Confirmation)
        if t1_alert and t2_result and t2_result.is_anomaly:
            combined_confidence = min(1.0, max(t1_alert.confidence_score, t2_result.confidence) + 0.05)
            # Merge feature attributions
            merged_attributions = t1_alert.top_contributions or t2_result.top_deviations

            return ThreatAlert(
                window_id=window.window_id,
                detector_tier=DetectorTier.HYBRID_ENSEMBLE,
                severity=SeverityLevel.CRITICAL,
                confidence_score=round(combined_confidence, 4),
                anomaly_score=round(t1_alert.anomaly_score + (t2_result.anomaly_score * 5.0), 2),
                decision_threshold=t1_alert.decision_threshold,
                top_contributions=merged_attributions,
                threat_hypothesis=f"Hybrid Confirmed: {t1_alert.threat_hypothesis}",
                explanation_narrative=(
                    f"Corroborated by Dual-Tier Detection. Statistical Z-Score deviation ({t1_alert.anomaly_score:.1f}σ) "
                    f"combined with Multivariate Isolation Forest anomaly score ({t2_result.anomaly_score:.2f})."
                ),
                recommended_mitigation=t1_alert.recommended_mitigation,
                raw_window_metrics=window,
            )

        # Case 2: Only Tier-1 Statistical Filter flagged
        if t1_alert:
            return t1_alert

        # Case 3: Only Tier-2 Multivariate Isolation Forest flagged (Contextual / Subtle Attack)
        if t2_result and t2_result.is_anomaly:
            hypothesis, narrative, mitigation = self._infer_contextual_semantics(window, t2_result)

            return ThreatAlert(
                window_id=window.window_id,
                detector_tier=DetectorTier.TIER2_UNSUPERVISED_IFOREST,
                severity=t2_result.severity,
                confidence_score=t2_result.confidence,
                anomaly_score=t2_result.anomaly_score,
                decision_threshold=t2_result.decision_threshold,
                top_contributions=t2_result.top_deviations,
                threat_hypothesis=hypothesis,
                explanation_narrative=narrative,
                recommended_mitigation=mitigation,
                raw_window_metrics=window,
            )

        # Normal window
        return None

    def _infer_contextual_semantics(
        self,
        window: WindowAggregate,
        result: IsolationForestResult,
    ) -> Tuple[str, str, str]:
        """Infer threat hypothesis for multivariate anomalies flagged by Isolation Forest."""
        top_feature = result.top_deviations[0].feature_name if result.top_deviations else ""

        if top_feature == "off_hour_ratio" or window.off_hour_ratio > 0.4:
            return (
                "Off-Hours Reconnaissance Probing",
                f"Abnormal workload access patterns detected during off-peak hours ({window.off_hour_ratio * 100:.1f}% off-hour traffic).",
                "Review CloudTrail IAM role assumption events and enforce IP CIDR restriction on administrative endpoints.",
            )
        elif top_feature in ("error_rate_4xx", "auth_failure_count") or window.auth_failure_count > 5:
            return (
                "Sub-Threshold Credential Probing",
                f"Multivariate correlation reveals elevated authorization failures ({window.auth_failure_count}) and client error ratios.",
                "Enforce MFA challenge on suspicious IAM identities and inspect authentication logs.",
            )
        elif top_feature in ("total_response_bytes", "mean_response_bytes"):
            return (
                "Contextual Data Exfiltration Anomaly",
                f"Payload response size ({window.mean_response_bytes:.0f} bytes/req) deviates significantly from multivariate baseline.",
                "Review outbound VPC Flow logs and restrict IAM S3 GetObject batch size permissions.",
            )
        elif top_feature == "bot_agent_ratio":
            return (
                "Automated Bot Infiltration",
                f"Unusual surge in automated bot client user agents ({window.bot_agent_ratio * 100:.1f}%) detected across multiple endpoints.",
                "Deploy AWS WAF Bot Control managed rule group to challenge non-browser automation.",
            )
        else:
            return (
                "Multivariate Contextual Anomaly",
                f"Isolation Forest identified multi-dimensional deviation (score: {result.anomaly_score:.2f}) across correlated metrics.",
                "Perform SOC drill-down analysis on raw window metrics and inspect microservice traces.",
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or test Isolation Forest detector.")
    parser.add_argument("--train-baseline", action="store_true", help="Train baseline model")
    parser.add_argument("--save-path", type=str, default="models/isolation_forest_baseline.joblib", help="Model save path")
    args = parser.parse_args()

    if args.train_baseline:
        print(f"Training baseline Isolation Forest model from synthetic diurnal telemetry...")
        detector, windows = train_baseline_model(output_model_path=args.save_path)
        print(f"Model successfully trained on {len(windows)} windows and saved to: {args.save_path}")
