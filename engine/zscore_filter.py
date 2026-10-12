"""Tier-1 Real-Time Statistical Filter Engine (Z-Score & EWMA).

Provides sub-millisecond volumetric and rate-based threat filtering:
1. SlidingWindowZScoreFilter: Rolling parametric Z-score and robust MAD filter.
2. EWMAVolatilityTracker: Exponentially Weighted Moving Average for rapid step-change detection.
3. StatisticalFilterEngine: Unified Tier-1 coordinator emitting ThreatAlert records.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from api.schemas import (
    DetectorTier,
    FeatureAttribution,
    SeverityLevel,
    ThreatAlert,
    WindowAggregate,
)


@dataclass
class ZScoreAnomalyResult:
    """Result of evaluating a single metric with the Z-score filter."""

    metric_name: str
    observed_value: float
    rolling_mean: float
    rolling_std: float
    z_score: float
    threshold: float
    is_anomaly: bool
    severity: SeverityLevel
    confidence: float


@dataclass
class EWMAResult:
    """Result of evaluating a single metric with the EWMA volatility tracker."""

    metric_name: str
    observed_value: float
    smoothed_mean: float
    smoothed_std: float
    upper_control_limit: float
    lower_control_limit: float
    is_anomaly: bool
    volatility_index: float
    deviation_ratio: float
    velocity_spread: float = 0.0  # Fast EWMA minus Slow EWMA indicating acceleration


class SlidingWindowZScoreFilter:
    """Sliding-window Z-Score anomaly filter with robust calibration and outlier dampening."""

    def __init__(
        self,
        history_size: int = 50,
        z_threshold: float = 3.0,
        min_warmup_samples: int = 10,
        use_robust_mad: bool = False,
    ) -> None:
        """Initialize sliding-window Z-Score filter.

        Args:
            history_size: Number of past normal window observations in FIFO memory.
            z_threshold: Number of standard deviations to trigger anomaly flag (default 3.0).
            min_warmup_samples: Minimum baseline observations required before alerting.
            use_robust_mad: If True, uses Median Absolute Deviation (resists outlier contamination).
        """
        self.history_size = max(10, history_size)
        self.z_threshold = max(1.0, float(z_threshold))
        self.min_warmup_samples = max(3, min_warmup_samples)
        self.use_robust_mad = use_robust_mad

        # Memory buffer per monitored feature
        self._buffers: Dict[str, deque[float]] = {}
        # Dynamic sensitivity multiplier (1.0 = default)
        self.sensitivity_multiplier: float = 1.0

    def get_effective_threshold(self) -> float:
        """Calculate active threshold adjusted by sensitivity multiplier."""
        return max(1.0, self.z_threshold / self.sensitivity_multiplier)

    def update_history(self, metric_name: str, value: float, is_confirmed_anomaly: bool = False) -> None:
        """Add observation to history buffer.

        If is_confirmed_anomaly is True, skips buffer insertion to prevent poisoning the baseline.
        """
        if is_confirmed_anomaly:
            return

        if metric_name not in self._buffers:
            self._buffers[metric_name] = deque(maxlen=self.history_size)
        self._buffers[metric_name].append(float(value))

    def evaluate_metric(
        self,
        metric_name: str,
        value: float,
        auto_update_baseline: bool = True,
    ) -> ZScoreAnomalyResult:
        """Evaluate a single metric against its rolling statistical baseline.

        Args:
            metric_name: Name of metric being evaluated.
            value: Current observed numeric value.
            auto_update_baseline: If True, updates baseline if record is normal.

        Returns:
            ZScoreAnomalyResult with deviation calculus and severity.
        """
        buffer = self._buffers.get(metric_name)
        val = float(value)

        # Warm-up phase check
        if not buffer or len(buffer) < self.min_warmup_samples:
            if auto_update_baseline:
                self.update_history(metric_name, val, is_confirmed_anomaly=False)
            return ZScoreAnomalyResult(
                metric_name=metric_name,
                observed_value=val,
                rolling_mean=val,
                rolling_std=0.0,
                z_score=0.0,
                threshold=self.get_effective_threshold(),
                is_anomaly=False,
                severity=SeverityLevel.INFO,
                confidence=0.0,
            )

        arr = np.array(buffer, dtype=np.float64)

        if self.use_robust_mad:
            # Median Absolute Deviation (MAD) scaled by 1.4826 for normal consistency
            median = float(np.median(arr))
            mad = float(np.median(np.abs(arr - median)))
            dispersion = max(1e-4, 1.4826 * mad)
            center = median
            z_score = (val - center) / dispersion
        else:
            center = float(np.mean(arr))
            dispersion = max(1e-4, float(np.std(arr)))
            z_score = (val - center) / dispersion

        effective_threshold = self.get_effective_threshold()
        abs_z = abs(z_score)
        is_anomaly = abs_z >= effective_threshold

        # Compute severity tier
        if abs_z >= 5.0:
            severity = SeverityLevel.CRITICAL
        elif abs_z >= 4.0:
            severity = SeverityLevel.HIGH
        elif abs_z >= 3.0:
            severity = SeverityLevel.MEDIUM
        elif abs_z >= 2.0:
            severity = SeverityLevel.LOW
        else:
            severity = SeverityLevel.INFO

        # Confidence based on threshold exceedance
        confidence = min(1.0, max(0.5, abs_z / (effective_threshold * 1.5))) if is_anomaly else 0.0

        # Update baseline buffer only if observation is not anomalous
        if auto_update_baseline:
            self.update_history(metric_name, val, is_confirmed_anomaly=is_anomaly)

        return ZScoreAnomalyResult(
            metric_name=metric_name,
            observed_value=round(val, 4),
            rolling_mean=round(center, 4),
            rolling_std=round(dispersion, 4),
            z_score=round(z_score, 4),
            threshold=round(effective_threshold, 2),
            is_anomaly=is_anomaly,
            severity=severity,
            confidence=round(confidence, 4),
        )

    def reset(self) -> None:
        """Clear all metric histories."""
        self._buffers.clear()


class EWMAVolatilityTracker:
    """Exponentially Weighted Moving Average (EWMA) control chart for volatility tracking."""

    def __init__(
        self,
        alpha: float = 0.20,
        alpha_fast: float = 0.35,
        alpha_slow: float = 0.08,
        control_limit_factor: float = 3.0,
    ) -> None:
        """Initialize EWMA filter.

        Args:
            alpha: Primary exponential smoothing weight [0.01, 1.0].
            alpha_fast: Fast reaction coefficient for acceleration detection.
            alpha_slow: Slow anchor baseline coefficient.
            control_limit_factor: Multiplier for standard deviation upper control limit (UCL).
        """
        self.alpha = min(1.0, max(0.01, float(alpha)))
        self.alpha_fast = min(1.0, max(0.01, float(alpha_fast)))
        self.alpha_slow = min(1.0, max(0.01, float(alpha_slow)))
        self.control_limit_factor = max(1.0, float(control_limit_factor))

        # State tracking: metric_name -> (mean, variance, fast_mean, slow_mean, sample_count)
        self._state: Dict[str, Tuple[float, float, float, float, int]] = {}

    def update_and_evaluate(
        self,
        metric_name: str,
        value: float,
    ) -> EWMAResult:
        """Update EWMA state and evaluate control limit breaches.

        Formula:
            mean_t = alpha * x_t + (1 - alpha) * mean_{t-1}
            var_t = alpha * (x_t - mean_t)^2 + (1 - alpha) * var_{t-1}
            velocity = fast_mean_t - slow_mean_t
        """
        val = float(value)

        if metric_name not in self._state:
            self._state[metric_name] = (val, 1.0, val, val, 1)
            return EWMAResult(
                metric_name=metric_name,
                observed_value=val,
                smoothed_mean=val,
                smoothed_std=1.0,
                upper_control_limit=val + (self.control_limit_factor * 1.0),
                lower_control_limit=max(0.0, val - (self.control_limit_factor * 1.0)),
                is_anomaly=False,
                volatility_index=0.0,
                deviation_ratio=0.0,
                velocity_spread=0.0,
            )

        prev_mean, prev_var, prev_fast, prev_slow, count = self._state[metric_name]
        prev_std = max(1e-4, math.sqrt(max(0.0, prev_var)))

        # Evaluate Upper and Lower Control Limits (UCL / LCL) against prior baseline
        ucl = prev_mean + (self.control_limit_factor * prev_std)
        lcl = max(0.0, prev_mean - (self.control_limit_factor * prev_std))
        is_anomaly = val > ucl

        # Update primary mean and variance (dampen update if anomaly to protect baseline)
        effective_alpha = self.alpha * 0.2 if is_anomaly else self.alpha
        new_mean = (effective_alpha * val) + ((1.0 - effective_alpha) * prev_mean)
        delta_sq = (val - new_mean) ** 2
        new_var = (effective_alpha * delta_sq) + ((1.0 - effective_alpha) * prev_var)
        new_std = max(1e-4, math.sqrt(max(0.0, new_var)))

        # Update dual-horizon fast and slow estimators
        new_fast = (self.alpha_fast * val) + ((1.0 - self.alpha_fast) * prev_fast)
        new_slow = (self.alpha_slow * val) + ((1.0 - self.alpha_slow) * prev_slow)
        velocity_spread = new_fast - new_slow

        # Update state
        self._state[metric_name] = (new_mean, new_var, new_fast, new_slow, count + 1)

        volatility_index = new_std / max(1e-3, abs(new_mean))
        deviation_ratio = (val - prev_mean) / prev_std if prev_std > 0 else 0.0

        return EWMAResult(
            metric_name=metric_name,
            observed_value=round(val, 4),
            smoothed_mean=round(new_mean, 4),
            smoothed_std=round(new_std, 4),
            upper_control_limit=round(ucl, 4),
            lower_control_limit=round(lcl, 4),
            is_anomaly=is_anomaly,
            volatility_index=round(volatility_index, 4),
            deviation_ratio=round(deviation_ratio, 4),
            velocity_spread=round(velocity_spread, 4),
        )

    def reset(self) -> None:
        """Clear all EWMA states."""
        self._state.clear()


class StatisticalFilterEngine:
    """Unified Tier-1 Statistical Threat Filter coordinating Z-Score and EWMA."""

    MONITORED_METRICS: List[str] = [
        "request_count",
        "request_rate_per_sec",
        "error_rate_4xx",
        "error_rate_5xx",
        "mean_duration_ms",
        "p95_duration_ms",
        "total_response_bytes",
        "auth_failure_count",
        "bot_agent_ratio",
    ]

    def __init__(
        self,
        z_threshold: float = 3.0,
        history_size: int = 50,
        ewma_alpha: float = 0.20,
    ) -> None:
        """Initialize Tier-1 filter engine."""
        self.zscore_filter = SlidingWindowZScoreFilter(
            history_size=history_size,
            z_threshold=z_threshold,
        )
        self.ewma_tracker = EWMAVolatilityTracker(
            alpha=ewma_alpha,
            control_limit_factor=z_threshold,
        )

    def evaluate_window(
        self,
        window: WindowAggregate,
    ) -> Optional[ThreatAlert]:
        """Evaluate a temporal window across Tier-1 statistical filters.

        Returns ThreatAlert if any monitored metric exceeds statistical bounds, else None.
        """
        fmap = window.to_feature_vector()
        flagged_results: List[ZScoreAnomalyResult] = []
        attributions: List[FeatureAttribution] = []

        # 1. Evaluate Z-Score across monitored features
        for metric in self.MONITORED_METRICS:
            val = fmap.get(metric, 0.0)
            res = self.zscore_filter.evaluate_metric(metric, val)
            if res.is_anomaly:
                flagged_results.append(res)

        # 2. Update EWMA volatility tracker
        ewma_breached = False
        for metric in self.MONITORED_METRICS:
            val = fmap.get(metric, 0.0)
            ewma_res = self.ewma_tracker.update_and_evaluate(metric, val)
            if ewma_res.is_anomaly:
                ewma_breached = True

        if not flagged_results and not ewma_breached:
            return None

        # Sort flagged metrics by highest absolute Z-score
        flagged_results.sort(key=lambda r: abs(r.z_score), reverse=True)

        # Build FeatureAttribution for top contributors
        rank = 1
        for res in flagged_results[:4]:
            attributions.append(
                FeatureAttribution(
                    feature_name=res.metric_name,
                    observed_value=res.observed_value,
                    baseline_expected=res.rolling_mean,
                    shap_value=round(res.z_score / 10.0, 4),  # Scaled importance
                    importance_rank=rank,
                )
            )
            rank += 1

        primary = flagged_results[0] if flagged_results else None
        max_z = primary.z_score if primary else 3.5
        severity = primary.severity if primary else SeverityLevel.MEDIUM
        confidence = primary.confidence if primary else 0.85

        # Infer Threat Hypothesis based on primary violated metric
        hypothesis, narrative, mitigation = self._infer_threat_semantics(primary, window)
        detector_tier = (
            DetectorTier.TIER1_STATISTICAL_ZSCORE
            if flagged_results
            else DetectorTier.TIER1_STATISTICAL_EWMA
        )

        return ThreatAlert(
            window_id=window.window_id,
            detector_tier=detector_tier,
            severity=severity,
            confidence_score=confidence,
            anomaly_score=round(max_z, 2),
            decision_threshold=self.zscore_filter.get_effective_threshold(),
            top_contributions=attributions,
            threat_hypothesis=hypothesis,
            explanation_narrative=narrative,
            recommended_mitigation=mitigation,
            raw_window_metrics=window,
        )

    def _infer_threat_semantics(
        self,
        primary: Optional[ZScoreAnomalyResult],
        window: WindowAggregate,
    ) -> Tuple[str, str, str]:
        """Generate human-readable threat hypothesis and incident response mitigation."""
        if not primary:
            return (
                "Statistical Volatility Surge",
                "Temporal EWMA control limits breached due to rapid metric fluctuations.",
                "Review ingress traffic patterns and scale worker capacity if sustained.",
            )

        metric = primary.metric_name
        z = primary.z_score
        obs = primary.observed_value
        exp = primary.rolling_mean

        if metric in ("request_count", "request_rate_per_sec"):
            return (
                "High Volumetric HTTP Flood (DDoS)",
                f"Request volume surged to {obs} ({z:.1f}σ deviation over historical baseline of {exp:.1f}).",
                "Apply rate limiting on CloudFront / ALB and inspect top client IPs for bot signatures.",
            )
        elif metric == "auth_failure_count":
            return (
                "Distributed Credential Stuffing Campaign",
                f"Authentication failures spiked to {int(obs)} ({z:.1f}σ deviation, expected {exp:.1f}).",
                "Trigger CAPTCHA verification on /auth/login and temporarily lock repeatedly failing accounts.",
            )
        elif metric == "total_response_bytes":
            return (
                "Privileged Data Exfiltration Surge",
                f"Outbound byte throughput surged to {int(obs):,} bytes ({z:.1f}σ deviation).",
                "Audit S3 GetObject and bulk export API calls from originating IAM identities.",
            )
        elif metric in ("error_rate_4xx", "error_rate_5xx"):
            return (
                "API Reconnaissance or Backend Saturation",
                f"Error rate escalated to {obs * 100:.1f}% ({z:.1f}σ deviation over baseline {exp * 100:.1f}%).",
                "Inspect WAF logs for injection/probing payloads and verify upstream service health.",
            )
        elif metric in ("mean_duration_ms", "p95_duration_ms"):
            return (
                "Workload Resource Exhaustion / Denial of Service",
                f"Response latency degraded to {obs:.1f}ms ({z:.1f}σ deviation, baseline {exp:.1f}ms).",
                "Check CPU/Memory utilization on backend pods and enable autoscaling triggers.",
            )
        else:
            return (
                f"Statistical Anomaly on {metric}",
                f"{metric} reached {obs} ({z:.1f}σ deviation over baseline {exp:.1f}).",
                "Investigate correlating metrics and monitor workload behavior.",
            )
