"""Unit and validation tests for Tier-1 Z-Score filter and EWMA volatility tracking."""

from datetime import datetime, timedelta, timezone
import random
import pytest

from api.schemas import (
    DetectorTier,
    SeverityLevel,
    ThreatAlert,
    WindowAggregate,
)
from engine.zscore_filter import (
    EWMAVolatilityTracker,
    SlidingWindowZScoreFilter,
    StatisticalFilterEngine,
)


def create_dummy_window(
    offset_min: int,
    request_count: int = 120,
    duration_sec: float = 60.0,
    auth_failures: int = 1,
    resp_bytes: int = 120000,
    err_4xx: int = 2,
    err_5xx: int = 0,
    mean_duration: float = 45.0,
) -> WindowAggregate:
    """Generate mock WindowAggregate instances for filter evaluation."""
    now = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc) + timedelta(minutes=offset_min)
    return WindowAggregate(
        window_start=now,
        window_end=now + timedelta(seconds=duration_sec),
        duration_seconds=duration_sec,
        request_count=request_count,
        request_rate_per_sec=round(request_count / duration_sec, 4),
        error_count_4xx=err_4xx,
        error_count_5xx=err_5xx,
        error_rate_4xx=round(err_4xx / max(1, request_count), 4),
        error_rate_5xx=round(err_5xx / max(1, request_count), 4),
        mean_duration_ms=mean_duration,
        p95_duration_ms=mean_duration * 1.5,
        unique_source_ips=max(1, request_count // 3),
        unique_users=max(1, request_count // 5),
        total_request_bytes=request_count * 500,
        total_response_bytes=resp_bytes,
        mean_response_bytes=round(resp_bytes / max(1, request_count), 2),
        auth_failure_count=auth_failures,
        off_hour_ratio=0.0,
        bot_agent_ratio=0.05,
    )


def test_zscore_filter_warmup_phase():
    """Verify filter suppresses alerts during warm-up phase until min samples met."""
    zfilter = SlidingWindowZScoreFilter(history_size=30, z_threshold=3.0, min_warmup_samples=10)

    # First 9 samples are in warm-up
    for i in range(9):
        res = zfilter.evaluate_metric("request_count", 100.0 + i)
        assert res.is_anomaly is False
        assert res.severity == SeverityLevel.INFO

    # Sample 10 completes warm-up
    res_tenth = zfilter.evaluate_metric("request_count", 105.0)
    assert len(zfilter._buffers["request_count"]) == 10


def test_zscore_filter_normal_baseline_stability():
    """Verify normal random fluctuations within 3 sigma produce zero false positive anomalies."""
    random.seed(42)
    zfilter = SlidingWindowZScoreFilter(history_size=50, z_threshold=3.0, min_warmup_samples=10)

    # Feed 40 baseline values ~ Normal(100, 5)
    for _ in range(40):
        val = random.normalvariate(100.0, 5.0)
        res = zfilter.evaluate_metric("request_count", val)
        assert res.is_anomaly is False


def test_zscore_filter_volumetric_spike_detection():
    """Verify a massive surge (>5 sigma) triggers an anomaly with CRITICAL severity without buffer contamination."""
    random.seed(43)
    zfilter = SlidingWindowZScoreFilter(history_size=50, z_threshold=3.0, min_warmup_samples=10)

    # Warm-up and establish baseline around 100
    for _ in range(25):
        val = random.normalvariate(100.0, 4.0)
        zfilter.evaluate_metric("request_count", val)

    initial_buf_len = len(zfilter._buffers["request_count"])

    # Inject volumetric surge (600 requests instead of 100)
    spike_result = zfilter.evaluate_metric("request_count", 650.0)

    assert spike_result.is_anomaly is True
    assert spike_result.z_score > 5.0
    assert spike_result.severity == SeverityLevel.CRITICAL
    assert spike_result.confidence >= 0.90

    # Ensure anomalous point was NOT appended to history (prevents baseline drift poisoning)
    assert 650.0 not in zfilter._buffers["request_count"]


def test_dynamic_sensitivity_multiplier():
    """Verify sensitivity multiplier scales threshold dynamically."""
    zfilter = SlidingWindowZScoreFilter(history_size=50, z_threshold=3.0, min_warmup_samples=10)

    # Default threshold = 3.0
    assert zfilter.get_effective_threshold() == 3.0

    # Increase sensitivity by 1.5x -> threshold drops to 2.0
    zfilter.sensitivity_multiplier = 1.5
    assert zfilter.get_effective_threshold() == 2.0

    # Decrease sensitivity by 0.5x -> threshold rises to 6.0
    zfilter.sensitivity_multiplier = 0.5
    assert zfilter.get_effective_threshold() == 6.0


def test_robust_mad_filter():
    """Verify Median Absolute Deviation (MAD) calculates robust dispersion resistant to skew."""
    zfilter = SlidingWindowZScoreFilter(history_size=30, z_threshold=3.0, min_warmup_samples=5, use_robust_mad=True)

    # Baseline with a tight cluster
    for _ in range(15):
        zfilter.evaluate_metric("latency", 50.0)

    res = zfilter.evaluate_metric("latency", 250.0)
    assert res.is_anomaly is True
    assert res.z_score > 3.0


def test_ewma_volatility_step_change():
    """Verify EWMAVolatilityTracker detects rapid upper control limit breaches and tracks velocity."""
    tracker = EWMAVolatilityTracker(alpha=0.25, control_limit_factor=3.0)

    # Feed initial steady state
    for _ in range(20):
        tracker.update_and_evaluate("req_rate", 10.0)

    # Sudden surge to 80.0
    res_surge = tracker.update_and_evaluate("req_rate", 80.0)

    assert res_surge.is_anomaly is True
    assert res_surge.observed_value > res_surge.upper_control_limit
    assert res_surge.velocity_spread > 0.0  # Fast EWMA moves higher than slow EWMA


def test_statistical_filter_engine_ddos_alert():
    """Verify StatisticalFilterEngine flags HTTP Flood DDoS and attributes to request_count."""
    engine = StatisticalFilterEngine(z_threshold=3.0, history_size=30)

    # Warm-up with 15 normal windows (approx 120 requests)
    for i in range(15):
        w = create_dummy_window(offset_min=i, request_count=120)
        alert = engine.evaluate_window(w)
        assert alert is None

    # Inject DDoS window (1500 requests)
    ddos_window = create_dummy_window(offset_min=16, request_count=1500, err_5xx=80)
    alert = engine.evaluate_window(ddos_window)

    assert alert is not None
    assert alert.detector_tier == DetectorTier.TIER1_STATISTICAL_ZSCORE
    assert alert.severity in (SeverityLevel.HIGH, SeverityLevel.CRITICAL)
    assert alert.threat_hypothesis == "High Volumetric HTTP Flood (DDoS)"
    assert len(alert.top_contributions) >= 1
    assert alert.top_contributions[0].feature_name == "request_count"
    assert "rate limiting" in alert.recommended_mitigation.lower()


def test_statistical_filter_engine_credential_stuffing_alert():
    """Verify StatisticalFilterEngine flags credential stuffing surge on auth_failure_count."""
    engine = StatisticalFilterEngine(z_threshold=3.0, history_size=30)

    # Warm-up with normal auth failures (0 to 2)
    for i in range(15):
        w = create_dummy_window(offset_min=i, auth_failures=1)
        engine.evaluate_window(w)

    # Inject brute force window with 300 auth failures
    stuffing_window = create_dummy_window(offset_min=16, auth_failures=300)
    alert = engine.evaluate_window(stuffing_window)

    assert alert is not None
    assert alert.threat_hypothesis == "Distributed Credential Stuffing Campaign"
    assert any(c.feature_name == "auth_failure_count" for c in alert.top_contributions)
    assert "captcha" in alert.recommended_mitigation.lower()
