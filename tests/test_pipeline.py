"""Unit and integration tests for Temporal Threat Detector Phase 1."""

from datetime import datetime, timezone
import pytest

from api.schemas import (
    AttackType,
    CloudTelemetryLog,
    HttpMethod,
    UserAgentCategory,
    WindowAggregate,
    ThreatAlert,
    DetectorTier,
    SeverityLevel,
    FeatureAttribution,
)
from data.synthetic_generator import (
    SyntheticTelemetryGenerator,
    LEGITIMATE_ENDPOINTS,
    SENSITIVE_ADMIN_ENDPOINTS,
)


def test_cloud_telemetry_schema_validation():
    """Verify CloudTelemetryLog Pydantic schema constraints and default values."""
    log = CloudTelemetryLog(
        source_ip="192.168.1.100",
        user_identity="arn:aws:iam::123456789012:user/tester",
        endpoint="/api/v1/workloads/status",
        status_code=200,
        request_bytes=350,
        response_bytes=1500,
        duration_ms=42.5,
    )

    assert log.event_id is not None
    assert log.http_method == HttpMethod.GET
    assert log.user_agent_category == UserAgentCategory.BROWSER
    assert log.is_synthetic_attack is False
    assert log.status_code == 200


def test_window_aggregate_feature_vector_conversion():
    """Verify WindowAggregate calculates feature vector dictionary correctly."""
    now = datetime.now(timezone.utc)
    window = WindowAggregate(
        window_start=now,
        window_end=now,
        duration_seconds=60.0,
        request_count=120,
        request_rate_per_sec=2.0,
        error_count_4xx=5,
        error_count_5xx=1,
        error_rate_4xx=0.0416,
        error_rate_5xx=0.0083,
        mean_duration_ms=45.2,
        p95_duration_ms=88.0,
        unique_source_ips=25,
        unique_users=15,
        total_request_bytes=60000,
        total_response_bytes=180000,
        mean_response_bytes=1500.0,
        auth_failure_count=2,
        off_hour_ratio=0.0,
        bot_agent_ratio=0.05,
    )

    vector = window.to_feature_vector()
    assert isinstance(vector, dict)
    assert vector["request_count"] == 120.0
    assert vector["unique_source_ips"] == 25.0
    assert vector["mean_duration_ms"] == 45.2


def test_threat_alert_schema():
    """Verify ThreatAlert schema and serialization with SHAP attribution elements."""
    now = datetime.now(timezone.utc)
    dummy_window = WindowAggregate(
        window_start=now,
        window_end=now,
        duration_seconds=60.0,
        request_count=1000,
        request_rate_per_sec=16.6,
        error_count_4xx=10,
        error_count_5xx=200,
        error_rate_4xx=0.01,
        error_rate_5xx=0.20,
        mean_duration_ms=850.0,
        p95_duration_ms=1800.0,
        unique_source_ips=650,
        unique_users=1,
        total_request_bytes=450000,
        total_response_bytes=120000,
        mean_response_bytes=120.0,
        auth_failure_count=0,
        off_hour_ratio=0.1,
        bot_agent_ratio=0.88,
    )

    attribution = FeatureAttribution(
        feature_name="request_rate_per_sec",
        observed_value=16.6,
        baseline_expected=2.1,
        shap_value=0.42,
        importance_rank=1,
    )

    alert = ThreatAlert(
        window_id="w-001",
        detector_tier=DetectorTier.HYBRID_ENSEMBLE,
        severity=SeverityLevel.CRITICAL,
        confidence_score=0.96,
        anomaly_score=4.85,
        decision_threshold=3.0,
        top_contributions=[attribution],
        threat_hypothesis="High Volumetric HTTP Flood (DDoS)",
        explanation_narrative="Request rate surged 8x over historical baseline with elevated 5xx error spikes.",
        recommended_mitigation="Apply AWS WAF rate-limiting rule on ingress ALB.",
        raw_window_metrics=dummy_window,
    )

    assert alert.severity == SeverityLevel.CRITICAL
    assert len(alert.top_contributions) == 1
    assert alert.top_contributions[0].feature_name == "request_rate_per_sec"


def test_synthetic_baseline_generation():
    """Verify synthetic baseline generation adheres to legitimate endpoints and ranges."""
    gen = SyntheticTelemetryGenerator(seed=101)
    now = datetime.now(timezone.utc)
    event = gen.generate_baseline_event(now)

    assert isinstance(event, CloudTelemetryLog)
    assert event.is_synthetic_attack is False
    assert (event.endpoint, event.http_method) in LEGITIMATE_ENDPOINTS
    assert event.duration_ms > 0
    assert event.status_code in [200, 201, 400, 500]


def test_synthetic_ddos_attack_injection():
    """Verify DDoS injection sets automated bot UA, high latency, and optional 503s."""
    gen = SyntheticTelemetryGenerator(seed=102)
    now = datetime.now(timezone.utc)
    event = gen.generate_attack_event(AttackType.DDOS_HTTP_FLOOD, now, intensity=4.0)

    assert event.is_synthetic_attack is True
    assert event.attack_label == AttackType.DDOS_HTTP_FLOOD
    assert event.user_agent_category == UserAgentCategory.AUTOMATED_BOT
    assert event.duration_ms >= 100.0  # Latency under load


def test_synthetic_credential_stuffing_injection():
    """Verify credential stuffing injection produces 401/403 errors on auth endpoints."""
    gen = SyntheticTelemetryGenerator(seed=103)
    now = datetime.now(timezone.utc)
    event = gen.generate_attack_event(AttackType.CREDENTIAL_STUFFING, now)

    assert event.is_synthetic_attack is True
    assert event.attack_label == AttackType.CREDENTIAL_STUFFING
    assert event.endpoint == "/api/v1/auth/login"
    assert event.status_code in [401, 403]
    assert event.http_method == HttpMethod.POST


def test_synthetic_off_hour_probing_injection():
    """Verify off-hour reconnaissance hits sensitive admin/debug endpoints."""
    gen = SyntheticTelemetryGenerator(seed=104)
    now = datetime.now(timezone.utc)
    event = gen.generate_attack_event(AttackType.OFF_HOUR_PROBING, now)

    assert event.is_synthetic_attack is True
    assert event.attack_label == AttackType.OFF_HOUR_PROBING
    assert event.endpoint in SENSITIVE_ADMIN_ENDPOINTS
    assert event.status_code in [401, 403, 404]


def test_synthetic_data_exfiltration_injection():
    """Verify data exfiltration produces anomalous byte response volumes."""
    gen = SyntheticTelemetryGenerator(seed=105)
    now = datetime.now(timezone.utc)
    event = gen.generate_attack_event(AttackType.DATA_EXFILTRATION, now, intensity=2.0)

    assert event.is_synthetic_attack is True
    assert event.attack_label == AttackType.DATA_EXFILTRATION
    assert event.response_bytes > 5 * 1024 * 1024  # Greater than 5 MB


def test_batch_generator_and_dataframe_conversion():
    """Verify batch generator output integrity and dataframe conversion."""
    gen = SyntheticTelemetryGenerator(seed=106)
    batch = gen.generate_batch(count=40, inject_attack=AttackType.DDOS_HTTP_FLOOD, attack_ratio=0.25)

    assert len(batch) == 40
    # Injected count: 10 attack events, 30 baseline
    attack_events = [e for e in batch if e.is_synthetic_attack]
    assert len(attack_events) == 10

    # Ensure chronological order
    timestamps = [e.timestamp for e in batch]
    assert timestamps == sorted(timestamps)

    df = gen.to_dataframe(batch)
    assert len(df) == 40
    assert "source_ip" in df.columns
    assert "status_code" in df.columns
