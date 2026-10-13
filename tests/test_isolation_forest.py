"""Unit and integration tests for Tier-2 Isolation Forest and Hybrid Threat Ensemble."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
import numpy as np

from api.schemas import (
    DetectorTier,
    SeverityLevel,
    ThreatAlert,
    WindowAggregate,
)
from engine.isolation_forest import (
    HybridThreatEnsemble,
    IsolationForestDetector,
    train_baseline_model,
)
from engine.zscore_filter import StatisticalFilterEngine


def create_baseline_window(
    offset_min: int,
    req_count: int = 100,
    off_hour: float = 0.0,
    resp_bytes: int = 150000,
    err_4xx: int = 2,
    auth_fail: int = 1,
) -> WindowAggregate:
    """Helper to generate baseline window aggregates."""
    now = datetime(2026, 10, 13, 14, 0, tzinfo=timezone.utc) + timedelta(minutes=offset_min)
    return WindowAggregate(
        window_start=now,
        window_end=now + timedelta(seconds=60),
        duration_seconds=60.0,
        request_count=req_count,
        request_rate_per_sec=req_count / 60.0,
        error_count_4xx=err_4xx,
        error_count_5xx=0,
        error_rate_4xx=err_4xx / max(1, req_count),
        error_rate_5xx=0.0,
        mean_duration_ms=45.0,
        p95_duration_ms=75.0,
        unique_source_ips=30,
        unique_users=15,
        total_request_bytes=req_count * 500,
        total_response_bytes=resp_bytes,
        mean_response_bytes=resp_bytes / max(1, req_count),
        auth_failure_count=auth_fail,
        off_hour_ratio=off_hour,
        bot_agent_ratio=0.02,
    )


def test_isolation_forest_fit_and_anomaly_detection():
    """Verify Isolation Forest learns baseline distribution and flags multi-feature outliers."""
    # Build 30 clean baseline windows
    baseline = [
        create_baseline_window(i, req_count=100 + (i % 10), resp_bytes=150000 + (i * 1000))
        for i in range(30)
    ]

    detector = IsolationForestDetector(n_estimators=50, contamination=0.05, random_state=42)
    detector.fit(baseline)
    assert detector.is_fitted is True

    # 1. Normal window should be classified as inlier
    normal_window = create_baseline_window(35, req_count=102, resp_bytes=155000)
    normal_res = detector.evaluate_window(normal_window)
    assert normal_res.is_anomaly is False
    assert normal_res.severity in (SeverityLevel.INFO, SeverityLevel.LOW)

    # 2. Exfiltration outlier (abnormal bytes + abnormal duration)
    exfil_window = create_baseline_window(36, req_count=100, resp_bytes=45000000)
    exfil_res = detector.evaluate_window(exfil_window)
    assert exfil_res.is_anomaly is True
    assert exfil_res.raw_score < 0.0
    assert len(exfil_res.top_deviations) >= 1
    # Check top deviation caught byte surge
    top_features = [d.feature_name for d in exfil_res.top_deviations]
    assert any("response_bytes" in f for f in top_features)


def test_isolation_forest_persistence_roundtrip(tmp_path: Path):
    """Verify saving and restoring IsolationForestDetector preserves model weights and inference."""
    baseline = [create_baseline_window(i) for i in range(25)]
    detector = IsolationForestDetector(n_estimators=40, contamination=0.05, random_state=42)
    detector.fit(baseline)
    detector.calibrate_threshold(baseline, target_fpr=0.04)

    test_window = create_baseline_window(99, req_count=800, resp_bytes=9000000)
    orig_res = detector.evaluate_window(test_window)

    # Save to disk
    model_path = tmp_path / "iforest_test.joblib"
    detector.save_model(str(model_path))
    assert model_path.exists()

    # Load from disk
    loaded_detector = IsolationForestDetector.load_model(str(model_path))
    assert loaded_detector.is_fitted is True
    loaded_res = loaded_detector.evaluate_window(test_window)

    assert loaded_res.is_anomaly == orig_res.is_anomaly
    assert loaded_res.raw_score == pytest.approx(orig_res.raw_score, rel=1e-4)
    assert loaded_res.anomaly_score == pytest.approx(orig_res.anomaly_score, rel=1e-4)


def test_train_baseline_model_pipeline(tmp_path: Path):
    """Verify automated baseline model training routine with synthetic telemetry."""
    save_file = tmp_path / "baseline_model.joblib"
    detector, metadata = train_baseline_model(
        output_model_path=str(save_file),
        n_baseline_events=250,
        contamination=0.04,
    )

    assert detector.is_fitted is True
    assert save_file.exists()
    assert metadata["n_baseline_events"] == 250
    assert metadata["n_baseline_windows"] > 5
    assert metadata["feature_count"] == 13


def test_hybrid_ensemble_dual_confirmation():
    """Verify HybridThreatEnsemble escalates to HYBRID_ENSEMBLE with CRITICAL severity on dual confirmation."""
    # Setup trained AI detector
    baseline = [create_baseline_window(i, req_count=100) for i in range(30)]
    ai_detector = IsolationForestDetector(n_estimators=50, contamination=0.05, random_state=42)
    ai_detector.fit(baseline)

    # Statistical engine warmed up on baseline
    stat_engine = StatisticalFilterEngine(z_threshold=3.0, history_size=30)
    for w in baseline:
        stat_engine.evaluate_window(w)

    ensemble = HybridThreatEnsemble(statistical_engine=stat_engine, ai_detector=ai_detector)

    # Inject massive DDoS window (triggers both Z-Score and Isolation Forest)
    attack_window = create_baseline_window(35, req_count=2000, resp_bytes=3000000, err_4xx=50)
    alert = ensemble.evaluate(attack_window)

    assert alert is not None
    assert alert.detector_tier == DetectorTier.HYBRID_ENSEMBLE
    assert alert.severity == SeverityLevel.CRITICAL
    assert alert.confidence_score >= 0.90
    assert "Hybrid Confirmed" in alert.threat_hypothesis
    assert "Dual-Tier Detection" in alert.explanation_narrative


def test_hybrid_ensemble_contextual_attack_tier2_only():
    """Verify HybridThreatEnsemble flags subtle contextual attacks (off-hour probing) via Tier-2 AI when Tier-1 is silent."""
    baseline = [create_baseline_window(i) for i in range(25)]
    ai_detector = IsolationForestDetector(n_estimators=50, contamination=0.05, random_state=42)
    ai_detector.fit(baseline)

    # Empty/silent statistical engine simulating a sub-threshold stealthy event
    stat_engine = StatisticalFilterEngine(z_threshold=10.0, history_size=30)
    ensemble = HybridThreatEnsemble(statistical_engine=stat_engine, ai_detector=ai_detector)

    # Off-hour reconnaissance probing window (subtle volume but multi-feature exfiltration payload)
    recon_window = create_baseline_window(35, req_count=100, off_hour=1.0, resp_bytes=35000000)
    alert = ensemble.evaluate(recon_window)

    assert alert is not None
    assert alert.detector_tier == DetectorTier.TIER2_UNSUPERVISED_IFOREST
    assert alert.confidence_score > 0.0
    assert len(alert.top_contributions) >= 1
    assert "response_bytes" in alert.top_contributions[0].feature_name or "off_hour" in alert.top_contributions[0].feature_name
