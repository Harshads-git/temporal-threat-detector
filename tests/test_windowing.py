"""Tests for sliding and tumbling temporal window aggregation and feature vectorization."""

from datetime import datetime, timedelta, timezone
import numpy as np
import pytest

from api.schemas import (
    CloudTelemetryLog,
    HttpMethod,
    UserAgentCategory,
    WindowAggregate,
)
from engine.windowing import (
    CANONICAL_FEATURE_NAMES,
    FeatureMatrixVectorizer,
    StreamingWindowBuffer,
    TemporalWindowAggregator,
    WindowType,
    compute_window_features,
)


def create_test_event(
    offset_seconds: float,
    base_time: datetime,
    status_code: int = 200,
    duration_ms: float = 40.0,
    source_ip: str = "10.0.1.5",
    user_identity: str = "user-alice",
    req_bytes: int = 500,
    resp_bytes: int = 2000,
    ua: UserAgentCategory = UserAgentCategory.BROWSER,
    error_code: str = None,
) -> CloudTelemetryLog:
    """Helper to create deterministic telemetry events for window tests."""
    return CloudTelemetryLog(
        timestamp=base_time + timedelta(seconds=offset_seconds),
        source_ip=source_ip,
        user_identity=user_identity,
        endpoint="/api/v1/test",
        http_method=HttpMethod.GET,
        status_code=status_code,
        request_bytes=req_bytes,
        response_bytes=resp_bytes,
        duration_ms=duration_ms,
        user_agent_category=ua,
        error_code=error_code,
    )


def test_compute_window_features_exact_values():
    """Verify compute_window_features calculates exact statistical and rate metrics."""
    base_time = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)
    w_start = base_time
    w_end = base_time + timedelta(seconds=60)

    events = [
        # Normal 200 event
        create_test_event(5, base_time, status_code=200, duration_ms=20.0, resp_bytes=1000, source_ip="10.0.1.1"),
        # Another normal event
        create_test_event(15, base_time, status_code=200, duration_ms=40.0, resp_bytes=2000, source_ip="10.0.1.2"),
        # Auth failure (401)
        create_test_event(25, base_time, status_code=401, duration_ms=30.0, resp_bytes=500, source_ip="10.0.1.3", error_code="InvalidCredentials"),
        # Server error (503) from Bot
        create_test_event(35, base_time, status_code=503, duration_ms=100.0, resp_bytes=300, source_ip="10.0.1.4", ua=UserAgentCategory.AUTOMATED_BOT),
        # Normal event
        create_test_event(45, base_time, status_code=200, duration_ms=60.0, resp_bytes=1200, source_ip="10.0.1.1"),
    ]

    agg = compute_window_features(events, w_start, w_end)

    assert agg.request_count == 5
    assert agg.duration_seconds == 60.0
    assert agg.request_rate_per_sec == pytest.approx(5 / 60.0, rel=1e-3)
    assert agg.error_count_4xx == 1
    assert agg.error_rate_4xx == pytest.approx(0.20, rel=1e-3)
    assert agg.error_count_5xx == 1
    assert agg.error_rate_5xx == pytest.approx(0.20, rel=1e-3)
    assert agg.auth_failure_count == 1
    assert agg.bot_agent_ratio == pytest.approx(0.20, rel=1e-3)
    assert agg.unique_source_ips == 4
    assert agg.total_response_bytes == (1000 + 2000 + 500 + 300 + 1200)
    assert agg.mean_response_bytes == 1000.0
    assert agg.mean_duration_ms == 50.0  # (20 + 40 + 30 + 100 + 60) / 5


def test_empty_window_handling():
    """Verify empty event window yields valid non-null zero metrics."""
    base_time = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)
    agg = compute_window_features([], base_time, base_time + timedelta(seconds=60))

    assert agg.request_count == 0
    assert agg.request_rate_per_sec == 0.0
    assert agg.error_rate_4xx == 0.0
    assert agg.mean_duration_ms == 0.0


def test_temporal_aggregator_sliding_vs_tumbling():
    """Verify distinct window counts between sliding (overlapping) and tumbling windows."""
    base_time = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)
    # Generate 1 event every 2 seconds for 120 seconds (60 events)
    events = [create_test_event(offset_seconds=i * 2, base_time=base_time) for i in range(60)]

    # Sliding aggregator: 60s window with 10s hop
    sliding_agg = TemporalWindowAggregator(window_size_seconds=60.0, hop_seconds=10.0, window_type=WindowType.SLIDING)
    sliding_windows = sliding_agg.aggregate_batch(events)

    # Tumbling aggregator: 60s window with 60s hop
    tumbling_agg = TemporalWindowAggregator(window_size_seconds=60.0, window_type=WindowType.TUMBLING)
    tumbling_windows = tumbling_agg.aggregate_batch(events)

    # Sliding should have significantly more overlapping windows than tumbling
    assert len(sliding_windows) >= 7
    assert len(tumbling_windows) == 2

    # Verify sliding hop step
    hop = (sliding_windows[1].window_start - sliding_windows[0].window_start).total_seconds()
    assert hop == 10.0

    # Verify tumbling hop step equals window size
    tumb_hop = (tumbling_windows[1].window_start - tumbling_windows[0].window_start).total_seconds()
    assert tumb_hop == 60.0


def test_streaming_window_buffer():
    """Verify StreamingWindowBuffer ingests events and emits mature sliding windows."""
    base_time = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)
    buf = StreamingWindowBuffer(window_size_seconds=60.0, hop_seconds=10.0)

    # Ingest events spanning only 30 seconds -> no full 60s window mature yet
    for i in range(15):
        buf.add_event(create_test_event(i * 2, base_time))

    emitted_early = buf.check_and_emit_windows()
    assert len(emitted_early) == 0

    # Ingest events up to 90 seconds -> windows should now mature and emit
    for i in range(15, 45):
        buf.add_event(create_test_event(i * 2, base_time))

    emitted_later = buf.check_and_emit_windows()
    assert len(emitted_later) >= 2
    assert all(w.duration_seconds == 60.0 for w in emitted_later)

    # Clearing resets buffer
    buf.clear()
    assert len(buf.check_and_emit_windows()) == 0


def test_feature_matrix_vectorizer():
    """Verify FeatureMatrixVectorizer transforms windows into aligned 2D arrays and standardized matrices."""
    base_time = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)
    events = [create_test_event(offset_seconds=i, base_time=base_time) for i in range(180)]

    aggregator = TemporalWindowAggregator(window_size_seconds=60.0, hop_seconds=10.0)
    windows = aggregator.aggregate_batch(events)
    assert len(windows) >= 5

    vectorizer = FeatureMatrixVectorizer()
    vectorizer.fit_baseline(windows)

    # Baseline parameters fitted
    assert vectorizer.baseline_means is not None
    assert vectorizer.baseline_means.shape == (len(CANONICAL_FEATURE_NAMES),)
    assert "request_count" in vectorizer.feature_stats
    assert vectorizer.feature_stats["request_count"]["mean"] > 0

    # Transform batch
    matrix = vectorizer.transform(windows)
    assert matrix.shape == (len(windows), len(CANONICAL_FEATURE_NAMES))

    # Standardize matrix (should have approx mean 0)
    scaled = vectorizer.standardize(matrix)
    scaled_means = np.mean(scaled, axis=0)
    assert np.allclose(scaled_means, 0.0, atol=1e-2)

    # Single window transformation
    single = vectorizer.transform_single(windows[0])
    assert single.shape == (len(CANONICAL_FEATURE_NAMES),)
    assert np.array_equal(single, matrix[0])

    # Convert to DataFrame
    df = vectorizer.to_dataframe(windows)
    assert len(df) == len(windows)
    for col in CANONICAL_FEATURE_NAMES:
        assert col in df.columns
