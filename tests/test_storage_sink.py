"""Unit and integration tests for Cloud and Local Storage Sinks (Alerts & Telemetry)."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import gzip
import io
import json
from pathlib import Path
import shutil
from unittest.mock import MagicMock, patch
import pytest

from api.schemas import (
    CloudTelemetryLog,
    DetectorTier,
    FeatureAttribution,
    HttpMethod,
    SeverityLevel,
    ThreatAlert,
    UserAgentCategory,
    WindowAggregate,
)
from cloud.storage_sink import (
    DynamoDBAlertSink,
    InMemoryAlertSink,
    InMemoryTelemetryArchiver,
    LocalFileTelemetryArchiver,
    LocalJsonAlertSink,
    S3TelemetryArchiver,
    _convert_decimals_to_floats,
    _convert_floats_to_decimals,
    get_alert_sink,
    get_telemetry_archiver,
)
from data.synthetic_generator import SyntheticTelemetryGenerator


def create_sample_alert(
    offset_min: int = 0,
    severity: SeverityLevel = SeverityLevel.HIGH,
    threat_hypothesis: str = "High Volumetric HTTP Flood (DDoS)",
) -> ThreatAlert:
    """Helper creating a fully populated ThreatAlert instance."""
    now = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc) + timedelta(minutes=offset_min)
    window = WindowAggregate(
        window_start=now - timedelta(seconds=60),
        window_end=now,
        duration_seconds=60.0,
        request_count=1500,
        request_rate_per_sec=25.0,
        error_count_4xx=10,
        error_count_5xx=5,
        error_rate_4xx=0.0067,
        error_rate_5xx=0.0033,
        mean_duration_ms=55.0,
        p95_duration_ms=120.0,
        unique_source_ips=300,
        unique_users=150,
        total_request_bytes=750000,
        total_response_bytes=3000000,
        mean_response_bytes=2000.0,
        auth_failure_count=2,
        off_hour_ratio=0.0,
        bot_agent_ratio=0.05,
    )
    return ThreatAlert(
        window_id=f"win-{offset_min}",
        timestamp=now,
        detector_tier=DetectorTier.HYBRID_ENSEMBLE,
        severity=severity,
        confidence_score=0.92,
        anomaly_score=4.85,
        decision_threshold=3.0,
        top_contributions=[
            FeatureAttribution(
                feature_name="request_count",
                observed_value=1500.0,
                baseline_expected=105.0,
                shap_value=0.88,
                importance_rank=1,
            )
        ],
        threat_hypothesis=threat_hypothesis,
        explanation_narrative="Request volume escalated 14x above diurnal baseline.",
        recommended_mitigation="Deploy AWS WAF rate-limiting rule.",
        raw_window_metrics=window,
    )


# ==============================================================================
# IN-MEMORY ALERT SINK TESTS
# ==============================================================================


def test_in_memory_alert_sink_crud():
    """Verify in-memory sink handles store, retrieval, non-existent key, and clear."""
    sink = InMemoryAlertSink(max_capacity=10)
    alert = create_sample_alert(offset_min=0)

    # Save
    assert sink.save_alert(alert) is True
    assert sink.count_alerts() == 1

    # Get
    retrieved = sink.get_alert(alert.alert_id)
    assert retrieved is not None
    assert retrieved.alert_id == alert.alert_id
    assert retrieved.threat_hypothesis == alert.threat_hypothesis

    # Non-existent
    assert sink.get_alert("unknown-id") is None

    # Clear
    sink.clear()
    assert sink.count_alerts() == 0
    assert sink.get_alert(alert.alert_id) is None


def test_in_memory_alert_sink_filtering_and_sorting():
    """Verify in-memory sink filters by severity, hypothesis, time range, and sorts descending."""
    sink = InMemoryAlertSink()
    alerts = [
        create_sample_alert(0, SeverityLevel.LOW, "Low Anomaly"),
        create_sample_alert(5, SeverityLevel.MEDIUM, "Distributed Credential Stuffing Campaign"),
        create_sample_alert(10, SeverityLevel.HIGH, "High Volumetric HTTP Flood (DDoS)"),
        create_sample_alert(15, SeverityLevel.CRITICAL, "Privileged Data Exfiltration Surge"),
    ]
    assert sink.save_alerts(alerts) == 4

    # Timestamp descending order
    all_alerts = sink.list_alerts(limit=10)
    assert len(all_alerts) == 4
    assert all_alerts[0].timestamp > all_alerts[1].timestamp

    # Severity filtering
    high_alerts = sink.list_alerts(severity=SeverityLevel.HIGH)
    assert len(high_alerts) == 1
    assert high_alerts[0].severity == SeverityLevel.HIGH

    # Threat hypothesis filtering
    ddos_alerts = sink.list_alerts(threat_hypothesis="High Volumetric HTTP Flood (DDoS)")
    assert len(ddos_alerts) == 1
    assert ddos_alerts[0].threat_hypothesis == "High Volumetric HTTP Flood (DDoS)"

    # Time range filtering
    start_t = datetime(2026, 10, 15, 12, 6, tzinfo=timezone.utc)
    ranged = sink.list_alerts(start_time=start_t)
    assert len(ranged) == 2


def test_in_memory_alert_sink_capacity_eviction():
    """Verify oldest alert is evicted when capacity is reached."""
    sink = InMemoryAlertSink(max_capacity=3)
    a1 = create_sample_alert(0)
    a2 = create_sample_alert(1)
    a3 = create_sample_alert(2)
    a4 = create_sample_alert(3)

    sink.save_alert(a1)
    sink.save_alert(a2)
    sink.save_alert(a3)
    assert sink.count_alerts() == 3

    # Add 4th, should evict a1
    sink.save_alert(a4)
    assert sink.count_alerts() == 3
    assert sink.get_alert(a1.alert_id) is None
    assert sink.get_alert(a4.alert_id) is not None


# ==============================================================================
# LOCAL JSON ALERT SINK TESTS
# ==============================================================================


def test_local_json_alert_sink_persistence_and_reload(tmp_path: Path):
    """Verify local JSONL sink persists to disk and reloads on startup."""
    jsonl_file = tmp_path / "alerts.jsonl"
    sink = LocalJsonAlertSink(file_path=jsonl_file)

    a1 = create_sample_alert(0, SeverityLevel.MEDIUM)
    a2 = create_sample_alert(1, SeverityLevel.HIGH)
    assert sink.save_alerts([a1, a2]) == 2
    assert jsonl_file.exists()

    # Re-instantiate sink pointing to same file
    sink2 = LocalJsonAlertSink(file_path=jsonl_file)
    assert sink2.count_alerts() == 2
    reloaded_a1 = sink2.get_alert(a1.alert_id)
    assert reloaded_a1 is not None
    assert reloaded_a1.alert_id == a1.alert_id
    assert reloaded_a1.severity == SeverityLevel.MEDIUM

    # Clear
    sink2.clear()
    assert sink2.count_alerts() == 0
    assert jsonl_file.read_text(encoding="utf-8") == ""


def test_local_json_alert_sink_corrupt_line_handling(tmp_path: Path):
    """Verify corrupt JSON lines do not crash index loader."""
    jsonl_file = tmp_path / "corrupt_alerts.jsonl"
    valid_alert = create_sample_alert(0)
    jsonl_file.write_text(
        f"{valid_alert.model_dump_json()}\n"
        f"{{INVALID JSON LINE}}\n"
        f"{{}}\n",
        encoding="utf-8",
    )

    sink = LocalJsonAlertSink(file_path=jsonl_file)
    assert sink.count_alerts() == 1
    assert sink.get_alert(valid_alert.alert_id) is not None


# ==============================================================================
# DYNAMODB ALERT SINK & DECIMAL CONVERSION TESTS
# ==============================================================================


def test_decimal_conversion_utilities():
    """Verify bidirectional conversion between float and Decimal."""
    data = {
        "float_val": 42.5,
        "nested": {"nested_float": 0.001, "int_val": 10},
        "list_vals": [1.1, 2.2, "string"],
    }
    dynamo_ready = _convert_floats_to_decimals(data)
    assert isinstance(dynamo_ready["float_val"], Decimal)
    assert isinstance(dynamo_ready["nested"]["nested_float"], Decimal)
    assert dynamo_ready["nested"]["int_val"] == 10
    assert isinstance(dynamo_ready["list_vals"][0], Decimal)

    restored = _convert_decimals_to_floats(dynamo_ready)
    assert isinstance(restored["float_val"], float)
    assert restored["float_val"] == 42.5
    assert restored["nested"]["nested_float"] == 0.001


def test_mock_dynamodb_alert_sink():
    """Verify DynamoDBAlertSink using mock Boto3 DynamoDB resource."""
    mock_resource = MagicMock()
    mock_table = MagicMock()
    mock_resource.Table.return_value = mock_table

    sink = DynamoDBAlertSink(
        table_name="TestThreatAlerts",
        dynamodb_resource=mock_resource,
        auto_create_table=False,
    )

    alert = create_sample_alert(0)

    # Save
    mock_table.put_item.return_value = {}
    assert sink.save_alert(alert) is True
    mock_table.put_item.assert_called_once()

    # Get
    item_dict = sink._alert_to_item(alert)
    mock_table.get_item.return_value = {"Item": item_dict}
    retrieved = sink.get_alert(alert.alert_id)
    assert retrieved is not None
    assert retrieved.alert_id == alert.alert_id

    # Scan / list
    mock_table.scan.return_value = {"Items": [item_dict]}
    alerts = sink.list_alerts(limit=5)
    assert len(alerts) == 1
    assert alerts[0].alert_id == alert.alert_id


# ==============================================================================
# IN-MEMORY & LOCAL FILE TELEMETRY ARCHIVER TESTS
# ==============================================================================


def test_in_memory_telemetry_archiver():
    """Verify in-memory telemetry archiver roundtrip, prefix listing, and errors."""
    archiver = InMemoryTelemetryArchiver()
    gen = SyntheticTelemetryGenerator(seed=42)
    logs = gen.generate_batch(count=15)

    uri = archiver.archive_batch(logs)
    assert uri.startswith("mem://telemetry/")

    read_logs = archiver.read_batch(uri)
    assert len(read_logs) == 15
    assert read_logs[0].event_id == logs[0].event_id

    archives = archiver.list_archives()
    assert len(archives) == 1
    assert archives[0] == uri

    # Non-existent archive raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        archiver.read_batch("mem://telemetry/non-existent.jsonl.gz")

    archiver.clear()
    assert len(archiver.list_archives()) == 0


def test_local_file_telemetry_archiver_gzip_partitioning(tmp_path: Path):
    """Verify date partitioning, gzip compression, and readback for LocalFileTelemetryArchiver."""
    base_dir = tmp_path / "raw_telemetry"
    archiver = LocalFileTelemetryArchiver(base_dir=base_dir)

    gen = SyntheticTelemetryGenerator(seed=42)
    logs = gen.generate_batch(count=20)
    archive_date = datetime(2026, 10, 15, 10, 0, tzinfo=timezone.utc)

    uri = archiver.archive_batch(logs, partition_date=archive_date, batch_id="batch123")
    assert "year=2026/month=10/day=15" in uri
    assert uri.endswith(".jsonl.gz")

    # Verify gzip compression on disk
    file_path = Path(uri)
    assert file_path.exists()
    with open(file_path, "rb") as f:
        magic = f.read(2)
        assert magic == b"\x1f\x8b"  # Gzip magic number

    # Read back
    read_logs = archiver.read_batch(uri)
    assert len(read_logs) == 20
    assert read_logs[0].source_ip == logs[0].source_ip

    # Listing
    archives = archiver.list_archives(prefix="year=2026")
    assert len(archives) == 1

    # Clear
    archiver.clear()
    assert len(archiver.list_archives()) == 0


# ==============================================================================
# S3 TELEMETRY ARCHIVER TESTS (MOCK BOTO3 CLIENT)
# ==============================================================================


def test_mock_s3_telemetry_archiver():
    """Verify S3TelemetryArchiver with mocked Boto3 S3 client."""
    mock_s3 = MagicMock()
    archiver = S3TelemetryArchiver(
        bucket_name="mock-telemetry-bucket",
        auto_create_bucket=False,
        s3_client=mock_s3,
    )

    gen = SyntheticTelemetryGenerator(seed=42)
    logs = gen.generate_batch(count=10)
    archive_date = datetime(2026, 10, 15, 14, 0, tzinfo=timezone.utc)

    # Archive
    uri = archiver.archive_batch(logs, partition_date=archive_date, batch_id="test001")
    assert uri.startswith("s3://mock-telemetry-bucket/telemetry/year=2026/month=10/day=15/")
    mock_s3.put_object.assert_called_once()
    put_call = mock_s3.put_object.call_args[1]
    assert put_call["Bucket"] == "mock-telemetry-bucket"
    assert put_call["ContentEncoding"] == "gzip"

    # Read back
    raw_json = "\n".join(log.model_dump_json() for log in logs) + "\n"
    compressed_body = gzip.compress(raw_json.encode("utf-8"))
    mock_body = MagicMock()
    mock_body.read.return_value = compressed_body
    mock_s3.get_object.return_value = {"Body": mock_body}

    read_logs = archiver.read_batch(uri)
    assert len(read_logs) == 10
    assert read_logs[0].event_id == logs[0].event_id


# ==============================================================================
# FACTORY RESOLVER TESTS
# ==============================================================================


def test_get_alert_sink_factory(tmp_path: Path):
    """Verify get_alert_sink resolver across in_memory, local_json, and fallback."""
    sink1 = get_alert_sink(driver="in_memory")
    assert isinstance(sink1, InMemoryAlertSink)

    jsonl_path = tmp_path / "factory_alerts.jsonl"
    sink2 = get_alert_sink(driver="local_json", file_path=str(jsonl_path))
    assert isinstance(sink2, LocalJsonAlertSink)

    with patch.dict("os.environ", {"ALERT_STORAGE_DRIVER": "in_memory"}):
        sink3 = get_alert_sink(driver="auto")
        assert isinstance(sink3, InMemoryAlertSink)


def test_get_telemetry_archiver_factory(tmp_path: Path):
    """Verify get_telemetry_archiver resolver across in_memory, local_file, and fallback."""
    arch1 = get_telemetry_archiver(driver="in_memory")
    assert isinstance(arch1, InMemoryTelemetryArchiver)

    arch2 = get_telemetry_archiver(driver="local_file", base_dir=str(tmp_path / "raw"))
    assert isinstance(arch2, LocalFileTelemetryArchiver)

    with patch.dict("os.environ", {"TELEMETRY_ARCHIVE_DRIVER": "in_memory"}):
        arch3 = get_telemetry_archiver(driver="auto")
        assert isinstance(arch3, InMemoryTelemetryArchiver)
