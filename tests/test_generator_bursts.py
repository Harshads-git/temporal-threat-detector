"""Tests for stochastic baseline variance, burst shaping, and dataset export stability."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import pytest

from api.schemas import (
    AttackType,
    BurstMode,
    CloudTelemetryLog,
    HttpMethod,
)
from data.synthetic_generator import (
    SyntheticTelemetryGenerator,
)


def test_burst_multiplier_constant():
    """Verify constant burst profile maintains stable scale factor around 1.0."""
    gen = SyntheticTelemetryGenerator(seed=201)
    # With 0 jitter
    val_mid = gen.calculate_burst_multiplier(BurstMode.CONSTANT, progress_ratio=0.5, jitter_pct=0.0)
    assert val_mid == pytest.approx(1.0, rel=1e-3)

    # With jitter
    val_jitter = gen.calculate_burst_multiplier(BurstMode.CONSTANT, progress_ratio=0.5, jitter_pct=0.2)
    assert 0.75 <= val_jitter <= 1.25


def test_burst_multiplier_sudden_spike():
    """Verify sudden spike concentrates amplification at center apex (t=0.5)."""
    gen = SyntheticTelemetryGenerator(seed=202)
    val_start = gen.calculate_burst_multiplier(BurstMode.SUDDEN_SPIKE, progress_ratio=0.0, jitter_pct=0.0)
    val_peak = gen.calculate_burst_multiplier(BurstMode.SUDDEN_SPIKE, progress_ratio=0.5, jitter_pct=0.0)
    val_end = gen.calculate_burst_multiplier(BurstMode.SUDDEN_SPIKE, progress_ratio=1.0, jitter_pct=0.0)

    assert val_start < 0.5
    assert val_peak > 3.0
    assert val_end < 0.5
    assert val_peak > (val_start * 6.0)


def test_burst_multiplier_gradual_ramp():
    """Verify gradual ramp escalates monotonically across elapsed progress."""
    gen = SyntheticTelemetryGenerator(seed=203)
    val_early = gen.calculate_burst_multiplier(BurstMode.GRADUAL_RAMP, progress_ratio=0.1, jitter_pct=0.0)
    val_mid = gen.calculate_burst_multiplier(BurstMode.GRADUAL_RAMP, progress_ratio=0.5, jitter_pct=0.0)
    val_late = gen.calculate_burst_multiplier(BurstMode.GRADUAL_RAMP, progress_ratio=0.9, jitter_pct=0.0)

    assert val_early < val_mid < val_late
    assert val_late > 2.0


def test_burst_multiplier_periodic_pulses():
    """Verify periodic pulse alternates between elevated attack pulses and quiet valleys."""
    gen = SyntheticTelemetryGenerator(seed=204)
    # Sine wave pulse peaks when sin > 0 and troughs when sin <= 0
    # t = 0.05 -> sin(0.5 pi) = 1.0 > 0
    # t = 0.15 -> sin(1.5 pi) = -1.0 < 0
    val_pulse = gen.calculate_burst_multiplier(BurstMode.PERIODIC_PULSES, progress_ratio=0.05, jitter_pct=0.0)
    val_valley = gen.calculate_burst_multiplier(BurstMode.PERIODIC_PULSES, progress_ratio=0.15, jitter_pct=0.0)

    assert val_pulse > 2.0
    assert val_valley < 0.5


def test_diurnal_curve_variation():
    """Verify diurnal multiplier adheres to day/night circadian cycles."""
    gen = SyntheticTelemetryGenerator(seed=205)
    # Peak at 14:00 UTC
    day_peak = datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc)
    # Trough at 03:00 UTC
    night_trough = datetime(2026, 10, 10, 3, 0, tzinfo=timezone.utc)

    mult_day = gen._get_diurnal_multiplier(day_peak)
    mult_night = gen._get_diurnal_multiplier(night_trough)

    assert mult_day > mult_night
    assert mult_day >= 1.5
    assert mult_night <= 0.6


def test_baseline_latency_and_status_distribution():
    """Verify baseline distribution satisfies SLA latency and low error criteria."""
    gen = SyntheticTelemetryGenerator(seed=206)
    events = gen.generate_batch(count=300, inject_attack=None)

    durations = [e.duration_ms for e in events]
    status_codes = [e.status_code for e in events]

    # Latencies strictly positive
    assert all(d > 0.0 for d in durations)

    # Median latency is realistic for cloud services (under 100ms)
    sorted_durations = sorted(durations)
    median_duration = sorted_durations[len(durations) // 2]
    assert 20.0 <= median_duration <= 90.0

    # Healthy status code ratio (> 85% success 2xx)
    success_ratio = sum(1 for s in status_codes if 200 <= s < 300) / len(status_codes)
    assert success_ratio >= 0.85

    # No synthetic attack flags in baseline
    assert not any(e.is_synthetic_attack for e in events)


def test_dataset_jsonl_and_csv_export(tmp_path: Path):
    """Verify JSONL and CSV dataset export formats and integrity."""
    gen = SyntheticTelemetryGenerator(seed=207)
    events = gen.generate_batch(count=25, inject_attack=AttackType.CREDENTIAL_STUFFING, attack_ratio=0.4)

    jsonl_path = tmp_path / "test_out.jsonl"
    csv_path = tmp_path / "test_out.csv"

    # Export JSONL
    written_jsonl = gen.export_dataset_jsonl(str(jsonl_path), events)
    assert written_jsonl == 25
    assert jsonl_path.exists()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 25
        first_obj = json.loads(lines[0])
        assert "event_id" in first_obj
        assert "source_ip" in first_obj

    # Export CSV
    written_csv = gen.export_dataset_csv(str(csv_path), events)
    assert written_csv == 25
    assert csv_path.exists()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 25
        assert "endpoint" in rows[0]
        assert "duration_ms" in rows[0]


def test_benchmark_suite_manifest_generation(tmp_path: Path):
    """Verify export_benchmark_suite produces valid manifest and ground truth indices."""
    gen = SyntheticTelemetryGenerator(seed=208)
    suite_dir = tmp_path / "benchmark_run"

    manifest = gen.export_benchmark_suite(
        output_dir=str(suite_dir),
        count=60,
        inject_attack=AttackType.OFF_HOUR_PROBING,
        attack_ratio=0.3,
        burst_mode=BurstMode.GRADUAL_RAMP,
        intensity=4.0,
    )

    assert manifest["record_count"] == 60
    assert manifest["attack_count"] == 18
    assert manifest["baseline_count"] == 42
    assert manifest["injected_attack_type"] == "OFF_HOUR_PROBING"
    assert manifest["burst_mode"] == "GRADUAL_RAMP"
    assert len(manifest["ground_truth_anomalous_indices"]) == 18

    # Verify manifest file on disk
    manifest_file = suite_dir / "manifest.json"
    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        assert loaded["dataset_id"] == manifest["dataset_id"]


def test_realtime_streaming_playback():
    """Verify realtime playback generator yields events with callback triggers."""
    gen = SyntheticTelemetryGenerator(seed=209)
    events = gen.generate_batch(count=10, inject_attack=None)

    collected = []

    def on_record(record: CloudTelemetryLog):
        collected.append(record.event_id)

    stream = gen.stream_realtime_events(
        events,
        speed_multiplier=100.0,  # Fast forward replay
        max_delay_seconds=0.01,
        on_event=on_record,
    )

    streamed_records = list(stream)
    assert len(streamed_records) == 10
    assert len(collected) == 10
    assert collected[0] == events[0].event_id
