"""Unit and integration test suite for benchmarking engines, profilers, and accuracy evaluators."""

from __future__ import annotations

import pytest
from benchmarks.benchmark_accuracy import (
    AccuracyBenchmarkReport,
    DetectionAccuracyBenchmarkRunner,
    DetectionAccuracyMetrics,
)
from benchmarks.benchmark_pipeline import (
    LatencyMetrics,
    PipelineBenchmarkReport,
    PipelineBenchmarkRunner,
    PipelineLatencyProfiler,
)


def test_latency_metrics_known_values() -> None:
    """Verify LatencyMetrics computes correct mean, percentiles, and boundaries."""
    # 5 samples in seconds: [0.010, 0.020, 0.030, 0.040, 0.050] -> [10ms, 20ms, 30ms, 40ms, 50ms]
    durations = [0.010, 0.020, 0.030, 0.040, 0.050]
    metrics = LatencyMetrics.from_durations_sec(durations)

    assert metrics.sample_count == 5
    assert metrics.min_ms == 10.0
    assert metrics.max_ms == 50.0
    assert metrics.mean_ms == 30.0
    assert metrics.p50_ms == 30.0
    assert metrics.p95_ms > 40.0


def test_latency_metrics_empty() -> None:
    """Verify LatencyMetrics handles empty list gracefully."""
    metrics = LatencyMetrics.from_durations_sec([])
    assert metrics.sample_count == 0
    assert metrics.mean_ms == 0.0
    assert metrics.p50_ms == 0.0


def test_pipeline_latency_profiler_recording() -> None:
    """Verify PipelineLatencyProfiler records durations across phases and summarizes."""
    profiler = PipelineLatencyProfiler()
    profiler.record("ingestion", 0.002)
    profiler.record("ingestion", 0.004)
    profiler.record("tier1_statistical", 0.001)

    summary = profiler.get_summary()
    assert "ingestion" in summary
    assert "tier1_statistical" in summary
    assert summary["ingestion"].sample_count == 2
    assert summary["ingestion"].mean_ms == 3.0
    assert summary["tier1_statistical"].sample_count == 1
    assert summary["tier1_statistical"].mean_ms == 1.0


def test_pipeline_benchmark_runner_smoke() -> None:
    """Verify PipelineBenchmarkRunner executes benchmark run and formats reports."""
    runner = PipelineBenchmarkRunner(seed=42)
    report = runner.run_benchmark(batch_size=50, iterations=2, warmup_runs=1)

    assert isinstance(report, PipelineBenchmarkReport)
    assert report.total_events_processed == 100
    assert report.throughput_events_per_sec > 0.0
    assert "ingestion" in report.component_latencies
    assert "tier1_statistical" in report.component_latencies
    assert "tier2_isolation_forest" in report.component_latencies
    assert "xai_tree_explainer" in report.component_latencies

    # Serialization tests
    report_dict = report.to_dict()
    assert report_dict["total_events_processed"] == 100
    assert "component_latencies" in report_dict

    md_table = report.to_markdown()
    assert "Pipeline Performance Benchmark" in md_table
    assert "Tier 1: Z-Score & MAD Filter" in md_table


def test_detection_accuracy_metrics_computation() -> None:
    """Verify DetectionAccuracyMetrics computes precision, recall, f1, and fpr accurately."""
    # 4 True Positives, 1 False Positive, 4 True Negatives, 1 False Negative
    y_true = [True, True, True, True, False, False, False, False, False, True]
    y_pred = [True, True, True, True, True,  False, False, False, False, False]

    m = DetectionAccuracyMetrics.compute("TestDetector", y_true, y_pred)
    assert m.true_positives == 4
    assert m.false_positives == 1
    assert m.true_negatives == 4
    assert m.false_negatives == 1
    assert m.total_samples == 10

    # Accuracy = 8/10 = 0.8
    assert m.accuracy == 0.8
    # Precision = 4 / (4 + 1) = 0.8
    assert m.precision == 0.8
    # Recall = 4 / (4 + 1) = 0.8
    assert m.recall == 0.8
    # F1 = 0.8
    assert m.f1_score == 0.8
    # FPR = 1 / (1 + 4) = 0.2
    assert m.false_positive_rate == 0.2


def test_detection_accuracy_benchmark_runner_smoke() -> None:
    """Verify DetectionAccuracyBenchmarkRunner evaluates multi-vector scenarios and formats output."""
    runner = DetectionAccuracyBenchmarkRunner(seed=42)
    report = runner.run_benchmark(events_per_scenario=60)

    assert isinstance(report, AccuracyBenchmarkReport)
    assert "normal" in report.sample_counts
    assert "ddos" in report.sample_counts

    # Verify all 3 detector tiers evaluated
    assert "Tier 1 (Statistical Filter)" in report.overall_metrics
    assert "Tier 2 (Isolation Forest AI)" in report.overall_metrics
    assert "Hybrid Threat Ensemble" in report.overall_metrics

    # Verify per-vector recall mapping
    assert "ddos" in report.per_vector_recall
    assert "credential_stuffing" in report.per_vector_recall

    # Markdown format check
    md = report.to_markdown()
    assert "Threat Detection Accuracy & Architectural Comparison Benchmark" in md
    assert "Overall Architectural Comparison" in md
    assert "Threat Vector Recall Breakdown" in md
