"""High-throughput pipeline benchmarking engine and component latency profiler.

Evaluates end-to-end ingestion throughput and latency breakdowns across:
1. Ingestion & Event Deserialization
2. Temporal Window Aggregation
3. Tier 1 Statistical Filter (MAD / EWMA)
4. Tier 2 Multivariate Isolation Forest
5. SHAP TreeExplainer Feature Importance Attribution
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np

from api.schemas import CloudTelemetryLog, WindowAggregate
from data.synthetic_generator import SyntheticTelemetryGenerator
from engine.explainability import ShapExplainerEngine
from engine.isolation_forest import HybridThreatEnsemble, IsolationForestDetector
from engine.windowing import TemporalWindowAggregator
from engine.zscore_filter import StatisticalFilterEngine

logger = logging.getLogger("benchmarks.pipeline")


@dataclass
class LatencyMetrics:
    """Latency distribution summary in milliseconds."""

    mean_ms: float
    std_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    sample_count: int

    @classmethod
    def from_durations_sec(cls, durations_sec: List[float]) -> LatencyMetrics:
        """Compute latency percentiles and distribution stats from a list of durations in seconds."""
        if not durations_sec:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

        ms_vals = np.array(durations_sec, dtype=np.float64) * 1000.0
        return cls(
            mean_ms=round(float(np.mean(ms_vals)), 3),
            std_ms=round(float(np.std(ms_vals)), 3),
            p50_ms=round(float(np.percentile(ms_vals, 50)), 3),
            p90_ms=round(float(np.percentile(ms_vals, 90)), 3),
            p95_ms=round(float(np.percentile(ms_vals, 95)), 3),
            p99_ms=round(float(np.percentile(ms_vals, 99)), 3),
            min_ms=round(float(np.min(ms_vals)), 3),
            max_ms=round(float(np.max(ms_vals)), 3),
            sample_count=len(durations_sec),
        )


class PipelineLatencyProfiler:
    """Fine-grained multi-phase execution timer."""

    def __init__(self) -> None:
        self.timings: Dict[str, List[float]] = {
            "ingestion": [],
            "windowing": [],
            "tier1_statistical": [],
            "tier2_isolation_forest": [],
            "xai_tree_explainer": [],
            "end_to_end": [],
        }

    def record(self, phase: str, duration_sec: float) -> None:
        """Record duration for a specific execution phase."""
        if phase not in self.timings:
            self.timings[phase] = []
        self.timings[phase].append(duration_sec)

    def get_summary(self) -> Dict[str, LatencyMetrics]:
        """Compute statistical latency distribution for all phases."""
        return {
            phase: LatencyMetrics.from_durations_sec(durations)
            for phase, durations in self.timings.items()
        }


@dataclass
class PipelineBenchmarkReport:
    """Comprehensive performance and throughput benchmark report."""

    timestamp: str
    total_events_processed: int
    batch_count: int
    duration_seconds: float
    throughput_events_per_sec: float
    component_latencies: Dict[str, LatencyMetrics]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        data = asdict(self)
        return data

    def to_markdown(self) -> str:
        """Format benchmark summary into GitHub-flavored Markdown table."""
        lines = [
            "# Temporal Threat Detector - Pipeline Performance Benchmark",
            f"**Timestamp:** `{self.timestamp}` | **Total Events Processed:** `{self.total_events_processed:,}` | **Throughput:** `{self.throughput_events_per_sec:,.1f} events/sec`\n",
            "### Component Latency Breakdown (milliseconds)",
            "| Pipeline Component | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        phase_labels = {
            "ingestion": "Ingestion & Deserialization",
            "windowing": "Temporal Window Aggregation",
            "tier1_statistical": "Tier 1: Z-Score & MAD Filter",
            "tier2_isolation_forest": "Tier 2: Isolation Forest AI",
            "xai_tree_explainer": "Novelty: SHAP TreeExplainer XAI",
            "end_to_end": "**Total End-to-End Pipeline**",
        }
        for phase, m in self.component_latencies.items():
            label = phase_labels.get(phase, phase)
            lines.append(
                f"| {label} | {m.mean_ms:.2f} | {m.p50_ms:.2f} | {m.p95_ms:.2f} | {m.p99_ms:.2f} | {m.min_ms:.2f} | {m.max_ms:.2f} |"
            )
        return "\n".join(lines)


class PipelineBenchmarkRunner:
    """Coordinates automated multi-tier pipeline performance benchmarks."""

    def __init__(self, seed: int = 42) -> None:
        self.generator = SyntheticTelemetryGenerator(seed=seed)
        self.aggregator = TemporalWindowAggregator(window_size_seconds=60.0, hop_seconds=10.0)
        self.stat_engine = StatisticalFilterEngine(z_threshold=3.0)

        # Pre-fit Isolation Forest baseline for inference testing
        baseline_events = self.generator.generate_batch(count=1200, inject_attack=None)
        baseline_windows = self.aggregator.aggregate_batch(baseline_events)
        self.ai_detector = IsolationForestDetector(n_estimators=50, contamination=0.03, random_state=seed)
        self.ai_detector.fit(baseline_windows)
        self.ai_detector.calibrate_threshold(baseline_windows, target_fpr=0.02)

        # Initialize SHAP explainer
        self.explainer = ShapExplainerEngine(
            model=self.ai_detector.model,
            scaler=self.ai_detector.scaler,
            background_data=self.ai_detector.scaler.inverse_transform(self.ai_detector.scaler.transform(
                [[w.to_feature_vector().get(c, 0.0) for c in self.ai_detector.feature_names] for w in baseline_windows]
            )),
            feature_names=self.ai_detector.feature_names,
        )

        self.ensemble = HybridThreatEnsemble(
            statistical_engine=self.stat_engine,
            ai_detector=self.ai_detector,
        )

    def run_benchmark(
        self,
        batch_size: int = 500,
        iterations: int = 10,
        warmup_runs: int = 2,
    ) -> PipelineBenchmarkReport:
        """Run performance benchmark measuring throughput and latency per component."""
        logger.info(
            "Executing benchmark: batch_size=%d, iterations=%d, warmup=%d...",
            batch_size,
            iterations,
            warmup_runs,
        )

        # 1. Warm-up cycles (JIT and cache priming)
        for _ in range(warmup_runs):
            warmup_batch = self.generator.generate_batch(count=100, inject_attack="ddos")
            windows = self.aggregator.aggregate_batch(warmup_batch)
            for w in windows:
                self.ensemble.evaluate(w)
                if self.ai_detector.is_fitted:
                    self.explainer.explain_window(w, top_k=3)

        # 2. Benchmark execution
        profiler = PipelineLatencyProfiler()
        total_events = 0
        overall_start = time.perf_counter()

        for i in range(iterations):
            iter_start = time.perf_counter()

            # Phase 1: Ingestion
            t0 = time.perf_counter()
            attack_type = "ddos" if i % 2 == 0 else "credential_stuffing"
            events = self.generator.generate_batch(count=batch_size, inject_attack=attack_type)
            t1 = time.perf_counter()
            profiler.record("ingestion", t1 - t0)
            total_events += len(events)

            # Phase 2: Window Aggregation
            t0 = time.perf_counter()
            windows = self.aggregator.aggregate_batch(events)
            t1 = time.perf_counter()
            profiler.record("windowing", t1 - t0)

            # Evaluate each aggregated window
            for w in windows:
                # Phase 3: Tier 1 Statistical Filter
                t0 = time.perf_counter()
                self.stat_engine.evaluate_window(w)
                t1 = time.perf_counter()
                profiler.record("tier1_statistical", t1 - t0)

                # Phase 4: Tier 2 Isolation Forest
                t0 = time.perf_counter()
                self.ai_detector.evaluate_window(w)
                t1 = time.perf_counter()
                profiler.record("tier2_isolation_forest", t1 - t0)

                # Phase 5: SHAP XAI Attribution
                t0 = time.perf_counter()
                self.explainer.explain_window(w, top_k=4)
                t1 = time.perf_counter()
                profiler.record("xai_tree_explainer", t1 - t0)

            iter_end = time.perf_counter()
            profiler.record("end_to_end", iter_end - iter_start)

        overall_duration = max(1e-4, time.perf_counter() - overall_start)
        throughput = total_events / overall_duration

        return PipelineBenchmarkReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_events_processed=total_events,
            batch_count=iterations,
            duration_seconds=round(overall_duration, 4),
            throughput_events_per_sec=round(throughput, 2),
            component_latencies=profiler.get_summary(),
            metadata={
                "batch_size": batch_size,
                "iterations": iterations,
                "warmup_runs": warmup_runs,
            },
        )


def main() -> None:
    """CLI Entry point for running performance benchmarks."""
    parser = argparse.ArgumentParser(description="Temporal Threat Detector Pipeline Latency Benchmark")
    parser.add_argument("--batch-size", type=int, default=500, help="Event batch size per iteration")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark evaluation iterations")
    parser.add_argument("--output", default=None, help="Path to save JSON benchmark report")
    args = parser.parse_args()

    runner = PipelineBenchmarkRunner()
    report = runner.run_benchmark(batch_size=args.batch_size, iterations=args.iterations)

    print("\n" + report.to_markdown() + "\n")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"[+] Saved benchmark report to: {out_path}")


if __name__ == "__main__":
    main()
