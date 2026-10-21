"""Threat detection accuracy benchmark across multi-vector cyber attack scenarios.

Evaluates and compares:
1. Tier 1 Statistical Filter alone (Z-Score / MAD)
2. Tier 2 Multivariate Isolation Forest alone
3. Hybrid Dual-Tier Threat Ensemble

Computes Precision, Recall, F1-Score, and False Positive Rate (FPR) across:
- Clean Diurnal Baseline
- Volumetric HTTP Flood (DDoS)
- Credential Stuffing (Auth Failures)
- Off-Hour Reconnaissance Probing
- Large-Scale Data Exfiltration
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from api.schemas import ThreatAlert, WindowAggregate
from data.synthetic_generator import SyntheticTelemetryGenerator
from engine.isolation_forest import HybridThreatEnsemble, IsolationForestDetector
from engine.windowing import TemporalWindowAggregator
from engine.zscore_filter import StatisticalFilterEngine

logger = logging.getLogger("benchmarks.accuracy")


@dataclass
class DetectionAccuracyMetrics:
    """Confusion matrix and classification performance metrics."""

    detector_name: str
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_samples: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float

    @classmethod
    def compute(
        cls,
        detector_name: str,
        y_true: List[bool],
        y_pred: List[bool],
    ) -> DetectionAccuracyMetrics:
        """Calculate binary classification performance metrics."""
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)
        total = len(y_true)

        accuracy = round((tp + tn) / max(1, total), 4)
        precision = round(tp / max(1, tp + fp), 4)
        recall = round(tp / max(1, tp + fn), 4)
        f1 = round((2 * precision * recall) / max(1e-4, precision + recall), 4)
        fpr = round(fp / max(1, fp + tn), 4)

        return cls(
            detector_name=detector_name,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            total_samples=total,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            false_positive_rate=fpr,
        )


@dataclass
class AccuracyBenchmarkReport:
    """Aggregated detection accuracy benchmark results."""

    timestamp: str
    sample_counts: Dict[str, int]
    overall_metrics: Dict[str, DetectionAccuracyMetrics]
    per_vector_recall: Dict[str, Dict[str, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return asdict(self)

    def to_markdown(self) -> str:
        """Format accuracy matrix into GitHub-flavored Markdown tables."""
        lines = [
            "# Threat Detection Accuracy & Architectural Comparison Benchmark",
            f"**Timestamp:** `{self.timestamp}` | **Total Evaluation Windows:** `{sum(self.sample_counts.values())}`\n",
            "### 1. Overall Architectural Comparison",
            "| Detector Architecture | Accuracy | Precision | Recall | F1-Score | FPR | TP | FP | FN |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for name, m in self.overall_metrics.items():
            lines.append(
                f"| **{m.detector_name}** | {m.accuracy:.1%} | {m.precision:.1%} | {m.recall:.1%} | **{m.f1_score:.3f}** | {m.false_positive_rate:.1%} | {m.true_positives} | {m.false_positives} | {m.false_negatives} |"
            )

        lines.extend([
            "\n### 2. Threat Vector Recall Breakdown (%)",
            "| Threat Attack Vector | Tier 1 (MAD/EWMA) | Tier 2 (iForest) | Hybrid Ensemble |",
            "| :--- | :--- | :--- | :--- |",
        ])

        vectors = ["ddos", "credential_stuffing", "off_hour_probing", "data_exfiltration"]
        vector_labels = {
            "ddos": "High Volumetric HTTP Flood (DDoS)",
            "credential_stuffing": "Credential Stuffing & Auth Failure",
            "off_hour_probing": "Off-Hour Reconnaissance Probing",
            "data_exfiltration": "Massive Data Exfiltration",
        }

        for vec in vectors:
            v_name = vector_labels.get(vec, vec)
            t1_r = self.per_vector_recall.get(vec, {}).get("tier1_statistical", 0.0)
            t2_r = self.per_vector_recall.get(vec, {}).get("tier2_iforest", 0.0)
            hyb_r = self.per_vector_recall.get(vec, {}).get("hybrid_ensemble", 0.0)
            lines.append(f"| {v_name} | {t1_r:.1%} | {t2_r:.1%} | **{hyb_r:.1%}** |")

        return "\n".join(lines)


class DetectionAccuracyBenchmarkRunner:
    """Evaluates multi-vector detection accuracy and FPR across architectures."""

    def __init__(self, seed: int = 42) -> None:
        self.generator = SyntheticTelemetryGenerator(seed=seed)
        self.aggregator = TemporalWindowAggregator(window_size_seconds=60.0, hop_seconds=10.0)

        # Baseline training for Isolation Forest
        baseline_events = self.generator.generate_batch(count=1500, inject_attack=None)
        self.baseline_windows = self.aggregator.aggregate_batch(baseline_events)

        self.stat_engine = StatisticalFilterEngine(z_threshold=3.0)
        for w in self.baseline_windows:
            self.stat_engine.evaluate_window(w)

        self.ai_detector = IsolationForestDetector(n_estimators=50, contamination=0.03, random_state=seed)
        self.ai_detector.fit(self.baseline_windows)
        self.ai_detector.calibrate_threshold(self.baseline_windows, target_fpr=0.02)

        self.ensemble = HybridThreatEnsemble(
            statistical_engine=self.stat_engine,
            ai_detector=self.ai_detector,
        )

    def run_benchmark(self, events_per_scenario: int = 400) -> AccuracyBenchmarkReport:
        """Execute accuracy benchmark across normal baseline and all 4 attack vectors."""
        logger.info("Executing detection accuracy benchmark across multi-vector scenarios...")

        # 1. Generate test scenarios
        scenarios: Dict[str, Tuple[List[WindowAggregate], bool]] = {}

        # Scenario A: Clean Normal Windows (Ground truth: False)
        clean_events = self.generator.generate_batch(count=events_per_scenario * 2, inject_attack=None)
        clean_windows = self.aggregator.aggregate_batch(clean_events)
        scenarios["normal"] = (clean_windows, False)

        # Scenarios B-E: Injected Attacks (Ground truth: True)
        for attack in ("ddos", "credential_stuffing", "off_hour_probing", "data_exfiltration"):
            events = self.generator.generate_batch(count=events_per_scenario, inject_attack=attack)
            windows = self.aggregator.aggregate_batch(events)
            scenarios[attack] = (windows, True)

        # 2. Evaluate across all detectors
        y_true_all: List[bool] = []
        y_pred_t1: List[bool] = []
        y_pred_t2: List[bool] = []
        y_pred_hyb: List[bool] = []

        vector_recalls: Dict[str, Dict[str, float]] = {}
        sample_counts: Dict[str, int] = {}

        for scenario_name, (windows, is_attack) in scenarios.items():
            sample_counts[scenario_name] = len(windows)
            t1_detected = 0
            t2_detected = 0
            hyb_detected = 0

            for w in windows:
                # Tier 1
                t1_alert = self.stat_engine.evaluate_window(w)
                t1_flag = t1_alert is not None

                # Tier 2
                t2_res = self.ai_detector.evaluate_window(w)
                t2_flag = t2_res.is_anomaly if t2_res else False

                # Hybrid Ensemble
                hyb_alert = self.ensemble.evaluate(w)
                hyb_flag = hyb_alert is not None

                y_true_all.append(is_attack)
                y_pred_t1.append(t1_flag)
                y_pred_t2.append(t2_flag)
                y_pred_hyb.append(hyb_flag)

                if is_attack:
                    if t1_flag:
                        t1_detected += 1
                    if t2_flag:
                        t2_detected += 1
                    if hyb_flag:
                        hyb_detected += 1

            if is_attack and len(windows) > 0:
                vector_recalls[scenario_name] = {
                    "tier1_statistical": round(t1_detected / len(windows), 4),
                    "tier2_iforest": round(t2_detected / len(windows), 4),
                    "hybrid_ensemble": round(hyb_detected / len(windows), 4),
                }

        # 3. Compute metrics
        overall = {
            "Tier 1 (Statistical Filter)": DetectionAccuracyMetrics.compute(
                "Tier 1 (Statistical Filter)", y_true_all, y_pred_t1
            ),
            "Tier 2 (Isolation Forest AI)": DetectionAccuracyMetrics.compute(
                "Tier 2 (Isolation Forest AI)", y_true_all, y_pred_t2
            ),
            "Hybrid Threat Ensemble": DetectionAccuracyMetrics.compute(
                "Hybrid Threat Ensemble", y_true_all, y_pred_hyb
            ),
        }

        return AccuracyBenchmarkReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sample_counts=sample_counts,
            overall_metrics=overall,
            per_vector_recall=vector_recalls,
            metadata={"events_per_scenario": events_per_scenario},
        )


def main() -> None:
    """CLI Entry point for detection accuracy benchmarking."""
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Temporal Threat Detector Accuracy Evaluation")
    parser.add_argument("--events", type=int, default=400, help="Events per evaluation scenario")
    parser.add_argument("--output", default=None, help="Path to save JSON accuracy report")
    args = parser.parse_args()

    runner = DetectionAccuracyBenchmarkRunner()
    report = runner.run_benchmark(events_per_scenario=args.events)

    print("\n" + report.to_markdown() + "\n")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"[+] Saved accuracy report to: {out_path}")


if __name__ == "__main__":
    main()
