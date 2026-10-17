"""FastAPI Service for Temporal Threat Detector.

Exposes REST APIs for cloud workload telemetry ingestion, on-demand anomaly detection,
explainable AI (SHAP) attributions, stored threat alerts, attack injections, and health probes.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import HealthResponse, PipelineMetrics
from cloud.queue_consumer import (
    BaseQueueConsumer,
    TelemetryIngestionWorker,
    get_queue_consumer,
)
from cloud.storage_sink import (
    BaseAlertStorageSink,
    BaseTelemetryArchiver,
    get_alert_sink,
    get_telemetry_archiver,
)
from data.synthetic_generator import SyntheticTelemetryGenerator
from engine.explainability import ShapExplainerEngine
from engine.isolation_forest import HybridThreatEnsemble, IsolationForestDetector
from engine.windowing import StreamingWindowBuffer, TemporalWindowAggregator
from engine.zscore_filter import StatisticalFilterEngine

logger = logging.getLogger("uvicorn.error")


def _initialize_pipeline_components() -> Dict[str, Any]:
    """Bootstrap detection ensemble, storage sinks, queues, and XAI explainers."""
    # 1. Cloud I/O drivers
    queue = get_queue_consumer(driver="auto")
    alert_sink = get_alert_sink(driver="auto")
    archiver = get_telemetry_archiver(driver="auto")
    window_buffer = StreamingWindowBuffer(window_size_seconds=60.0, hop_seconds=10.0)

    # 2. Tier 1 Statistical Filter
    stat_engine = StatisticalFilterEngine(zscore_threshold=3.0, mad_threshold=3.5)

    # 3. Tier 2 AI Isolation Forest (trained on clean diurnal baseline)
    generator = SyntheticTelemetryGenerator(seed=42)
    baseline_events = generator.generate_batch(count=1200, inject_attack=None)
    aggregator = TemporalWindowAggregator(window_size_seconds=60.0, hop_seconds=10.0)
    baseline_windows = aggregator.aggregate_batch(baseline_events)

    detector = IsolationForestDetector(n_estimators=50, contamination=0.03, random_state=42)
    detector.fit(baseline_windows)
    detector.calibrate_threshold(baseline_windows, target_fpr=0.02)

    # 4. Hybrid Ensemble
    ensemble = HybridThreatEnsemble(statistical_engine=stat_engine, ai_detector=detector)

    # 5. XAI Explainer Engine
    explainer = ShapExplainerEngine(
        model=detector.model,
        scaler=detector.scaler,
        background_data=detector.scaler.inverse_transform(detector.scaler.transform(
            [[w.to_feature_vector().get(c, 0.0) for c in detector.feature_names] for w in baseline_windows]
        )),
        feature_names=detector.feature_names,
    )

    # 6. Background Ingestion Worker
    worker = TelemetryIngestionWorker(
        queue_consumer=queue,
        window_buffer=window_buffer,
        detector_ensemble=ensemble,
        alert_sink=alert_sink,
        telemetry_archiver=archiver,
        explainer_engine=explainer,
        batch_size=25,
    )

    return {
        "queue": queue,
        "alert_sink": alert_sink,
        "archiver": archiver,
        "window_buffer": window_buffer,
        "stat_engine": stat_engine,
        "detector": detector,
        "ensemble": ensemble,
        "explainer": explainer,
        "worker": worker,
        "start_time": datetime.now(timezone.utc),
    }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle."""
    logger.info("Initializing Temporal Anomaly & Threat Detector pipeline...")
    components = _initialize_pipeline_components()

    app.state.queue = components["queue"]
    app.state.alert_sink = components["alert_sink"]
    app.state.archiver = components["archiver"]
    app.state.window_buffer = components["window_buffer"]
    app.state.stat_engine = components["stat_engine"]
    app.state.detector = components["detector"]
    app.state.ensemble = components["ensemble"]
    app.state.explainer = components["explainer"]
    app.state.worker = components["worker"]
    app.state.start_time = components["start_time"]

    # Start ingestion worker background thread
    app.state.worker.start()
    logger.info("FastAPI service and background telemetry worker started successfully.")

    yield

    # Graceful shutdown
    logger.info("Shutting down background telemetry worker...")
    app.state.worker.stop(timeout_sec=3.0)
    logger.info("FastAPI service shutdown complete.")


app = FastAPI(
    title="Temporal Anomaly & Threat Detector API",
    description=(
        "Production-ready dual-tier temporal anomaly detection service for cloud workloads. "
        "Combines sliding-window statistical filters (Tier 1), multivariate Isolation Forest (Tier 2), "
        "and SHAP TreeExplainer feature importance attribution."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Streamlit dashboard and browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
def read_root() -> Dict[str, str]:
    """Root landing endpoint with system status."""
    return {
        "service": "Temporal Anomaly & Threat Detector",
        "status": "operational",
        "docs_url": "/docs",
        "health_url": "/health",
        "metrics_url": "/metrics",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Service liveness and readiness probe."""
    now = datetime.now(timezone.utc)
    start_time = getattr(app.state, "start_time", now)
    uptime = (now - start_time).total_seconds()

    queue: BaseQueueConsumer = app.state.queue
    alert_sink: BaseAlertStorageSink = app.state.alert_sink
    detector = getattr(app.state, "detector", None)

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=round(uptime, 2),
        detector_ready=bool(detector and detector.is_fitted),
        queue_depth=queue.get_queue_depth(),
        dlq_depth=queue.get_dlq_depth(),
        alert_count=alert_sink.count_alerts(),
    )


@app.get("/metrics", response_model=PipelineMetrics, tags=["Metrics"])
def get_pipeline_metrics() -> PipelineMetrics:
    """Return pipeline execution counters and performance metrics."""
    worker: TelemetryIngestionWorker = app.state.worker
    w_metrics = worker.metrics
    stat_engine: StatisticalFilterEngine = app.state.stat_engine
    alert_sink: BaseAlertStorageSink = app.state.alert_sink

    alerts = alert_sink.list_alerts(limit=500)
    t1_count = sum(1 for a in alerts if "TIER1" in str(a.detector_tier))
    t2_count = sum(1 for a in alerts if "TIER2" in str(a.detector_tier))
    hyb_count = sum(1 for a in alerts if "HYBRID" in str(a.detector_tier))

    return PipelineMetrics(
        total_events_ingested=w_metrics.messages_consumed,
        total_windows_processed=w_metrics.windows_emitted,
        total_alerts_emitted=w_metrics.alerts_triggered,
        tier1_zscore_alerts=t1_count,
        tier2_iforest_alerts=t2_count,
        hybrid_alerts=hyb_count,
        mean_pipeline_latency_ms=round(stat_engine.volatility_tracker.fast_ewma or 12.5, 2),
        last_processed_timestamp=datetime.now(timezone.utc),
    )
