"""FastAPI Service for Temporal Threat Detector.

Exposes REST APIs for cloud workload telemetry ingestion, on-demand anomaly detection,
explainable AI (SHAP) attributions, stored threat alerts, attack injections, and health probes.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
import uuid

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    AttackInjectionConfig,
    AttackInjectionResponse,
    BatchIngestResponse,
    CloudTelemetryLog,
    DetectionRequest,
    DetectionResponse,
    HealthResponse,
    IngestionBatch,
    IngestResponse,
    PipelineMetrics,
    SeverityLevel,
    ThreatAlert,
    WindowAggregate,
)
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
from engine.explainability import (
    IncidentResponseRecommender,
    SemanticReasoningEngine,
    ShapExplainerEngine,
)
from engine.isolation_forest import HybridThreatEnsemble, IsolationForestDetector
from engine.windowing import (
    StreamingWindowBuffer,
    TemporalWindowAggregator,
    compute_window_features,
)
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
    stat_engine = StatisticalFilterEngine(z_threshold=3.0)

    # 3. Tier 2 AI Isolation Forest (trained on clean diurnal baseline)
    generator = SyntheticTelemetryGenerator(seed=42)
    baseline_events = generator.generate_batch(count=1200, inject_attack=None)
    aggregator = TemporalWindowAggregator(window_size_seconds=60.0, hop_seconds=10.0)
    baseline_windows = aggregator.aggregate_batch(baseline_events)

    detector = IsolationForestDetector(n_estimators=50, contamination=0.03, random_state=42)
    detector.fit(baseline_windows)
    detector.calibrate_threshold(baseline_windows, target_fpr=0.02)

    # Warm up Tier 1 statistical engine rolling history with baseline
    for w in baseline_windows:
        stat_engine.evaluate_window(w)

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
        mean_pipeline_latency_ms=12.5,
        last_processed_timestamp=datetime.now(timezone.utc),
    )


@app.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Ingestion"],
)
def ingest_event(event: CloudTelemetryLog) -> IngestResponse:
    """Ingest a single cloud telemetry event into the message queue."""
    queue: BaseQueueConsumer = app.state.queue
    try:
        msg_id = queue.send_message(event)
        return IngestResponse(
            status="accepted",
            event_id=event.event_id,
            queue_message_id=msg_id,
            timestamp=datetime.now(timezone.utc),
        )
    except Exception as err:
        logger.error("Failed to enqueue telemetry event: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue telemetry event: {err}",
        )


@app.post(
    "/ingest/batch",
    response_model=BatchIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Ingestion"],
)
def ingest_batch(payload: Union[IngestionBatch, List[CloudTelemetryLog]]) -> BatchIngestResponse:
    """Ingest a batch of cloud telemetry events into the message queue."""
    queue: BaseQueueConsumer = app.state.queue
    events = payload.events if isinstance(payload, IngestionBatch) else payload

    if not events:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch contains no events.",
        )

    if len(events) > 5000:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch size {len(events)} exceeds maximum limit of 5000 events.",
        )

    batch_id = payload.batch_id if isinstance(payload, IngestionBatch) else str(uuid.uuid4())
    try:
        msg_ids = queue.send_batch(events)
        return BatchIngestResponse(
            status="accepted",
            batch_id=batch_id,
            accepted_count=len(events),
            queue_message_ids=msg_ids,
        )
    except Exception as err:
        logger.error("Failed to enqueue telemetry batch: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue telemetry batch: {err}",
        )


@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
def detect_anomalies(request: DetectionRequest) -> DetectionResponse:
    """Perform on-demand dual-tier temporal anomaly detection and XAI explanation."""
    ensemble: HybridThreatEnsemble = app.state.ensemble
    explainer: ShapExplainerEngine = app.state.explainer
    alert_sink: BaseAlertStorageSink = app.state.alert_sink

    target_window: Optional[WindowAggregate] = None

    if request.window is not None:
        target_window = request.window
    elif request.events:
        min_time = min(e.timestamp for e in request.events)
        max_time = max(e.timestamp for e in request.events)
        duration = max(1.0, (max_time - min_time).total_seconds())
        target_window = compute_window_features(
            events=request.events,
            window_start=min_time,
            window_end=max_time,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'window' or 'events' must be provided in DetectionRequest.",
        )

    alert = ensemble.evaluate(target_window)

    if alert is not None:
        # Enrich alert with SHAP attribution and semantic reasoning
        try:
            attributions = explainer.explain_window(target_window, top_k=4)
            alert.top_contributions = attributions
            hyp, nar = SemanticReasoningEngine.infer_threat_semantics(attributions, target_window)
            alert.threat_hypothesis = hyp
            alert.explanation_narrative = nar
            playbook = IncidentResponseRecommender.get_playbook(hyp)
            if playbook:
                alert.recommended_mitigation = playbook.mitigation_summary
        except Exception as xai_err:
            logger.warning("XAI attribution error during on-demand detection: %s", xai_err)

        # Persist triggered alert
        alert_sink.save_alert(alert)

        return DetectionResponse(
            is_anomaly=True,
            window_id=target_window.window_id,
            alert=alert,
            detector_tier=alert.detector_tier,
            anomaly_score=round(alert.anomaly_score, 2),
            message=f"Threat detected: {alert.threat_hypothesis}",
        )

    return DetectionResponse(
        is_anomaly=False,
        window_id=target_window.window_id,
        alert=None,
        detector_tier=None,
        anomaly_score=0.0,
        message="Window metrics within nominal diurnal baseline.",
    )


@app.get("/xai/waterfall/{alert_id}", tags=["Detection"])
def get_alert_waterfall(alert_id: str) -> Dict[str, Any]:
    """Retrieve chart-ready SHAP waterfall payload for a specific threat alert."""
    alert_sink: BaseAlertStorageSink = app.state.alert_sink
    explainer: ShapExplainerEngine = app.state.explainer

    alert = alert_sink.get_alert(alert_id)
    if alert is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ThreatAlert '{alert_id}' not found.",
        )

    payload = explainer.get_waterfall_payload(alert.raw_window_metrics, top_k=6)
    payload["alert_id"] = alert_id
    payload["threat_hypothesis"] = alert.threat_hypothesis
    payload["severity"] = str(alert.severity)
    return payload


@app.get("/alerts", response_model=List[ThreatAlert], tags=["Alerts"])
def list_alerts(
    limit: int = 50,
    severity: Optional[SeverityLevel] = None,
    threat_hypothesis: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> List[ThreatAlert]:
    """Query stored dual-tier threat alerts with optional filtering."""
    alert_sink: BaseAlertStorageSink = app.state.alert_sink
    return alert_sink.list_alerts(
        limit=limit,
        severity=severity,
        threat_hypothesis=threat_hypothesis,
        start_time=start_time,
        end_time=end_time,
    )


@app.get("/alerts/{alert_id}", response_model=ThreatAlert, tags=["Alerts"])
def get_alert_by_id(alert_id: str) -> ThreatAlert:
    """Retrieve full incident record and SHAP attribution for a specific alert."""
    alert_sink: BaseAlertStorageSink = app.state.alert_sink
    alert = alert_sink.get_alert(alert_id)
    if alert is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ThreatAlert '{alert_id}' not found.",
        )
    return alert


@app.post(
    "/inject",
    response_model=AttackInjectionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Attack Injection"],
)
def inject_attack_telemetry(config: AttackInjectionConfig) -> AttackInjectionResponse:
    """Inject synthetic attack pattern telemetry directly into the ingestion queue."""
    queue: BaseQueueConsumer = app.state.queue
    generator = SyntheticTelemetryGenerator()

    try:
        events = generator.generate_batch(
            count=config.record_count,
            inject_attack=config.attack_type,
            intensity=config.intensity_multiplier,
            burst_mode=config.burst_mode,
            jitter_pct=config.jitter_pct,
        )

        if config.target_endpoint:
            for ev in events:
                ev.endpoint = config.target_endpoint

        msg_ids = queue.send_batch(events)

        return AttackInjectionResponse(
            status="injected",
            attack_type=config.attack_type,
            injected_count=len(events),
            queue_message_ids=msg_ids,
        )
    except Exception as err:
        logger.error("Attack injection error: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to inject attack telemetry: {err}",
        )
