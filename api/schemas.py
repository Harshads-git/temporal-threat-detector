"""Pydantic schemas for the Temporal Anomaly & Threat Detector pipeline.

Defines standardized contracts for:
- Raw cloud telemetry events (CloudTrail / ALB / VPC Flow inspired)
- Temporal rolling window aggregated features
- Threat detection alerts with Explainable AI (SHAP) attributions
- Attack injection commands and ingestion batches
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import uuid


class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class UserAgentCategory(str, Enum):
    BROWSER = "BROWSER"
    CLI_SDK = "CLI_SDK"
    AUTOMATED_BOT = "AUTOMATED_BOT"
    MOBILE_APP = "MOBILE_APP"
    UNKNOWN = "UNKNOWN"


class AttackType(str, Enum):
    BASELINE_NORMAL = "BASELINE_NORMAL"
    DDOS_HTTP_FLOOD = "DDOS_HTTP_FLOOD"
    CREDENTIAL_STUFFING = "CREDENTIAL_STUFFING"
    OFF_HOUR_PROBING = "OFF_HOUR_PROBING"
    DATA_EXFILTRATION = "DATA_EXFILTRATION"


class DetectorTier(str, Enum):
    TIER1_STATISTICAL_ZSCORE = "TIER1_STATISTICAL_ZSCORE"
    TIER1_STATISTICAL_EWMA = "TIER1_STATISTICAL_EWMA"
    TIER2_UNSUPERVISED_IFOREST = "TIER2_UNSUPERVISED_IFOREST"
    HYBRID_ENSEMBLE = "HYBRID_ENSEMBLE"


class SeverityLevel(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CloudTelemetryLog(BaseModel):
    """Normalized cloud workload access & telemetry event record."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Globally unique identifier for the event",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the event occurred",
    )
    source_ip: str = Field(
        ...,
        examples=["198.51.100.45", "10.0.4.12"],
        description="Originating IP address of the request",
    )
    user_identity: str = Field(
        ...,
        examples=["arn:aws:iam::123456789012:user/alice", "service-role/worker-svc"],
        description="IAM identity, API client, or token subject",
    )
    service_name: str = Field(
        default="cloud-ingress-api",
        description="Target microservice or cloud resource name",
    )
    endpoint: str = Field(
        ...,
        examples=["/api/v1/auth/login", "/api/v1/data/export", "/admin/debug"],
        description="API path or resource method accessed",
    )
    http_method: HttpMethod = Field(
        default=HttpMethod.GET,
        description="HTTP method used in the request",
    )
    status_code: int = Field(
        default=200,
        ge=100,
        le=599,
        description="HTTP response status code",
    )
    request_bytes: int = Field(
        default=512,
        ge=0,
        description="Size of incoming payload in bytes",
    )
    response_bytes: int = Field(
        default=2048,
        ge=0,
        description="Size of response payload in bytes",
    )
    duration_ms: float = Field(
        default=45.0,
        ge=0.0,
        description="Server processing latency in milliseconds",
    )
    geo_country: str = Field(
        default="US",
        description="ISO 2-letter country code resolved from IP",
    )
    user_agent_category: UserAgentCategory = Field(
        default=UserAgentCategory.BROWSER,
        description="Categorized client user agent",
    )
    error_code: Optional[str] = Field(
        default=None,
        description="Cloud or application error code (e.g. AccessDenied, RateLimited)",
    )
    is_synthetic_attack: bool = Field(
        default=False,
        description="Flag indicating if record was synthetically injected (for ground-truth eval)",
    )
    attack_label: Optional[AttackType] = Field(
        default=None,
        description="Specific injected attack type for verification benchmarks",
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, value):
        if isinstance(value, str):
            # Parse ISO string
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value


class WindowAggregate(BaseModel):
    """Aggregated temporal feature vector across a discrete rolling/tumbling window."""

    window_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Short identifier for the temporal window",
    )
    window_start: datetime = Field(..., description="UTC start time of the window")
    window_end: datetime = Field(..., description="UTC end time of the window")
    duration_seconds: float = Field(..., description="Window duration in seconds")

    # Volumetric and rate metrics
    request_count: int = Field(..., ge=0, description="Total requests within window")
    request_rate_per_sec: float = Field(..., ge=0.0, description="Requests per second")
    error_count_4xx: int = Field(..., ge=0, description="Count of 4xx client errors")
    error_count_5xx: int = Field(..., ge=0, description="Count of 5xx server errors")
    error_rate_4xx: float = Field(..., ge=0.0, le=1.0, description="Ratio of 4xx responses")
    error_rate_5xx: float = Field(..., ge=0.0, le=1.0, description="Ratio of 5xx responses")

    # Latency metrics
    mean_duration_ms: float = Field(..., ge=0.0, description="Mean response latency (ms)")
    p95_duration_ms: float = Field(..., ge=0.0, description="95th percentile latency (ms)")

    # Cardinality & Identity metrics
    unique_source_ips: int = Field(..., ge=0, description="Distinct source IP addresses")
    unique_users: int = Field(..., ge=0, description="Distinct authenticated users")

    # Byte throughput
    total_request_bytes: int = Field(..., ge=0, description="Sum of request payload bytes")
    total_response_bytes: int = Field(..., ge=0, description="Sum of response payload bytes")
    mean_response_bytes: float = Field(..., ge=0.0, description="Mean bytes per response")

    # Contextual behavioral signals
    auth_failure_count: int = Field(
        ..., ge=0, description="Count of 401/403 authorization failures"
    )
    off_hour_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Proportion of events during off-hours (00:00-06:00)"
    )
    bot_agent_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Ratio of automated bot user-agents"
    )

    def to_feature_vector(self) -> Dict[str, float]:
        """Convert into canonical numeric feature map for ML / Statistical engines."""
        return {
            "request_count": float(self.request_count),
            "request_rate_per_sec": float(self.request_rate_per_sec),
            "error_rate_4xx": float(self.error_rate_4xx),
            "error_rate_5xx": float(self.error_rate_5xx),
            "mean_duration_ms": float(self.mean_duration_ms),
            "p95_duration_ms": float(self.p95_duration_ms),
            "unique_source_ips": float(self.unique_source_ips),
            "unique_users": float(self.unique_users),
            "total_response_bytes": float(self.total_response_bytes),
            "mean_response_bytes": float(self.mean_response_bytes),
            "auth_failure_count": float(self.auth_failure_count),
            "off_hour_ratio": float(self.off_hour_ratio),
            "bot_agent_ratio": float(self.bot_agent_ratio),
        }


class FeatureAttribution(BaseModel):
    """Explainable AI (SHAP) feature contribution breakdown."""

    feature_name: str = Field(..., description="Name of the evaluated metric")
    observed_value: float = Field(..., description="Actual observed metric value in this window")
    baseline_expected: float = Field(..., description="Baseline expected value from historical distribution")
    shap_value: float = Field(
        ...,
        description="Signed SHAP attribution (positive values push towards anomaly classification)",
    )
    importance_rank: int = Field(..., ge=1, description="Rank ordered by absolute importance")


class ThreatAlert(BaseModel):
    """Dual-tier detection alert record accompanied by explainable AI reasoning."""

    alert_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique alert identifier",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC time alert was triggered",
    )
    window_id: str = Field(..., description="Correlated temporal window ID")
    detector_tier: DetectorTier = Field(
        ..., description="Which detection tier or ensemble flagged the incident"
    )
    severity: SeverityLevel = Field(..., description="Computed incident triage severity")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in anomaly assessment"
    )
    anomaly_score: float = Field(
        ..., description="Raw algorithmic anomaly score (Z-score deviation or negative IF score)"
    )
    decision_threshold: float = Field(
        ..., description="Threshold exceeded to trigger detection"
    )

    # Explainability & Semantic Reasoning
    top_contributions: List[FeatureAttribution] = Field(
        default_factory=list,
        description="Rank-ordered SHAP feature importance attributions",
    )
    threat_hypothesis: str = Field(
        ...,
        examples=[
            "High Volumetric HTTP Flood (DDoS)",
            "Distributed Credential Stuffing Campaign",
            "Off-Hours Reconnaissance Probing",
            "Privileged Exfiltration Surge",
        ],
        description="Inferred attack category based on feature attribution dominance",
    )
    explanation_narrative: str = Field(
        ...,
        description="Human-readable plain English justification for faculty & SOC analysts",
    )
    recommended_mitigation: str = Field(
        ...,
        description="Actionable cloud incident response playbook suggestion",
    )
    raw_window_metrics: WindowAggregate = Field(
        ...,
        description="Underlying raw window metric snapshot for drill-down investigation",
    )


class AttackInjectionConfig(BaseModel):
    """Payload to instruct synthetic telemetry generator to inject attack patterns."""

    attack_type: AttackType = Field(
        default=AttackType.DDOS_HTTP_FLOOD,
        description="Type of threat pattern to inject",
    )
    record_count: int = Field(
        default=200, ge=1, le=5000, description="Number of synthetic attack events to inject"
    )
    intensity_multiplier: float = Field(
        default=3.5,
        ge=1.0,
        le=50.0,
        description="Volume/anomaly magnitude amplification factor",
    )
    target_endpoint: Optional[str] = Field(
        default=None,
        description="Specific target API endpoint override (e.g. /api/v1/auth/login)",
    )


class IngestionBatch(BaseModel):
    """Batch container for bulk telemetry transmission."""

    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str = Field(default="synthetic-telemetry-engine")
    events: List[CloudTelemetryLog] = Field(..., min_length=1)


class PipelineMetrics(BaseModel):
    """Real-time pipeline monitoring statistics."""

    total_events_ingested: int
    total_windows_processed: int
    total_alerts_emitted: int
    tier1_zscore_alerts: int
    tier2_iforest_alerts: int
    hybrid_alerts: int
    mean_pipeline_latency_ms: float
    last_processed_timestamp: Optional[datetime] = None
