"""Unified Cloud Environment Configuration and Resource Orchestrator.

Manages dynamic detection of LocalStack, live AWS, or local zero-cost in-memory
drivers for seamless vivas, offline presentations, and cloud deployments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional
import urllib.request

from cloud.provisioner import CloudResourceProvisioner
from cloud.queue_consumer import (
    BaseQueueConsumer,
    InMemoryQueueConsumer,
    SQSQueueConsumer,
    get_queue_consumer,
)
from cloud.storage_sink import (
    BaseAlertStorageSink,
    BaseTelemetryArchiver,
    DynamoDBAlertSink,
    InMemoryAlertSink,
    InMemoryTelemetryArchiver,
    LocalFileTelemetryArchiver,
    LocalJsonAlertSink,
    S3TelemetryArchiver,
    get_alert_sink,
    get_telemetry_archiver,
)

logger = logging.getLogger("cloud.config")


DriverMode = Literal["auto", "in-memory", "local", "aws", "localstack"]


def is_endpoint_reachable(endpoint_url: str, timeout_sec: float = 0.5) -> bool:
    """Test if a local or remote HTTP endpoint (e.g., LocalStack port 4566) is responding."""
    try:
        req = urllib.request.Request(
            f"{endpoint_url.rstrip('/')}/_localstack/health",
            headers={"User-Agent": "TemporalThreatDetector/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            return response.status in (200, 404)
    except Exception:
        # Fallback probe to root
        try:
            with urllib.request.urlopen(endpoint_url, timeout=timeout_sec) as response:
                return response.status in (200, 404, 403)
        except Exception:
            return False


def detect_active_environment() -> DriverMode:
    """Detect whether LocalStack, AWS, or in-memory emulation should be activated."""
    explicit_driver = os.getenv("TELEMETRY_DRIVER_MODE", "").lower()
    if explicit_driver in ("in-memory", "local", "aws", "localstack"):
        return explicit_driver  # type: ignore

    # Check for LocalStack endpoint environment variable
    custom_endpoint = os.getenv("AWS_ENDPOINT_URL") or os.getenv("LOCALSTACK_URL")
    if custom_endpoint and is_endpoint_reachable(custom_endpoint):
        logger.info("Detected responsive LocalStack endpoint at %s", custom_endpoint)
        return "localstack"

    # Check default localhost LocalStack port 4566
    if is_endpoint_reachable("http://localhost:4566"):
        logger.info("Detected local LocalStack instance on port 4566")
        return "localstack"

    # Check for live AWS credentials
    has_aws_keys = bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
    is_real_aws = has_aws_keys and not custom_endpoint and not os.getenv("AWS_ACCESS_KEY_ID", "").startswith("test")
    if is_real_aws:
        logger.info("Detected live AWS credentials in environment")
        return "aws"

    # Default to local zero-cost in-memory mode for offline vivas and fast tests
    return "in-memory"


@dataclass
class CloudSettings:
    """Configuration container for cloud queues, tables, buckets, and endpoints."""

    driver_mode: DriverMode = field(default_factory=detect_active_environment)
    aws_region: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    aws_endpoint_url: Optional[str] = os.getenv("AWS_ENDPOINT_URL")
    aws_access_key_id: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")

    # Resource Identifiers
    queue_name: str = os.getenv("TELEMETRY_QUEUE_NAME", "cloud-workload-telemetry-queue")
    dlq_name: str = os.getenv("TELEMETRY_DLQ_NAME", "cloud-workload-telemetry-queue-dlq")
    table_name: str = os.getenv("ALERTS_TABLE_NAME", "temporal-threat-alerts")
    bucket_name: str = os.getenv("TELEMETRY_BUCKET_NAME", "temporal-threat-telemetry-archive")

    # Local Persistence Fallbacks
    local_storage_dir: Path = field(default_factory=lambda: Path("data"))

    def __post_init__(self) -> None:
        """Normalize endpoint and keys for LocalStack if active."""
        if self.driver_mode == "localstack":
            if not self.aws_endpoint_url:
                self.aws_endpoint_url = "http://localhost:4566"
            if not self.aws_access_key_id:
                self.aws_access_key_id = "test"
            if not self.aws_secret_access_key:
                self.aws_secret_access_key = "test"


@dataclass
class CloudInfrastructureStack:
    """Active runtime drivers and provisioner instance."""

    queue_consumer: BaseQueueConsumer
    alert_sink: BaseAlertStorageSink
    archiver: BaseTelemetryArchiver
    provisioner: Optional[CloudResourceProvisioner]
    settings: CloudSettings

    @property
    def is_cloud_active(self) -> bool:
        """True if backed by live AWS or LocalStack Boto3 services."""
        return self.settings.driver_mode in ("aws", "localstack")

    def to_status_dict(self) -> Dict[str, Any]:
        """Summarize active stack components for telemetry health checks."""
        return {
            "driver_mode": self.settings.driver_mode,
            "is_cloud_active": self.is_cloud_active,
            "region": self.settings.aws_region,
            "endpoint_url": self.settings.aws_endpoint_url,
            "queue_name": self.settings.queue_name,
            "table_name": self.settings.table_name,
            "bucket_name": self.settings.bucket_name,
        }


def build_cloud_stack(
    settings: Optional[CloudSettings] = None,
    auto_provision: bool = True,
) -> CloudInfrastructureStack:
    """Build and wire the entire cloud infrastructure stack according to active environment."""
    cfg = settings or CloudSettings()
    logger.info("Building Cloud Infrastructure Stack with driver mode: %s", cfg.driver_mode)

    provisioner: Optional[CloudResourceProvisioner] = None

    if cfg.driver_mode in ("aws", "localstack"):
        provisioner = CloudResourceProvisioner(
            endpoint_url=cfg.aws_endpoint_url,
            region_name=cfg.aws_region,
            aws_access_key_id=cfg.aws_access_key_id,
            aws_secret_access_key=cfg.aws_secret_access_key,
        )
        if auto_provision:
            try:
                provisioner.provision_all(
                    queue_name=cfg.queue_name,
                    dlq_name=cfg.dlq_name,
                    table_name=cfg.table_name,
                    bucket_name=cfg.bucket_name,
                )
            except Exception as prov_err:
                logger.warning("Could not auto-provision cloud resources: %s", prov_err)

        queue = SQSQueueConsumer(
            queue_name=cfg.queue_name,
            dlq_name=cfg.dlq_name,
            endpoint_url=cfg.aws_endpoint_url,
            region_name=cfg.aws_region,
            aws_access_key_id=cfg.aws_access_key_id,
            aws_secret_access_key=cfg.aws_secret_access_key,
        )
        alert_sink = DynamoDBAlertSink(
            table_name=cfg.table_name,
            endpoint_url=cfg.aws_endpoint_url,
            region_name=cfg.aws_region,
            aws_access_key_id=cfg.aws_access_key_id,
            aws_secret_access_key=cfg.aws_secret_access_key,
        )
        archiver = S3TelemetryArchiver(
            bucket_name=cfg.bucket_name,
            endpoint_url=cfg.aws_endpoint_url,
            region_name=cfg.aws_region,
            aws_access_key_id=cfg.aws_access_key_id,
            aws_secret_access_key=cfg.aws_secret_access_key,
        )
    elif cfg.driver_mode == "local":
        queue = InMemoryQueueConsumer()
        alert_sink = LocalJsonAlertSink(filepath=str(cfg.local_storage_dir / "threat_alerts.jsonl"))
        archiver = LocalFileTelemetryArchiver(base_dir=str(cfg.local_storage_dir / "raw"))
    else:  # "in-memory" or fallback
        queue = InMemoryQueueConsumer()
        alert_sink = InMemoryAlertSink()
        archiver = InMemoryTelemetryArchiver()

    return CloudInfrastructureStack(
        queue_consumer=queue,
        alert_sink=alert_sink,
        archiver=archiver,
        provisioner=provisioner,
        settings=cfg,
    )
