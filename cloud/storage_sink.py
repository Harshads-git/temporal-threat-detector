"""Pluggable cloud and local storage sinks for ThreatAlert incident records.

Supports:
1. InMemoryAlertSink: Ultra-fast thread-safe storage for tests and local runtime.
2. LocalJsonAlertSink: Durable append-only JSONL file storage with directory rotation.
3. DynamoDBAlertSink: AWS DynamoDB driver supporting LocalStack or live AWS with auto-table provisioning.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
import json
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Union

import boto3
from botocore.exceptions import ClientError

from api.schemas import SeverityLevel, ThreatAlert

logger = logging.getLogger(__name__)


def _convert_floats_to_decimals(obj: Any) -> Any:
    """Recursively convert float values to Decimal for Boto3 DynamoDB compatibility."""
    if isinstance(obj, float):
        # Convert through string to prevent precision artifacts (e.g. 0.1 -> 0.10000000000000000555)
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _convert_floats_to_decimals(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_floats_to_decimals(v) for v in obj]
    return obj


def _convert_decimals_to_floats(obj: Any) -> Any:
    """Recursively convert Decimal values back to float/int for Pydantic parsing."""
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    if isinstance(obj, dict):
        return {k: _convert_decimals_to_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_decimals_to_floats(v) for v in obj]
    return obj


class BaseAlertStorageSink(ABC):
    """Abstract interface for storing and querying ThreatAlert incident records."""

    @abstractmethod
    def save_alert(self, alert: ThreatAlert) -> bool:
        """Persist a single threat alert.

        Args:
            alert: Validated ThreatAlert instance.

        Returns:
            True if persistence succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def save_alerts(self, alerts: List[ThreatAlert]) -> int:
        """Batch persist multiple threat alerts.

        Args:
            alerts: List of ThreatAlert instances.

        Returns:
            Count of successfully stored alerts.
        """
        pass

    @abstractmethod
    def get_alert(self, alert_id: str) -> Optional[ThreatAlert]:
        """Fetch alert by unique ID.

        Args:
            alert_id: UUID string of the alert.

        Returns:
            ThreatAlert instance if found, None otherwise.
        """
        pass

    @abstractmethod
    def list_alerts(
        self,
        limit: int = 50,
        severity: Optional[SeverityLevel] = None,
        threat_hypothesis: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[ThreatAlert]:
        """List alerts matching optional filter criteria, sorted by timestamp descending.

        Args:
            limit: Maximum records to return.
            severity: Optional filter by SeverityLevel.
            threat_hypothesis: Optional filter by hypothesis string.
            start_time: Filter alerts occurring at or after start_time.
            end_time: Filter alerts occurring at or before end_time.

        Returns:
            List of matching ThreatAlert instances.
        """
        pass

    @abstractmethod
    def count_alerts(self) -> int:
        """Return total number of stored alerts."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Purge all stored alerts (primarily for testing and resets)."""
        pass


class InMemoryAlertSink(BaseAlertStorageSink):
    """Thread-safe in-memory alert sink for fast local testing and ephemeral deployments."""

    def __init__(self, max_capacity: int = 10000):
        self.max_capacity = max_capacity
        self._alerts: Dict[str, ThreatAlert] = {}
        self._lock = threading.RLock()

    def save_alert(self, alert: ThreatAlert) -> bool:
        with self._lock:
            if len(self._alerts) >= self.max_capacity and alert.alert_id not in self._alerts:
                # Evict oldest alert
                oldest_key = next(iter(self._alerts))
                del self._alerts[oldest_key]
            self._alerts[alert.alert_id] = alert
            return True

    def save_alerts(self, alerts: List[ThreatAlert]) -> int:
        count = 0
        with self._lock:
            for alert in alerts:
                if self.save_alert(alert):
                    count += 1
        return count

    def get_alert(self, alert_id: str) -> Optional[ThreatAlert]:
        with self._lock:
            return self._alerts.get(alert_id)

    def list_alerts(
        self,
        limit: int = 50,
        severity: Optional[SeverityLevel] = None,
        threat_hypothesis: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[ThreatAlert]:
        with self._lock:
            alerts = list(self._alerts.values())

        # Sort descending by timestamp
        alerts.sort(key=lambda a: a.timestamp, reverse=True)

        filtered: List[ThreatAlert] = []
        for alert in alerts:
            if severity is not None and alert.severity != severity:
                continue
            if threat_hypothesis is not None and alert.threat_hypothesis != threat_hypothesis:
                continue
            if start_time is not None and alert.timestamp < start_time:
                continue
            if end_time is not None and alert.timestamp > end_time:
                continue

            filtered.append(alert)
            if len(filtered) >= limit:
                break

        return filtered

    def count_alerts(self) -> int:
        with self._lock:
            return len(self._alerts)

    def clear(self) -> None:
        with self._lock:
            self._alerts.clear()


class LocalJsonAlertSink(BaseAlertStorageSink):
    """Durable append-only JSONL file sink for local audit logs and edge deployments."""

    def __init__(self, file_path: Union[str, Path] = "data/alerts/threat_alerts.jsonl"):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._index: Dict[str, ThreatAlert] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Scan existing JSONL file into memory index if file exists."""
        if not self.file_path.exists():
            return

        with self._lock:
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            alert = ThreatAlert.model_validate(data)
                            self._index[alert.alert_id] = alert
                        except Exception as err:
                            logger.warning("Corrupt JSON line %d in %s: %s", line_num, self.file_path, err)
            except Exception as err:
                logger.error("Failed loading alert index from %s: %s", self.file_path, err)

    def save_alert(self, alert: ThreatAlert) -> bool:
        with self._lock:
            self._index[alert.alert_id] = alert
            try:
                with open(self.file_path, "a", encoding="utf-8") as f:
                    f.write(alert.model_dump_json() + "\n")
                return True
            except Exception as err:
                logger.error("Failed to append alert %s to %s: %s", alert.alert_id, self.file_path, err)
                return False

    def save_alerts(self, alerts: List[ThreatAlert]) -> int:
        count = 0
        with self._lock:
            try:
                with open(self.file_path, "a", encoding="utf-8") as f:
                    for alert in alerts:
                        self._index[alert.alert_id] = alert
                        f.write(alert.model_dump_json() + "\n")
                        count += 1
            except Exception as err:
                logger.error("Batch append failed for %s: %s", self.file_path, err)
        return count

    def get_alert(self, alert_id: str) -> Optional[ThreatAlert]:
        with self._lock:
            return self._index.get(alert_id)

    def list_alerts(
        self,
        limit: int = 50,
        severity: Optional[SeverityLevel] = None,
        threat_hypothesis: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[ThreatAlert]:
        with self._lock:
            alerts = list(self._index.values())

        alerts.sort(key=lambda a: a.timestamp, reverse=True)

        filtered: List[ThreatAlert] = []
        for alert in alerts:
            if severity is not None and alert.severity != severity:
                continue
            if threat_hypothesis is not None and alert.threat_hypothesis != threat_hypothesis:
                continue
            if start_time is not None and alert.timestamp < start_time:
                continue
            if end_time is not None and alert.timestamp > end_time:
                continue

            filtered.append(alert)
            if len(filtered) >= limit:
                break

        return filtered

    def count_alerts(self) -> int:
        with self._lock:
            return len(self._index)

    def clear(self) -> None:
        with self._lock:
            self._index.clear()
            if self.file_path.exists():
                self.file_path.write_text("", encoding="utf-8")


class DynamoDBAlertSink(BaseAlertStorageSink):
    """AWS DynamoDB driver for ThreatAlert persistence with LocalStack and live AWS support."""

    def __init__(
        self,
        table_name: str = "ThreatAlerts",
        endpoint_url: Optional[str] = None,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        auto_create_table: bool = True,
        dynamodb_resource: Optional[Any] = None,
    ):
        self.table_name = table_name
        self.endpoint_url = endpoint_url
        self.region_name = region_name

        if dynamodb_resource is not None:
            self._dynamodb = dynamodb_resource
        else:
            self._dynamodb = boto3.resource(
                "dynamodb",
                region_name=self.region_name,
                endpoint_url=self.endpoint_url,
                aws_access_key_id=aws_access_key_id or "test",
                aws_secret_access_key=aws_secret_access_key or "test",
            )

        self._table = self._dynamodb.Table(self.table_name)
        if auto_create_table:
            self.ensure_table_exists()

    def ensure_table_exists(self) -> None:
        """Create the DynamoDB table if it does not already exist."""
        try:
            self._table.load()
        except ClientError as err:
            error_code = err.response.get("Error", {}).get("Code", "")
            if error_code in ("ResourceNotFoundException", "404"):
                logger.info("Table %s not found. Creating table on DynamoDB...", self.table_name)
                try:
                    self._table = self._dynamodb.create_table(
                        TableName=self.table_name,
                        KeySchema=[
                            {"AttributeName": "alert_id", "KeyType": "HASH"},
                        ],
                        AttributeDefinitions=[
                            {"AttributeName": "alert_id", "AttributeType": "S"},
                        ],
                        BillingMode="PAY_PER_REQUEST",
                    )
                    self._table.wait_until_exists()
                    logger.info("DynamoDB table %s created successfully.", self.table_name)
                except Exception as create_err:
                    logger.warning("Could not auto-create DynamoDB table %s: %s", self.table_name, create_err)
            else:
                logger.warning("Could not query DynamoDB table %s: %s", self.table_name, err)
        except Exception as err:
            logger.warning("DynamoDB connection or initialization error: %s", err)

    def _alert_to_item(self, alert: ThreatAlert) -> Dict[str, Any]:
        """Convert ThreatAlert to DynamoDB-compatible item with Decimals."""
        raw_dict = alert.model_dump(mode="json")
        item = _convert_floats_to_decimals(raw_dict)
        return item

    def _item_to_alert(self, item: Dict[str, Any]) -> ThreatAlert:
        """Convert DynamoDB item back to validated ThreatAlert."""
        cleaned = _convert_decimals_to_floats(item)
        return ThreatAlert.model_validate(cleaned)

    def save_alert(self, alert: ThreatAlert) -> bool:
        try:
            item = self._alert_to_item(alert)
            self._table.put_item(Item=item)
            return True
        except Exception as err:
            logger.error("Failed to save alert %s to DynamoDB: %s", alert.alert_id, err)
            return False

    def save_alerts(self, alerts: List[ThreatAlert]) -> int:
        if not alerts:
            return 0
        count = 0
        try:
            with self._table.batch_writer() as batch:
                for alert in alerts:
                    item = self._alert_to_item(alert)
                    batch.put_item(Item=item)
                    count += 1
            return count
        except Exception as err:
            logger.error("DynamoDB batch save error: %s", err)
            # Fallback to individual put_items
            saved = 0
            for alert in alerts:
                if self.save_alert(alert):
                    saved += 1
            return saved

    def get_alert(self, alert_id: str) -> Optional[ThreatAlert]:
        try:
            resp = self._table.get_item(Key={"alert_id": alert_id})
            item = resp.get("Item")
            if not item:
                return None
            return self._item_to_alert(item)
        except Exception as err:
            logger.error("Error retrieving alert %s from DynamoDB: %s", alert_id, err)
            return None

    def list_alerts(
        self,
        limit: int = 50,
        severity: Optional[SeverityLevel] = None,
        threat_hypothesis: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[ThreatAlert]:
        try:
            scan_kwargs: Dict[str, Any] = {"Limit": max(100, limit * 2)}
            resp = self._table.scan(**scan_kwargs)
            items = resp.get("Items", [])
            alerts = [self._item_to_alert(it) for it in items]

            alerts.sort(key=lambda a: a.timestamp, reverse=True)

            filtered: List[ThreatAlert] = []
            for alert in alerts:
                if severity is not None and alert.severity != severity:
                    continue
                if threat_hypothesis is not None and alert.threat_hypothesis != threat_hypothesis:
                    continue
                if start_time is not None and alert.timestamp < start_time:
                    continue
                if end_time is not None and alert.timestamp > end_time:
                    continue
                filtered.append(alert)
                if len(filtered) >= limit:
                    break
            return filtered
        except Exception as err:
            logger.error("Error scanning DynamoDB alerts: %s", err)
            return []

    def count_alerts(self) -> int:
        try:
            self._table.load()
            return int(self._table.item_count or 0)
        except Exception:
            return len(self.list_alerts(limit=1000))

    def clear(self) -> None:
        try:
            alerts = self.list_alerts(limit=500)
            with self._table.batch_writer() as batch:
                for a in alerts:
                    batch.delete_item(Key={"alert_id": a.alert_id})
        except Exception as err:
            logger.warning("Error clearing DynamoDB table: %s", err)


def get_alert_sink(
    driver: str = "auto",
    file_path: Optional[str] = None,
    table_name: str = "ThreatAlerts",
    endpoint_url: Optional[str] = None,
    region_name: str = "us-east-1",
    **kwargs: Any,
) -> BaseAlertStorageSink:
    """Factory creating an alert sink based on requested driver or environment.

    Args:
        driver: 'in_memory', 'local_json', 'dynamodb', or 'auto'.
        file_path: File path for local_json sink.
        table_name: Table name for DynamoDB sink.
        endpoint_url: Endpoint URL (e.g. 'http://localhost:4566' for LocalStack).
        region_name: AWS region name.

    Returns:
        Configured BaseAlertStorageSink instance.
    """
    driver = driver.lower().strip()

    if driver == "in_memory":
        return InMemoryAlertSink()

    if driver == "local_json":
        path = file_path or "data/alerts/threat_alerts.jsonl"
        return LocalJsonAlertSink(file_path=path)

    if driver == "dynamodb":
        return DynamoDBAlertSink(
            table_name=table_name,
            endpoint_url=endpoint_url,
            region_name=region_name,
            **kwargs,
        )

    # Auto mode: attempt DynamoDB if endpoint/env provided, else fallback to local_json or in_memory
    import os

    env_driver = os.environ.get("ALERT_STORAGE_DRIVER", "").lower()
    if env_driver:
        return get_alert_sink(
            driver=env_driver,
            file_path=file_path,
            table_name=table_name,
            endpoint_url=endpoint_url,
            region_name=region_name,
            **kwargs,
        )

    localstack_endpoint = os.environ.get("LOCALSTACK_ENDPOINT", os.environ.get("AWS_ENDPOINT_URL"))
    if endpoint_url or localstack_endpoint:
        try:
            return DynamoDBAlertSink(
                table_name=table_name,
                endpoint_url=endpoint_url or localstack_endpoint,
                region_name=region_name,
                auto_create_table=True,
                **kwargs,
            )
        except Exception as err:
            logger.warning("Could not connect to DynamoDB (%s), falling back to LocalJsonAlertSink: %s", endpoint_url, err)

    # Default fallback
    return LocalJsonAlertSink(file_path=file_path or "data/alerts/threat_alerts.jsonl")
