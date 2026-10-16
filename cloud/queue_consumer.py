"""Event-driven telemetry queue consumer drivers (In-Memory FIFO & AWS SQS).

Supports local zero-cost simulation with simulated visibility timeout and dead-letter queue (DLQ),
as well as AWS SQS via Boto3 with automated queue provisioning, redrive policies, and batch operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import boto3
from botocore.exceptions import ClientError

from api.schemas import CloudTelemetryLog

logger = logging.getLogger(__name__)


@dataclass
class QueueMessage:
    """Wrapper representing a message retrieved from a telemetry queue."""

    message_id: str
    body: str
    receipt_handle: str
    receive_count: int = 1
    attributes: Dict[str, Any] = field(default_factory=dict)
    enqueued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_log(self) -> CloudTelemetryLog:
        """Deserialize message payload into a validated CloudTelemetryLog."""
        return CloudTelemetryLog.model_validate_json(self.body)


class BaseQueueConsumer(ABC):
    """Abstract interface for event-driven telemetry message queues."""

    @abstractmethod
    def send_message(self, log: CloudTelemetryLog) -> str:
        """Publish a single telemetry log event to the queue.

        Args:
            log: Validated CloudTelemetryLog event.

        Returns:
            Unique message ID assigned by the queue.
        """
        pass

    @abstractmethod
    def send_batch(self, logs: List[CloudTelemetryLog]) -> List[str]:
        """Publish a batch of telemetry logs to the queue.

        Args:
            logs: List of CloudTelemetryLog events.

        Returns:
            List of message IDs published.
        """
        pass

    @abstractmethod
    def poll_batch(self, max_messages: int = 10, wait_time_seconds: int = 1) -> List[QueueMessage]:
        """Poll a batch of messages from the queue.

        Args:
            max_messages: Maximum messages to retrieve (up to 10 for SQS compatibility).
            wait_time_seconds: Long polling wait timeout in seconds.

        Returns:
            List of QueueMessage items.
        """
        pass

    @abstractmethod
    def acknowledge(self, receipt_handle: str) -> bool:
        """Delete/acknowledge a successfully processed message.

        Args:
            receipt_handle: Identifier returned during poll.

        Returns:
            True if deleted successfully, False otherwise.
        """
        pass

    @abstractmethod
    def acknowledge_batch(self, receipt_handles: List[str]) -> int:
        """Batch delete/acknowledge multiple processed messages.

        Args:
            receipt_handles: List of receipt handles to delete.

        Returns:
            Count of successfully acknowledged messages.
        """
        pass

    @abstractmethod
    def reject(self, receipt_handle: str, requeue: bool = True) -> bool:
        """Reject message on processing error. If requeue=False, move to DLQ.

        Args:
            receipt_handle: Identifier of message to reject.
            requeue: Whether to make message immediately available for retry.

        Returns:
            True if handled successfully, False otherwise.
        """
        pass

    @abstractmethod
    def get_queue_depth(self) -> int:
        """Return approximate number of messages available for consumption."""
        pass

    @abstractmethod
    def get_dlq_depth(self) -> int:
        """Return approximate number of messages in the dead-letter queue."""
        pass

    @abstractmethod
    def poll_dlq(self, max_messages: int = 10) -> List[QueueMessage]:
        """Inspect/drain messages currently residing in the dead-letter queue."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Purge all messages in primary queue and dead-letter queue."""
        pass


class InMemoryQueueConsumer(BaseQueueConsumer):
    """Thread-safe in-memory message queue with simulated visibility timeout and DLQ routing.

    Suitable for local unit testing, rapid prototyping, and offline faculty demonstrations.
    """

    def __init__(
        self,
        max_receive_count: int = 3,
        visibility_timeout_sec: float = 30.0,
    ) -> None:
        self.max_receive_count = max_receive_count
        self.visibility_timeout_sec = visibility_timeout_sec

        self._lock = threading.RLock()
        self._ready_queue: List[QueueMessage] = []
        # Maps receipt_handle -> (visibility_deadline, QueueMessage)
        self._in_flight: Dict[str, Tuple[float, QueueMessage]] = {}
        self._dlq: List[QueueMessage] = []

    def _reclaim_expired_in_flight(self) -> None:
        """Check in-flight messages and requeue any whose visibility timeout has passed."""
        now = time.monotonic()
        expired_handles = [
            handle
            for handle, (deadline, _) in self._in_flight.items()
            if now > deadline
        ]
        for handle in expired_handles:
            _, msg = self._in_flight.pop(handle)
            if msg.receive_count >= self.max_receive_count:
                logger.warning(
                    "Message %s exceeded max receives (%d). Moving to DLQ.",
                    msg.message_id,
                    self.max_receive_count,
                )
                self._dlq.append(msg)
            else:
                self._ready_queue.append(msg)

    def send_message(self, log: CloudTelemetryLog) -> str:
        msg_id = str(uuid.uuid4())
        msg = QueueMessage(
            message_id=msg_id,
            body=log.model_dump_json(),
            receipt_handle=str(uuid.uuid4()),
            receive_count=0,
            enqueued_at=datetime.now(timezone.utc),
        )
        with self._lock:
            self._ready_queue.append(msg)
        return msg_id

    def send_batch(self, logs: List[CloudTelemetryLog]) -> List[str]:
        msg_ids: List[str] = []
        with self._lock:
            for log in logs:
                msg_ids.append(self.send_message(log))
        return msg_ids

    def poll_batch(self, max_messages: int = 10, wait_time_seconds: int = 1) -> List[QueueMessage]:
        start_time = time.monotonic()
        with self._lock:
            self._reclaim_expired_in_flight()

            while not self._ready_queue and (time.monotonic() - start_time) < wait_time_seconds:
                # Release lock briefly to allow producers to send
                self._lock.release()
                time.sleep(0.05)
                self._lock.acquire()
                self._reclaim_expired_in_flight()

            messages: List[QueueMessage] = []
            now = time.monotonic()
            deadline = now + self.visibility_timeout_sec

            while self._ready_queue and len(messages) < max_messages:
                msg = self._ready_queue.pop(0)
                msg.receive_count += 1

                # If receive count exceeded, send directly to DLQ
                if msg.receive_count > self.max_receive_count:
                    logger.warning("Message %s routed to DLQ (receive_count=%d)", msg.message_id, msg.receive_count)
                    self._dlq.append(msg)
                    continue

                receipt_handle = str(uuid.uuid4())
                msg.receipt_handle = receipt_handle
                self._in_flight[receipt_handle] = (deadline, msg)
                messages.append(msg)

            return messages

    def acknowledge(self, receipt_handle: str) -> bool:
        with self._lock:
            if receipt_handle in self._in_flight:
                del self._in_flight[receipt_handle]
                return True
            return False

    def acknowledge_batch(self, receipt_handles: List[str]) -> int:
        count = 0
        with self._lock:
            for handle in receipt_handles:
                if self.acknowledge(handle):
                    count += 1
        return count

    def reject(self, receipt_handle: str, requeue: bool = True) -> bool:
        with self._lock:
            if receipt_handle not in self._in_flight:
                return False

            _, msg = self._in_flight.pop(receipt_handle)
            if not requeue or msg.receive_count >= self.max_receive_count:
                self._dlq.append(msg)
            else:
                self._ready_queue.insert(0, msg)
            return True

    def get_queue_depth(self) -> int:
        with self._lock:
            self._reclaim_expired_in_flight()
            return len(self._ready_queue)

    def get_dlq_depth(self) -> int:
        with self._lock:
            return len(self._dlq)

    def poll_dlq(self, max_messages: int = 10) -> List[QueueMessage]:
        with self._lock:
            drain = self._dlq[:max_messages]
            self._dlq = self._dlq[max_messages:]
            return drain

    def clear(self) -> None:
        with self._lock:
            self._ready_queue.clear()
            self._in_flight.clear()
            self._dlq.clear()


class SQSQueueConsumer(BaseQueueConsumer):
    """AWS SQS message queue driver supporting LocalStack and live AWS deployments.

    Includes automatic dead-letter queue (DLQ) provisioning with redrive policy configuration.
    """

    def __init__(
        self,
        queue_name: str = "cloud-workload-telemetry-queue",
        dlq_name: Optional[str] = None,
        max_receive_count: int = 3,
        visibility_timeout_sec: int = 30,
        endpoint_url: Optional[str] = None,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        auto_create_queues: bool = True,
        sqs_client: Optional[Any] = None,
    ) -> None:
        self.queue_name = queue_name
        self.dlq_name = dlq_name or f"{queue_name}-dlq"
        self.max_receive_count = max_receive_count
        self.visibility_timeout_sec = visibility_timeout_sec
        self.endpoint_url = endpoint_url
        self.region_name = region_name

        if sqs_client is not None:
            self._sqs = sqs_client
        else:
            self._sqs = boto3.client(
                "sqs",
                region_name=self.region_name,
                endpoint_url=self.endpoint_url,
                aws_access_key_id=aws_access_key_id or "test",
                aws_secret_access_key=aws_secret_access_key or "test",
            )

        self.queue_url: str = ""
        self.dlq_url: str = ""

        if auto_create_queues:
            self.ensure_queues_exist()

    def ensure_queues_exist(self) -> None:
        """Create primary SQS queue and DLQ with redrive policy if they do not exist."""
        try:
            # 1. Create or resolve DLQ
            try:
                dlq_resp = self._sqs.get_queue_url(QueueName=self.dlq_name)
                self.dlq_url = dlq_resp["QueueUrl"]
            except ClientError:
                dlq_resp = self._sqs.create_queue(QueueName=self.dlq_name)
                self.dlq_url = dlq_resp["QueueUrl"]
                logger.info("Created SQS Dead-Letter Queue: %s", self.dlq_url)

            # Retrieve DLQ ARN for redrive policy
            attrs = self._sqs.get_queue_attributes(
                QueueUrl=self.dlq_url,
                AttributeNames=["QueueArn"],
            )
            dlq_arn = attrs.get("Attributes", {}).get("QueueArn", "")

            # 2. Create or resolve primary queue
            redrive_policy = json.dumps(
                {
                    "deadLetterTargetArn": dlq_arn,
                    "maxReceiveCount": self.max_receive_count,
                }
            )

            try:
                q_resp = self._sqs.get_queue_url(QueueName=self.queue_name)
                self.queue_url = q_resp["QueueUrl"]
            except ClientError:
                create_attrs = {
                    "VisibilityTimeout": str(self.visibility_timeout_sec),
                }
                if dlq_arn:
                    create_attrs["RedrivePolicy"] = redrive_policy

                q_resp = self._sqs.create_queue(
                    QueueName=self.queue_name,
                    Attributes=create_attrs,
                )
                self.queue_url = q_resp["QueueUrl"]
                logger.info("Created primary SQS Queue: %s with DLQ redrive", self.queue_url)

        except Exception as err:
            logger.warning("Could not auto-provision SQS queues: %s", err)

    def send_message(self, log: CloudTelemetryLog) -> str:
        resp = self._sqs.send_message(
            QueueUrl=self.queue_url,
            MessageBody=log.model_dump_json(),
        )
        return resp.get("MessageId", str(uuid.uuid4()))

    def send_batch(self, logs: List[CloudTelemetryLog]) -> List[str]:
        if not logs:
            return []

        message_ids: List[str] = []
        # SQS send_message_batch allows at most 10 messages per batch
        for i in range(0, len(logs), 10):
            chunk = logs[i : i + 10]
            entries = [
                {
                    "Id": str(idx),
                    "MessageBody": item.model_dump_json(),
                }
                for idx, item in enumerate(chunk)
            ]
            resp = self._sqs.send_message_batch(
                QueueUrl=self.queue_url,
                Entries=entries,
            )
            for success in resp.get("Successful", []):
                message_ids.append(success.get("MessageId", ""))
        return message_ids

    def poll_batch(self, max_messages: int = 10, wait_time_seconds: int = 1) -> List[QueueMessage]:
        if not self.queue_url:
            return []

        resp = self._sqs.receive_message(
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=min(10, max(1, max_messages)),
            WaitTimeSeconds=min(20, max(0, wait_time_seconds)),
            AttributeNames=["ApproximateReceiveCount", "SentTimestamp"],
        )

        messages: List[QueueMessage] = []
        for raw_msg in resp.get("Messages", []):
            attrs = raw_msg.get("Attributes", {})
            receive_count = int(attrs.get("ApproximateReceiveCount", 1))

            messages.append(
                QueueMessage(
                    message_id=raw_msg.get("MessageId", ""),
                    body=raw_msg.get("Body", ""),
                    receipt_handle=raw_msg.get("ReceiptHandle", ""),
                    receive_count=receive_count,
                    attributes=attrs,
                )
            )
        return messages

    def acknowledge(self, receipt_handle: str) -> bool:
        try:
            self._sqs.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle,
            )
            return True
        except ClientError as err:
            logger.error("Failed to delete SQS message: %s", err)
            return False

    def acknowledge_batch(self, receipt_handles: List[str]) -> int:
        if not receipt_handles:
            return 0

        success_count = 0
        for i in range(0, len(receipt_handles), 10):
            chunk = receipt_handles[i : i + 10]
            entries = [{"Id": str(idx), "ReceiptHandle": handle} for idx, handle in enumerate(chunk)]
            try:
                resp = self._sqs.delete_message_batch(
                    QueueUrl=self.queue_url,
                    Entries=entries,
                )
                success_count += len(resp.get("Successful", []))
            except ClientError as err:
                logger.error("SQS batch delete failed: %s", err)
        return success_count

    def reject(self, receipt_handle: str, requeue: bool = True) -> bool:
        try:
            if requeue:
                # Reset visibility timeout to 0 so other workers can process it immediately
                self._sqs.change_message_visibility(
                    QueueUrl=self.queue_url,
                    ReceiptHandle=receipt_handle,
                    VisibilityTimeout=0,
                )
            else:
                # If explicit non-requeue, drop or let SQS redrive to DLQ
                self._sqs.delete_message(
                    QueueUrl=self.queue_url,
                    ReceiptHandle=receipt_handle,
                )
            return True
        except ClientError as err:
            logger.error("Failed to reject SQS message: %s", err)
            return False

    def get_queue_depth(self) -> int:
        if not self.queue_url:
            return 0
        try:
            attrs = self._sqs.get_queue_attributes(
                QueueUrl=self.queue_url,
                AttributeNames=["ApproximateNumberOfMessages"],
            )
            return int(attrs.get("Attributes", {}).get("ApproximateNumberOfMessages", 0))
        except Exception:
            return 0

    def get_dlq_depth(self) -> int:
        if not self.dlq_url:
            return 0
        try:
            attrs = self._sqs.get_queue_attributes(
                QueueUrl=self.dlq_url,
                AttributeNames=["ApproximateNumberOfMessages"],
            )
            return int(attrs.get("Attributes", {}).get("ApproximateNumberOfMessages", 0))
        except Exception:
            return 0

    def poll_dlq(self, max_messages: int = 10) -> List[QueueMessage]:
        if not self.dlq_url:
            return []
        try:
            resp = self._sqs.receive_message(
                QueueUrl=self.dlq_url,
                MaxNumberOfMessages=min(10, max(1, max_messages)),
                WaitTimeSeconds=1,
            )
            messages: List[QueueMessage] = []
            for raw_msg in resp.get("Messages", []):
                messages.append(
                    QueueMessage(
                        message_id=raw_msg.get("MessageId", ""),
                        body=raw_msg.get("Body", ""),
                        receipt_handle=raw_msg.get("ReceiptHandle", ""),
                        receive_count=1,
                    )
                )
            return messages
        except Exception as err:
            logger.error("Failed to poll DLQ: %s", err)
            return []

    def clear(self) -> None:
        try:
            if self.queue_url:
                self._sqs.purge_queue(QueueUrl=self.queue_url)
            if self.dlq_url:
                self._sqs.purge_queue(QueueUrl=self.dlq_url)
        except Exception as err:
            logger.warning("Could not purge SQS queue: %s", err)


def get_queue_consumer(
    driver: str = "auto",
    queue_name: str = "cloud-workload-telemetry-queue",
    dlq_name: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    region_name: str = "us-east-1",
    **kwargs: Any,
) -> BaseQueueConsumer:
    """Factory resolving queue consumer driver based on explicit request or environment.

    Args:
        driver: 'in_memory', 'sqs', or 'auto'.
        queue_name: Primary queue name.
        dlq_name: Optional dead-letter queue name.
        endpoint_url: Endpoint URL (e.g. 'http://localhost:4566' for LocalStack).
        region_name: AWS region name.

    Returns:
        Configured BaseQueueConsumer instance.
    """
    driver = driver.lower().strip()

    if driver == "in_memory":
        return InMemoryQueueConsumer(**kwargs)

    if driver == "sqs":
        return SQSQueueConsumer(
            queue_name=queue_name,
            dlq_name=dlq_name,
            endpoint_url=endpoint_url,
            region_name=region_name,
            **kwargs,
        )

    # Auto resolution
    import os

    env_driver = os.environ.get("QUEUE_CONSUMER_DRIVER", "").lower()
    if env_driver:
        return get_queue_consumer(
            driver=env_driver,
            queue_name=queue_name,
            dlq_name=dlq_name,
            endpoint_url=endpoint_url,
            region_name=region_name,
            **kwargs,
        )

    localstack_endpoint = os.environ.get("LOCALSTACK_ENDPOINT", os.environ.get("AWS_ENDPOINT_URL"))
    if endpoint_url or localstack_endpoint:
        try:
            return SQSQueueConsumer(
                queue_name=queue_name,
                dlq_name=dlq_name,
                endpoint_url=endpoint_url or localstack_endpoint,
                region_name=region_name,
                auto_create_queues=True,
                **kwargs,
            )
        except Exception as err:
            logger.warning("Could not connect to SQS (%s), falling back to InMemoryQueueConsumer: %s", endpoint_url, err)

    return InMemoryQueueConsumer(**kwargs)
