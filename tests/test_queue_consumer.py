"""Unit and integration test suite for Event-Driven Queue Consumers and Telemetry Worker."""

from datetime import datetime, timedelta, timezone
import json
import time
from unittest.mock import MagicMock, patch
import pytest

from api.schemas import AttackType, CloudTelemetryLog
from cloud.queue_consumer import (
    BackoffPolicy,
    InMemoryQueueConsumer,
    QueueMessage,
    SQSQueueConsumer,
    TelemetryIngestionWorker,
    get_queue_consumer,
)
from cloud.storage_sink import InMemoryAlertSink, InMemoryTelemetryArchiver
from data.synthetic_generator import SyntheticTelemetryGenerator
from engine.isolation_forest import HybridThreatEnsemble, IsolationForestDetector
from engine.windowing import StreamingWindowBuffer, TemporalWindowAggregator
from engine.zscore_filter import StatisticalFilterEngine


# ==============================================================================
# QUEUE MESSAGE & IN-MEMORY QUEUE CONSUMER TESTS
# ==============================================================================


def test_queue_message_serialization():
    """Verify QueueMessage deserializes correctly into CloudTelemetryLog."""
    gen = SyntheticTelemetryGenerator(seed=42)
    sample_log = gen.generate_batch(count=1)[0]

    msg = QueueMessage(
        message_id="msg-101",
        body=sample_log.model_dump_json(),
        receipt_handle="rh-101",
        receive_count=1,
    )
    deserialized = msg.to_log()
    assert deserialized.event_id == sample_log.event_id
    assert deserialized.source_ip == sample_log.source_ip
    assert deserialized.endpoint == sample_log.endpoint


def test_in_memory_queue_basic_send_and_poll():
    """Verify in-memory queue send, batch send, poll, and depth tracking."""
    queue = InMemoryQueueConsumer(visibility_timeout_sec=5.0)
    gen = SyntheticTelemetryGenerator(seed=42)
    logs = gen.generate_batch(count=5)

    assert queue.get_queue_depth() == 0

    # Single send
    mid1 = queue.send_message(logs[0])
    assert mid1 is not None
    assert queue.get_queue_depth() == 1

    # Batch send
    mids = queue.send_batch(logs[1:])
    assert len(mids) == 4
    assert queue.get_queue_depth() == 5

    # Poll 3 messages
    polled = queue.poll_batch(max_messages=3, wait_time_seconds=0)
    assert len(polled) == 3
    assert queue.get_queue_depth() == 2  # 2 ready, 3 in-flight

    # Acknowledge 3 polled messages
    handles = [m.receipt_handle for m in polled]
    ack_count = queue.acknowledge_batch(handles)
    assert ack_count == 3
    assert queue.get_queue_depth() == 2

    # Drain remaining 2
    rem = queue.poll_batch(max_messages=10, wait_time_seconds=0)
    assert len(rem) == 2
    assert queue.acknowledge(rem[0].receipt_handle) is True
    assert queue.acknowledge(rem[1].receipt_handle) is True
    assert queue.get_queue_depth() == 0


def test_in_memory_queue_rejection_and_dlq_routing():
    """Verify rejection routing and automatic dead-letter queue escalation after max retries."""
    queue = InMemoryQueueConsumer(max_receive_count=2, visibility_timeout_sec=0.2)
    gen = SyntheticTelemetryGenerator(seed=42)
    log = gen.generate_batch(count=1)[0]

    queue.send_message(log)
    assert queue.get_queue_depth() == 1
    assert queue.get_dlq_depth() == 0

    # 1st Receive
    batch1 = queue.poll_batch(max_messages=1, wait_time_seconds=0)
    assert len(batch1) == 1
    assert batch1[0].receive_count == 1

    # Reject with requeue=True
    assert queue.reject(batch1[0].receipt_handle, requeue=True) is True
    assert queue.get_queue_depth() == 1
    assert queue.get_dlq_depth() == 0

    # 2nd Receive
    batch2 = queue.poll_batch(max_messages=1, wait_time_seconds=0)
    assert len(batch2) == 1
    assert batch2[0].receive_count == 2

    # Reject again (now at max_receive_count -> routes to DLQ)
    assert queue.reject(batch2[0].receipt_handle, requeue=True) is True
    assert queue.get_queue_depth() == 0
    assert queue.get_dlq_depth() == 1

    # Poll DLQ
    dlq_items = queue.poll_dlq(max_messages=5)
    assert len(dlq_items) == 1
    assert queue.get_dlq_depth() == 0

    # Clear
    queue.clear()
    assert queue.get_queue_depth() == 0
    assert queue.get_dlq_depth() == 0


def test_in_memory_queue_visibility_timeout_expiration():
    """Verify in-flight messages are automatically reclaimed after visibility timeout expires."""
    queue = InMemoryQueueConsumer(max_receive_count=3, visibility_timeout_sec=0.1)
    gen = SyntheticTelemetryGenerator(seed=42)
    log = gen.generate_batch(count=1)[0]

    queue.send_message(log)
    batch = queue.poll_batch(max_messages=1, wait_time_seconds=0)
    assert len(batch) == 1
    assert queue.get_queue_depth() == 0  # in-flight

    # Sleep past visibility timeout
    time.sleep(0.15)

    # Queue depth should now reflect reclaimed message
    assert queue.get_queue_depth() == 1
    reclaimed = queue.poll_batch(max_messages=1, wait_time_seconds=0)
    assert len(reclaimed) == 1
    assert reclaimed[0].receive_count == 2
    queue.acknowledge(reclaimed[0].receipt_handle)


# ==============================================================================
# SQS QUEUE CONSUMER (MOCK BOTO3) TESTS
# ==============================================================================


def test_mock_sqs_queue_consumer():
    """Verify SQSQueueConsumer interactions using mocked Boto3 client."""
    mock_sqs = MagicMock()
    mock_sqs.get_queue_url.side_effect = [
        {"QueueUrl": "https://sqs.us-east-1.amazonaws.com/123/mock-dlq"},
        {"QueueUrl": "https://sqs.us-east-1.amazonaws.com/123/mock-queue"},
    ]
    mock_sqs.get_queue_attributes.return_value = {
        "Attributes": {
            "QueueArn": "arn:aws:sqs:us-east-1:123:mock-dlq",
            "ApproximateNumberOfMessages": "5",
        }
    }

    consumer = SQSQueueConsumer(
        queue_name="mock-queue",
        auto_create_queues=True,
        sqs_client=mock_sqs,
    )

    assert consumer.queue_url == "https://sqs.us-east-1.amazonaws.com/123/mock-queue"
    assert consumer.dlq_url == "https://sqs.us-east-1.amazonaws.com/123/mock-dlq"

    # Send message
    gen = SyntheticTelemetryGenerator(seed=42)
    log = gen.generate_batch(count=1)[0]
    mock_sqs.send_message.return_value = {"MessageId": "sqs-msg-1"}

    mid = consumer.send_message(log)
    assert mid == "sqs-msg-1"
    mock_sqs.send_message.assert_called_once()

    # Send batch
    mock_sqs.send_message_batch.return_value = {
        "Successful": [{"Id": "0", "MessageId": "batch-1"}, {"Id": "1", "MessageId": "batch-2"}]
    }
    bids = consumer.send_batch([log, log])
    assert len(bids) == 2

    # Poll batch
    mock_sqs.receive_message.return_value = {
        "Messages": [
            {
                "MessageId": "m1",
                "ReceiptHandle": "rh1",
                "Body": log.model_dump_json(),
                "Attributes": {"ApproximateReceiveCount": "1"},
            }
        ]
    }
    polled = consumer.poll_batch(max_messages=1)
    assert len(polled) == 1
    assert polled[0].message_id == "m1"
    assert polled[0].receive_count == 1

    # Acknowledge
    assert consumer.acknowledge("rh1") is True
    mock_sqs.delete_message.assert_called_once()

    # Queue depth
    assert consumer.get_queue_depth() == 5


# ==============================================================================
# BACKOFF POLICY & METRICS TESTS
# ==============================================================================


def test_backoff_policy_delay_calculation():
    """Verify exponential backoff progression, max bound, and non-negative jitter."""
    policy = BackoffPolicy(
        initial_delay_sec=1.0,
        max_delay_sec=8.0,
        multiplier=2.0,
        jitter_factor=0.1,
    )

    # Attempt 0: ~1.0 +/- 0.1
    d0 = policy.compute_delay(0)
    assert 0.85 <= d0 <= 1.15

    # Attempt 1: ~2.0 +/- 0.2
    d1 = policy.compute_delay(1)
    assert 1.75 <= d1 <= 2.25

    # Attempt 2: ~4.0 +/- 0.4
    d2 = policy.compute_delay(2)
    assert 3.5 <= d2 <= 4.5

    # Attempt 5: bounded by max_delay_sec (8.0 +/- 0.8)
    d5 = policy.compute_delay(5)
    assert 7.0 <= d5 <= 8.8


# ==============================================================================
# TELEMETRY INGESTION WORKER INTEGRATION TESTS
# ==============================================================================


def test_telemetry_ingestion_worker_end_to_end():
    """Verify complete pipeline integration from queue ingestion to windowing and alert generation."""
    queue = InMemoryQueueConsumer()
    window_buffer = StreamingWindowBuffer(window_size_seconds=60.0, hop_seconds=10.0)
    stat_engine = StatisticalFilterEngine()
    ensemble = HybridThreatEnsemble(statistical_engine=stat_engine)
    alert_sink = InMemoryAlertSink()
    archiver = InMemoryTelemetryArchiver()

    worker = TelemetryIngestionWorker(
        queue_consumer=queue,
        window_buffer=window_buffer,
        detector_ensemble=ensemble,
        alert_sink=alert_sink,
        telemetry_archiver=archiver,
        batch_size=100,
    )

    # Generate attack burst (DDoS HTTP Flood)
    gen = SyntheticTelemetryGenerator(seed=42)
    attack_logs = gen.generate_batch(
        count=300,
        inject_attack=AttackType.DDOS_HTTP_FLOOD,
        intensity=4.0,
    )

    # Producer publishes to queue
    queue.send_batch(attack_logs)
    assert queue.get_queue_depth() == 300

    # Worker consumes and processes
    stats = worker.process_batch(max_messages=300)
    assert stats["messages_polled"] == 300
    assert stats["messages_processed"] == 300
    assert stats["messages_failed"] == 0
    assert queue.get_queue_depth() == 0  # Queue fully drained

    # Verify telemetry was archived
    assert len(archiver.list_archives()) >= 1

    # Verify windows and alerts were produced
    assert stats["windows_emitted"] > 0
    assert alert_sink.count_alerts() > 0

    alerts = alert_sink.list_alerts(limit=5)
    assert len(alerts) > 0
    assert alerts[0].threat_hypothesis is not None


def test_telemetry_ingestion_worker_poison_pill_handling():
    """Verify malformed JSON messages are routed to DLQ without terminating worker."""
    queue = InMemoryQueueConsumer(max_receive_count=3)
    worker = TelemetryIngestionWorker(
        queue_consumer=queue,
        batch_size=10,
    )

    gen = SyntheticTelemetryGenerator(seed=42)
    valid_log = gen.generate_batch(count=1)[0]

    # Queue 1 valid, 1 poison pill, 1 valid
    queue.send_message(valid_log)

    # Directly insert raw malformed message into queue
    corrupt_msg = QueueMessage(
        message_id="bad-msg-1",
        body="INVALID NOT JSON TELEMETRY PAYLOAD {",
        receipt_handle="bad-rh-1",
        receive_count=0,
    )
    with queue._lock:
        queue._ready_queue.append(corrupt_msg)

    queue.send_message(valid_log)
    assert queue.get_queue_depth() == 3

    # Process batch
    stats = worker.process_batch(max_messages=10)
    assert stats["messages_polled"] == 3
    assert stats["messages_processed"] == 2
    assert stats["messages_failed"] == 1

    # Corrupt message was moved to DLQ
    assert queue.get_dlq_depth() == 1
    dlq_items = queue.poll_dlq(max_messages=5)
    assert len(dlq_items) == 1
    assert dlq_items[0].message_id == "bad-msg-1"


def test_telemetry_ingestion_worker_background_thread():
    """Verify worker start and stop background daemon thread lifecycle."""
    queue = InMemoryQueueConsumer()
    worker = TelemetryIngestionWorker(
        queue_consumer=queue,
        batch_size=5,
    )

    worker.start()
    assert worker._worker_thread is not None
    assert worker._worker_thread.is_alive()

    # Send messages while worker is active
    gen = SyntheticTelemetryGenerator(seed=42)
    logs = gen.generate_batch(count=10)
    queue.send_batch(logs)

    # Wait briefly for worker thread to drain
    time.sleep(0.3)
    assert queue.get_queue_depth() == 0
    assert worker.metrics.messages_consumed >= 10

    # Stop worker
    worker.stop(timeout_sec=2.0)
    assert worker._worker_thread is None


# ==============================================================================
# QUEUE CONSUMER FACTORY RESOLVER TESTS
# ==============================================================================


def test_get_queue_consumer_factory():
    """Verify get_queue_consumer factory across in_memory, sqs, and environment fallback."""
    q1 = get_queue_consumer(driver="in_memory")
    assert isinstance(q1, InMemoryQueueConsumer)

    with patch.dict("os.environ", {"QUEUE_CONSUMER_DRIVER": "in_memory"}):
        q2 = get_queue_consumer(driver="auto")
        assert isinstance(q2, InMemoryQueueConsumer)
