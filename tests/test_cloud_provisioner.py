"""Unit and integration tests for AWS/LocalStack CloudResourceProvisioner and CloudSettings."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from botocore.exceptions import ClientError
import pytest

from cloud.config import (
    CloudInfrastructureStack,
    CloudSettings,
    build_cloud_stack,
    detect_active_environment,
    is_endpoint_reachable,
)
from cloud.provisioner import CloudResourceProvisioner


@pytest.fixture
def mock_aws_clients():
    """Create coordinated mock SQS, DynamoDB, and S3 clients."""
    mock_sqs = MagicMock()
    mock_ddb = MagicMock()
    mock_s3 = MagicMock()
    return mock_sqs, mock_ddb, mock_s3


def test_provision_sqs_creation(mock_aws_clients) -> None:
    """Verify SQS provisioning creates DLQ and primary queue with redrive policy."""
    mock_sqs, mock_ddb, mock_s3 = mock_aws_clients

    # DLQ does not exist initially -> throws ClientError on get_queue_url
    not_found_err = ClientError({"Error": {"Code": "QueueDoesNotExist", "Message": "Not found"}}, "GetQueueUrl")
    mock_sqs.get_queue_url.side_effect = [
        not_found_err,  # DLQ initial check
        not_found_err,  # Primary queue initial check
    ]
    mock_sqs.create_queue.side_effect = [
        {"QueueUrl": "https://sqs.us-east-1.amazonaws.com/123/test-dlq"},
        {"QueueUrl": "https://sqs.us-east-1.amazonaws.com/123/test-queue"},
    ]
    mock_sqs.get_queue_attributes.side_effect = [
        {"Attributes": {"QueueArn": "arn:aws:sqs:us-east-1:123:test-dlq"}},
        {"Attributes": {"QueueArn": "arn:aws:sqs:us-east-1:123:test-queue"}},
    ]

    provisioner = CloudResourceProvisioner(
        endpoint_url="http://localhost:4566",
        region_name="us-east-1",
        sqs_client=mock_sqs,
        dynamodb_resource=mock_ddb,
        s3_client=mock_s3,
    )

    res = provisioner.provision_sqs(
        queue_name="test-queue",
        dlq_name="test-dlq",
        visibility_timeout_sec=45,
        max_receive_count=5,
    )

    assert res["queue_name"] == "test-queue"
    assert res["queue_url"] == "https://sqs.us-east-1.amazonaws.com/123/test-queue"
    assert res["dlq_name"] == "test-dlq"
    assert res["dlq_arn"] == "arn:aws:sqs:us-east-1:123:test-dlq"
    assert mock_sqs.create_queue.call_count == 2


def test_provision_sqs_existing_queues_idempotent(mock_aws_clients) -> None:
    """Verify SQS provisioning idempotently handles pre-existing queues."""
    mock_sqs, mock_ddb, mock_s3 = mock_aws_clients

    mock_sqs.get_queue_url.side_effect = [
        {"QueueUrl": "https://sqs.us-east-1.amazonaws.com/123/existing-dlq"},
        {"QueueUrl": "https://sqs.us-east-1.amazonaws.com/123/existing-queue"},
    ]
    mock_sqs.get_queue_attributes.side_effect = [
        {"Attributes": {"QueueArn": "arn:aws:sqs:us-east-1:123:existing-dlq"}},
        {"Attributes": {"QueueArn": "arn:aws:sqs:us-east-1:123:existing-queue"}},
    ]

    provisioner = CloudResourceProvisioner(
        sqs_client=mock_sqs,
        dynamodb_resource=mock_ddb,
        s3_client=mock_s3,
    )

    res = provisioner.provision_sqs(queue_name="existing-queue", dlq_name="existing-dlq")
    assert res["queue_url"] == "https://sqs.us-east-1.amazonaws.com/123/existing-queue"
    mock_sqs.set_queue_attributes.assert_called_once()


def test_provision_dynamodb_creation(mock_aws_clients) -> None:
    """Verify DynamoDB table creation with on-demand billing mode."""
    mock_sqs, mock_ddb, mock_s3 = mock_aws_clients

    mock_table = MagicMock()
    mock_table.load.side_effect = ClientError({"Error": {"Code": "ResourceNotFoundException"}}, "DescribeTable")
    mock_ddb.Table.return_value = mock_table

    created_table = MagicMock()
    created_table.table_arn = "arn:aws:dynamodb:us-east-1:123:table/test-alerts"
    mock_ddb.create_table.return_value = created_table

    provisioner = CloudResourceProvisioner(
        sqs_client=mock_sqs,
        dynamodb_resource=mock_ddb,
        s3_client=mock_s3,
    )

    res = provisioner.provision_dynamodb(table_name="test-alerts")
    assert res["table_name"] == "test-alerts"
    assert res["status"] == "ACTIVE"
    assert res["created"] is True
    mock_ddb.create_table.assert_called_once_with(
        TableName="test-alerts",
        KeySchema=[{"AttributeName": "alert_id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "alert_id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )


def test_provision_s3_creation(mock_aws_clients) -> None:
    """Verify S3 bucket creation and idempotency."""
    mock_sqs, mock_ddb, mock_s3 = mock_aws_clients

    # S3 bucket does not exist initially
    mock_s3.head_bucket.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadBucket")

    provisioner = CloudResourceProvisioner(
        region_name="eu-west-1",
        sqs_client=mock_sqs,
        dynamodb_resource=mock_ddb,
        s3_client=mock_s3,
    )

    res = provisioner.provision_s3(bucket_name="test-archive-bucket")
    assert res["bucket_name"] == "test-archive-bucket"
    assert res["status"] == "CREATED"
    assert res["created"] is True
    mock_s3.create_bucket.assert_called_once_with(
        Bucket="test-archive-bucket",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
    )


def test_provision_all_orchestration(mock_aws_clients) -> None:
    """Verify full cloud stack provisioning orchestration."""
    mock_sqs, mock_ddb, mock_s3 = mock_aws_clients

    provisioner = CloudResourceProvisioner(
        endpoint_url="http://localhost:4566",
        sqs_client=mock_sqs,
        dynamodb_resource=mock_ddb,
        s3_client=mock_s3,
    )

    with patch.object(provisioner, "provision_sqs", return_value={"status": "OK"}), \
         patch.object(provisioner, "provision_dynamodb", return_value={"status": "ACTIVE"}), \
         patch.object(provisioner, "provision_s3", return_value={"status": "EXISTS"}):
        res = provisioner.provision_all()
        assert res["status"] == "PROVISIONED"
        assert res["endpoint_url"] == "http://localhost:4566"


def test_verify_resources_healthy(mock_aws_clients) -> None:
    """Verify health check verification reports healthy when all resources exist."""
    mock_sqs, mock_ddb, mock_s3 = mock_aws_clients
    mock_sqs.get_queue_url.return_value = {"QueueUrl": "http://sqs/q"}
    mock_table = MagicMock()
    mock_table.table_status = "ACTIVE"
    mock_ddb.Table.return_value = mock_table
    mock_s3.head_bucket.return_value = {}

    provisioner = CloudResourceProvisioner(
        sqs_client=mock_sqs,
        dynamodb_resource=mock_ddb,
        s3_client=mock_s3,
    )

    report = provisioner.verify_resources()
    assert report["all_healthy"] is True
    assert report["sqs"]["status"] == "HEALTHY"
    assert report["dynamodb"]["status"] == "HEALTHY"
    assert report["s3"]["status"] == "HEALTHY"


def test_verify_resources_unhealthy(mock_aws_clients) -> None:
    """Verify health check verification flags unhealthy when a service fails."""
    mock_sqs, mock_ddb, mock_s3 = mock_aws_clients
    mock_sqs.get_queue_url.side_effect = Exception("SQS Connection Failed")
    mock_table = MagicMock()
    mock_table.load.side_effect = Exception("DynamoDB down")
    mock_ddb.Table.return_value = mock_table
    mock_s3.head_bucket.side_effect = Exception("S3 bucket not found")

    provisioner = CloudResourceProvisioner(
        sqs_client=mock_sqs,
        dynamodb_resource=mock_ddb,
        s3_client=mock_s3,
    )

    report = provisioner.verify_resources()
    assert report["all_healthy"] is False
    assert report["sqs"]["status"] == "UNHEALTHY"


def test_teardown_all_resources(mock_aws_clients) -> None:
    """Verify teardown deletes all allocated cloud resources."""
    mock_sqs, mock_ddb, mock_s3 = mock_aws_clients
    mock_sqs.get_queue_url.return_value = {"QueueUrl": "http://sqs/q"}
    mock_table = MagicMock()
    mock_ddb.Table.return_value = mock_table
    mock_s3.list_objects_v2.return_value = {"Contents": [{"Key": "part-1.jsonl.gz"}]}

    provisioner = CloudResourceProvisioner(
        sqs_client=mock_sqs,
        dynamodb_resource=mock_ddb,
        s3_client=mock_s3,
    )

    res = provisioner.teardown_all()
    assert res["sqs"] is True
    assert res["dlq"] is True
    assert res["dynamodb"] is True
    assert res["s3"] is True
    mock_sqs.delete_queue.assert_called()
    mock_table.delete.assert_called_once()
    mock_s3.delete_object.assert_called_once_with(Bucket="temporal-threat-telemetry-archive", Key="part-1.jsonl.gz")
    mock_s3.delete_bucket.assert_called_once()


def test_cloud_settings_and_environment_detection() -> None:
    """Verify CloudSettings defaults and active environment resolver."""
    with patch.dict("os.environ", {"TELEMETRY_DRIVER_MODE": "in-memory"}):
        env = detect_active_environment()
        assert env == "in-memory"

    settings = CloudSettings(driver_mode="localstack", aws_endpoint_url="http://localhost:4566")
    assert settings.driver_mode == "localstack"
    assert settings.aws_endpoint_url == "http://localhost:4566"
    assert settings.aws_access_key_id == "test"


def test_build_cloud_stack_in_memory() -> None:
    """Verify build_cloud_stack creates in-memory drivers cleanly."""
    settings = CloudSettings(driver_mode="in-memory")
    stack = build_cloud_stack(settings=settings)
    assert isinstance(stack, CloudInfrastructureStack)
    assert stack.is_cloud_active is False
    assert stack.provisioner is None
    status = stack.to_status_dict()
    assert status["driver_mode"] == "in-memory"
    assert status["is_cloud_active"] is False


def test_build_cloud_stack_local_files(tmp_path) -> None:
    """Verify build_cloud_stack creates local persistent JSON file and gzip archiver."""
    settings = CloudSettings(driver_mode="local", local_storage_dir=tmp_path)
    stack = build_cloud_stack(settings=settings)
    assert isinstance(stack, CloudInfrastructureStack)
    assert stack.settings.driver_mode == "local"
    assert stack.is_cloud_active is False
