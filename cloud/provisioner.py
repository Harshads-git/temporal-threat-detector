"""Automated AWS and LocalStack resource provisioner for Temporal Threat Detector.

Handles idempotent creation, configuration, verification, and teardown of:
1. SQS Telemetry Ingestion Queue & Dead-Letter Queue (DLQ) with Redrive Policy
2. DynamoDB ThreatAlerts table with On-Demand billing (PAY_PER_REQUEST)
3. S3 Telemetry Archive Bucket for partitioned gzip JSONL storage
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger("cloud.provisioner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class CloudResourceProvisioner:
    """Manages cloud infrastructure lifecycle across live AWS and LocalStack environments."""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        sqs_client: Optional[Any] = None,
        dynamodb_resource: Optional[Any] = None,
        s3_client: Optional[Any] = None,
    ) -> None:
        """Initialize provisioner with AWS credentials or LocalStack endpoint."""
        self.endpoint_url = endpoint_url or os.getenv("AWS_ENDPOINT_URL")
        self.region_name = region_name or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID", "test")
        secret_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY", "test")

        boto_kwargs: Dict[str, Any] = {
            "region_name": self.region_name,
            "aws_access_key_id": key_id,
            "aws_secret_access_key": secret_key,
        }
        if self.endpoint_url:
            boto_kwargs["endpoint_url"] = self.endpoint_url

        self.sqs = sqs_client or boto3.client("sqs", **boto_kwargs)
        self.dynamodb = dynamodb_resource or boto3.resource("dynamodb", **boto_kwargs)
        self.s3 = s3_client or boto3.client("s3", **boto_kwargs)

    def provision_sqs(
        self,
        queue_name: str = "cloud-workload-telemetry-queue",
        dlq_name: Optional[str] = None,
        visibility_timeout_sec: int = 30,
        max_receive_count: int = 3,
    ) -> Dict[str, str]:
        """Idempotently provision primary SQS queue and DLQ with redrive policy."""
        dlq_name = dlq_name or f"{queue_name}-dlq"
        logger.info("Provisioning SQS queues (Primary: %s, DLQ: %s)...", queue_name, dlq_name)

        # 1. Resolve or Create DLQ
        try:
            dlq_resp = self.sqs.get_queue_url(QueueName=dlq_name)
            dlq_url = dlq_resp["QueueUrl"]
            logger.info("Existing SQS DLQ resolved: %s", dlq_url)
        except ClientError:
            dlq_resp = self.sqs.create_queue(QueueName=dlq_name)
            dlq_url = dlq_resp["QueueUrl"]
            logger.info("Created SQS DLQ: %s", dlq_url)

        # Retrieve DLQ ARN
        dlq_attrs = self.sqs.get_queue_attributes(QueueUrl=dlq_url, AttributeNames=["QueueArn"])
        dlq_arn = dlq_attrs.get("Attributes", {}).get("QueueArn", "")

        # 2. Resolve or Create Primary Queue with Redrive Policy
        redrive_policy = json.dumps({
            "deadLetterTargetArn": dlq_arn,
            "maxReceiveCount": max_receive_count,
        })

        try:
            q_resp = self.sqs.get_queue_url(QueueName=queue_name)
            queue_url = q_resp["QueueUrl"]
            logger.info("Existing SQS Primary Queue resolved: %s", queue_url)
            # Update attributes to ensure redrive policy matches
            self.sqs.set_queue_attributes(
                QueueUrl=queue_url,
                Attributes={
                    "VisibilityTimeout": str(visibility_timeout_sec),
                    "RedrivePolicy": redrive_policy,
                },
            )
        except ClientError:
            create_attrs = {
                "VisibilityTimeout": str(visibility_timeout_sec),
            }
            if dlq_arn:
                create_attrs["RedrivePolicy"] = redrive_policy

            q_resp = self.sqs.create_queue(QueueName=queue_name, Attributes=create_attrs)
            queue_url = q_resp["QueueUrl"]
            logger.info("Created SQS Primary Queue: %s", queue_url)

        q_attrs = self.sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["QueueArn"])
        queue_arn = q_attrs.get("Attributes", {}).get("QueueArn", "")

        return {
            "queue_name": queue_name,
            "queue_url": queue_url,
            "queue_arn": queue_arn,
            "dlq_name": dlq_name,
            "dlq_url": dlq_url,
            "dlq_arn": dlq_arn,
        }

    def provision_dynamodb(self, table_name: str = "temporal-threat-alerts") -> Dict[str, Any]:
        """Idempotently provision DynamoDB table with PAY_PER_REQUEST on-demand billing."""
        logger.info("Provisioning DynamoDB table: %s...", table_name)
        table = self.dynamodb.Table(table_name)
        try:
            table.load()
            logger.info("Existing DynamoDB table resolved: %s (Status: %s)", table_name, table.table_status)
            return {
                "table_name": table_name,
                "table_arn": table.table_arn,
                "status": table.table_status,
                "created": False,
            }
        except ClientError as err:
            err_code = err.response.get("Error", {}).get("Code", "")
            if err_code in ("ResourceNotFoundException", "404"):
                table = self.dynamodb.create_table(
                    TableName=table_name,
                    KeySchema=[
                        {"AttributeName": "alert_id", "KeyType": "HASH"},
                    ],
                    AttributeDefinitions=[
                        {"AttributeName": "alert_id", "AttributeType": "S"},
                    ],
                    BillingMode="PAY_PER_REQUEST",
                )
                table.wait_until_exists()
                logger.info("Created DynamoDB table: %s", table_name)
                return {
                    "table_name": table_name,
                    "table_arn": table.table_arn,
                    "status": "ACTIVE",
                    "created": True,
                }
            raise

    def provision_s3(self, bucket_name: str = "temporal-threat-telemetry-archive") -> Dict[str, Any]:
        """Idempotently provision S3 archive bucket."""
        logger.info("Provisioning S3 archive bucket: %s...", bucket_name)
        try:
            self.s3.head_bucket(Bucket=bucket_name)
            logger.info("Existing S3 bucket resolved: %s", bucket_name)
            return {"bucket_name": bucket_name, "status": "EXISTS", "created": False}
        except ClientError as err:
            err_code = str(err.response.get("Error", {}).get("Code", ""))
            if err_code in ("404", "NoSuchBucket"):
                create_kwargs: Dict[str, Any] = {"Bucket": bucket_name}
                if self.region_name != "us-east-1":
                    create_kwargs["CreateBucketConfiguration"] = {
                        "LocationConstraint": self.region_name
                    }
                self.s3.create_bucket(**create_kwargs)
                logger.info("Created S3 bucket: %s", bucket_name)
                return {"bucket_name": bucket_name, "status": "CREATED", "created": True}
            raise

    def provision_all(
        self,
        queue_name: str = "cloud-workload-telemetry-queue",
        dlq_name: Optional[str] = None,
        table_name: str = "temporal-threat-alerts",
        bucket_name: str = "temporal-threat-telemetry-archive",
    ) -> Dict[str, Any]:
        """Provision complete cloud telemetry infrastructure suite."""
        logger.info("Executing full cloud infrastructure provisioning...")
        sqs_res = self.provision_sqs(queue_name=queue_name, dlq_name=dlq_name)
        ddb_res = self.provision_dynamodb(table_name=table_name)
        s3_res = self.provision_s3(bucket_name=bucket_name)

        summary = {
            "endpoint_url": self.endpoint_url or "aws-standard",
            "region": self.region_name,
            "sqs": sqs_res,
            "dynamodb": ddb_res,
            "s3": s3_res,
            "status": "PROVISIONED",
        }
        logger.info("All cloud infrastructure resources provisioned successfully.")
        return summary

    def verify_resources(
        self,
        queue_name: str = "cloud-workload-telemetry-queue",
        dlq_name: Optional[str] = None,
        table_name: str = "temporal-threat-alerts",
        bucket_name: str = "temporal-threat-telemetry-archive",
    ) -> Dict[str, Any]:
        """Verify the health and presence of all cloud resources."""
        dlq_name = dlq_name or f"{queue_name}-dlq"
        report: Dict[str, Any] = {"endpoint_url": self.endpoint_url, "region": self.region_name}

        # 1. Verify SQS
        try:
            q_url = self.sqs.get_queue_url(QueueName=queue_name)["QueueUrl"]
            dlq_url = self.sqs.get_queue_url(QueueName=dlq_name)["QueueUrl"]
            report["sqs"] = {"status": "HEALTHY", "queue_url": q_url, "dlq_url": dlq_url}
        except Exception as err:
            report["sqs"] = {"status": "UNHEALTHY", "error": str(err)}

        # 2. Verify DynamoDB
        try:
            tbl = self.dynamodb.Table(table_name)
            tbl.load()
            report["dynamodb"] = {"status": "HEALTHY", "table_status": tbl.table_status}
        except Exception as err:
            report["dynamodb"] = {"status": "UNHEALTHY", "error": str(err)}

        # 3. Verify S3
        try:
            self.s3.head_bucket(Bucket=bucket_name)
            report["s3"] = {"status": "HEALTHY", "bucket_name": bucket_name}
        except Exception as err:
            report["s3"] = {"status": "UNHEALTHY", "error": str(err)}

        report["all_healthy"] = all(
            report[k].get("status") == "HEALTHY" for k in ("sqs", "dynamodb", "s3")
        )
        return report

    def teardown_all(
        self,
        queue_name: str = "cloud-workload-telemetry-queue",
        dlq_name: Optional[str] = None,
        table_name: str = "temporal-threat-alerts",
        bucket_name: str = "temporal-threat-telemetry-archive",
    ) -> Dict[str, bool]:
        """Delete all provisioned resources for clean test environments or resetting."""
        dlq_name = dlq_name or f"{queue_name}-dlq"
        logger.info("Executing teardown of provisioned cloud resources...")
        results = {"sqs": False, "dlq": False, "dynamodb": False, "s3": False}

        # 1. Teardown Primary Queue
        try:
            q_url = self.sqs.get_queue_url(QueueName=queue_name)["QueueUrl"]
            self.sqs.delete_queue(QueueUrl=q_url)
            results["sqs"] = True
        except Exception as err:
            logger.warning("Error deleting primary queue: %s", err)

        # 2. Teardown DLQ
        try:
            dlq_url = self.sqs.get_queue_url(QueueName=dlq_name)["QueueUrl"]
            self.sqs.delete_queue(QueueUrl=dlq_url)
            results["dlq"] = True
        except Exception as err:
            logger.warning("Error deleting DLQ: %s", err)

        # 3. Teardown DynamoDB
        try:
            tbl = self.dynamodb.Table(table_name)
            tbl.delete()
            results["dynamodb"] = True
        except Exception as err:
            logger.warning("Error deleting DynamoDB table: %s", err)

        # 4. Teardown S3
        try:
            # Empty bucket first
            objects = self.s3.list_objects_v2(Bucket=bucket_name)
            for item in objects.get("Contents", []):
                self.s3.delete_object(Bucket=bucket_name, Key=item["Key"])
            self.s3.delete_bucket(Bucket=bucket_name)
            results["s3"] = True
        except Exception as err:
            logger.warning("Error deleting S3 bucket: %s", err)

        return results


def main() -> None:
    """CLI Entry point for resource provisioning and health verification."""
    parser = argparse.ArgumentParser(description="Temporal Threat Detector Cloud Resource Provisioner")
    parser.add_argument("--action", choices=["provision", "verify", "teardown"], default="provision")
    parser.add_argument("--endpoint-url", default=None, help="LocalStack or custom AWS endpoint URL")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--queue", default="cloud-workload-telemetry-queue")
    parser.add_argument("--table", default="temporal-threat-alerts")
    parser.add_argument("--bucket", default="temporal-threat-telemetry-archive")
    args = parser.parse_args()

    provisioner = CloudResourceProvisioner(endpoint_url=args.endpoint_url, region_name=args.region)

    if args.action == "provision":
        res = provisioner.provision_all(queue_name=args.queue, table_name=args.table, bucket_name=args.bucket)
        print(json.dumps(res, indent=2))
    elif args.action == "verify":
        res = provisioner.verify_resources(queue_name=args.queue, table_name=args.table, bucket_name=args.bucket)
        print(json.dumps(res, indent=2))
        sys.exit(0 if res.get("all_healthy") else 1)
    elif args.action == "teardown":
        res = provisioner.teardown_all(queue_name=args.queue, table_name=args.table, bucket_name=args.bucket)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
