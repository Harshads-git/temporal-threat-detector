"""Synthetic Cloud Telemetry Generator with Realistic Diurnal Baseline & Attack Injections.

Simulates cloud-native workload access logs (inspired by AWS CloudTrail / ALB / VPC Flow logs).
Supports:
1. Diurnal circadian rhythm baseline (sinusoidal peak daytime, low night-time traffic).
2. Four parameterized cyber attack patterns:
   - DDOS_HTTP_FLOOD: Volumetric request flood with latency spikes & 503s.
   - CREDENTIAL_STUFFING: Rapid authentication failures (401/403) across accounts.
   - OFF_HOUR_PROBING: Low-and-slow reconnaissance targeting administrative endpoints at 03:00.
   - DATA_EXFILTRATION: Normal request frequency but anomalous response payload bytes.
3. Real-time streaming generator, batch list, or Pandas DataFrame exports.
"""

from datetime import datetime, timedelta, timezone
import math
import random
import uuid
from typing import Callable, Generator, List, Optional
import argparse
import csv
import json
import os
from pathlib import Path

from api.schemas import (
    AttackType,
    BurstMode,
    CloudTelemetryLog,
    HttpMethod,
    UserAgentCategory,
)

# Standard baseline configuration
LEGITIMATE_ENDPOINTS = [
    ("/api/v1/workloads/status", HttpMethod.GET),
    ("/api/v1/workloads/metrics", HttpMethod.GET),
    ("/api/v1/tasks/submit", HttpMethod.POST),
    ("/api/v1/auth/session/refresh", HttpMethod.POST),
    ("/api/v1/catalog/items", HttpMethod.GET),
    ("/api/v1/orders/create", HttpMethod.POST),
    ("/healthz", HttpMethod.GET),
]

LEGITIMATE_USERS = [
    "arn:aws:iam::123456789012:user/dev-alice",
    "arn:aws:iam::123456789012:user/dev-bob",
    "arn:aws:iam::123456789012:user/ops-carol",
    "service-role/worker-task-runner",
    "service-role/alb-health-checker",
]

LEGITIMATE_IP_POOLS = [
    "10.0.1.",  # Internal VPC Private Subnet
    "10.0.2.",
    "192.168.1.",
    "198.51.100.",  # Standard Documentation Public IPs
    "203.0.113.",
]

SENSITIVE_ADMIN_ENDPOINTS = [
    "/admin/debug/environment",
    "/actuator/env",
    "/.git/config",
    "/api/v1/secrets/dump",
    "/aws/iam/credentials",
    "/api/v1/database/raw-query",
]


class SyntheticTelemetryGenerator:
    """Configurable synthetic telemetry generator for cloud temporal threat detection."""

    def __init__(self, seed: Optional[int] = 42) -> None:
        if seed is not None:
            random.seed(seed)
        self.base_time = datetime.now(timezone.utc) - timedelta(hours=24)

    def _get_diurnal_multiplier(self, dt: datetime) -> float:
        """Compute diurnal traffic curve factor based on UTC hour.

        Peak at 14:00 UTC (multiplier ~ 1.8), trough at 03:00 UTC (multiplier ~ 0.25).
        """
        hour = dt.hour + (dt.minute / 60.0)
        # Cosine peak around 14:00
        cycle = math.cos((hour - 14) * 2 * math.pi / 24)
        # Scale between 0.25 and 1.75
        multiplier = 1.0 + (0.75 * cycle)
        return max(0.15, multiplier)

    def calculate_burst_multiplier(
        self,
        burst_mode: BurstMode,
        progress_ratio: float,
        jitter_pct: float = 0.15,
    ) -> float:
        """Calculate dynamic amplification multiplier according to temporal burst profile.

        Args:
            burst_mode: Profile shape (CONSTANT, SUDDEN_SPIKE, GRADUAL_RAMP, PERIODIC_PULSES)
            progress_ratio: Normalized progression of attack duration from 0.0 to 1.0
            jitter_pct: Bounded stochastic noise percentage [0.0, 0.9]

        Returns:
            Calculated volume scaling multiplier.
        """
        t = max(0.0, min(1.0, progress_ratio))

        if burst_mode == BurstMode.SUDDEN_SPIKE:
            # Concentrated Gaussian impulse centered at t = 0.5 (std dev ~ 0.1)
            sigma = 0.10
            envelope = math.exp(-((t - 0.5) ** 2) / (2 * (sigma ** 2)))
            base_multiplier = 0.2 + (3.3 * envelope)

        elif burst_mode == BurstMode.GRADUAL_RAMP:
            # Escalating curve from 0.2 to 2.5
            base_multiplier = 0.2 + (2.3 * t)

        elif burst_mode == BurstMode.PERIODIC_PULSES:
            # Pulsing on/off periods (5 complete cycle pulses across duration)
            pulse = math.sin(t * 10 * math.pi)
            base_multiplier = 2.4 if pulse > 0.0 else 0.25

        else:  # BurstMode.CONSTANT
            base_multiplier = 1.0

        if jitter_pct > 0.0:
            jitter = random.uniform(1.0 - jitter_pct, 1.0 + jitter_pct)
            base_multiplier *= jitter

        return max(0.05, base_multiplier)

    def generate_baseline_event(
        self, current_time: datetime, is_synthetic: bool = False
    ) -> CloudTelemetryLog:
        """Generate a single clean, legitimate cloud workload telemetry record."""
        endpoint, method = random.choice(LEGITIMATE_ENDPOINTS)
        ip_prefix = random.choice(LEGITIMATE_IP_POOLS)
        source_ip = f"{ip_prefix}{random.randint(2, 250)}"
        user = random.choice(LEGITIMATE_USERS)

        # Baseline response characteristics
        # Latency log-normal distribution centered around 35ms - 55ms
        duration_ms = max(5.0, random.lognormvariate(3.6, 0.4))
        # Mostly 200 OK, occasionally 201 Created (10%), rare 400 Bad Request (2%), rare 500 (0.2%)
        status_prob = random.random()
        if status_prob < 0.88:
            status_code = 200
            error_code = None
        elif status_prob < 0.96:
            status_code = 201 if method == HttpMethod.POST else 200
            error_code = None
        elif status_prob < 0.995:
            status_code = 400
            error_code = "InvalidParameterException"
        else:
            status_code = 500
            error_code = "InternalServiceError"

        request_bytes = int(random.normalvariate(500, 120))
        response_bytes = int(random.normalvariate(2200, 450))

        agent_prob = random.random()
        if agent_prob < 0.70:
            ua = UserAgentCategory.BROWSER
        elif agent_prob < 0.95:
            ua = UserAgentCategory.MOBILE_APP
        else:
            ua = UserAgentCategory.CLI_SDK

        return CloudTelemetryLog(
            timestamp=current_time,
            source_ip=source_ip,
            user_identity=user,
            service_name="cloud-ingress-api",
            endpoint=endpoint,
            http_method=method,
            status_code=status_code,
            request_bytes=max(64, request_bytes),
            response_bytes=max(128, response_bytes),
            duration_ms=round(duration_ms, 2),
            geo_country=random.choice(["US", "US", "DE", "IN", "GB"]),
            user_agent_category=ua,
            error_code=error_code,
            is_synthetic_attack=is_synthetic,
            attack_label=AttackType.BASELINE_NORMAL,
        )

    def generate_attack_event(
        self,
        attack_type: AttackType,
        current_time: datetime,
        intensity: float = 3.5,
        target_endpoint: Optional[str] = None,
    ) -> CloudTelemetryLog:
        """Generate a malicious event matching parameterized attack characteristics."""
        if attack_type == AttackType.DDOS_HTTP_FLOOD:
            # High rate, bot agents, latency degradation, occasional 503 Service Unavailable
            ep = target_endpoint or "/api/v1/catalog/items"
            bot_ip = f"{random.randint(11, 220)}.{random.randint(1, 250)}.{random.randint(1, 250)}.{random.randint(1, 250)}"
            status = 503 if random.random() < 0.45 else 200
            return CloudTelemetryLog(
                timestamp=current_time,
                source_ip=bot_ip,
                user_identity="anonymous-unauthenticated",
                service_name="cloud-ingress-api",
                endpoint=ep,
                http_method=HttpMethod.GET,
                status_code=status,
                request_bytes=random.randint(200, 600),
                response_bytes=1024 if status == 200 else 256,
                # Server latency spikes under saturation
                duration_ms=round(random.uniform(450.0, 1850.0) * (intensity / 3.0), 2),
                geo_country=random.choice(["CN", "RU", "BR", "RO", "US"]),
                user_agent_category=UserAgentCategory.AUTOMATED_BOT,
                error_code="ServiceUnavailable" if status == 503 else None,
                is_synthetic_attack=True,
                attack_label=attack_type,
            )

        elif attack_type == AttackType.CREDENTIAL_STUFFING:
            # Concentrated POSTs to auth endpoint with 401 Unauthorized errors
            ep = target_endpoint or "/api/v1/auth/login"
            attacker_ip = f"185.220.101.{random.randint(5, 240)}"
            targeted_user = f"victim-user-{random.randint(100, 999)}@company.com"
            # 85% failed attempts, 15% 403 or locked
            status = 401 if random.random() < 0.88 else 403
            return CloudTelemetryLog(
                timestamp=current_time,
                source_ip=attacker_ip,
                user_identity=targeted_user,
                service_name="auth-service",
                endpoint=ep,
                http_method=HttpMethod.POST,
                status_code=status,
                request_bytes=random.randint(450, 750),
                response_bytes=random.randint(300, 600),
                duration_ms=round(random.uniform(80.0, 220.0), 2),
                geo_country=random.choice(["NL", "RU", "UA", "SC"]),
                user_agent_category=UserAgentCategory.AUTOMATED_BOT,
                error_code="InvalidCredentials" if status == 401 else "AccountLocked",
                is_synthetic_attack=True,
                attack_label=attack_type,
            )

        elif attack_type == AttackType.OFF_HOUR_PROBING:
            # Low volume, scanning forbidden or admin debug endpoints
            ep = target_endpoint or random.choice(SENSITIVE_ADMIN_ENDPOINTS)
            probe_ip = f"45.33.32.{random.randint(10, 150)}"
            status = random.choice([404, 403, 401])
            return CloudTelemetryLog(
                timestamp=current_time,
                source_ip=probe_ip,
                user_identity="arn:aws:iam::123456789012:role/unauthorized-recon",
                service_name="cloud-ingress-api",
                endpoint=ep,
                http_method=HttpMethod.GET,
                status_code=status,
                request_bytes=random.randint(120, 380),
                response_bytes=random.randint(150, 400),
                duration_ms=round(random.uniform(20.0, 90.0), 2),
                geo_country="IR",
                user_agent_category=UserAgentCategory.CLI_SDK,
                error_code="AccessDenied" if status == 403 else "ResourceNotFound",
                is_synthetic_attack=True,
                attack_label=attack_type,
            )

        elif attack_type == AttackType.DATA_EXFILTRATION:
            # Massive response payload sizes via data export or s3 retrieval
            ep = target_endpoint or "/api/v1/data/export"
            rogue_user = "arn:aws:iam::123456789012:user/compromised-admin"
            rogue_ip = "198.51.100.99"
            # Megabyte scale payload
            megabytes = random.uniform(8.0, 45.0) * intensity
            return CloudTelemetryLog(
                timestamp=current_time,
                source_ip=rogue_ip,
                user_identity=rogue_user,
                service_name="storage-service",
                endpoint=ep,
                http_method=HttpMethod.GET,
                status_code=200,
                request_bytes=random.randint(300, 600),
                response_bytes=int(megabytes * 1024 * 1024),
                duration_ms=round(random.uniform(1500.0, 5200.0), 2),
                geo_country="US",
                user_agent_category=UserAgentCategory.CLI_SDK,
                error_code=None,
                is_synthetic_attack=True,
                attack_label=attack_type,
            )

        else:
            return self.generate_baseline_event(current_time, is_synthetic=True)

    def generate_timeline_stream(
        self,
        start_time: datetime,
        duration_minutes: int = 60,
        baseline_rate_per_min: int = 120,
        attack_injections: Optional[List[dict]] = None,
    ) -> Generator[CloudTelemetryLog, None, None]:
        """Generate a sequential time-ordered stream of telemetry events with scheduled attack phases.

        attack_injections format:
        [
            {
                "offset_minute": 20,
                "duration_minutes": 5,
                "attack_type": AttackType.DDOS_HTTP_FLOOD,
                "intensity": 5.0,
                "extra_rate_per_min": 1000
            }
        ]
        """
        attack_injections = attack_injections or []
        current_time = start_time
        total_seconds = duration_minutes * 60

        # Build schedule lookup
        for sec_offset in range(total_seconds):
            current_sec_time = start_time + timedelta(seconds=sec_offset)
            minute_offset = sec_offset // 60

            # Calculate diurnal rate for this second
            diurnal_factor = self._get_diurnal_multiplier(current_sec_time)
            sec_baseline_rate = (baseline_rate_per_min * diurnal_factor) / 60.0

            # Poisson-distributed event generation for natural randomness
            num_baseline_events = int(random.normalvariate(sec_baseline_rate, math.sqrt(max(0.1, sec_baseline_rate))))
            num_baseline_events = max(0, num_baseline_events)

            for _ in range(num_baseline_events):
                yield self.generate_baseline_event(current_sec_time)

            # Check if any attack is active during this minute
            for injection in attack_injections:
                inj_start = injection.get("offset_minute", 0)
                inj_dur = injection.get("duration_minutes", 1)
                if inj_start <= minute_offset < (inj_start + inj_dur):
                    inj_type = injection.get("attack_type", AttackType.DDOS_HTTP_FLOOD)
                    base_intensity = injection.get("intensity", 3.0)
                    burst_mode = injection.get("burst_mode", BurstMode.CONSTANT)
                    if isinstance(burst_mode, str):
                        burst_mode = BurstMode(burst_mode)
                    jitter_pct = injection.get("jitter_pct", 0.15)

                    # Compute progress ratio through this injection phase
                    elapsed_min = minute_offset - inj_start
                    progress = elapsed_min / max(1.0, float(inj_dur))
                    burst_factor = self.calculate_burst_multiplier(
                        burst_mode=burst_mode,
                        progress_ratio=progress,
                        jitter_pct=jitter_pct,
                    )
                    effective_intensity = base_intensity * burst_factor

                    extra_rate = (injection.get("extra_rate_per_min", 400) * burst_factor) / 60.0
                    num_attack_events = max(1, int(random.normalvariate(extra_rate, math.sqrt(max(0.1, extra_rate)))))

                    for _ in range(num_attack_events):
                        yield self.generate_attack_event(
                            attack_type=inj_type,
                            current_time=current_sec_time,
                            intensity=effective_intensity,
                            target_endpoint=injection.get("target_endpoint"),
                        )

    def generate_batch(
        self,
        count: int = 500,
        start_time: Optional[datetime] = None,
        inject_attack: Optional[AttackType] = None,
        attack_ratio: float = 0.25,
        intensity: float = 3.5,
        burst_mode: BurstMode = BurstMode.CONSTANT,
        jitter_pct: float = 0.15,
    ) -> List[CloudTelemetryLog]:
        """Generate a fixed-size batch of events, optionally blending an injected attack with burst shaping."""
        start_time = start_time or (datetime.now(timezone.utc) - timedelta(hours=1))
        events: List[CloudTelemetryLog] = []

        attack_count = int(count * attack_ratio) if inject_attack else 0
        baseline_count = count - attack_count

        time_step = timedelta(seconds=max(1, 3600 // count))
        current = start_time

        # Generate baseline
        for _ in range(baseline_count):
            events.append(self.generate_baseline_event(current))
            current += time_step

        # Generate injected attack events if requested with burst shaping
        if inject_attack:
            for i in range(attack_count):
                progress = i / max(1.0, float(attack_count - 1))
                burst_factor = self.calculate_burst_multiplier(
                    burst_mode=burst_mode,
                    progress_ratio=progress,
                    jitter_pct=jitter_pct,
                )
                effective_intensity = intensity * burst_factor

                events.append(
                    self.generate_attack_event(
                        attack_type=inject_attack,
                        current_time=current,
                        intensity=effective_intensity,
                    )
                )
                current += timedelta(milliseconds=random.randint(50, 400))

        # Sort chronologically
        events.sort(key=lambda e: e.timestamp)
        return events

    def to_dataframe(self, events: List[CloudTelemetryLog]):
        """Convert a list of telemetry events to a Pandas DataFrame."""
        import pandas as pd

        records = [e.model_dump() for e in events]
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def export_dataset_jsonl(self, filepath: str, events: List[CloudTelemetryLog]) -> int:
        """Export events to JSON Lines (JSONL) format for cloud log ingestion.

        Args:
            filepath: Destination file path.
            events: List of telemetry event models.

        Returns:
            Number of written records.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event.model_dump(), default=str) + "\n")
        return len(events)

    def export_dataset_csv(self, filepath: str, events: List[CloudTelemetryLog]) -> int:
        """Export events to flat CSV format for analytics and scikit-learn training.

        Args:
            filepath: Destination CSV path.
            events: List of telemetry event models.

        Returns:
            Number of written records.
        """
        if not events:
            return 0
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(events[0].model_dump().keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for event in events:
                writer.writerow(event.model_dump(mode="json"))
        return len(events)

    def export_benchmark_suite(
        self,
        output_dir: str,
        count: int = 1000,
        inject_attack: Optional[AttackType] = None,
        attack_ratio: float = 0.25,
        burst_mode: BurstMode = BurstMode.CONSTANT,
        intensity: float = 3.5,
        jitter_pct: float = 0.15,
    ) -> dict:
        """Generate a complete labeled benchmark dataset with JSONL, CSV, and manifest.json.

        Used for faculty demonstrations, model training benchmarks, and Viva defense evaluation.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        events = self.generate_batch(
            count=count,
            inject_attack=inject_attack,
            attack_ratio=attack_ratio,
            intensity=intensity,
            burst_mode=burst_mode,
            jitter_pct=jitter_pct,
        )

        jsonl_file = out_path / "telemetry_stream.jsonl"
        csv_file = out_path / "telemetry_stream.csv"
        manifest_file = out_path / "manifest.json"

        self.export_dataset_jsonl(str(jsonl_file), events)
        self.export_dataset_csv(str(csv_file), events)

        # Build ground-truth validation manifest
        attack_indices = [i for i, e in enumerate(events) if e.is_synthetic_attack]
        durations = [e.duration_ms for e in events]
        bytes_out = [e.response_bytes for e in events]
        error_count = sum(1 for e in events if e.status_code >= 400)

        manifest = {
            "dataset_id": str(uuid.uuid4()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "record_count": len(events),
            "baseline_count": len(events) - len(attack_indices),
            "attack_count": len(attack_indices),
            "attack_ratio": round(len(attack_indices) / max(1, len(events)), 4),
            "injected_attack_type": inject_attack.value if inject_attack else None,
            "burst_mode": burst_mode.value,
            "intensity_multiplier": intensity,
            "jitter_pct": jitter_pct,
            "ground_truth_anomalous_indices": attack_indices,
            "temporal_bounds": {
                "start": events[0].timestamp.isoformat() if events else None,
                "end": events[-1].timestamp.isoformat() if events else None,
            },
            "summary_metrics": {
                "mean_duration_ms": round(sum(durations) / max(1, len(durations)), 2),
                "total_response_bytes": sum(bytes_out),
                "error_rate": round(error_count / max(1, len(events)), 4),
            },
            "files": {
                "jsonl": str(jsonl_file.name),
                "csv": str(csv_file.name),
            },
        }

        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def stream_realtime_events(
        self,
        events: List[CloudTelemetryLog],
        speed_multiplier: float = 1.0,
        max_delay_seconds: float = 1.0,
        on_event: Optional[Callable[[CloudTelemetryLog], None]] = None,
    ) -> Generator[CloudTelemetryLog, None, None]:
        """Stream telemetry events sequentially with time-scaled pacing for live visualizer replay.

        Args:
            events: Sequence of events sorted chronologically.
            speed_multiplier: Replay acceleration factor (e.g. 5.0 = 5x speed).
            max_delay_seconds: Upper cap on inter-arrival delay sleep.
            on_event: Optional callback executed as each record is yielded.
        """
        import time

        for i, event in enumerate(events):
            if i > 0 and speed_multiplier > 0:
                delta = (event.timestamp - events[i - 1].timestamp).total_seconds()
                scaled_delay = max(0.0, delta / speed_multiplier)
                time.sleep(min(scaled_delay, max_delay_seconds))

            if on_event:
                on_event(event)
            yield event


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic cloud telemetry.")
    parser.add_argument("--count", type=int, default=100, help="Number of records to generate")
    parser.add_argument(
        "--attack",
        type=str,
        default="none",
        choices=["none", "ddos", "credential_stuffing", "off_hour", "data_exfil"],
        help="Attack injection scenario",
    )
    parser.add_argument(
        "--burst-mode",
        type=str,
        default="constant",
        choices=["constant", "sudden_spike", "gradual_ramp", "periodic_pulses"],
        help="Temporal burst profile shape",
    )
    parser.add_argument("--intensity", type=float, default=3.5, help="Anomaly intensity multiplier")
    parser.add_argument("--jitter", type=float, default=0.15, help="Stochastic jitter noise factor")
    parser.add_argument("--out", type=str, default="", help="Optional JSONL file output path")
    parser.add_argument("--export-dir", type=str, default="", help="Export complete labeled benchmark suite directory")
    args = parser.parse_args()

    attack_map = {
        "ddos": AttackType.DDOS_HTTP_FLOOD,
        "credential_stuffing": AttackType.CREDENTIAL_STUFFING,
        "off_hour": AttackType.OFF_HOUR_PROBING,
        "data_exfil": AttackType.DATA_EXFILTRATION,
        "none": None,
    }

    burst_map = {
        "constant": BurstMode.CONSTANT,
        "sudden_spike": BurstMode.SUDDEN_SPIKE,
        "gradual_ramp": BurstMode.GRADUAL_RAMP,
        "periodic_pulses": BurstMode.PERIODIC_PULSES,
    }

    generator = SyntheticTelemetryGenerator(seed=42)
    selected_attack = attack_map.get(args.attack)
    selected_burst = burst_map.get(args.burst_mode, BurstMode.CONSTANT)

    if args.export_dir:
        manifest = generator.export_benchmark_suite(
            output_dir=args.export_dir,
            count=args.count,
            inject_attack=selected_attack,
            attack_ratio=0.25 if selected_attack else 0.0,
            burst_mode=selected_burst,
            intensity=args.intensity,
            jitter_pct=args.jitter,
        )
        print(f"Benchmark suite exported to: {args.export_dir}")
        print(f"Manifest: Total={manifest['record_count']}, Baseline={manifest['baseline_count']}, Attacks={manifest['attack_count']}")
    else:
        batch = generator.generate_batch(
            count=args.count,
            inject_attack=selected_attack,
            burst_mode=selected_burst,
            intensity=args.intensity,
            jitter_pct=args.jitter,
        )
        print(f"Generated {len(batch)} telemetry events. Injected attack: {args.attack} (Burst: {args.burst_mode})")
        sample = batch[0]
        print(f"Sample Event: ID={sample.event_id} Time={sample.timestamp} IP={sample.source_ip} Path={sample.endpoint} Status={sample.status_code}")

        if args.out:
            written = generator.export_dataset_jsonl(args.out, batch)
            print(f"Successfully saved {written} records to {args.out}")
