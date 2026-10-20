"""Validation test suite for Dockerfile, .dockerignore, and docker-compose orchestration."""

from __future__ import annotations

from pathlib import Path
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_dockerfile_multi_stage_structure() -> None:
    """Verify Dockerfile contains multi-stage builder and hardened non-root runtime."""
    dockerfile_path = PROJECT_ROOT / "docker" / "Dockerfile"
    assert dockerfile_path.exists(), "docker/Dockerfile must exist"

    content = dockerfile_path.read_text(encoding="utf-8")

    # Multi-stage check
    assert "AS builder" in content, "Must define a builder stage"
    assert "AS runtime" in content, "Must define a runtime stage"

    # Security: unprivileged non-root user
    assert "useradd" in content or "adduser" in content, "Must create an unprivileged user"
    assert "USER appuser" in content, "Must drop root privileges and switch to appuser"

    # Port exposure
    assert "EXPOSE 8000 8501" in content or ("8000" in content and "8501" in content), "Must expose API and UI ports"

    # Healthcheck
    assert "HEALTHCHECK" in content, "Must include container healthcheck instruction"


def test_dockerignore_exclusions() -> None:
    """Verify .dockerignore blocks cache, logs, virtual environments, and raw data."""
    dockerignore_path = PROJECT_ROOT / ".dockerignore"
    assert dockerignore_path.exists(), ".dockerignore must exist"

    ignored_patterns = set(
        line.strip()
        for line in dockerignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )

    required_exclusions = {
        "__pycache__",
        ".pytest_cache",
        ".env",
        ".venv",
        ".git",
        "data/raw",
    }

    for req in required_exclusions:
        assert any(req in pat for pat in ignored_patterns), f".dockerignore must exclude {req}"


def test_docker_compose_service_definitions() -> None:
    """Verify docker-compose.yml defines all 4 services with correct dependencies and ports."""
    compose_path = PROJECT_ROOT / "docker" / "docker-compose.yml"
    assert compose_path.exists(), "docker/docker-compose.yml must exist"

    with open(compose_path, "r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    assert "services" in compose_data, "docker-compose must have 'services' section"
    services = compose_data["services"]

    # 1. LocalStack service
    assert "localstack" in services, "Must declare 'localstack' service"
    ls_svc = services["localstack"]
    assert "4566:4566" in ls_svc.get("ports", []), "LocalStack must map port 4566"
    assert "healthcheck" in ls_svc, "LocalStack must declare a healthcheck"

    # 2. Cloud Provisioner Init service
    assert "cloud-init" in services, "Must declare 'cloud-init' service"
    init_svc = services["cloud-init"]
    assert "localstack" in init_svc.get("depends_on", {}), "cloud-init must depend on localstack"

    # 3. FastAPI API service
    assert "api" in services, "Must declare 'api' service"
    api_svc = services["api"]
    assert "8000:8000" in api_svc.get("ports", []), "API must map port 8000"
    assert "localstack" in api_svc.get("depends_on", {}), "api must depend on localstack"
    assert "cloud-init" in api_svc.get("depends_on", {}), "api must depend on cloud-init"
    assert "healthcheck" in api_svc, "api must declare a healthcheck"

    # 4. Streamlit Dashboard service
    assert "dashboard" in services, "Must declare 'dashboard' service"
    dash_svc = services["dashboard"]
    assert "8501:8501" in dash_svc.get("ports", []), "Dashboard must map port 8501"
    assert "api" in dash_svc.get("depends_on", {}), "dashboard must depend on api"

    # Networks
    assert "networks" in compose_data, "docker-compose must declare networks"
    assert "threat-net" in compose_data["networks"], "Must declare threat-net bridge network"
