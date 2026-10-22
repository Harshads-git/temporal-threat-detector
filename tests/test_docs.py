"""Validation test suite for documentation files, links, and API consistency."""

from __future__ import annotations

from pathlib import Path
import pytest
from api.main import app
from engine.windowing import CANONICAL_FEATURE_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_documentation_files_exist() -> None:
    """Verify all critical documentation artifacts exist and are non-empty."""
    expected_docs = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docker" / "README.md",
        PROJECT_ROOT / "docs" / "ARCHITECTURE.md",
        PROJECT_ROOT / "docs" / "VIVA_PRESENTATION_GUIDE.md",
    ]

    for doc_path in expected_docs:
        assert doc_path.exists(), f"Missing expected document: {doc_path}"
        content = doc_path.read_text(encoding="utf-8")
        assert len(content.strip()) > 100, f"Document appears empty or stub: {doc_path}"


def test_readme_documents_active_api_endpoints() -> None:
    """Verify that all endpoints documented in README.md are implemented in FastAPI app."""
    readme_path = PROJECT_ROOT / "README.md"
    content = readme_path.read_text(encoding="utf-8")

    registered_paths = {route.path for route in app.routes}

    expected_endpoints = [
        "/",
        "/health",
        "/metrics",
        "/ingest",
        "/ingest/batch",
        "/detect",
        "/alerts",
        "/inject",
    ]

    for ep in expected_endpoints:
        assert ep in content, f"README.md does not mention expected API endpoint: {ep}"
        assert ep in registered_paths, f"FastAPI application is missing documented route: {ep}"


def test_architecture_feature_vector_alignment() -> None:
    """Verify that the 13-feature vector in docs/ARCHITECTURE.md matches CANONICAL_FEATURE_NAMES."""
    arch_path = PROJECT_ROOT / "docs" / "ARCHITECTURE.md"
    content = arch_path.read_text(encoding="utf-8")

    assert len(CANONICAL_FEATURE_NAMES) == 13
    for feature in CANONICAL_FEATURE_NAMES:
        assert feature in content, f"docs/ARCHITECTURE.md is missing documented feature: {feature}"


def test_viva_presentation_guide_coverage() -> None:
    """Verify that viva guide includes live demo steps and examiner Q&A."""
    guide_path = PROJECT_ROOT / "docs" / "VIVA_PRESENTATION_GUIDE.md"
    content = guide_path.read_text(encoding="utf-8")

    assert "run_demo.bat" in content
    assert "DDoS" in content
    assert "Credential Stuffing" in content
    assert "Off-Hour" in content
    assert "Isolation Forest" in content
    assert "Tree-SHAP" in content
    assert "Median Absolute Deviation" in content
