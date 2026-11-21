"""Fixtures de las pruebas de la API: aplicación en modo mock, servicio fake y datos temporales."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from docuerp.api.app import create_app
from docuerp.api.datasources import ApiPaths, FileDataSource
from docuerp.api.mock import MockService, build_pdf

# Con --import-mode=importlib el directorio de pruebas no entra en sys.path: se agrega para
# poder importar el módulo auxiliar ``api_helpers``.
sys.path.insert(0, str(Path(__file__).parent))

from api_helpers import FakeService


@pytest.fixture
def fake_service() -> FakeService:
    return FakeService()


@pytest.fixture
def mock_client() -> TestClient:
    """Cliente sobre la app en modo mock y sin retardos."""
    app = create_app(mock=True, service=MockService(delay_scale=0.0), rate_limit_per_min=1000)
    return TestClient(app)


@pytest.fixture
def api_paths(tmp_path: Path) -> ApiPaths:
    """Árbol de archivos temporal con manifiesto, PDF/DOCX, chunks, evaluación y seguridad."""
    manifest = tmp_path / "manifest"
    pdf_dir = tmp_path / "build" / "pdf"
    docx_dir = tmp_path / "runbooks" / "docx"
    processed = tmp_path / "processed"
    for directory in (manifest, pdf_dir, docx_dir, processed, tmp_path / "index" / "structure"):
        directory.mkdir(parents=True)

    entries = [
        {
            "id": "odoo-18.0-inventario",
            "title": "Odoo 18.0 Inventario",
            "source_file": "odoo-18.0-inventario.pdf",
            "allowed_roles": ["portal_user", "n1", "n2", "admin"],
        },
        {
            "id": "RB-ADM-001@v1.0",
            "title": "Runbook restringido",
            "source_file": "RB-ADM-001_v1.0.docx",
            "allowed_roles": ["admin"],
        },
        {"id": "sin-roles", "title": "Sin roles", "source_file": "sin-roles.pdf"},
    ]
    (manifest / "odoo.yaml").write_text(yaml.safe_dump(entries), encoding="utf-8")
    (pdf_dir / "odoo-18.0-inventario.pdf").write_bytes(build_pdf([["Página uno"]]))
    (docx_dir / "RB-ADM-001_v1.0.docx").write_bytes(b"PK-fake-docx")
    (pdf_dir / "sin-roles.pdf").write_bytes(build_pdf([["x"]]))
    (tmp_path / "secreto.pdf").write_bytes(b"NO DEBE SERVIRSE")

    chunk = {
        "chunk_id": "odoo-18.0-inventario::p0001::c000",
        "doc_id": "odoo-18.0-inventario",
        "text": "texto",
        "source": "odoo-18.0-inventario.pdf",
        "allowed_roles": ["portal_user", "n1", "n2", "admin"],
    }
    restricted = {**chunk, "chunk_id": "RB-ADM-001@v1.0::p0001::c000", "allowed_roles": ["admin"]}
    (processed / "chunks.structure.jsonl").write_text(
        json.dumps(chunk) + "\n\nlínea corrupta\n" + json.dumps(restricted) + "\n", encoding="utf-8"
    )

    results = tmp_path / "eval_results"
    for run_id, recall in (("20261110-0900_baseline", 0.64), ("20261115-1030_final", 0.88)):
        run_dir = results / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(
            json.dumps({"retrieval": {"recall@5": recall}}), encoding="utf-8"
        )
        (run_dir / "config.json").write_text(json.dumps({"mode": run_id}), encoding="utf-8")
    (results / "20261116-0800_incompleta").mkdir()
    (results / "no es una corrida").mkdir()

    security = tmp_path / "security_results"
    security.mkdir()
    rows = [
        {"id": "RT-1", "vector": "injection_direct", "expected_outcome": "blocked", "passed": True},
        {
            "id": "RT-2",
            "vector": "injection_direct",
            "expected_outcome": "blocked",
            "passed": False,
        },
        {
            "id": "RT-3",
            "vector": "permission_evasion",
            "expected_outcome": "denied",
            "passed": True,
        },
    ]
    (security / "redteam.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    return ApiPaths(
        manifest=manifest,
        document_dirs=(pdf_dir, tmp_path / "runbooks" / "pdf", docx_dir),
        processed=processed,
        index=tmp_path / "index",
        tickets_db=tmp_path / "tickets.db",
        eval_results=results,
        security_results=security,
        audit_log=tmp_path / "audit.jsonl",
    )


@pytest.fixture
def real_client(fake_service: FakeService, api_paths: ApiPaths) -> TestClient:
    """Cliente con servicio fake y archivos temporales (sin modo mock)."""
    app = create_app(
        mock=False,
        service=fake_service,
        data=FileDataSource(api_paths),
        rate_limit_per_min=1000,
    )
    return TestClient(app)
