"""Fixtures compartidas de pruebas."""

from __future__ import annotations

from typing import Any

import pytest

from docuerp.schemas import make_chunk_id


@pytest.fixture
def chunk_payload() -> dict[str, Any]:
    """Payload válido de ``Chunk`` para construir variantes en pruebas."""
    doc_id = "odoo-18.0-inventario"
    return {
        "chunk_id": make_chunk_id(doc_id, 12, 0),
        "doc_id": doc_id,
        "text": "Inventario > Operaciones > Recepciones\nPara validar una recepción...",
        "source": "odoo-18.0-inventario.pdf",
        "page": 12,
        "page_end": 13,
        "version": "18.0@a1b2c3d",
        "erp_version": "18.0",
        "status": "vigente",
        "section_path": "Inventario > Operaciones > Recepciones",
        "module": "inventario",
        "doc_type": "odoo_doc",
        "language": "en",
        "allowed_roles": ["portal_user", "n1", "n2", "admin"],
        "content_hash": "a" * 64,
        "ingested_at": "2026-11-17T08:00:00Z",
        "token_count": 312,
    }
