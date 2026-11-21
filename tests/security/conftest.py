"""Fixtures del módulo de seguridad: guard con embedder ``fake`` y fábrica de chunks."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from docuerp.embeddings import FakeEmbedder
from docuerp.schemas import Chunk, QueryRequest, ScoredChunk, make_chunk_id
from docuerp.security.input_guard import InputGuard, InputGuardConfig
from docuerp.security.permissions import get_policy


@pytest.fixture(autouse=True)
def _fresh_policy() -> Iterator[None]:
    """La política RBAC se cachea por proceso; cada prueba parte de la real."""
    get_policy.cache_clear()
    yield
    get_policy.cache_clear()


@pytest.fixture(scope="session")
def guard() -> InputGuard:
    """Guard completo con embedder ``fake`` (rápido, determinista, sin modelos)."""
    return InputGuard(InputGuardConfig(embedder="fake"), embedder=FakeEmbedder())


@pytest.fixture(scope="session")
def patterns_guard() -> InputGuard:
    """Guard sin capa semántica: aísla el aporte de los patrones."""
    return InputGuard(InputGuardConfig(semantic_enabled=False))


@pytest.fixture
def ask(guard: InputGuard) -> Callable[..., Any]:
    """Evalúa una consulta con el guard ``fake`` y devuelve el ``InputAssessment``."""

    def _ask(text: str, role: str = "n1", user_id: str = "USR-0101") -> Any:
        return guard.assess(QueryRequest(query=text, role=role, user_id=user_id))  # type: ignore[arg-type]

    return _ask


@pytest.fixture
def make_hit() -> Callable[..., ScoredChunk]:
    """Fábrica de ``ScoredChunk`` válidos con los campos que importan a seguridad."""

    def _make(
        doc_id: str = "odoo-18.0-inventario",
        *,
        page: int = 12,
        page_end: int | None = None,
        idx: int = 0,
        text: str = "Para validar una recepción abra la operación y pulse Validar.",
        allowed_roles: list[str] | None = None,
        erp_version: str = "18.0",
        status: str = "vigente",
        flags: list[str] | None = None,
        source: str | None = None,
        rank: int = 1,
        score: float = 0.8,
    ) -> ScoredChunk:
        chunk = Chunk(
            chunk_id=make_chunk_id(doc_id, page, idx),
            doc_id=doc_id,
            text=text,
            source=source or f"{doc_id}.pdf",
            page=page,
            page_end=page_end or page,
            version="v1.0",
            erp_version=erp_version,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            section_path="Inventario > Recepciones",
            module="inventario",
            doc_type="odoo_doc" if doc_id.startswith("odoo") else "runbook",
            language="es",
            allowed_roles=allowed_roles or ["portal_user", "n1", "n2", "admin"],  # type: ignore[arg-type]
            content_hash="a" * 64,
            ingested_at="2026-11-17T08:00:00Z",
            token_count=len(text.split()),
            flags=flags or [],
        )
        return ScoredChunk(chunk=chunk, score=score, rank=rank, retriever="rerank")

    return _make
