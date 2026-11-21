"""Metadatos de chunk: ``chunk_id`` determinista, sha256, versionado e ``ingested_at``.

Cada ``ChunkDraft`` (texto + páginas + ruta de sección) se combina con la entrada del
manifiesto de su documento para producir un :class:`~docuerp.schemas.Chunk` completo:
procedencia (``source``, ``version``), vigencia (``status``), control de acceso
(``allowed_roles``) y huella (``content_hash``) para detectar duplicados.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from docuerp.ingest.chunk import ChunkDraft
from docuerp.ingest.manifest import SourceEntry
from docuerp.schemas import Chunk, make_chunk_id
from docuerp.text import normalize_text

__all__ = ["build_chunks", "content_hash", "make_chunk_id", "utc_now_iso"]


def content_hash(text: str) -> str:
    """sha256 hexadecimal del texto normalizado (NFC, espacios colapsados).

    Normalizar antes de hashear hace que dos chunks que solo difieren en espacios o en la
    forma Unicode de una tilde se reconozcan como duplicados.
    """
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    """Marca de tiempo ISO-8601 UTC (``2026-11-17T08:00:00Z``)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_chunks(
    drafts: list[ChunkDraft],
    entry: SourceEntry,
    *,
    flags_by_index: dict[int, list[str]] | None = None,
    ingested_at: str | None = None,
) -> list[Chunk]:
    """Convierte borradores en ``Chunk`` con metadatos del manifiesto.

    ``idx`` del ``chunk_id`` es el índice del chunk dentro de su página inicial, de modo
    que el identificador es estable mientras no cambie el contenido de esa página.

    Args:
        drafts: borradores en orden de documento.
        entry: entrada del manifiesto del documento.
        flags_by_index: marcas por posición del borrador (p. ej. ``suspicious_instruction``).
        ingested_at: marca de tiempo común a toda la corrida (por defecto, ahora).

    Raises:
        ValueError: si una página tiene más de 1000 chunks (límite del formato del id).
    """
    stamp = ingested_at or utc_now_iso()
    per_page: dict[int, int] = {}
    chunks: list[Chunk] = []
    for index, draft in enumerate(drafts):
        idx = per_page.get(draft.page, 0)
        per_page[draft.page] = idx + 1
        flags = sorted(set((flags_by_index or {}).get(index, [])))
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(entry.id, draft.page, idx),
                doc_id=entry.id,
                text=draft.text,
                source=entry.source_file,
                page=draft.page,
                page_end=max(draft.page, draft.page_end),
                version=entry.version,
                erp_version=entry.erp_version,
                status=entry.status,
                section_path=draft.section_path,
                module=entry.module,
                doc_type=entry.doc_type,
                language=entry.language,
                allowed_roles=list(entry.allowed_roles),
                content_hash=content_hash(draft.text),
                ingested_at=stamp,
                token_count=draft.token_count,
                flags=flags,
            )
        )
    return chunks
