"""Pre-filtro de metadatos: rol ∈ allowed_roles, versión ∈ {q.erp_version, "any"} y vigencia.

Son las tres reglas de gobernanza del SPEC §5.2. Se aplican **antes** de fusionar y
reordenar, de modo que un fragmento no autorizado nunca llega al reranker ni al LLM.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from docuerp.retrieval.contracts import RetrievalQuery
from docuerp.schemas import Chunk


def passes_filters(chunk: Chunk, q: RetrievalQuery) -> bool:
    """``True`` si el chunk es visible para el rol, aplica a la versión y está vigente.

    - Rol: ``q.role`` debe estar en ``chunk.allowed_roles`` (RBAC documental).
    - Versión: el chunk aplica si es de ``q.erp_version`` o es independiente de versión
      (``"any"``, p. ej. un runbook interno).
    - Vigencia: solo ``status == "vigente"`` salvo que ``q.include_obsolete`` sea cierto.
    """
    if q.role not in chunk.allowed_roles:
        return False
    if chunk.erp_version not in (q.erp_version, "any"):
        return False
    return q.include_obsolete or chunk.status == "vigente"


def split_by_filters(chunks: Iterable[Chunk], q: RetrievalQuery) -> tuple[list[Chunk], list[Chunk]]:
    """Separa ``chunks`` en ``(autorizados, descartados)`` conservando el orden."""
    kept: list[Chunk] = []
    dropped: list[Chunk] = []
    for chunk in chunks:
        (kept if passes_filters(chunk, q) else dropped).append(chunk)
    return kept, dropped


def chroma_where(q: RetrievalQuery) -> dict[str, Any]:
    """Traduce los filtros de ``q`` a una cláusula ``where`` de Chroma.

    Solo cubre versión y vigencia, que son metadatos escalares y se pueden filtrar de forma
    nativa dentro del índice HNSW. El rol **no** se incluye: ``allowed_roles`` se guarda como
    texto ``"|n1|n2|"`` y Chroma no admite "contiene" sobre metadatos, así que el rol se
    verifica en Python con :func:`passes_filters` (SPEC §8).
    """
    clauses: list[dict[str, Any]] = [{"erp_version": {"$in": [q.erp_version, "any"]}}]
    if not q.include_obsolete:
        clauses.append({"status": {"$eq": "vigente"}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
