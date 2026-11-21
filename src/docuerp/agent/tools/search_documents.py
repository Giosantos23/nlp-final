"""Herramienta ``search_documents(query, role, erp_version, top_k≤8)``.

Envuelve al recuperador híbrido de ``docuerp.retrieval`` (que ya aplica los filtros de rol,
versión y vigencia y el gate de evidencia τ). La herramienta devuelve datos serializables
a JSON: los hits completos (para que el generador cite) y un resumen por etapa (para el
inspector de traza), nunca objetos internos.

El rol de los argumentos debe coincidir con el rol autenticado del contexto: si alguien
intenta buscar "como admin" siendo ``n1``, la llamada se deniega.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from docuerp.agent.tools.base import ToolContext, ToolResult, ToolSpec, tool_roles
from docuerp.schemas import ErpVersion, UserRole

if TYPE_CHECKING:
    from docuerp.config import Settings
    from docuerp.retrieval.contracts import RetrievalQuery, RetrievalResult, Retriever

MAX_TOP_K = 8
"""Tope duro de fragmentos por búsqueda (acota contexto y costo)."""

STAGE_PREVIEW = 5
"""Cuántos candidatos por etapa se resumen en la traza."""


class SearchDocumentsInput(BaseModel):
    """Argumentos de ``search_documents``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=3, max_length=2000)
    role: UserRole
    erp_version: ErpVersion = "18.0"
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)


class LazyRetriever:
    """Construye el recuperador configurado en ``settings.retrieval`` en el primer uso.

    Así importar el agente (o listar herramientas) no carga bge-m3 ni el reranker.
    """

    def __init__(self) -> None:
        self._retriever: Retriever | None = None
        self._lock = threading.Lock()

    def _build(self) -> Retriever:
        """Crea (una sola vez, de forma segura entre hilos) el recuperador real."""
        with self._lock:
            if self._retriever is None:
                from docuerp.config import get_settings
                from docuerp.retrieval import build_retriever

                cfg = get_settings().retrieval
                self._retriever = build_retriever(
                    cfg.mode, chunker=cfg.chunker, embedder=cfg.embedder
                )
            return self._retriever

    def retrieve(self, q: RetrievalQuery) -> RetrievalResult:
        """Delegado al recuperador real."""
        return self._build().retrieve(q)


def summarize_stages(result: RetrievalResult) -> dict[str, list[dict[str, Any]]]:
    """Resumen sanitizado por etapa: ``chunk_id``, fuente, página, rank y score (sin texto)."""
    return {
        stage: [
            {
                "chunk_id": hit.chunk.chunk_id,
                "source": hit.chunk.source,
                "page": hit.chunk.page,
                "rank": hit.rank,
                "score": round(hit.score, 4),
            }
            for hit in hits[:STAGE_PREVIEW]
        ]
        for stage, hits in result.stages.items()
    }


async def search_documents(
    args: SearchDocumentsInput,
    *,
    retriever: Retriever | None = None,
    candidates_k: int = 20,
    include_obsolete: bool = False,
) -> ToolResult:
    """Busca en el corpus autorizado para el rol."""
    from docuerp.retrieval.contracts import RetrievalQuery

    active = retriever or LazyRetriever()
    query = RetrievalQuery(
        text=args.query,
        role=args.role,
        erp_version=args.erp_version,
        top_k=args.top_k,
        candidates_k=max(candidates_k, args.top_k),
        include_obsolete=include_obsolete,
    )
    # La recuperación (embeddings, reranker) es bloqueante: se ejecuta en un hilo.
    result = await asyncio.to_thread(active.retrieve, query)
    return ToolResult(
        tool="search_documents",
        ok=True,
        data={
            "hits": [hit.model_dump(mode="json") for hit in result.hits],
            "evidence_ok": result.evidence_ok,
            "top_score": result.top_score,
            "filtered_out": result.filtered_out,
            "timings_ms": dict(result.timings_ms),
            "stages": summarize_stages(result),
        },
    )


def build_search_spec(retriever: Retriever, *, settings: Settings) -> ToolSpec:
    """Especificación registrable de ``search_documents`` ligada a ``retriever``."""

    async def _handler(args: SearchDocumentsInput, ctx: ToolContext) -> ToolResult:
        if args.role != ctx.role:
            return ToolResult(
                tool="search_documents",
                ok=False,
                error="El rol de la búsqueda no coincide con el del usuario autenticado.",
                data={"denied": True},
            )
        return await search_documents(
            args,
            retriever=retriever,
            candidates_k=settings.retrieval.candidates_k,
            include_obsolete=settings.retrieval.include_obsolete,
        )

    return ToolSpec(
        name="search_documents",
        description="Busca procedimientos en manuales Odoo y runbooks autorizados para el rol "
        f"y la versión (top_k ≤ {MAX_TOP_K}).",
        input_model=SearchDocumentsInput,
        allowed_roles=tool_roles("search_documents"),
        # La primera búsqueda carga bge-m3 y el reranker: se da un margen mínimo de 30 s.
        timeout_s=max(settings.agent.tool_timeout_s, 30.0),
        max_calls_per_query=2,
        handler=_handler,
    )
