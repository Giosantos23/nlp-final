"""Precalentamiento del agente al arrancar la API.

Sin precalentar, la primera consulta paga la carga de bge-m3 (guard semántico y router), del
índice BM25/Chroma y del reranker: unos 10 a 20 s en este equipo. Al iniciar la API se lanza
en segundo plano una consulta "de mentira" por cada componente, **sobre las mismas
instancias que usa el agente** (``get_agent_service().deps``), así los modelos quedan
cargados una sola vez en el proceso y la primera consulta real ya va en caliente.

Pasos (en orden, cada uno con su duración): ``agent`` (construir el grafo), ``input_guard``
(banco de ataques + bge-m3), ``router`` (clasificador de intención), ``retrieval`` (BM25,
Chroma, embedder y reranker vía ``search_documents``), ``classifier`` (clasificador de
tickets) y ``docx_pdf`` (versiones PDF de los runbooks DOCX). Ninguno llama al LLM ni
escribe en la BD de tickets.

El estado se expone en ``GET /api/health`` (campo ``warmup``). Un paso que falla no detiene
los siguientes: la API sigue atendiendo y ese componente se cargará en la primera consulta.
Se desactiva con ``DOCUERP_API_WARMUP=0``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

logger = logging.getLogger("docuerp.api")

WarmupState = Literal["disabled", "pending", "running", "ready", "partial"]
StepStatus = Literal["pending", "running", "ok", "error", "skipped"]

PROBE_QUERY = "¿Cómo valido una recepción de compra en inventario?"
"""Consulta neutra (documental, sin PII) para ejercitar guard, router y recuperación."""

PROBE_TICKET = "Al validar la factura de proveedor aparece un error de impuestos."
"""Texto neutro para el clasificador de tickets."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class WarmupStep:
    """Un componente a precalentar."""

    name: str
    label: str
    status: StepStatus = "pending"
    duration_ms: float | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "status": self.status,
            "duration_ms": None if self.duration_ms is None else round(self.duration_ms, 1),
            "error": self.error,
        }


@dataclass
class Warmup:
    """Estado observable del precalentamiento (lo lee ``/api/health``)."""

    enabled: bool = True
    started_at: str | None = None
    finished_at: str | None = None
    total_ms: float | None = None
    steps: list[WarmupStep] = field(default_factory=list)

    @property
    def state(self) -> WarmupState:
        """``ready`` si todo cargó, ``partial`` si algún paso falló."""
        if not self.enabled:
            return "disabled"
        if self.started_at is None:
            return "pending"
        if self.finished_at is None:
            return "running"
        return "ready" if all(s.status in ("ok", "skipped") for s in self.steps) else "partial"

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_ms": None if self.total_ms is None else round(self.total_ms, 1),
            "steps": [s.as_dict() for s in self.steps],
        }

    async def _run_step(self, step: WarmupStep, action: Callable[[], Awaitable[None]]) -> None:
        step.status = "running"
        started = time.perf_counter()
        try:
            await action()
        except Exception as exc:  # un paso fallido no detiene el resto
            step.status = "error"
            step.error = type(exc).__name__
            logger.warning("precalentamiento: %s falló (%s)", step.name, type(exc).__name__)
        else:
            step.status = "ok"
        finally:
            step.duration_ms = (time.perf_counter() - started) * 1000

    async def run(self) -> None:
        """Ejecuta todos los pasos sobre el servicio por defecto del agente."""
        if not self.enabled:
            return
        from docuerp.agent.service import get_agent_service
        from docuerp.schemas import QueryRequest

        self.started_at = _now()
        started = time.perf_counter()
        holder: dict[str, Any] = {}
        request = QueryRequest(query=PROBE_QUERY, role="n1", user_id="USR-0101")

        async def agent() -> None:
            holder["service"] = await asyncio.to_thread(get_agent_service)

        async def input_guard() -> None:
            verdict = await asyncio.to_thread(holder["service"].deps.input_guard, request)
            if not verdict.allowed:  # la consulta neutra nunca debería bloquearse
                raise RuntimeError("consulta de prueba bloqueada")

        async def router() -> None:
            await asyncio.to_thread(holder["service"].deps.router.decide, PROBE_QUERY)

        async def tool(name: str, args: dict[str, Any]) -> None:
            result = await holder["service"].deps.registry.invoke(
                name, args, role="n1", user_id="USR-0101"
            )
            if not result.ok:
                raise RuntimeError(result.error or name)

        async def retrieval() -> None:
            await tool(
                "search_documents",
                {"query": PROBE_QUERY, "role": "n1", "erp_version": "18.0", "top_k": 5},
            )

        async def classifier() -> None:
            await tool("classify_ticket", {"text": PROBE_TICKET})

        async def docx_pdf() -> None:
            from docuerp import paths
            from docuerp.api.docx_pdf import docx_rendition

            sources = sorted(p.name for p in (paths.RUNBOOKS / "docx").glob("*.docx"))
            for source in sources:
                await asyncio.to_thread(docx_rendition, source)

        plan: list[tuple[WarmupStep, Callable[[], Awaitable[None]]]] = [
            (WarmupStep("agent", "Grafo del agente"), agent),
            (WarmupStep("input_guard", "Guard de entrada (bge-m3)"), input_guard),
            (WarmupStep("router", "Router de intención"), router),
            (WarmupStep("retrieval", "Índices, embedder y reranker"), retrieval),
            (WarmupStep("classifier", "Clasificador de tickets"), classifier),
            (WarmupStep("docx_pdf", "PDF de los runbooks DOCX"), docx_pdf),
        ]
        self.steps = [step for step, _ in plan]
        for step, action in plan:
            if step.name != "agent" and "service" not in holder:
                step.status = "skipped"
                step.error = "agente no disponible"
                continue
            await self._run_step(step, action)
        self.total_ms = (time.perf_counter() - started) * 1000
        self.finished_at = _now()
        logger.info(
            "precalentamiento %s en %.0f ms (%s)",
            self.state,
            self.total_ms,
            ", ".join(f"{s.name}={s.duration_ms or 0:.0f}ms" for s in self.steps),
        )
