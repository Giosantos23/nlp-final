"""Fachada única del agente usada por la API, la CLI y la evaluación.

* :func:`answer_query` — procesa una consulta y devuelve el ``QueryResponse`` completo.
* :func:`answer_query_stream` — generador asíncrono que emite cada ``TraceEvent`` en cuanto
  termina su etapa y, al final, el ``QueryResponse`` (lo usa el SSE de la API).
* :func:`confirm_action` — confirmación humana (HITL) de un borrador de ticket.

Las tres funciones de módulo delegan en un :class:`AgentService` por defecto (perezoso).
Las pruebas y la API pueden instalar otro con :func:`set_agent_service`.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator

from docuerp.agent import templates
from docuerp.agent.deps import AgentDeps, build_default_deps
from docuerp.agent.graph import build_graph, initial_state
from docuerp.agent.tools.base import UNAVAILABLE_MESSAGE, ToolResult
from docuerp.schemas import Answer, QueryRequest, QueryResponse, TraceEvent, Usage

log = logging.getLogger(__name__)


def _ms(started: float) -> float:
    """Milisegundos desde ``started``."""
    return max(0.0, (time.perf_counter() - started) * 1000)


class AgentService:
    """Agente compilado sobre unas dependencias (reales por defecto, *fakes* en pruebas)."""

    def __init__(self, deps: AgentDeps | None = None) -> None:
        self.deps = deps or build_default_deps()
        self.graph = build_graph(self.deps)

    # ------------------------------------------------------------------ consultas
    async def answer_query_stream(
        self, req: QueryRequest
    ) -> AsyncIterator[TraceEvent | QueryResponse]:
        """Emite los ``TraceEvent`` a medida que ocurren y al final el ``QueryResponse``.

        Si la petición trae ``confirmation_token`` se trata como confirmación HITL.
        Pase lo que pase, el último elemento emitido es siempre un ``QueryResponse``.
        """
        if req.confirmation_token:
            response = await self.confirm_action(req.confirmation_token, req)
            for event in response.events:
                yield event
            yield response
            return
        state = initial_state(req)
        emitted: list[TraceEvent] = []
        final: QueryResponse | None = None
        try:
            async for chunk in self.graph.astream(state, stream_mode="updates"):
                for update in chunk.values():
                    if not update:
                        continue
                    for event in update.get("events", []):
                        emitted.append(event)
                        yield event
                    if "response" in update:
                        final = update["response"]
        except Exception as exc:  # última red de seguridad: el grafo no debe romper la API
            log.exception("Falla no controlada del grafo del agente")
            error_event = TraceEvent(
                stage="error",
                status="error",
                duration_ms=_ms(state["started_at"]),
                data={"failed_stage": "graph", "error_type": type(exc).__name__},
            )
            emitted.append(error_event)
            yield error_event
        if final is None:
            final = QueryResponse(
                trace_id=state["trace_id"],
                intent="out_of_domain",
                response=templates.degraded("ocurrió un error interno del agente."),
                latency_ms=_ms(state["started_at"]),
                usage=Usage(),
                events=emitted,
            )
        yield final

    async def answer_query(self, req: QueryRequest) -> QueryResponse:
        """Procesa una consulta de extremo a extremo y devuelve la respuesta con su traza."""
        final: QueryResponse | None = None
        async for item in self.answer_query_stream(req):
            if isinstance(item, QueryResponse):
                final = item
        if final is None:  # answer_query_stream siempre termina con un QueryResponse
            raise RuntimeError("El agente no produjo una respuesta")
        return final

    # ------------------------------------------------------------------ HITL
    async def confirm_action(self, token: str, req: QueryRequest) -> QueryResponse:
        """Confirma (HITL) un borrador pendiente: valida el token HMAC, su vigencia, que sea
        de un solo uso y que pertenezca al mismo usuario, rol y sesión; luego escribe."""
        started = time.perf_counter()
        trace_id = uuid.uuid4().hex
        registry = self.deps.registry
        step = time.perf_counter()
        blocked, block_reason = False, None
        if req.role not in registry.get("create_ticket").allowed_roles:
            result = ToolResult(
                tool="create_ticket",
                ok=False,
                error=f"el rol {req.role} no puede confirmar tickets.",
                data={"denied": True},
            )
        elif registry.is_unavailable("create_ticket"):
            result = ToolResult(
                tool="create_ticket",
                ok=False,
                error=UNAVAILABLE_MESSAGE,
                data={"unavailable": True, "simulated": True},
            )
        else:
            result = await asyncio.to_thread(
                self.deps.ticket_service.commit,
                token,
                user_id=req.user_id,
                session_id=req.session_id,
                role=req.role,
            )
        answer: Answer
        if result.ok:
            answer = templates.ticket_committed(result.data, review=result.needs_human_review)
            status = "ok"
        elif result.data.get("denied"):
            answer = templates.denied("create_ticket", result.error)
            status, blocked, block_reason = "blocked", True, "permiso_denegado"
        elif result.data.get("rejected"):
            answer = templates.confirmation_rejected(result.error or "token inválido.")
            status, blocked, block_reason = "blocked", True, "confirmacion_rechazada"
        else:
            answer = templates.degraded(result.error or UNAVAILABLE_MESSAGE, tool="create_ticket")
            status = "error"
        tool_event = TraceEvent(
            stage="tool.create_ticket",
            status=status,
            duration_ms=_ms(step),
            data={"action": "confirm", "ok": result.ok, "error": result.error, **result.data},
        )
        base = QueryResponse(
            trace_id=trace_id,
            intent="ticket_create",
            response=answer,
            blocked=blocked,
            block_reason=block_reason,
            latency_ms=_ms(started),
            usage=Usage(),
            events=[tool_event],
        )
        respond_started = time.perf_counter()
        audit_status = "ok"
        try:
            self.deps.audit.log(base, request=req)
        except Exception as exc:  # la auditoría nunca impide responder
            log.warning("No se pudo escribir la auditoría: %s", type(exc).__name__)
            audit_status = "error"
        respond_event = TraceEvent(
            stage="respond",
            status="ok",
            duration_ms=_ms(respond_started),
            data={"audit": audit_status, "needs_human_review": answer.needs_human_review},
        )
        return base.model_copy(
            update={"events": [tool_event, respond_event], "latency_ms": _ms(started)}
        )


_SERVICE: AgentService | None = None
_SERVICE_LOCK = threading.Lock()


def get_agent_service() -> AgentService:
    """Servicio por defecto del proceso (se construye en el primer uso)."""
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = AgentService()
        return _SERVICE


def set_agent_service(service: AgentService | None) -> None:
    """Instala un servicio (pruebas, API en modo mock) o, con ``None``, vuelve al defecto."""
    global _SERVICE
    with _SERVICE_LOCK:
        _SERVICE = service


async def answer_query(req: QueryRequest) -> QueryResponse:
    """Procesa una consulta de extremo a extremo y devuelve la respuesta con su traza."""
    return await get_agent_service().answer_query(req)


async def answer_query_stream(req: QueryRequest) -> AsyncIterator[TraceEvent | QueryResponse]:
    """Emite cada ``TraceEvent`` al completarse su etapa y al final el ``QueryResponse``."""
    async for item in get_agent_service().answer_query_stream(req):
        yield item


async def confirm_action(token: str, req: QueryRequest) -> QueryResponse:
    """Confirma (HITL) un borrador pendiente: valida el token HMAC, su vigencia, que sea de
    un solo uso y que pertenezca a la misma sesión/usuario; luego escribe el ticket."""
    return await get_agent_service().confirm_action(token, req)
