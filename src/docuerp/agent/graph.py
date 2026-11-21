"""Grafo LangGraph del agente.

Flujo (SPEC §5.4)::

    input_guard → router ─┬─ doc_query ──────→ retrieve → evidence_gate → permission_check
                          │                                                      │
                          ├─ ticket_status ┐                                     ▼
                          ├─ ticket_classify├→ tool ──────────→ permission_check → generate
                          ├─ ticket_create ┘                                     │
                          │                                                      ▼
                          └─ out_of_domain → reject ──────────────→ respond ← output_guard

Reglas transversales:

* **Presupuesto**: máximo ``agent.max_steps`` (4) pasos de acción —cada llamada a
  herramienta y cada generación cuenta uno— y ``agent.max_tool_calls`` (3) llamadas a
  herramientas por consulta. Si se agota, se responde de forma segura en lugar de seguir.
* **Cortocircuito**: cuando un nodo ya decidió la respuesta (bloqueo, abstención,
  denegación, degradación), marca ``short_circuit`` y el grafo salta directo a ``respond``.
* **Degradación controlada**: cada nodo está envuelto por :func:`_guarded`; una excepción
  inesperada se convierte en una respuesta segura con ``needs_human_review=true`` y un
  ``TraceEvent`` de error, nunca en un error 500. Si el LLM falla o hace timeout se reintenta
  con el proveedor ``offline`` (extractivo, sin LLM).
* **Trazabilidad**: cada nodo emite ``TraceEvent`` con su duración y datos sanitizados (ids,
  scores, banderas; nunca el texto de la consulta ni PII).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from docuerp.agent import templates
from docuerp.agent.deps import AgentDeps, build_default_deps
from docuerp.agent.router import TICKET_ID_RE
from docuerp.agent.state import AgentState
from docuerp.agent.tools.base import ToolResult
from docuerp.agent.tools.create_ticket import extract_account_id
from docuerp.agent.tools.search_documents import MAX_TOP_K
from docuerp.llm.contracts import LLMProvider, LLMResult
from docuerp.schemas import (
    Answer,
    PendingAction,
    QueryRequest,
    QueryResponse,
    ScoredChunk,
    TraceEvent,
    TraceStatus,
    Usage,
)

log = logging.getLogger(__name__)

NodeFn = Callable[[AgentState], Awaitable[dict[str, Any]]]

TOOL_FOR_INTENT: dict[str, str] = {
    "ticket_status": "get_ticket_status",
    "ticket_classify": "classify_ticket",
    "ticket_create": "create_ticket",
}
"""Herramienta principal de cada intención de tickets."""

RETRIEVAL_STAGES: dict[str, str] = {
    "bm25": "retrieval.bm25",
    "dense": "retrieval.dense",
    "fusion": "retrieval.fusion",
    "rerank": "rerank",
    "mmr": "mmr",
}
"""Nombre de etapa en la traza para cada clave de ``RetrievalResult.timings_ms``."""

DEFAULT_DRAFT: dict[str, str] = {
    "category": "configuracion",
    "priority": "media",
    "module": "general",
}
"""Clasificación por defecto de un borrador si ``classify_ticket`` no está disponible
(el borrador queda marcado para revisión humana)."""

SUMMARY_MAX_CHARS = 200

# Prefijo de orden ("crea un ticket para...", "clasifica este caso:") que se quita del texto
# antes de clasificarlo o usarlo como resumen del borrador.
_COMMAND_PREFIX_RE = re.compile(
    r"^\s*(?:por\s+favor[,\s]+)?(?:(?:me\s+)?(?:puedes|podr[ií]as)\s+)?"
    r"(?:cr[eé]a(?:me)?|crear|abr[eé](?:me)?|abrir|registr[aá](?:me)?|registrar|"
    r"levant[aá](?:me)?|levantar|gener[aá]|generar|report[aá]|reportar|"
    r"clasific[aá](?:me)?|clasificar|categoriz[aá](?:me)?|categorizar)"
    r"(?:\s+(?:un|una|el|la|este|esta|nuevo|nueva|mi))*"
    r"(?:\s+(?:ticket|tiquete|caso|incidencia|incidente)s?)?"
    r"(?:\s+(?:para|de|sobre|por|con))?\s*[:\-–,]?\s*",
    re.IGNORECASE,
)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s")


def strip_command(text: str) -> str:
    """Quita la orden inicial ("crea un ticket:") y devuelve solo el contenido del ticket.

    Devuelve ``""`` si la consulta era solo la orden (p. ej. "clasifica"): el nodo de
    herramientas pide entonces más detalle en lugar de inventar un ticket vacío.
    """
    return _COMMAND_PREFIX_RE.sub("", text, count=1).strip()


def make_summary(text: str) -> str:
    """Primera oración (si tiene ≥10 caracteres) recortada a 200 caracteres."""
    first = _SENTENCE_END_RE.split(text.strip(), maxsplit=1)[0]
    summary = first if len(first) >= 10 else text.strip()
    if len(summary) > SUMMARY_MAX_CHARS:
        summary = summary[: SUMMARY_MAX_CHARS - 1].rstrip() + "…"
    return summary


def _elapsed_ms(started: float) -> float:
    """Milisegundos transcurridos desde ``started`` (``time.perf_counter``)."""
    return max(0.0, (time.perf_counter() - started) * 1000)


def _event(stage: str, status: TraceStatus, started: float, **data: Any) -> TraceEvent:
    """Crea un ``TraceEvent`` con la duración desde ``started``."""
    return TraceEvent(stage=stage, status=status, duration_ms=_elapsed_ms(started), data=data)


def _short(answer: Answer, *, degraded: bool = False, **extra: Any) -> dict[str, Any]:
    """Actualización de estado que fija la respuesta y salta a ``respond``.

    ``degraded`` solo se escribe cuando es ``True`` para no borrar una degradación previa.
    """
    update: dict[str, Any] = {"answer": answer, "short_circuit": True, **extra}
    if degraded:
        update["degraded"] = True
    return update


def _guarded(stage: str, fn: NodeFn) -> NodeFn:
    """Envuelve un nodo: cualquier excepción → respuesta segura + evento de error."""

    async def node(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            return await fn(state)
        except Exception as exc:  # degradación controlada: ninguna etapa tumba la consulta
            log.exception("Falla en la etapa %s", stage)
            event = _event(
                "error", "error", started, failed_stage=stage, error_type=type(exc).__name__
            )
            return _short(
                templates.degraded(f"ocurrió un error interno en la etapa «{stage}»."),
                degraded=True,
                events=[event],
            )

    return node


def _usage_of(result: LLMResult) -> Usage:
    """Convierte el consumo de una llamada LLM en ``Usage``."""
    return Usage(
        llm_calls=1,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_read_tokens=result.cache_read_tokens,
        cache_creation_tokens=result.cache_creation_tokens,
        cost_usd=result.cost_usd,
    )


class AgentNodes:
    """Implementación de los nodos del grafo sobre unas :class:`AgentDeps` inyectadas."""

    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps
        self.limits = deps.settings.agent

    # ------------------------------------------------------------------ presupuesto
    def _budget_issue(self, state: AgentState, *, tool: bool) -> str | None:
        """Motivo (en español) si el siguiente paso excede el presupuesto; ``None`` si cabe."""
        if state.get("steps", 0) >= self.limits.max_steps:
            return f"se alcanzó el máximo de {self.limits.max_steps} pasos por consulta."
        if tool and state.get("tool_calls", 0) >= self.limits.max_tool_calls:
            return (
                f"se alcanzó el máximo de {self.limits.max_tool_calls} llamadas a "
                "herramientas por consulta."
            )
        return None

    async def _call_tool(
        self, state: AgentState, local: dict[str, Any], name: str, args: dict[str, Any]
    ) -> tuple[ToolResult, TraceEvent]:
        """Invoca ``name`` respetando el presupuesto; actualiza los contadores en ``local``.

        ``local`` acumula ``steps``, ``tool_calls``, ``tool_call_counts`` y ``tool_results``
        dentro de un mismo nodo (un nodo puede encadenar dos herramientas).
        """
        started = time.perf_counter()
        merged: AgentState = {**state, **local}  # type: ignore[typeddict-item]
        issue = self._budget_issue(merged, tool=True)
        if issue is not None:
            result = ToolResult(tool=name, ok=False, error=issue, data={"limit_exceeded": True})
            return result, _event(f"tool.{name}", "skipped", started, reason="budget")
        req = state["request"]
        counts = dict(merged.get("tool_call_counts", {}))
        result = await self.deps.registry.invoke(
            name,
            args,
            role=req.role,
            user_id=req.user_id,
            session_id=req.session_id,
            erp_version=req.erp_version,
            calls_in_query=counts.get(name, 0),
        )
        counts[name] = counts.get(name, 0) + 1
        local["tool_call_counts"] = counts
        local["steps"] = merged.get("steps", 0) + 1
        local["tool_calls"] = merged.get("tool_calls", 0) + 1
        local["tool_results"] = [*merged.get("tool_results", []), result]
        status: TraceStatus = "ok" if result.ok else ("blocked" if _denied(result) else "error")
        summary = {
            k: v
            for k, v in result.data.items()
            if k not in {"hits", "stages", "pending_action", "summary", "description"}
        }
        event = TraceEvent(
            stage=f"tool.{name}",
            status=status,
            duration_ms=result.latency_ms,
            data={"ok": result.ok, "error": result.error, **summary},
        )
        return result, event

    # ------------------------------------------------------------------ nodos
    async def input_guard(self, state: AgentState) -> dict[str, Any]:
        """Valida la entrada (inyección, PII, longitud). Falla cerrado: si el guard falla,
        la consulta no se procesa (lo hace :func:`_guarded`)."""
        started = time.perf_counter()
        req = state["request"]
        verdict = self.deps.input_guard(req)
        data = {
            "allowed": verdict.allowed,
            "risk_score": verdict.risk_score,
            "flags": list(verdict.flags),
        }
        if not verdict.allowed:
            return _short(
                templates.blocked(verdict.reason),
                blocked=True,
                block_reason=verdict.reason or "input_guard",
                guard_flags=list(verdict.flags),
                events=[_event("input_guard", "blocked", started, **data)],
            )
        return {
            "sanitized_query": verdict.sanitized_text.strip() or req.query,
            "guard_flags": list(verdict.flags),
            "events": [_event("input_guard", "ok", started, **data)],
        }

    async def router(self, state: AgentState) -> dict[str, Any]:
        """Decide la intención (reglas → clasificador → fuera de dominio)."""
        started = time.perf_counter()
        decision = self.deps.router.decide(state["sanitized_query"])
        return {
            "intent": decision.intent,
            "intent_confidence": decision.confidence,
            "intent_rationale": decision.rationale,
            "route_method": decision.method,
            "events": [
                _event(
                    "router",
                    "ok",
                    started,
                    intent=decision.intent,
                    confidence=decision.confidence,
                    method=decision.method,
                    rationale=decision.rationale,
                )
            ],
        }

    async def retrieve(self, state: AgentState) -> dict[str, Any]:
        """Recupera evidencia con ``search_documents`` (cuenta como paso y como tool)."""
        req = state["request"]
        local: dict[str, Any] = {}
        top_k = min(self.deps.settings.retrieval.top_k, MAX_TOP_K)
        result, tool_event = await self._call_tool(
            state,
            local,
            "search_documents",
            {
                "query": state["sanitized_query"],
                "role": req.role,
                "erp_version": req.erp_version,
                "top_k": top_k,
            },
        )
        events = [*_retrieval_events(result), tool_event]
        update: dict[str, Any] = {**local, "tool_name": "search_documents", "tool_result": result}
        if not result.ok:
            reason = result.error or "el buscador de documentos no está disponible."
            answer = (
                templates.denied("search_documents", result.error)
                if _denied(result)
                else templates.degraded(reason, tool="search_documents")
            )
            return {**update, **_short(answer, degraded=not _denied(result)), "events": events}
        hits = [ScoredChunk.model_validate(h) for h in result.data.get("hits", [])]
        summary = {
            key: result.data.get(key) for key in ("evidence_ok", "top_score", "filtered_out")
        }
        return {
            **update,
            "hits": hits,
            "evidence_ok": bool(result.data.get("evidence_ok")) and bool(hits),
            "retrieval": summary,
            "events": events,
        }

    async def evidence_gate(self, state: AgentState) -> dict[str, Any]:
        """Gate τ: sin evidencia suficiente → abstención con mensaje fijo."""
        started = time.perf_counter()
        retrieval = state.get("retrieval", {})
        data = {
            "evidence_ok": state.get("evidence_ok", False),
            "top_score": retrieval.get("top_score"),
            "tau": self.deps.settings.retrieval.tau,
            "n_hits": len(state.get("hits", [])),
        }
        if not state.get("evidence_ok", False):
            return _short(
                templates.abstention(), events=[_event("evidence_gate", "blocked", started, **data)]
            )
        return {"events": [_event("evidence_gate", "ok", started, **data)]}

    async def permission_check(self, state: AgentState) -> dict[str, Any]:
        """Revalida permisos después de recuperar o de usar la herramienta."""
        started = time.perf_counter()
        req = state["request"]
        if state.get("intent") == "doc_query":
            authorized, dropped = self.deps.filter_authorized(req.role, state.get("hits", []))
            data: dict[str, Any] = {"kept": len(authorized), "dropped": dropped}
            if self.deps.context_guard is not None and authorized:
                # Inyección indirecta: se retiran las oraciones con instrucciones dirigidas al
                # modelo (el LLM nunca las ve) y se conserva el contenido legítimo del chunk;
                # si no queda nada confiable, el chunk sale completo (cuarentena).
                context = self.deps.context_guard(authorized)
                if context.sanitized or context.quarantined:
                    authorized = context.hits
                    data.update(
                        kept=len(authorized),
                        sanitized=context.sanitized,
                        quarantined=context.quarantined,
                    )
            if not authorized:
                return _short(
                    templates.abstention(),
                    authorized_hits=[],
                    events=[_event("permission_check", "blocked", started, **data)],
                )
            return {
                "authorized_hits": authorized,
                "events": [_event("permission_check", "ok", started, **data)],
            }
        tool = state.get("tool_name") or ""
        result: ToolResult | None = state.get("tool_result")
        role_ok = tool in self.deps.registry.names() and req.role in (
            self.deps.registry.get(tool).allowed_roles
        )
        if result is None or not role_ok or _denied(result):
            reason = result.error if result is not None else None
            answer = (
                templates.limit_reached(tool, reason)
                if result is not None and _session_limit(result)
                else templates.denied(tool, reason)
            )
            return _short(
                answer,
                events=[_event("permission_check", "blocked", started, tool=tool, allowed=False)],
            )
        visible = result.data.get("visible", True) and result.data.get("identity_verified", True)
        if not visible:
            # Fuera de alcance es una denegación explícita: respuesta fija que no confirma si el
            # ticket existe, sin pasar por la generación ni exponer datos del resultado.
            return _short(
                templates.ticket_status(result.data),
                events=[
                    _event(
                        "permission_check",
                        "blocked",
                        started,
                        tool=tool,
                        allowed=True,
                        in_scope=False,
                    )
                ],
            )
        return {
            "events": [
                _event("permission_check", "ok", started, tool=tool, allowed=True, in_scope=True)
            ]
        }

    async def tool(self, state: AgentState) -> dict[str, Any]:
        """Planifica y ejecuta la herramienta de tickets según la intención."""
        intent = state["intent"]
        query = state["sanitized_query"]
        name = TOOL_FOR_INTENT[intent]
        local: dict[str, Any] = {"tool_name": name}
        events: list[TraceEvent] = []
        if intent == "ticket_status":
            ids = TICKET_ID_RE.findall(query)
            if not ids:
                started = time.perf_counter()
                skipped = _event(f"tool.{name}", "skipped", started, reason="missing_ticket_id")
                return {**local, **_short(templates.ask_ticket_id()), "events": [skipped]}
            result, event = await self._call_tool(state, local, name, {"ticket_id": ids[0].upper()})
            events.append(event)
        else:
            text = strip_command(query)
            min_len = 10 if intent == "ticket_create" else 3
            if len(text) < min_len:
                started = time.perf_counter()
                skipped = _event(f"tool.{name}", "skipped", started, reason="text_too_short")
                return {**local, **_short(templates.ask_ticket_text(name)), "events": [skipped]}
            role = state["request"].role
            can_classify = "classify_ticket" in self.deps.registry.names() and role in (
                self.deps.registry.get("classify_ticket").allowed_roles
            )
            if intent == "ticket_create" and not can_classify:
                # El rol puede pedir un ticket (escalado) pero no clasificar: el borrador usa la
                # clasificación por defecto y la revisión humana la corrige. No se intenta una
                # llamada que el registro va a denegar (ensuciaría la traza con un bloqueo).
                started = time.perf_counter()
                classified = ToolResult(
                    tool="classify_ticket", ok=False, error=None, data={"skipped": "role"}
                )
                events.append(
                    _event("tool.classify_ticket", "skipped", started, reason="role_not_allowed")
                )
            else:
                classified, event = await self._call_tool(
                    state, local, "classify_ticket", {"text": text[:2000]}
                )
                events.append(event)
            result = classified
            if intent == "ticket_create":
                result, event = await self._call_tool(
                    state, local, name, _draft_args(text, query, classified)
                )
                events.append(event)
        local["tool_result"] = result
        if not result.ok and not _denied(result):
            reason = result.error or "la herramienta no está disponible."
            answer = templates.degraded(reason, tool=name)
            return {**local, **_short(answer, degraded=True), "events": events}
        pending_raw = result.data.get("pending_action")
        if pending_raw is not None:
            local["pending_action"] = PendingAction.model_validate(pending_raw)
        return {**local, "events": events}

    async def generate(self, state: AgentState) -> dict[str, Any]:
        """Genera la respuesta: LLM fundamentado (documental) o plantilla (tickets)."""
        started = time.perf_counter()
        issue = self._budget_issue(state, tool=False)
        if issue is not None:
            return _short(
                templates.degraded(issue, tool=state.get("tool_name")),
                degraded=True,
                events=[_event("generation", "skipped", started, reason="budget")],
            )
        steps = state.get("steps", 0) + 1
        if state.get("intent") != "doc_query":
            answer = _template_answer(state)
            return {
                "answer": answer,
                "steps": steps,
                "generation_mode": "template",
                "events": [_event("generation", "ok", started, mode="template")],
            }
        return await self._generate_with_llm(state, steps, started)

    async def _generate_with_llm(
        self, state: AgentState, steps: int, started: float
    ) -> dict[str, Any]:
        """Generación documental con timeout y degradación al proveedor ``offline``."""
        deps = self.deps
        question = state["sanitized_query"]
        hits = state.get("authorized_hits", [])
        usage = state.get("usage", Usage())
        primary_error = ""
        try:
            answer, llm = await self._run_generator(
                deps.provider, question, hits, erp_version=state["request"].erp_version
            )
        except Exception as exc:  # timeout o caída del LLM → proveedor offline
            primary_error = type(exc).__name__
            log.warning("Generación principal falló (%s); se degrada", primary_error)
        else:
            answer = answer.model_copy(update={"tool_used": "search_documents"})
            return {
                "answer": answer,
                "steps": steps,
                "usage": usage + _usage_of(llm),
                "generation_mode": "llm",
                "events": [_generation_event("ok", started, "llm", llm)],
            }
        fallback_issue = self._budget_issue({**state, "steps": steps}, tool=False)
        if deps.fallback_provider is None or fallback_issue is not None:
            return _short(
                templates.abstention(review=True),
                degraded=True,
                steps=steps,
                generation_mode="safe",
                events=[
                    _event("generation", "error", started, mode="safe", error_type=primary_error)
                ],
            )
        try:
            answer, llm = await self._run_generator(
                deps.fallback_provider, question, hits, erp_version=state["request"].erp_version
            )
        except Exception as exc:
            log.warning("El proveedor de respaldo también falló (%s)", type(exc).__name__)
            return _short(
                templates.abstention(review=True),
                degraded=True,
                steps=steps + 1,
                generation_mode="safe",
                events=[
                    _event(
                        "generation",
                        "error",
                        started,
                        mode="safe",
                        error_type=primary_error,
                        fallback_error_type=type(exc).__name__,
                    )
                ],
            )
        answer = answer.model_copy(
            update={"tool_used": "search_documents", "needs_human_review": True}
        )
        return {
            "answer": answer,
            "steps": steps + 1,
            "usage": usage + _usage_of(llm),
            "degraded": True,
            "generation_mode": "fallback",
            "events": [
                _generation_event("retry", started, "fallback", llm, error_type=primary_error)
            ],
        }

    async def _run_generator(
        self,
        provider: LLMProvider,
        question: str,
        hits: list[ScoredChunk],
        *,
        erp_version: str | None = None,
    ) -> tuple[Answer, LLMResult]:
        """Llama al generador con un timeout duro (``llm.timeout_s``)."""
        cfg = self.deps.settings.llm
        return await asyncio.wait_for(
            self.deps.generator(
                question,
                hits,
                intent="doc_query",
                provider=provider,
                model=cfg.model,
                erp_version=erp_version,
            ),
            timeout=cfg.timeout_s,
        )

    async def output_guard(self, state: AgentState) -> dict[str, Any]:
        """Valida citas, PII, canario y coherencia de abstención de la respuesta final."""
        started = time.perf_counter()
        answer = state["answer"]
        if answer is None:
            raise ValueError("output_guard sin respuesta que validar")
        retrieved = state.get("authorized_hits", []) if state.get("intent") == "doc_query" else []
        verdict = self.deps.output_guard(
            answer,
            retrieved,
            tool_result=state.get("tool_result"),
            role=state["request"].role,
        )
        if verdict.valid:
            return {"answer": verdict.answer, "events": [_event("output_guard", "ok", started)]}
        corrected = verdict.answer.model_copy(update={"needs_human_review": True})
        return {
            "answer": corrected,
            "events": [_event("output_guard", "blocked", started, errors=list(verdict.errors))],
        }

    async def reject(self, state: AgentState) -> dict[str, Any]:
        """Rechazo controlado de consultas fuera de dominio."""
        started = time.perf_counter()
        return _short(
            templates.out_of_domain(),
            events=[_event("reject", "ok", started, reason=state.get("intent_rationale"))],
        )

    async def respond(self, state: AgentState) -> dict[str, Any]:
        """Arma el ``QueryResponse`` final y lo registra en la auditoría."""
        started = time.perf_counter()
        answer = state.get("answer") or templates.degraded("no se generó una respuesta.")
        base = QueryResponse(
            trace_id=state["trace_id"],
            intent=state.get("intent", "out_of_domain"),
            response=answer,
            pending_action=state.get("pending_action"),
            blocked=state.get("blocked", False),
            block_reason=state.get("block_reason"),
            latency_ms=_elapsed_ms(state["started_at"]),
            usage=state.get("usage", Usage()),
            events=list(state.get("events", [])),
        )
        audit_status = "ok"
        try:
            self.deps.audit.log(redact_for_audit(base), request=state["request"])
        except Exception as exc:  # la auditoría nunca debe impedir responder
            log.warning("No se pudo escribir la auditoría: %s", type(exc).__name__)
            audit_status = "error"
        event = _event(
            "respond",
            "ok",
            started,
            steps=state.get("steps", 0),
            tool_calls=state.get("tool_calls", 0),
            degraded=state.get("degraded", False),
            abstained=answer.abstained,
            needs_human_review=answer.needs_human_review,
            audit=audit_status,
        )
        final = base.model_copy(
            update={
                "events": [*base.events, event],
                "latency_ms": _elapsed_ms(state["started_at"]),
            }
        )
        return {"response": final, "events": [event]}


REDACTED_TOKEN = "[TOKEN-REDACTADO]"
"""Reemplazo del token de confirmación en la auditoría (es una credencial de un solo uso)."""


def redact_for_audit(response: QueryResponse) -> QueryResponse:
    """Copia de ``response`` sin el token de confirmación: la bitácora nunca guarda
    credenciales, ni siquiera de un solo uso."""
    if response.pending_action is None:
        return response
    pending = response.pending_action.model_copy(update={"confirmation_token": REDACTED_TOKEN})
    return response.model_copy(update={"pending_action": pending})


def _denied(result: ToolResult) -> bool:
    """``True`` si el resultado es una denegación de política (no una caída): permisos o el
    límite de llamadas por sesión, que frena el abuso (p. ej. crear tickets en masa)."""
    return bool(result.data.get("denied")) or _session_limit(result)


def _session_limit(result: ToolResult) -> bool:
    """``True`` si el registro rechazó la llamada por el límite de la sesión."""
    return bool(result.data.get("limit_exceeded")) and result.data.get("scope") == "session"


def _retrieval_events(result: ToolResult) -> list[TraceEvent]:
    """Un evento por etapa de recuperación con su latencia y top de candidatos."""
    timings: dict[str, float] = result.data.get("timings_ms", {}) or {}
    stages: dict[str, Any] = result.data.get("stages", {}) or {}
    ordered = [k for k in RETRIEVAL_STAGES if k in timings] + [
        k for k in timings if k not in RETRIEVAL_STAGES
    ]
    return [
        TraceEvent(
            stage=RETRIEVAL_STAGES.get(key, f"retrieval.{key}"),
            status="ok",
            duration_ms=max(0.0, float(timings[key])),
            data={"top": stages.get(key, [])},
        )
        for key in ordered
    ]


def _generation_event(
    status: TraceStatus, started: float, mode: str, llm: LLMResult, **extra: Any
) -> TraceEvent:
    """Evento de generación con proveedor, modelo, tokens y latencias (sin texto)."""
    return _event(
        "generation",
        status,
        started,
        mode=mode,
        provider=llm.provider,
        model=llm.model,
        input_tokens=llm.input_tokens,
        output_tokens=llm.output_tokens,
        cost_usd=llm.cost_usd,
        llm_latency_ms=llm.latency_ms,
        api_latency_ms=llm.api_latency_ms,
        **extra,
    )


def _draft_args(text: str, query: str, classified: ToolResult) -> dict[str, Any]:
    """Arma los argumentos de ``create_ticket`` a partir del texto y la clasificación."""
    if classified.ok:
        labels = {key: classified.data[key] for key in DEFAULT_DRAFT}
        review = bool(classified.data.get("needs_human_review"))
    else:
        labels, review = dict(DEFAULT_DRAFT), True
    return {
        "summary": make_summary(text),
        "description": query[:2000],
        **labels,
        "account_id": extract_account_id(query),
        "flag_for_review": review,
    }


def _template_answer(state: AgentState) -> Answer:
    """Respuesta determinista según la herramienta y su resultado."""
    tool = state.get("tool_name")
    result: ToolResult = state["tool_result"]
    if tool == "get_ticket_status":
        return templates.ticket_status(result.data)
    if tool == "classify_ticket":
        return templates.classification(result.data)
    if tool == "create_ticket":
        return templates.ticket_proposal(result.data, review=result.needs_human_review)
    raise ValueError(f"Sin plantilla para la herramienta {tool!r}")


def _after(default: str) -> Callable[[AgentState], str]:
    """Función de ruteo: ``respond`` si hubo cortocircuito; si no, ``default``."""

    def route(state: AgentState) -> str:
        return "respond" if state.get("short_circuit") else default

    return route


def _route_intent(state: AgentState) -> str:
    """Rama según la intención decidida por el router (o ``respond`` si el router falló)."""
    if state.get("short_circuit"):
        return "respond"
    intent = state.get("intent")
    if intent == "doc_query":
        return "retrieve"
    if intent in TOOL_FOR_INTENT:
        return "tool"
    return "reject"


def build_graph(deps: AgentDeps | None = None) -> Any:
    """Construye y compila el ``StateGraph`` sobre :class:`docuerp.agent.state.AgentState`."""
    nodes = AgentNodes(deps or build_default_deps())
    graph = StateGraph(AgentState)
    for name in (
        "input_guard",
        "router",
        "retrieve",
        "evidence_gate",
        "permission_check",
        "tool",
        "generate",
        "output_guard",
        "reject",
    ):
        graph.add_node(name, _guarded(name, getattr(nodes, name)))
    graph.add_node("respond", nodes.respond)

    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges("input_guard", _after("router"), ["router", "respond"])
    graph.add_conditional_edges("router", _route_intent, ["retrieve", "tool", "reject", "respond"])
    graph.add_conditional_edges("retrieve", _after("evidence_gate"), ["evidence_gate", "respond"])
    graph.add_conditional_edges(
        "evidence_gate", _after("permission_check"), ["permission_check", "respond"]
    )
    graph.add_conditional_edges("tool", _after("permission_check"), ["permission_check", "respond"])
    graph.add_conditional_edges("permission_check", _after("generate"), ["generate", "respond"])
    graph.add_conditional_edges("generate", _after("output_guard"), ["output_guard", "respond"])
    graph.add_edge("output_guard", "respond")
    graph.add_edge("reject", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


def initial_state(req: QueryRequest, *, trace_id: str | None = None) -> AgentState:
    """Estado inicial de una consulta (contadores en cero)."""
    return {
        "trace_id": trace_id or uuid.uuid4().hex,
        "request": req,
        "steps": 0,
        "tool_calls": 0,
        "tool_call_counts": {},
        "tool_results": [],
        "usage": Usage(),
        "events": [],
        "blocked": False,
        "block_reason": None,
        "short_circuit": False,
        "degraded": False,
        "pending_action": None,
        "started_at": time.perf_counter(),
    }
