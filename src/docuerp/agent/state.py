"""Estado compartido del grafo LangGraph.

LangGraph pasa un único diccionario entre nodos; cada nodo devuelve **solo las claves que
cambia** y LangGraph las fusiona. La clave ``events`` usa un *reducer* (``operator.add``):
en lugar de reemplazarse, las listas que devuelve cada nodo se **concatenan**, así cada
nodo agrega sus ``TraceEvent`` sin tener que copiar los anteriores.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from docuerp.schemas import (
    Answer,
    Intent,
    PendingAction,
    QueryRequest,
    QueryResponse,
    ScoredChunk,
    TraceEvent,
    Usage,
)


class AgentState(TypedDict, total=False):
    """Estado del agente por consulta. Los contadores imponen ≤4 pasos y ≤3 tools."""

    trace_id: str
    request: QueryRequest
    sanitized_query: str
    guard_flags: list[str]
    intent: Intent
    intent_confidence: float
    intent_rationale: str
    route_method: str
    retrieval: dict[str, Any]  # resumen serializable de RetrievalResult (sin textos)
    hits: list[ScoredChunk]
    authorized_hits: list[ScoredChunk]
    evidence_ok: bool
    tool_name: str | None
    tool_args: dict[str, Any]
    tool_result: Any  # docuerp.agent.tools.ToolResult de la herramienta principal
    tool_results: list[Any]  # todas las ToolResult de la consulta, en orden
    tool_call_counts: dict[str, int]  # llamadas por herramienta en esta consulta
    answer: Answer | None
    pending_action: PendingAction | None
    blocked: bool
    block_reason: str | None
    short_circuit: bool  # True → la respuesta ya está decidida: saltar a ``respond``
    degraded: bool  # True → hubo degradación controlada (falla, timeout, límite)
    generation_mode: str  # llm | fallback | template | safe
    steps: int
    tool_calls: int
    usage: Usage
    events: Annotated[list[TraceEvent], operator.add]
    started_at: float
    response: QueryResponse
