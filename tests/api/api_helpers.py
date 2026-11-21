"""Utilidades compartidas por las pruebas de la API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from docuerp.api.service import ConfirmationRejectedError
from docuerp.schemas import Answer, QueryRequest, QueryResponse, TraceEvent, Usage

QUERY = {
    "query": "¿Cómo valido una recepción parcial en Odoo 18?",
    "role": "n1",
    "user_id": "USR-0101",
    "erp_version": "18.0",
}


def make_response(
    trace_id: str = "fake-trace-0001", *, text: str = "respuesta fake"
) -> QueryResponse:
    """Respuesta mínima válida con dos eventos de traza."""
    return QueryResponse(
        trace_id=trace_id,
        intent="doc_query",
        response=Answer(
            answer=text, sources=[], abstained=False, tool_used=None, needs_human_review=False
        ),
        latency_ms=12.5,
        usage=Usage(),
        events=[
            TraceEvent(stage="router", status="ok", duration_ms=1.0, data={"intent": "doc_query"}),
            TraceEvent(stage="respond", status="ok", duration_ms=0.5),
        ],
    )


def parse_sse(lines: Any) -> list[tuple[str, dict[str, Any]]]:
    """Convierte las líneas de un flujo SSE en pares ``(evento, datos JSON)``."""
    events: list[tuple[str, dict[str, Any]]] = []
    name = ""
    for line in lines:
        line = line.strip()
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and name:
            events.append((name, json.loads(line.split(":", 1)[1].strip())))
            name = ""
    return events


class FakeService:
    """Servicio del agente de pruebas: cuenta llamadas y puede fallar a voluntad."""

    def __init__(self) -> None:
        self.calls = 0
        self.error: Exception | None = None
        self.stream_error_after: int | None = None
        self._counter = 0

    def _next_response(self) -> QueryResponse:
        self._counter += 1
        return make_response(f"fake-trace-{self._counter:04d}")

    async def answer_query(self, req: QueryRequest) -> QueryResponse:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self._next_response()

    async def answer_query_stream(
        self, req: QueryRequest
    ) -> AsyncIterator[TraceEvent | QueryResponse]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        response = self._next_response()
        for index, event in enumerate(response.events):
            if self.stream_error_after == index:
                raise RuntimeError("falla simulada a mitad del flujo")
            yield event
        yield response

    async def confirm_action(self, token: str, req: QueryRequest) -> QueryResponse:
        if token != "token-valido-0123456789":
            raise ConfirmationRejectedError("Token inválido.")
        return self._next_response()
