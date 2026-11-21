"""Fachada del servicio: streaming, funciones de módulo y red de seguridad final."""

from __future__ import annotations

import types
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import pytest

from docuerp.agent import service as service_mod
from docuerp.agent.service import AgentService
from docuerp.schemas import QueryResponse, TraceEvent

DOC_QUERY = "¿Cómo valido una recepción parcial en inventario?"


@pytest.fixture
def installed(make_service: Callable[..., AgentService]) -> Iterator[AgentService]:
    svc = make_service()
    service_mod.set_agent_service(svc)
    yield svc
    service_mod.set_agent_service(None)


async def test_stream_emits_events_then_final(
    installed: AgentService, kit: types.SimpleNamespace
) -> None:
    items = [item async for item in service_mod.answer_query_stream(kit.request(DOC_QUERY))]
    *events, final = items
    assert isinstance(final, QueryResponse)
    assert all(isinstance(e, TraceEvent) for e in events)
    assert [e.stage for e in events] == [e.stage for e in final.events]
    assert events[0].stage == "input_guard"
    assert events[-1].stage == "respond"


async def test_module_answer_query_uses_installed_service(
    installed: AgentService, kit: types.SimpleNamespace
) -> None:
    resp = await service_mod.answer_query(kit.request(DOC_QUERY))
    assert resp.intent == "doc_query"
    assert service_mod.get_agent_service() is installed


async def test_module_confirm_action(installed: AgentService, kit: types.SimpleNamespace) -> None:
    req = kit.request(
        "Crea un ticket: no puedo validar la recepción en inventario, cuenta ACC-0001",
        session_id="sesion-modulo",
    )
    resp = await service_mod.answer_query(req)
    assert resp.pending_action is not None
    done = await service_mod.confirm_action(resp.pending_action.confirmation_token, req)
    assert done.response.tool_used == "create_ticket"
    assert not done.blocked


async def test_stream_of_confirmation(installed: AgentService, kit: types.SimpleNamespace) -> None:
    req = kit.request("confirmar", session_id="sesion-modulo", confirmation_token="v1.x.y" * 4)
    items = [item async for item in service_mod.answer_query_stream(req)]
    assert [type(i).__name__ for i in items] == ["TraceEvent", "TraceEvent", "QueryResponse"]
    assert items[-1].blocked  # token inválido


async def test_graph_crash_returns_safe_response(
    make_service: Callable[..., AgentService], kit: types.SimpleNamespace
) -> None:
    svc = make_service()

    class ExplodingGraph:
        async def astream(self, state: Any, stream_mode: str) -> AsyncIterator[dict[str, Any]]:
            yield {"input_guard": {"events": []}}
            raise RuntimeError("fallo interno")

    svc.graph = ExplodingGraph()
    items = [item async for item in svc.answer_query_stream(kit.request(DOC_QUERY))]
    final = items[-1]
    assert isinstance(final, QueryResponse)
    assert final.response.needs_human_review
    assert final.response.abstained
    assert final.events[-1].data == {"failed_stage": "graph", "error_type": "RuntimeError"}


def test_default_service_is_lazy_and_resettable() -> None:
    service_mod.set_agent_service(None)
    try:
        first = service_mod.get_agent_service()
        assert first is service_mod.get_agent_service()
        assert first.deps.registry.names() == [
            "classify_ticket",
            "create_ticket",
            "get_ticket_status",
            "search_documents",
        ]
    finally:
        service_mod.set_agent_service(None)
