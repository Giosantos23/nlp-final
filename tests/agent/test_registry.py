"""Registro de herramientas: contrato, RBAC, límites, timeouts y simulación de caídas."""

from __future__ import annotations

import asyncio
import types

import pytest
from pydantic import BaseModel, ConfigDict

from docuerp.agent.tools import TOOL_NAMES
from docuerp.agent.tools.base import (
    UNAVAILABLE_ENV,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    default_registry,
)


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


def _spec(handler=None, **kw) -> ToolSpec:  # type: ignore[no-untyped-def]
    async def echo(args: EchoInput, ctx: ToolContext) -> ToolResult:
        return ToolResult(tool="echo", ok=True, data={"text": args.text, "role": ctx.role})

    values = {
        "name": "echo",
        "description": "eco",
        "input_model": EchoInput,
        "allowed_roles": ["n1", "n2"],
        "handler": handler or echo,
        "timeout_s": 0.2,
    }
    values.update(kw)
    return ToolSpec(**values)


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_spec())
    return reg


async def test_invoke_ok_includes_latency(registry: ToolRegistry) -> None:
    result = await registry.invoke("echo", {"text": "hola"}, role="n1", user_id="USR-0101")
    assert result.ok
    assert result.data == {"text": "hola", "role": "n1"}
    assert result.latency_ms >= 0


def test_register_duplicate_and_get_unknown(registry: ToolRegistry) -> None:
    with pytest.raises(ValueError, match="ya está registrada"):
        registry.register(_spec())
    with pytest.raises(KeyError):
        registry.get("nope")
    assert registry.names() == ["echo"]


async def test_unknown_tool_never_raises(registry: ToolRegistry) -> None:
    result = await registry.invoke("nope", {}, role="n1", user_id="USR-0101")
    assert not result.ok
    assert result.data["unknown_tool"] is True


async def test_role_denied(registry: ToolRegistry) -> None:
    result = await registry.invoke("echo", {"text": "x"}, role="portal_user", user_id="USR-0001")
    assert not result.ok
    assert result.data["denied"] is True


async def test_invalid_args_rejected_by_contract(registry: ToolRegistry) -> None:
    result = await registry.invoke("echo", {"text": "x", "extra": 1}, role="n1", user_id="USR-0101")
    assert not result.ok
    assert result.data["invalid_args"] is True
    assert "extra" in result.data["fields"]


async def test_query_limit(registry: ToolRegistry) -> None:
    result = await registry.invoke(
        "echo", {"text": "x"}, role="n1", user_id="USR-0101", calls_in_query=1
    )
    assert not result.ok
    assert result.data == {"limit_exceeded": True, "scope": "query"}


async def test_session_limit_counts_per_session() -> None:
    reg = ToolRegistry()
    reg.register(_spec(max_calls_per_session=2))
    kwargs = {"role": "n1", "user_id": "USR-0101", "session_id": "sesion-0001"}
    assert (await reg.invoke("echo", {"text": "a"}, **kwargs)).ok
    assert (await reg.invoke("echo", {"text": "b"}, **kwargs)).ok
    third = await reg.invoke("echo", {"text": "c"}, **kwargs)
    assert not third.ok
    assert third.data["scope"] == "session"
    # Otra sesión tiene su propio cupo.
    other = await reg.invoke(
        "echo", {"text": "d"}, role="n1", user_id="USR-0101", session_id="x" * 8
    )
    assert other.ok
    assert reg.session_calls("sesion-0001", "echo") == 2


async def test_timeout_degrades() -> None:
    async def slow(args: EchoInput, ctx: ToolContext) -> ToolResult:
        await asyncio.sleep(1)
        return ToolResult(tool="echo", ok=True)

    reg = ToolRegistry()
    reg.register(_spec(handler=slow, timeout_s=0.05))
    result = await reg.invoke("echo", {"text": "x"}, role="n1", user_id="USR-0101")
    assert not result.ok
    assert result.data["timeout"] is True
    assert result.data["unavailable"] is True


async def test_handler_exception_degrades() -> None:
    async def boom(args: EchoInput, ctx: ToolContext) -> ToolResult:
        raise RuntimeError("explota")

    reg = ToolRegistry()
    reg.register(_spec(handler=boom))
    result = await reg.invoke("echo", {"text": "x"}, role="n1", user_id="USR-0101")
    assert not result.ok
    assert result.data["error_type"] == "RuntimeError"


async def test_invalid_return_type_degrades() -> None:
    async def wrong(args: EchoInput, ctx: ToolContext) -> dict[str, str]:
        return {"no": "es ToolResult"}

    reg = ToolRegistry()
    reg.register(_spec(handler=wrong))
    result = await reg.invoke("echo", {"text": "x"}, role="n1", user_id="USR-0101")
    assert not result.ok


async def test_unavailable_by_flag_and_env(
    registry: ToolRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry.unavailable.add("echo")
    flagged = await registry.invoke("echo", {"text": "x"}, role="n1", user_id="USR-0101")
    assert flagged.data["simulated"] is True
    registry.unavailable.clear()
    monkeypatch.setenv(UNAVAILABLE_ENV, "all")
    assert registry.is_unavailable("echo")
    monkeypatch.setenv(UNAVAILABLE_ENV, "otra, echo")
    assert registry.is_unavailable("echo")
    monkeypatch.setenv(UNAVAILABLE_ENV, "")
    assert not registry.is_unavailable("echo")


async def test_simulated_outage_is_scoped_to_the_context(registry: ToolRegistry) -> None:
    import asyncio

    from docuerp.agent.tools.base import simulate_unavailable

    async def probe(outages: list[str]) -> tuple[bool, bool]:
        with simulate_unavailable(outages):
            await asyncio.sleep(0)  # cede el turno: la otra tarea corre con su propio contexto
            in_thread = await asyncio.to_thread(registry.is_unavailable, "echo")
            return registry.is_unavailable("echo"), in_thread

    down, healthy = await asyncio.gather(probe(["echo"]), probe([]))
    assert down == (True, True)
    assert healthy == (False, False)
    assert not registry.is_unavailable("echo")  # fuera del bloque no queda nada


def test_default_registry_has_four_tools_with_contracts(kit: types.SimpleNamespace) -> None:
    reg = default_registry(settings=kit.make_settings())
    assert reg.names() == sorted(TOOL_NAMES)
    create = reg.get("create_ticket")
    assert create.writes
    assert create.requires_confirmation
    assert create.max_calls_per_session == 3
    assert "portal_user" in create.allowed_roles
    assert reg.get("search_documents").input_model.model_fields["top_k"].metadata
    assert not reg.get("get_ticket_status").writes


def test_tool_context_session_key() -> None:
    assert ToolContext(role="n1", user_id="USR-0101").session_key == "user:USR-0101"
    assert ToolContext(role="n1", user_id="USR-0101", session_id="abc12345").session_key == (
        "abc12345"
    )


def test_tool_roles_come_from_roles_yaml() -> None:
    from docuerp.agent.tools.base import tool_roles

    assert tool_roles("classify_ticket") == ["n1", "n2", "admin"]
    # create_ticket suma el rol escalado (portal_user) de ``create_ticket_escalated``.
    assert tool_roles("create_ticket") == ["portal_user", "n1", "n2", "admin"]
    with pytest.raises(KeyError):
        tool_roles("borrar_todo")
