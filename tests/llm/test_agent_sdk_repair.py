"""Reparación de la fuga de formato de herramienta que produce Haiku con salida estructurada."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from claude_agent_sdk import AssistantMessage, ResultError, ResultMessage, ToolUseBlock

from docuerp.llm import agent_sdk
from docuerp.llm.agent_sdk import AgentSDKProvider, _final_result, repair_structured_output

ResultFactory = Callable[..., ResultMessage]

REQUIRED = ["answer", "sources", "abstained", "tool_used", "needs_human_review"]
SOURCES_JSON = '[{"source": "a.pdf", "page": 12, "chunk_id": "a::p0012::c000"}]'


def leaked_input(**overrides: Any) -> dict[str, Any]:
    """Entrada de ``StructuredOutput`` con ``sources`` pegado dentro de ``answer``."""
    payload: dict[str, Any] = {
        "answer": f'Paso 1. Validar.</answer>\n<parameter name="sources">{SOURCES_JSON}',
        "abstained": False,
        "tool_used": None,
        "needs_human_review": False,
    }
    payload.update(overrides)
    return payload


def test_repair_recovers_leaked_sources_parameter() -> None:
    repaired = repair_structured_output(leaked_input(), REQUIRED)
    assert repaired is not None
    assert repaired["answer"] == "Paso 1. Validar."
    assert repaired["sources"][0]["chunk_id"] == "a::p0012::c000"
    assert repaired["abstained"] is False


def test_repair_handles_closing_tags_and_several_leaked_parameters() -> None:
    closing = f'Texto.</answer>\n<parameter name="sources">{SOURCES_JSON}</parameter>\n</invoke>'
    repaired = repair_structured_output(leaked_input(answer=closing), REQUIRED)
    assert repaired is not None
    assert repaired["answer"] == "Texto."
    two = leaked_input(
        answer=(
            f'Texto.</answer><parameter name="sources">{SOURCES_JSON}</parameter>'
            '<parameter name="needs_human_review">true</parameter>'
        )
    )
    repaired = repair_structured_output(two, REQUIRED)
    assert repaired is not None
    assert repaired["needs_human_review"] is True


@pytest.mark.parametrize(
    "tool_input",
    [
        {"answer": "respuesta limpia", "abstained": False},  # sin fuga: nada que reparar
        leaked_input(answer='x</answer><parameter name="sources">no es json'),
        leaked_input(answer='x</answer><parameter name="otro">[]'),  # sigue faltando `sources`
        {"abstained": False},  # sin answer
    ],
)
def test_repair_returns_none_when_it_cannot_fix(tool_input: dict[str, Any]) -> None:
    assert repair_structured_output(tool_input, REQUIRED) is None


def stream(*messages: Any, then_raise: Exception | None = None) -> AsyncIterator[Any]:
    """Flujo de mensajes simulado; opcionalmente termina lanzando ``then_raise``."""

    async def generate() -> AsyncIterator[Any]:
        for message in messages:
            yield message
        if then_raise is not None:
            raise then_raise

    return generate()


def failed_stream(
    result_message: ResultFactory, tool_input: dict[str, Any], *, raise_error: bool
) -> AsyncIterator[Any]:
    """Un intento cuya salida estructurada fue rechazada por el CLI (``error_max_turns``)."""
    assistant = AssistantMessage(
        content=[ToolUseBlock(id="t1", name="StructuredOutput", input=tool_input)],
        model="claude-haiku-5-5",
    )
    failed = result_message(is_error=True, subtype="error_max_turns", result=None)
    error = ResultError("max turns", data={"subtype": "error_max_turns"}) if raise_error else None
    return stream(assistant, failed, then_raise=error)


@pytest.mark.parametrize("raise_error", [False, True])
async def test_final_result_salvages_failed_structured_output(
    result_message: ResultFactory, raise_error: bool
) -> None:
    messages = failed_stream(result_message, leaked_input(), raise_error=raise_error)
    result = await _final_result(messages, REQUIRED)
    assert result is not None
    assert result.is_error is False
    assert result.structured_output["sources"][0]["page"] == 12
    assert json.loads(result.result or "")["answer"] == "Paso 1. Validar."


async def test_final_result_reraises_when_failure_is_not_repairable(
    result_message: ResultFactory,
) -> None:
    messages = failed_stream(result_message, {"answer": "sin fuga"}, raise_error=True)
    with pytest.raises(ResultError):
        await _final_result(messages, REQUIRED)


async def test_final_result_without_required_fields_never_repairs(
    result_message: ResultFactory,
) -> None:
    messages = failed_stream(result_message, leaked_input(), raise_error=False)
    result = await _final_result(messages)
    assert result is not None
    assert result.is_error is True


async def test_provider_returns_repaired_answer_end_to_end(
    monkeypatch: pytest.MonkeyPatch, result_message: ResultFactory
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "token-de-prueba-123")

    async def fake_query(*, prompt: str, options: Any = None) -> AsyncIterator[Any]:
        async for message in failed_stream(result_message, leaked_input(), raise_error=True):
            yield message

    monkeypatch.setattr(agent_sdk, "query", fake_query)
    schema = {"type": "object", "required": REQUIRED}
    result = await AgentSDKProvider().complete("s", "u", json_schema=schema)
    assert result.parsed is not None
    assert result.parsed["sources"][0]["chunk_id"] == "a::p0012::c000"
