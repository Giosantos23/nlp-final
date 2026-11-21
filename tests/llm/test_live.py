"""Pruebas con el LLM real (``@pytest.mark.llm``): excluidas de ``make test``.

Ejecutar con ``pytest tests/llm/test_live.py -m llm -s``. Usan la suscripción configurada en
``.env`` (``CLAUDE_CODE_OAUTH_TOKEN``) con ``claude-haiku-5-5`` vía ``agent_sdk`` y imprimen
latencia, tokens y costo de lista.
"""

from __future__ import annotations

import pytest

from docuerp.generation.demo import SCENARIOS
from docuerp.generation.generator import generate_answer
from docuerp.generation.prompts import ANSWER_JSON_SCHEMA, build_system_prompt, build_user_prompt
from docuerp.llm.agent_sdk import AgentSDKProvider
from docuerp.schemas import ABSTENTION_MESSAGE, CANARY_TOKEN, Answer

pytestmark = pytest.mark.llm

MODEL = "claude-haiku-5-5"


def report(label: str, result: object) -> None:
    """Imprime el consumo de una llamada real (visible con ``-s``)."""
    print(f"\n[{label}] {result}")


@pytest.fixture
def real_provider() -> AgentSDKProvider:
    """Proveedor real; toma el token de ``.env`` (el conftest limpia el del entorno)."""
    return AgentSDKProvider(default_model=MODEL)


async def test_real_structured_output_is_a_valid_answer(real_provider: AgentSDKProvider) -> None:
    case = SCENARIOS["recepcion"]
    result = await real_provider.complete(
        build_system_prompt(),
        build_user_prompt(case.question, case.hits, intent=case.intent, erp_version="18.0"),
        model=MODEL,
        json_schema=ANSWER_JSON_SCHEMA,
        effort="low",
    )
    answer = Answer.model_validate(result.parsed)
    report(
        "structured_output",
        f"latencia={result.latency_ms:.0f} ms api={result.api_latency_ms:.0f} ms "
        f"tokens(in/out/cache_r/cache_w)={result.input_tokens}/{result.output_tokens}/"
        f"{result.cache_read_tokens}/{result.cache_creation_tokens} costo=US${result.cost_usd:.5f}",
    )
    assert answer.abstained is False
    assert answer.sources[0].chunk_id == case.hits[0].chunk.chunk_id
    assert CANARY_TOKEN not in answer.answer
    assert result.cost_usd > 0
    assert result.api_latency_ms is not None
    assert result.latency_ms > result.api_latency_ms


async def test_real_end_to_end_grounded_answer_and_abstention(
    real_provider: AgentSDKProvider,
) -> None:
    grounded = SCENARIOS["recepcion"]
    answer, result = await generate_answer(
        grounded.question,
        grounded.hits,
        intent=grounded.intent,
        provider=real_provider,
        model=MODEL,
        erp_version="18.0",
    )
    report(
        "generate_answer/recepcion",
        f"latencia={result.latency_ms:.0f} ms llamadas={result.calls} "
        f"costo=US${result.cost_usd:.5f}",
    )
    # Si el LLM fallara y se degradara al proveedor offline, esta prueba debe fallar.
    assert result.provider == "agent_sdk"
    assert result.calls >= 1
    assert answer.abstained is False
    assert {s.chunk_id for s in answer.sources} <= {h.chunk.chunk_id for h in grounded.hits}
    assert answer.sources
    assert "validar" in answer.answer.lower()

    ungrounded = SCENARIOS["sin-evidencia"]
    answer, result = await generate_answer(
        ungrounded.question,
        ungrounded.hits,
        intent=ungrounded.intent,
        provider=real_provider,
        model=MODEL,
        erp_version="18.0",
    )
    report(
        "generate_answer/sin-evidencia",
        f"latencia={result.latency_ms:.0f} ms llamadas={result.calls} "
        f"costo=US${result.cost_usd:.5f}",
    )
    assert result.provider == "agent_sdk"
    assert result.calls >= 1
    assert answer.abstained is True
    assert answer.answer == ABSTENTION_MESSAGE
    assert answer.sources == []
