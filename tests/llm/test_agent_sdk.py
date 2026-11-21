"""AgentSDKProvider con el SDK simulado (sin red ni CLI)."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest
from claude_agent_sdk import CLINotFoundError, ResultError, ResultMessage

from docuerp.llm import agent_sdk
from docuerp.llm.agent_sdk import AgentSDKProvider, classify_failure
from docuerp.llm.contracts import LLMError, LLMTimeout

ResultFactory = Callable[..., ResultMessage]
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


@pytest.fixture
def token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Token ficticio ya presente en el entorno (no se toca ningún .env real)."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "token-de-prueba-123")


def patch_query(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: ResultMessage | None = None,
    error: Exception | None = None,
    delay: float = 0.0,
) -> dict[str, Any]:
    """Reemplaza ``claude_agent_sdk.query`` y devuelve un diccionario con las opciones vistas."""
    seen: dict[str, Any] = {"calls": 0}

    async def fake_query(*, prompt: str, options: Any = None) -> AsyncIterator[Any]:
        seen["calls"] += 1
        seen["prompt"] = prompt
        seen["options"] = options
        if delay:
            await asyncio.sleep(delay)
        if error is not None:
            raise error
        if result is not None:
            yield result

    monkeypatch.setattr(agent_sdk, "query", fake_query)
    return seen


# ---------------------------------------------------------------------------
# Clasificación de errores
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "status", "expected"),
    [
        (
            "There's an issue with the selected model (x). It may not exist or you may not "
            "have access to it.",
            None,
            "invalid_model",
        ),
        ("[claude-code:unrecognized_model]", None, "invalid_model"),
        ("lo que sea", 404, "invalid_model"),
        ("OAuth token has expired", None, "auth"),
        ("Invalid API key", None, "auth"),
        ("lo que sea", 401, "auth"),
        ("API Error: overloaded", None, "unavailable"),
        ("lo que sea", 529, "unavailable"),
        ("Reached maximum number of turns (2)", None, "schema"),
        ("Output does not match required schema: must have required property 'x'", None, "schema"),
        ("algo raro", None, "unknown"),
    ],
)
def test_classify_failure(message: str, status: int | None, expected: str) -> None:
    assert classify_failure(message, status) == expected


# ---------------------------------------------------------------------------
# Entorno y autenticación
# ---------------------------------------------------------------------------


def test_exports_token_from_dotenv_and_removes_claudecode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("CLAUDE_CODE_OAUTH_TOKEN=token-desde-archivo\n", encoding="utf-8")
    monkeypatch.setenv("CLAUDECODE", "1")
    provider = AgentSDKProvider(env_path=env_file)
    provider._prepare_environment()
    assert os.environ["CLAUDE_CODE_OAUTH_TOKEN"] == "token-desde-archivo"
    assert "CLAUDECODE" not in os.environ


def test_existing_environment_token_wins_over_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("CLAUDE_CODE_OAUTH_TOKEN=del-archivo\n", encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "del-entorno")
    AgentSDKProvider(env_path=env_file)._prepare_environment()
    assert os.environ["CLAUDE_CODE_OAUTH_TOKEN"] == "del-entorno"


async def test_missing_token_raises_auth_error_without_calling_sdk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = patch_query(monkeypatch)
    provider = AgentSDKProvider(env_path=tmp_path / "no-existe.env")
    with pytest.raises(LLMError) as caught:
        await provider.complete("sys", "user")
    assert caught.value.cause == "auth"
    assert seen["calls"] == 0


# ---------------------------------------------------------------------------
# Opciones del SDK
# ---------------------------------------------------------------------------


async def test_options_follow_the_probed_configuration(
    monkeypatch: pytest.MonkeyPatch, token_env: None, result_message: ResultFactory
) -> None:
    seen = patch_query(
        monkeypatch, result=result_message(structured_output={"ok": True}, result="{}")
    )
    provider = AgentSDKProvider(default_model="claude-haiku-5-5")
    await provider.complete("SISTEMA", "USUARIO", json_schema=SCHEMA, max_tokens=1024, effort="low")
    options = seen["options"]
    assert options.model == "claude-haiku-5-5"
    assert options.effort == "low"
    assert options.system_prompt == "SISTEMA"
    assert options.tools == []
    assert options.setting_sources == []
    assert options.permission_mode == "dontAsk"
    assert options.max_turns == 1  # una salida inválida falla rápido; se repara o se reintenta
    assert options.output_format == {"type": "json_schema", "schema": SCHEMA}
    # max_tokens=1024 se eleva al mínimo medido de 2048; y se fuerza la suscripción.
    assert options.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "2048"
    assert options.env["ANTHROPIC_API_KEY"] == ""
    assert seen["prompt"] == "USUARIO"


async def test_options_without_schema_use_one_turn_and_keep_large_max_tokens(
    monkeypatch: pytest.MonkeyPatch, token_env: None, result_message: ResultFactory
) -> None:
    seen = patch_query(monkeypatch, result=result_message())
    await AgentSDKProvider().complete("s", "u", model="claude-sonnet-5-5", max_tokens=4096)
    options = seen["options"]
    assert options.max_turns == 1
    assert options.output_format is None
    assert options.model == "claude-sonnet-5-5"
    assert options.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "4096"


async def test_invalid_effort_is_a_programming_error(
    monkeypatch: pytest.MonkeyPatch, token_env: None
) -> None:
    patch_query(monkeypatch)
    with pytest.raises(ValueError, match="effort"):
        await AgentSDKProvider().complete("s", "u", effort="turbo")


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------


async def test_maps_result_message_to_llm_result(
    monkeypatch: pytest.MonkeyPatch, token_env: None, result_message: ResultFactory
) -> None:
    patch_query(monkeypatch, result=result_message(structured_output={"ok": True}, result="{}"))
    result = await AgentSDKProvider().complete("s", "u", json_schema=SCHEMA)
    assert result.parsed == {"ok": True}
    assert result.provider == "agent_sdk"
    assert result.model == "claude-haiku-5-5"
    assert (result.input_tokens, result.output_tokens) == (2, 50)
    assert (result.cache_read_tokens, result.cache_creation_tokens) == (100, 1400)
    assert result.cost_usd == pytest.approx(0.0004)
    assert result.api_latency_ms == 1200.0
    assert result.latency_ms >= 0
    assert result.calls == 1


async def test_falls_back_to_parsing_result_text_when_no_structured_output(
    monkeypatch: pytest.MonkeyPatch, token_env: None, result_message: ResultFactory
) -> None:
    patch_query(monkeypatch, result=result_message(structured_output=None, result='{"ok": false}'))
    result = await AgentSDKProvider().complete("s", "u", json_schema=SCHEMA)
    assert result.parsed == {"ok": False}


async def test_schema_requested_but_missing_raises_schema_error(
    monkeypatch: pytest.MonkeyPatch, token_env: None, result_message: ResultFactory
) -> None:
    patch_query(monkeypatch, result=result_message(structured_output=None, result="no es json"))
    with pytest.raises(LLMError) as caught:
        await AgentSDKProvider().complete("s", "u", json_schema=SCHEMA)
    assert caught.value.cause == "schema"


async def test_cost_is_estimated_when_cli_reports_none(
    monkeypatch: pytest.MonkeyPatch, token_env: None, result_message: ResultFactory
) -> None:
    patch_query(
        monkeypatch,
        result=result_message(
            total_cost_usd=None,
            usage={"input_tokens": 1_000_000, "output_tokens": 0},
        ),
    )
    result = await AgentSDKProvider().complete("s", "u", model="claude-haiku-5-5")
    assert result.cost_usd == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Errores
# ---------------------------------------------------------------------------


async def test_timeout_becomes_llm_timeout(
    monkeypatch: pytest.MonkeyPatch, token_env: None, result_message: ResultFactory
) -> None:
    patch_query(monkeypatch, result=result_message(), delay=1.0)
    with pytest.raises(LLMTimeout) as caught:
        await AgentSDKProvider().complete("s", "u", timeout_s=0.05)
    assert caught.value.cause == "timeout"


async def test_invalid_model_result_error_is_typed(
    monkeypatch: pytest.MonkeyPatch, token_env: None
) -> None:
    error = ResultError(
        "exit 1",
        data={"subtype": "error", "result": "There's an issue with the selected model (x)."},
    )
    patch_query(monkeypatch, error=error)
    with pytest.raises(LLMError) as caught:
        await AgentSDKProvider().complete("s", "u", model="claude-inexistente")
    assert caught.value.cause == "invalid_model"
    assert "claude-inexistente" in str(caught.value)


async def test_missing_cli_is_unavailable(monkeypatch: pytest.MonkeyPatch, token_env: None) -> None:
    patch_query(monkeypatch, error=CLINotFoundError("sin cli"))
    with pytest.raises(LLMError) as caught:
        await AgentSDKProvider().complete("s", "u")
    assert caught.value.cause == "unavailable"


async def test_error_result_message_is_classified(
    monkeypatch: pytest.MonkeyPatch, token_env: None, result_message: ResultFactory
) -> None:
    patch_query(
        monkeypatch,
        result=result_message(
            is_error=True, result="API Error: Unauthorized", api_error_status=401
        ),
    )
    with pytest.raises(LLMError) as caught:
        await AgentSDKProvider().complete("s", "u")
    assert caught.value.cause == "auth"


async def test_no_result_message_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, token_env: None
) -> None:
    patch_query(monkeypatch)  # el flujo termina sin ResultMessage
    with pytest.raises(LLMError, match="ResultMessage"):
        await AgentSDKProvider().complete("s", "u")


async def test_unexpected_exception_is_wrapped_without_leaking_details(
    monkeypatch: pytest.MonkeyPatch, token_env: None
) -> None:
    patch_query(monkeypatch, error=OSError("fallo al lanzar el proceso"))
    with pytest.raises(LLMError) as caught:
        await AgentSDKProvider().complete("s", "u")
    assert caught.value.cause == "unknown"
    assert "OSError" in str(caught.value)


async def test_error_messages_never_contain_the_token(
    monkeypatch: pytest.MonkeyPatch, token_env: None
) -> None:
    secret = os.environ["CLAUDE_CODE_OAUTH_TOKEN"]
    patch_query(monkeypatch, error=ResultError(f"fallo con {secret}", data={"result": secret}))
    with pytest.raises(LLMError) as caught:
        await AgentSDKProvider().complete("s", "u")
    assert secret not in str(caught.value)


# ---------------------------------------------------------------------------
# Concurrencia
# ---------------------------------------------------------------------------


async def test_semaphore_caps_concurrent_calls(
    monkeypatch: pytest.MonkeyPatch, token_env: None, result_message: ResultFactory
) -> None:
    running = 0
    peak = 0

    async def slow_query(*, prompt: str, options: Any = None) -> AsyncIterator[Any]:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.02)
        running -= 1
        yield result_message()

    monkeypatch.setattr(agent_sdk, "query", slow_query)
    provider = AgentSDKProvider(max_concurrency=2)
    await asyncio.gather(*(provider.complete("s", f"u{i}") for i in range(6)))
    assert peak == 2


def test_concurrency_can_be_configured_with_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCUERP_LLM_MAX_CONCURRENCY", "3")
    assert AgentSDKProvider()._semaphore.limit == 3
    assert AgentSDKProvider(max_concurrency=5)._semaphore.limit == 5


# ---------------------------------------------------------------------------
# Pool de clientes (opcional)
# ---------------------------------------------------------------------------


class FakeClient:
    """Sustituto de ``ClaudeSDKClient`` que registra prompts y conexiones."""

    instances: ClassVar[list[FakeClient]] = []
    result: ResultMessage

    def __init__(self, options: Any) -> None:
        self.options = options
        self.prompts: list[str] = []
        self.connected = False
        FakeClient.instances.append(self)

    async def __aenter__(self) -> FakeClient:
        self.connected = True
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        self.connected = False
        return False

    async def query(self, prompt: str) -> None:
        self.prompts.append(prompt)

    async def receive_response(self) -> AsyncIterator[ResultMessage]:
        yield FakeClient.result


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch, result_message: ResultFactory) -> type[FakeClient]:
    FakeClient.instances = []
    FakeClient.result = result_message(result="respuesta del cliente")
    monkeypatch.setattr(agent_sdk, "ClaudeSDKClient", FakeClient)
    return FakeClient


async def test_pool_reuses_client_and_clears_between_queries(
    token_env: None, fake_client: type[FakeClient]
) -> None:
    provider = AgentSDKProvider(use_client_pool=True)
    first = await provider.complete("sys", "pregunta 1")
    second = await provider.complete("sys", "pregunta 2")
    assert first.text == second.text == "respuesta del cliente"
    assert len(fake_client.instances) == 1
    client = fake_client.instances[0]
    # Cada consulta va seguida de /clear: nada de la conversación se arrastra.
    assert client.prompts == ["pregunta 1", "/clear", "pregunta 2", "/clear"]
    await provider.aclose()
    assert client.connected is False


async def test_pool_creates_one_client_per_concurrent_call_and_per_configuration(
    token_env: None, fake_client: type[FakeClient]
) -> None:
    provider = AgentSDKProvider(use_client_pool=True)

    async def slow_receive(self: FakeClient) -> AsyncIterator[ResultMessage]:
        await asyncio.sleep(0.02)
        yield FakeClient.result

    fake_client.receive_response = slow_receive  # type: ignore[method-assign,assignment]
    await asyncio.gather(provider.complete("sys", "a"), provider.complete("sys", "b"))
    assert len(fake_client.instances) == 2
    await provider.complete("otro system", "c")  # otra configuración ⇒ otro cliente
    assert len(fake_client.instances) == 3
    await provider.aclose()


async def test_pool_discards_client_after_error(
    token_env: None, fake_client: type[FakeClient]
) -> None:
    provider = AgentSDKProvider(use_client_pool=True)

    async def broken_receive(self: FakeClient) -> AsyncIterator[ResultMessage]:
        raise ResultError("exit 1", data={"result": "There's an issue with the selected model"})
        yield FakeClient.result  # pragma: no cover

    fake_client.receive_response = broken_receive  # type: ignore[method-assign,assignment]
    with pytest.raises(LLMError) as caught:
        await provider.complete("sys", "q", model="claude-malo")
    assert caught.value.cause == "invalid_model"
    await asyncio.sleep(0)  # deja que la tarea del cliente muerto termine
    assert fake_client.instances[0].connected is False
    await provider.aclose()


async def test_pool_times_out_and_kills_the_client(
    token_env: None, fake_client: type[FakeClient]
) -> None:
    provider = AgentSDKProvider(use_client_pool=True)

    async def stuck_receive(self: FakeClient) -> AsyncIterator[ResultMessage]:
        await asyncio.sleep(5)
        yield FakeClient.result  # pragma: no cover

    fake_client.receive_response = stuck_receive  # type: ignore[method-assign,assignment]
    with pytest.raises(LLMTimeout):
        await provider.complete("sys", "q", timeout_s=0.05)
    await asyncio.sleep(0.05)
    assert fake_client.instances[0].connected is False
    await provider.aclose()


def test_pool_flag_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    assert AgentSDKProvider().use_client_pool is False
    monkeypatch.setenv("DOCUERP_LLM_CLIENT_POOL", "1")
    assert AgentSDKProvider().use_client_pool is True
    assert AgentSDKProvider(use_client_pool=False).use_client_pool is False
