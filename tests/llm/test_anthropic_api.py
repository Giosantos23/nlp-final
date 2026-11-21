"""AnthropicAPIProvider con un cliente simulado (sin red ni clave real)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from docuerp.llm.anthropic_api import AnthropicAPIProvider, supports_adaptive_thinking
from docuerp.llm.contracts import LLMError, LLMTimeout

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def make_message(text: str = '{"ok": true}', stop_reason: str = "end_turn") -> SimpleNamespace:
    """Respuesta mínima con la forma de ``anthropic.types.Message``."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=40,
            cache_read_input_tokens=500,
            cache_creation_input_tokens=0,
        ),
    )


class FakeClient:
    """Cliente simulado: registra los argumentos de ``messages.create``."""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response if response is not None else make_message()
        self.error = error
        self.requests: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def provider_with(client: FakeClient, model: str = "claude-haiku-5-5") -> AnthropicAPIProvider:
    return AnthropicAPIProvider(default_model=model, client=client)  # type: ignore[arg-type]


def test_adaptive_thinking_only_for_sonnet_and_opus() -> None:
    assert supports_adaptive_thinking("claude-sonnet-5-5")
    assert supports_adaptive_thinking("claude-opus-5-5")
    assert not supports_adaptive_thinking("claude-haiku-5-5")


async def test_structured_output_request_and_result() -> None:
    client = FakeClient()
    result = await provider_with(client).complete(
        "SISTEMA", "USUARIO", json_schema=SCHEMA, max_tokens=900, timeout_s=12
    )
    request = client.requests[0]
    assert request["model"] == "claude-haiku-5-5"
    assert request["system"] == "SISTEMA"
    assert request["messages"] == [{"role": "user", "content": "USUARIO"}]
    assert request["max_tokens"] == 900
    assert request["timeout"] == 12
    assert request["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    assert "thinking" not in request  # Haiku no admite thinking adaptativo
    assert "temperature" not in request
    assert result.parsed == {"ok": True}
    assert result.provider == "anthropic_api"
    assert (result.input_tokens, result.output_tokens, result.cache_read_tokens) == (100, 40, 500)
    # Haiku: 100*0.10 + 500*0.10*0.1 de entrada, 40*0.50 de salida (USD por millón).
    assert result.cost_usd == pytest.approx((100 * 0.10 + 500 * 0.01 + 40 * 0.50) / 1_000_000)


async def test_sonnet_gets_adaptive_thinking_and_effort() -> None:
    client = FakeClient()
    await provider_with(client).complete(
        "s", "u", model="claude-sonnet-5-5", json_schema=SCHEMA, effort="low"
    )
    request = client.requests[0]
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"]["effort"] == "low"
    assert request["output_config"]["format"]["type"] == "json_schema"


async def test_plain_text_request_has_no_output_config() -> None:
    client = FakeClient(make_message("hola"))
    result = await provider_with(client).complete("s", "u")
    assert "output_config" not in client.requests[0]
    assert result.text == "hola"
    assert result.parsed is None


async def test_invalid_effort_rejected() -> None:
    with pytest.raises(ValueError, match="effort"):
        await provider_with(FakeClient()).complete("s", "u", effort="turbo")


@pytest.mark.parametrize(
    "message",
    [
        make_message("esto no es json"),
        make_message("[1, 2]"),
        make_message('{"ok": tr', "max_tokens"),
    ],
)
async def test_schema_failures_are_typed(message: SimpleNamespace) -> None:
    with pytest.raises(LLMError) as caught:
        await provider_with(FakeClient(message)).complete("s", "u", json_schema=SCHEMA)
    assert caught.value.cause == "schema"


async def test_missing_api_key_gives_clear_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("docuerp.llm.anthropic_api.load_env", lambda: False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY") as caught:
        await AnthropicAPIProvider().complete("s", "u")
    assert caught.value.cause == "auth"


@pytest.mark.parametrize(
    ("error", "cause"),
    [
        (
            anthropic.AuthenticationError(
                "clave inválida", response=httpx.Response(401, request=REQUEST), body=None
            ),
            "auth",
        ),
        (
            anthropic.NotFoundError(
                "model: x", response=httpx.Response(404, request=REQUEST), body=None
            ),
            "invalid_model",
        ),
        (
            anthropic.BadRequestError(
                "invalid model", response=httpx.Response(400, request=REQUEST), body=None
            ),
            "invalid_model",
        ),
        (
            anthropic.BadRequestError(
                "otra cosa", response=httpx.Response(400, request=REQUEST), body=None
            ),
            "unknown",
        ),
        (
            anthropic.RateLimitError(
                "demasiadas", response=httpx.Response(429, request=REQUEST), body=None
            ),
            "unavailable",
        ),
        (anthropic.APIConnectionError(request=REQUEST), "unavailable"),
    ],
)
async def test_api_errors_are_mapped_to_causes(error: Exception, cause: str) -> None:
    with pytest.raises(LLMError) as caught:
        await provider_with(FakeClient(error=error)).complete("s", "u")
    assert caught.value.cause == cause


async def test_sdk_timeout_becomes_llm_timeout() -> None:
    with pytest.raises(LLMTimeout):
        await provider_with(FakeClient(error=anthropic.APITimeoutError(request=REQUEST))).complete(
            "s", "u"
        )


async def test_overall_timeout_becomes_llm_timeout() -> None:
    import asyncio

    class Slow(FakeClient):
        async def _create(self, **kwargs: Any) -> Any:
            await asyncio.sleep(1)

    with pytest.raises(LLMTimeout):
        await provider_with(Slow()).complete("s", "u", timeout_s=0.05)


async def test_parsed_json_roundtrip_text_is_kept() -> None:
    payload = {"ok": False}
    result = await provider_with(FakeClient(make_message(json.dumps(payload)))).complete(
        "s", "u", json_schema=SCHEMA
    )
    assert result.parsed == payload
    assert json.loads(result.text) == payload
