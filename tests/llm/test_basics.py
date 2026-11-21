"""Contratos, proveedor fake, precios, fábrica y semáforo por bucle."""

from __future__ import annotations

import asyncio

import pytest

from docuerp.config import LLMSettings, ModelPrice
from docuerp.llm import LLMError, LLMProvider, LLMResult, get_provider
from docuerp.llm.concurrency import LoopSemaphore, int_from_env
from docuerp.llm.contracts import LLMTimeout
from docuerp.llm.factory import clear_provider_cache, parse_provider_name
from docuerp.llm.fake import FakeProvider
from docuerp.llm.pricing import estimate_cost

# ---------------------------------------------------------------------------
# LLMResult / LLMError
# ---------------------------------------------------------------------------


def test_llm_result_to_usage_copies_every_counter() -> None:
    result = LLMResult(
        text="x",
        model="m",
        provider="p",
        input_tokens=10,
        output_tokens=20,
        cache_read_tokens=30,
        cache_creation_tokens=40,
        cost_usd=0.5,
        calls=2,
    )
    usage = result.to_usage()
    assert (usage.llm_calls, usage.input_tokens, usage.output_tokens) == (2, 10, 20)
    assert (usage.cache_read_tokens, usage.cache_creation_tokens, usage.cost_usd) == (30, 40, 0.5)


def test_llm_error_carries_typed_cause() -> None:
    assert LLMError("boom").cause == "unknown"
    assert LLMError("sin credencial", cause="auth").cause == "auth"
    timeout = LLMTimeout("tarde")
    assert isinstance(timeout, LLMError)
    assert timeout.cause == "timeout"


# ---------------------------------------------------------------------------
# FakeProvider
# ---------------------------------------------------------------------------


async def test_fake_returns_programmed_responses_in_order_and_records_calls() -> None:
    fake = FakeProvider([{"a": 1}, "texto libre"])
    first = await fake.complete("sys", "user", model="m1", json_schema={"type": "object"})
    second = await fake.complete("sys2", "user2")
    assert first.parsed == {"a": 1}
    assert second.text == "texto libre"
    assert second.parsed is None
    assert [call["system"] for call in fake.calls] == ["sys", "sys2"]
    assert fake.calls[0]["model"] == "m1"


async def test_fake_parses_json_strings_only_when_schema_requested() -> None:
    fake = FakeProvider(['{"k": "v"}', '{"k": "v"}', "no es json", "[1, 2]"])
    assert (await fake.complete("s", "u", json_schema={})).parsed == {"k": "v"}
    assert (await fake.complete("s", "u")).parsed is None
    assert (await fake.complete("s", "u", json_schema={})).parsed is None
    assert (await fake.complete("s", "u", json_schema={})).parsed is None  # lista, no objeto


async def test_fake_raises_programmed_exception_and_when_exhausted() -> None:
    fake = FakeProvider([LLMError("caído", cause="unavailable")])
    with pytest.raises(LLMError) as caught:
        await fake.complete("s", "u")
    assert caught.value.cause == "unavailable"
    with pytest.raises(LLMError, match="sin respuestas"):
        await fake.complete("s", "u")


def test_fake_satisfies_provider_protocol() -> None:
    assert isinstance(FakeProvider(), LLMProvider)


# ---------------------------------------------------------------------------
# Precios
# ---------------------------------------------------------------------------


def test_estimate_cost_uses_list_price_per_million_tokens() -> None:
    settings = LLMSettings()
    # Haiku: 0.10 USD/M entrada, 0.50 USD/M salida.
    assert estimate_cost("claude-haiku-5-5", 1_000_000, 0, settings=settings) == pytest.approx(0.10)
    assert estimate_cost("claude-haiku-5-5", 0, 1_000_000, settings=settings) == pytest.approx(0.50)


def test_estimate_cost_applies_cache_multipliers() -> None:
    settings = LLMSettings()
    cost = estimate_cost(
        "claude-sonnet-5-5",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=1_000_000,
        cache_creation_tokens=1_000_000,
        settings=settings,
    )
    # Sonnet: lectura con tarifa propia 0.10 + escritura 1.25x de 2.0
    assert cost == pytest.approx(0.10 + 2.0 * 1.25)
    haiku = estimate_cost("claude-haiku-5-5", 0, 0, cache_read_tokens=1_000_000, settings=settings)
    assert haiku == pytest.approx(0.10 * 0.1)  # sin tarifa propia: multiplicador global


def test_estimate_cost_unknown_model_is_zero_and_negative_tokens_rejected() -> None:
    settings = LLMSettings(pricing={"x": ModelPrice(input=1.0, output=1.0)})
    assert estimate_cost("modelo-desconocido", 100, 100, settings=settings) == 0.0
    with pytest.raises(ValueError, match="negativos"):
        estimate_cost("x", -1, 0, settings=settings)


def test_estimate_cost_reads_global_settings_by_default() -> None:
    assert estimate_cost("claude-opus-5-5", 1_000_000, 1_000_000) == pytest.approx(24.0)


# ---------------------------------------------------------------------------
# Fábrica
# ---------------------------------------------------------------------------


def test_parse_provider_name_validates() -> None:
    assert parse_provider_name("offline") == "offline"
    with pytest.raises(ValueError, match="desconocido"):
        parse_provider_name("gpt")


def test_get_provider_builds_each_kind_and_caches() -> None:
    clear_provider_cache()
    assert get_provider("offline").name == "offline"
    assert get_provider("fake").name == "fake"
    assert get_provider("agent_sdk").name == "agent_sdk"
    assert get_provider("anthropic_api").name == "anthropic_api"
    assert get_provider("offline") is get_provider("offline")


def test_get_provider_default_comes_from_settings() -> None:
    clear_provider_cache()
    assert get_provider().name == "agent_sdk"


def test_get_provider_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="desconocido"):
        get_provider("inexistente")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Concurrencia
# ---------------------------------------------------------------------------


def test_int_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAR_PRUEBA", "4")
    assert int_from_env("VAR_PRUEBA", 6) == 4
    monkeypatch.setenv("VAR_PRUEBA", "cero")
    assert int_from_env("VAR_PRUEBA", 6) == 6
    monkeypatch.setenv("VAR_PRUEBA", "0")
    assert int_from_env("VAR_PRUEBA", 6) == 6
    monkeypatch.delenv("VAR_PRUEBA")
    assert int_from_env("VAR_PRUEBA", 6) == 6


def test_loop_semaphore_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError, match="≥ 1"):
        LoopSemaphore(0)


async def test_loop_semaphore_limits_parallelism() -> None:
    semaphore = LoopSemaphore(2)
    running = 0
    peak = 0

    async def worker() -> None:
        nonlocal running, peak
        async with semaphore:
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1

    await asyncio.gather(*(worker() for _ in range(6)))
    assert peak == 2


def test_loop_semaphore_works_across_event_loops() -> None:
    semaphore = LoopSemaphore(1)

    async def use() -> str:
        async with semaphore:
            await asyncio.sleep(0)
        return "ok"

    # Dos bucles distintos (como dos llamadas a asyncio.run): no debe fallar.
    assert asyncio.run(use()) == "ok"
    assert asyncio.run(use()) == "ok"
