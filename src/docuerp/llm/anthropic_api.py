"""Proveedor ``anthropic_api``: SDK ``anthropic`` con ``ANTHROPIC_API_KEY`` (producción).

Es la alternativa de producción al proveedor ``agent_sdk`` (cuya suscripción es solo para
desarrollo y evaluación personal). Llama a ``messages.create`` directamente, sin
arrancar un CLI, por lo que evita ~1.4 s de sobrecarga por llamada.

Decisiones:

* Salida estructurada con ``output_config={"format": {"type": "json_schema", ...}}``: la
  respuesta llega como texto JSON ya validado contra el esquema.
* ``thinking`` adaptativo y ``effort`` solo en Sonnet y Opus; Haiku no los admite, así
  que en ese modelo se omiten.
* No se envía ``temperature``: con salida estructurada y esquema cerrado la variación es
  mínima y varios modelos recientes rechazan parámetros de muestreo.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import weakref
from typing import Any

import anthropic

from docuerp.config import load_env
from docuerp.llm.agent_sdk import VALID_EFFORTS
from docuerp.llm.contracts import LLMError, LLMErrorCause, LLMResult, LLMTimeout
from docuerp.llm.pricing import estimate_cost

API_KEY_VARIABLE = "ANTHROPIC_API_KEY"
"""Variable de entorno con la clave de la API (se lee de ``.env``; nunca se imprime)."""

_ADAPTIVE_THINKING_FAMILIES = ("sonnet", "opus")
"""Familias de modelo con ``thinking`` adaptativo y ``effort`` (Haiku no los admite)."""


def supports_adaptive_thinking(model: str) -> bool:
    """Indica si el modelo admite ``thinking`` adaptativo y el parámetro ``effort``."""
    return any(family in model for family in _ADAPTIVE_THINKING_FAMILIES)


def _cause_from_api_error(exc: anthropic.APIError) -> LLMErrorCause:
    """Traduce una excepción del SDK de Anthropic a la causa tipada de :class:`LLMError`."""
    if isinstance(exc, anthropic.APITimeoutError):
        return "timeout"
    if isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
        return "auth"
    if isinstance(exc, anthropic.NotFoundError):
        return "invalid_model"
    if isinstance(exc, anthropic.BadRequestError):
        return "invalid_model" if "model" in str(exc).lower() else "unknown"
    if isinstance(
        exc,
        anthropic.APIConnectionError | anthropic.RateLimitError | anthropic.InternalServerError,
    ):
        return "unavailable"
    return "unknown"


class AnthropicAPIProvider:
    """Implementación de ``LLMProvider`` sobre la API de Messages.

    Args:
        default_model: modelo cuando ``complete`` no recibe ``model``.
        client: cliente ``AsyncAnthropic`` inyectado (pruebas); por defecto se crea uno
            por bucle de eventos con la clave de ``ANTHROPIC_API_KEY``.
    """

    name = "anthropic_api"

    def __init__(
        self,
        default_model: str = "claude-haiku-5-5",
        *,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self.default_model = default_model
        self._injected_client = client
        self._clients: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, anthropic.AsyncAnthropic
        ] = weakref.WeakKeyDictionary()

    def _client(self) -> anthropic.AsyncAnthropic:
        """Devuelve el cliente del bucle actual (los clientes httpx no se comparten entre bucles).

        Raises:
            LLMError: con causa ``auth`` si no hay ``ANTHROPIC_API_KEY``.
        """
        if self._injected_client is not None:
            return self._injected_client
        load_env()
        api_key = os.environ.get(API_KEY_VARIABLE, "").strip()
        if not api_key:
            raise LLMError(
                f"Falta {API_KEY_VARIABLE}: defínela en .env o en el entorno para usar el "
                "proveedor anthropic_api (o cambia a agent_sdk / offline).",
                cause="auth",
            )
        loop = asyncio.get_running_loop()
        client = self._clients.get(loop)
        if client is None:
            client = self._clients[loop] = anthropic.AsyncAnthropic(api_key=api_key, max_retries=1)
        return client

    def _build_request(
        self,
        system: str,
        user: str,
        model: str,
        json_schema: dict[str, Any] | None,
        max_tokens: int,
        effort: str | None,
        timeout_s: float,
    ) -> dict[str, Any]:
        """Arma los argumentos de ``messages.create`` según el modelo y las opciones."""
        if effort is not None and effort not in VALID_EFFORTS:
            raise ValueError(f"effort inválido: {effort!r}; use uno de {sorted(VALID_EFFORTS)}")
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "timeout": timeout_s,
        }
        output_config: dict[str, Any] = {}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        if supports_adaptive_thinking(model):
            request["thinking"] = {"type": "adaptive"}
            if effort is not None:
                output_config["effort"] = effort
        if output_config:
            request["output_config"] = output_config
        return request

    async def complete(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
        effort: str | None = None,
        timeout_s: float = 45.0,
    ) -> LLMResult:
        """Llamada a Messages con salida estructurada si ``json_schema`` está presente.

        Raises:
            LLMError: ``auth`` (sin clave o rechazada), ``invalid_model``, ``schema``
                (respuesta truncada o que no es JSON), ``unavailable`` o ``unknown``;
                :class:`LLMTimeout` si se agota el tiempo.
            ValueError: si ``effort`` no es un valor válido.
        """
        model = model or self.default_model
        client = self._client()
        request = self._build_request(
            system, user, model, json_schema, max_tokens, effort, timeout_s
        )
        started = time.perf_counter()
        try:
            async with asyncio.timeout(timeout_s):
                message = await client.messages.create(**request)
        except TimeoutError as exc:
            raise LLMTimeout(f"Tiempo agotado ({timeout_s:g} s) con el modelo {model}.") from exc
        except anthropic.APIError as exc:
            cause = _cause_from_api_error(exc)
            if cause == "timeout":
                raise LLMTimeout(
                    f"Tiempo agotado ({timeout_s:g} s) con el modelo {model}."
                ) from exc
            raise LLMError(
                f"Error de la API de Anthropic ({type(exc).__name__}): {str(exc)[:300]}", cause
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        return self._to_llm_result(message, model, json_schema, latency_ms)

    def _to_llm_result(
        self,
        message: Any,
        model: str,
        json_schema: dict[str, Any] | None,
        latency_ms: float,
    ) -> LLMResult:
        """Convierte la respuesta de Messages en :class:`LLMResult` (con costo estimado)."""
        text = "".join(block.text for block in message.content if block.type == "text")
        parsed: dict[str, Any] | None = None
        if json_schema is not None:
            if message.stop_reason == "max_tokens":
                raise LLMError(
                    "La respuesta JSON se truncó por max_tokens; súbelo o acorta el contexto.",
                    cause="schema",
                )
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMError("La respuesta no es JSON válido.", cause="schema") from exc
            if not isinstance(value, dict):
                raise LLMError("La respuesta JSON no es un objeto.", cause="schema")
            parsed = value
        usage = message.usage
        input_tokens = usage.input_tokens or 0
        output_tokens = usage.output_tokens or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        return LLMResult(
            text=text,
            parsed=parsed,
            model=model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
            cost_usd=estimate_cost(model, input_tokens, output_tokens, cache_read, cache_creation),
            latency_ms=latency_ms,
            api_latency_ms=latency_ms,
        )
