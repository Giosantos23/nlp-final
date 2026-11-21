"""Proveedor ``fake`` para pruebas unitarias: devuelve respuestas programadas."""

from __future__ import annotations

import json
from typing import Any

from docuerp.llm.contracts import LLMError, LLMResult


class FakeProvider:
    """Devuelve en orden las respuestas de ``responses`` y registra las llamadas.

    Cada elemento de ``responses`` puede ser:

    * ``dict``: se serializa como texto y queda en ``parsed`` (simula salida estructurada).
    * ``str``: texto crudo; si la llamada pide ``json_schema`` se intenta parsear como JSON
      y, si no es un objeto JSON, ``parsed`` queda en ``None`` (simula un JSON roto).
    * ``Exception``: se lanza en esa llamada (simula timeouts o caídas del proveedor).

    Cuando se agotan las respuestas programadas se lanza :class:`LLMError` para que una
    prueba nunca quede esperando en silencio una respuesta que nadie programó.
    """

    name = "fake"

    def __init__(self, responses: list[str | dict[str, Any] | Exception] | None = None) -> None:
        self.responses: list[str | dict[str, Any] | Exception] = list(responses or [])
        self.calls: list[dict[str, Any]] = []

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
        """Devuelve la siguiente respuesta programada (o lanza la excepción programada)."""
        self.calls.append(
            {
                "system": system,
                "user": user,
                "model": model,
                "json_schema": json_schema,
                "max_tokens": max_tokens,
                "effort": effort,
                "timeout_s": timeout_s,
            }
        )
        if not self.responses:
            raise LLMError("FakeProvider sin respuestas programadas", cause="unknown")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        parsed: dict[str, Any] | None
        if isinstance(response, dict):
            text, parsed = json.dumps(response, ensure_ascii=False), response
        else:
            text = response
            parsed = _parse_object(response) if json_schema is not None else None
        return LLMResult(
            text=text,
            parsed=parsed,
            model=model or "fake-model",
            provider=self.name,
            input_tokens=max(1, len(system + user) // 4),
            output_tokens=max(1, len(text) // 4),
            latency_ms=1.0,
            api_latency_ms=1.0,
        )


def _parse_object(text: str) -> dict[str, Any] | None:
    """Parsea ``text`` como objeto JSON; devuelve ``None`` si no lo es."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
