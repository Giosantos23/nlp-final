"""Contratos del proveedor LLM (SPEC §5.3).

Define el resultado normalizado de una llamada (:class:`LLMResult`), la interfaz común de
los proveedores (:class:`LLMProvider`) y el error tipado (:class:`LLMError`) con su causa.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from docuerp.schemas import Usage

ProviderName = Literal["agent_sdk", "anthropic_api", "offline", "fake"]
"""Proveedores disponibles. ``agent_sdk`` es el predeterminado en desarrollo."""

KNOWN_MODELS: tuple[str, ...] = ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5")
"""IDs exactos de modelo admitidos."""

LLMErrorCause = Literal[
    "timeout",  # la llamada superó ``timeout_s``
    "auth",  # falta o es inválida la credencial (token OAuth / ANTHROPIC_API_KEY)
    "invalid_model",  # el modelo no existe o la cuenta no tiene acceso
    "schema",  # se pidió JSON y la respuesta no llegó como JSON utilizable
    "unavailable",  # el servicio o el CLI no responde (red, proceso, sobrecarga)
    "unknown",  # cualquier otra falla del proveedor
]
"""Causas tipadas de :class:`LLMError`; el generador decide según la causa si reintenta
(``schema``) o degrada al proveedor ``offline`` (las demás)."""


class LLMResult(BaseModel):
    """Resultado normalizado de una llamada al LLM, con consumo, costo y latencias.

    Attributes:
        calls: número de llamadas reales al LLM que resume este resultado (1 por defecto;
            0 para proveedores sin LLM como ``offline``; mayor que 1 cuando el generador
            acumula un reintento).
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    parsed: dict[str, Any] | None = None
    model: str
    provider: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_creation_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    api_latency_ms: float | None = None
    calls: int = Field(default=1, ge=0)

    def to_usage(self) -> Usage:
        """Convierte el resultado en el consumo agregado ``Usage`` del contrato del agente."""
        return Usage(
            llm_calls=self.calls,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens,
            cost_usd=self.cost_usd,
        )


@runtime_checkable
class LLMProvider(Protocol):
    """Interfaz asíncrona común de los proveedores."""

    name: str

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
        """Genera una respuesta; si ``json_schema`` se indica, ``parsed`` trae el JSON."""
        ...


class LLMError(RuntimeError):
    """Error del proveedor con una causa tipada (``timeout``, ``auth``, ``invalid_model``,
    ``schema``, ``unavailable`` o ``unknown``).

    Args:
        message: descripción legible en español (nunca incluye secretos).
        cause: causa tipada para que quien llama decida cómo reaccionar.
    """

    def __init__(self, message: str, cause: LLMErrorCause = "unknown") -> None:
        super().__init__(message)
        self.cause: LLMErrorCause = cause


class LLMTimeout(LLMError):  # noqa: N818 - nombre fijado por el sondeo y el SPEC
    """Atajo tipado para ``LLMError(cause="timeout")``."""

    def __init__(self, message: str) -> None:
        super().__init__(message, cause="timeout")
