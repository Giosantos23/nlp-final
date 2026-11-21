"""Fábrica de proveedores LLM."""

from __future__ import annotations

from typing import get_args

from docuerp.config import get_settings
from docuerp.llm.contracts import LLMProvider, ProviderName

_CACHE: dict[tuple[str, str], LLMProvider] = {}
"""Proveedores ya construidos, por ``(nombre, modelo por defecto)``."""


def _build(name: ProviderName, default_model: str) -> LLMProvider:
    """Construye un proveedor nuevo. Las importaciones son perezosas: cargar ``anthropic`` o
    ``claude_agent_sdk`` solo cuesta si ese proveedor se usa."""
    if name == "agent_sdk":
        from docuerp.llm.agent_sdk import AgentSDKProvider

        llm = get_settings().llm
        return AgentSDKProvider(
            default_model=default_model,
            max_concurrency=llm.max_concurrency,
            use_client_pool=llm.use_client_pool or None,
        )
    if name == "anthropic_api":
        from docuerp.llm.anthropic_api import AnthropicAPIProvider

        return AnthropicAPIProvider(default_model=default_model)
    if name == "offline":
        from docuerp.llm.offline import OfflineProvider

        return OfflineProvider()
    if name == "fake":
        from docuerp.llm.fake import FakeProvider

        return FakeProvider()
    raise ValueError(f"Proveedor LLM desconocido: {name!r}")


def parse_provider_name(value: str) -> ProviderName:
    """Valida un texto (p. ej. de la CLI) como nombre de proveedor.

    Raises:
        ValueError: si no es ``agent_sdk``, ``anthropic_api``, ``offline`` o ``fake``.
    """
    valid = get_args(ProviderName)
    if value not in valid:
        raise ValueError(f"Proveedor LLM desconocido: {value!r}. Opciones: {', '.join(valid)}")
    return value  # type: ignore[return-value]  # estrechado por la comprobación anterior


def get_provider(name: ProviderName | None = None) -> LLMProvider:
    """Devuelve el proveedor ``name`` (por defecto ``settings.llm.provider``), cacheado.

    El proveedor ``fake`` también se cachea (comparte la lista de llamadas); en pruebas
    que necesiten uno limpio conviene instanciar ``FakeProvider`` directamente.

    Raises:
        ValueError: si ``name`` no es un proveedor conocido.
    """
    settings = get_settings().llm
    resolved: ProviderName = name or settings.provider
    key = (resolved, settings.model)
    provider = _CACHE.get(key)
    if provider is None:
        provider = _CACHE[key] = _build(resolved, settings.model)
    return provider


def clear_provider_cache() -> None:
    """Vacía la caché de proveedores (útil en pruebas tras cambiar el entorno)."""
    _CACHE.clear()
