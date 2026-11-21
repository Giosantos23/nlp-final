"""Proveedores LLM (SPEC §5.3). Propiedad de la rama ``feat/llm``."""

from __future__ import annotations

from docuerp.llm.contracts import KNOWN_MODELS, LLMError, LLMProvider, LLMResult, ProviderName
from docuerp.llm.factory import get_provider

__all__ = ["KNOWN_MODELS", "LLMError", "LLMProvider", "LLMResult", "ProviderName", "get_provider"]
