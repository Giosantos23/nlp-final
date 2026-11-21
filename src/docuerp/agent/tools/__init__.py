"""Herramientas del agente: búsqueda, estado de ticket, clasificación y creación (HITL)."""

from __future__ import annotations

from docuerp.agent.tools.base import (
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    default_registry,
    tool_roles,
)

TOOL_NAMES: tuple[str, ...] = (
    "search_documents",
    "get_ticket_status",
    "classify_ticket",
    "create_ticket",
)
"""Nombres canónicos de herramienta (valores válidos de ``Answer.tool_used``)."""

__all__ = [
    "TOOL_NAMES",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "default_registry",
    "tool_roles",
]
