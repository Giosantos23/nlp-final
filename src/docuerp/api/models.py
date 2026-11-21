"""Modelos específicos de la API (los contratos de consulta viven en ``docuerp.schemas``)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from docuerp.schemas import SESSION_ID_PATTERN, USER_ID_PATTERN, QueryResponse, UserRole


class ConfirmRequest(BaseModel):
    """Cuerpo de ``POST /api/actions/confirm``."""

    model_config = ConfigDict(extra="forbid")

    confirmation_token: str = Field(min_length=16, max_length=512)
    role: UserRole
    user_id: str = Field(pattern=USER_ID_PATTERN)
    session_id: str | None = Field(default=None, pattern=SESSION_ID_PATTERN)


class HealthResponse(BaseModel):
    """Respuesta de ``GET /api/health``."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"] = "ok"
    version: str
    mock: bool
    llm_provider: str
    models: dict[str, str]
    index_ready: bool
    tickets_db_ready: bool
    traces_cached: int
    warmup: dict[str, Any] | None = None
    """Precalentamiento de modelos e índices (``docuerp.api.warmup.Warmup.as_dict``)."""


class TraceLookup(BaseModel):
    """Respuesta de ``GET /api/trace/{trace_id}``.

    ``origin`` indica de dónde salió: ``memory`` (caché LRU, trae el ``QueryResponse``
    completo) o ``audit_log`` (registro sanitizado de ``logs/audit.jsonl``).
    """

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    origin: Literal["memory", "audit_log"]
    response: QueryResponse | None = None
    audit: dict[str, Any] | None = None


class DocumentList(BaseModel):
    """Respuesta de ``GET /api/documents``."""

    model_config = ConfigDict(extra="forbid")

    count: int
    documents: list[dict[str, Any]]
    stats: dict[str, Any] | None = None
    """Estadísticas del corpus visible (del reporte de ingesta), si existen."""


class UserList(BaseModel):
    """Respuesta de ``GET /api/users``."""

    model_config = ConfigDict(extra="forbid")

    users: list[dict[str, Any]]
