"""Contratos de datos compartidos de DocuERP (SPEC §4).

Este módulo es la **fuente de verdad** de los tipos que cruzan fronteras entre módulos
(ingesta, recuperación, generación, agente, seguridad, API y evaluación). Cualquier cambio
aquí rompe a varios equipos: se solicita en la sección "Solicitudes de cambio" del reporte
del agente y lo aplica únicamente el orquestador en ``main``.

Decisiones de diseño:

* Todos los modelos usan ``extra="forbid"``: un campo desconocido es un error, no un dato
  silenciosamente ignorado. Así el contrato de salida §11 (``Answer``) contiene
  *exactamente* los cinco campos exigidos por el enunciado.
* Las entradas de usuario (``QueryRequest.query``, ``TicketDraft``) se normalizan a Unicode
  NFC y sin espacios en los extremos **antes** de validar longitudes.
* Los patrones de identificadores (usuarios, tickets, sesiones, chunks) se exponen como
  constantes para que el resto del código no duplique expresiones regulares.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Tipos literales (vocabularios cerrados)
# ---------------------------------------------------------------------------

UserRole = Literal["portal_user", "n1", "n2", "admin"]
"""Rol simulado del usuario. Jerarquía documental: portal_user ⊂ n1 ⊂ n2 ⊂ admin."""

ErpVersion = Literal["17.0", "18.0"]
"""Versiones de Odoo cubiertas por el corpus."""

ChunkErpVersion = Literal["17.0", "18.0", "any"]
"""Versión aplicable a un chunk; ``any`` = documento interno independiente de versión."""

DocStatus = Literal["vigente", "obsoleto"]
"""Vigencia documental. Por defecto solo se recupera lo ``vigente``."""

DocType = Literal["odoo_doc", "runbook", "policy"]
"""Tipo de documento fuente."""

Language = Literal["en", "es"]
"""Idioma del texto del chunk."""

Intent = Literal["doc_query", "ticket_status", "ticket_create", "ticket_classify", "out_of_domain"]
"""Intención detectada por el enrutador del agente."""

RiskLevel = Literal["low", "medium", "high", "critical"]
"""Nivel de riesgo de un caso de evaluación o de una acción."""

TicketCategory = Literal[
    "configuracion",
    "error_funcional",
    "acceso_permisos",
    "rendimiento",
    "migracion",
    "capacitacion",
    "integracion",
]
"""Categorías de ticket que predice ``classify_ticket``."""

TicketPriority = Literal["baja", "media", "alta", "critica"]
"""Prioridad de un ticket."""

ErpModule = Literal[
    "inventario", "compras", "ventas", "facturacion", "accesos", "actualizacion", "general"
]
"""Módulo funcional del ERP al que pertenece un documento o ticket."""

RetrieverName = Literal["bm25", "dense", "hybrid", "rerank", "mmr"]
"""Etapa de recuperación que produjo un ``ScoredChunk``."""

TraceStatus = Literal["ok", "blocked", "error", "skipped", "retry"]
"""Resultado de una etapa del pipeline en la traza."""

# ---------------------------------------------------------------------------
# Constantes compartidas
# ---------------------------------------------------------------------------

ROLE_LEVEL: dict[str, int] = {"portal_user": 0, "n1": 1, "n2": 2, "admin": 3}
"""Nivel jerárquico de cada rol (mayor = más privilegios documentales)."""

USER_ID_PATTERN = r"^USR-\d{4}$"
"""Identificador de usuario simulado, p. ej. ``USR-0101``."""

TICKET_ID_PATTERN = r"^TCK-\d{5}$"
"""Identificador de ticket, p. ej. ``TCK-00042``."""

SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"
"""Identificador de sesión opaco generado por el cliente."""

CHUNK_ID_PATTERN = r"^.+::p\d{4}::c\d{3}$"
"""Forma de ``chunk_id``: ``{doc_id}::p{page:04d}::c{idx:03d}``."""

SHA256_PATTERN = r"^[0-9a-f]{64}$"
"""Hash sha256 en hexadecimal minúscula."""

QUERY_MIN_CHARS = 3
QUERY_MAX_CHARS = 2000

ABSTENTION_MESSAGE = (
    "No encontré evidencia suficiente en la documentación autorizada para responder con "
    "seguridad. Reformula la consulta o escálala a un analista N2."
)
"""Mensaje fijo de abstención (el guard de salida exige este texto cuando ``abstained``)."""

CANARY_TOKEN = "DOCUERP-CANARY-7f3a"
"""Canario anti-fuga incrustado en el prompt de sistema; si aparece en una salida, se bloquea."""


def normalize_text(value: str) -> str:
    """Normaliza texto de usuario: Unicode NFC y sin espacios en los extremos."""
    return unicodedata.normalize("NFC", value).strip()


def make_chunk_id(doc_id: str, page: int, idx: int) -> str:
    """Construye el ``chunk_id`` determinista ``{doc_id}::p{page:04d}::c{idx:03d}``.

    Args:
        doc_id: identificador del documento en el manifiesto.
        page: página inicial (1-indexed).
        idx: índice del chunk dentro de esa página (0-indexed).
    """
    if page < 1:
        raise ValueError("page debe ser ≥ 1 (1-indexed)")
    if not 0 <= idx <= 999:
        raise ValueError("idx debe estar en [0, 999]")
    return f"{doc_id}::p{page:04d}::c{idx:03d}"


class _Contract(BaseModel):
    """Base de todos los contratos: campos extra prohibidos y asignación validada."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------


class QueryRequest(_Contract):
    """Contrato de entrada de una consulta (API, CLI y evaluación).

    ``query`` se normaliza (NFC + strip) antes de validar su longitud (3..2000).
    ``confirmation_token`` solo se usa para confirmar un borrador pendiente (HITL).
    """

    query: str = Field(min_length=QUERY_MIN_CHARS, max_length=QUERY_MAX_CHARS)
    role: UserRole
    user_id: str = Field(pattern=USER_ID_PATTERN)
    erp_version: ErpVersion = "18.0"
    session_id: str | None = Field(default=None, pattern=SESSION_ID_PATTERN)
    confirmation_token: str | None = Field(default=None, min_length=16, max_length=512)

    @field_validator("query", mode="before")
    @classmethod
    def _normalize_query(cls, value: Any) -> Any:
        """Normaliza la consulta (NFC + strip) antes de aplicar las restricciones."""
        return normalize_text(value) if isinstance(value, str) else value


# ---------------------------------------------------------------------------
# Corpus y recuperación
# ---------------------------------------------------------------------------


class Chunk(_Contract):
    """Fragmento indexado del corpus con todos sus metadatos de gobernanza y cita."""

    chunk_id: str = Field(pattern=CHUNK_ID_PATTERN)
    doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    page: int = Field(ge=1)
    page_end: int = Field(ge=1)
    version: str = Field(min_length=1)
    erp_version: ChunkErpVersion
    status: DocStatus
    section_path: str
    module: ErpModule
    doc_type: DocType
    language: Language
    allowed_roles: list[UserRole] = Field(min_length=1)
    content_hash: str = Field(pattern=SHA256_PATTERN)
    ingested_at: str
    token_count: int = Field(ge=0)
    flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> Chunk:
        """Verifica ``page_end ≥ page`` y que ``chunk_id`` empiece por ``doc_id``."""
        if self.page_end < self.page:
            raise ValueError("page_end debe ser ≥ page")
        if not self.chunk_id.startswith(f"{self.doc_id}::"):
            raise ValueError("chunk_id debe empezar por '{doc_id}::'")
        return self


class ScoredChunk(_Contract):
    """Chunk con su puntaje y posición en una etapa de recuperación."""

    chunk: Chunk
    score: float
    rank: int = Field(ge=1)
    retriever: RetrieverName
    stage_scores: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Salida (§11 del enunciado)
# ---------------------------------------------------------------------------


class SourceRef(_Contract):
    """Cita verificable: archivo, página (1-indexed) y ``chunk_id``."""

    source: str = Field(min_length=1)
    page: int = Field(ge=1)
    chunk_id: str = Field(pattern=CHUNK_ID_PATTERN)


class Answer(_Contract):
    """CONTRATO DE SALIDA §11: exactamente ``answer``, ``sources``, ``abstained``,
    ``tool_used`` y ``needs_human_review``. No se agregan campos."""

    answer: str
    sources: list[SourceRef]
    abstained: bool
    tool_used: str | None
    needs_human_review: bool


# ---------------------------------------------------------------------------
# Tickets y acciones con confirmación humana
# ---------------------------------------------------------------------------


class TicketDraft(_Contract):
    """Borrador de ticket propuesto por ``create_ticket`` (se escribe solo al confirmar)."""

    summary: str = Field(min_length=10, max_length=200)
    description: str = Field(default="", max_length=2000)
    category: TicketCategory
    priority: TicketPriority
    module: ErpModule
    account_id: str | None = Field(default=None, max_length=64)

    @field_validator("summary", "description", mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        """Normaliza texto libre (NFC + strip)."""
        return normalize_text(value) if isinstance(value, str) else value


class PendingAction(_Contract):
    """Acción crítica en espera de confirmación humana (token HMAC de un solo uso)."""

    action: Literal["create_ticket"]
    draft: TicketDraft
    confirmation_token: str = Field(min_length=16, max_length=512)
    expires_at: str


# ---------------------------------------------------------------------------
# Telemetría
# ---------------------------------------------------------------------------


class Usage(_Contract):
    """Consumo agregado de LLM de una consulta (costo estimado a precio de lista)."""

    llm_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_creation_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)

    def __add__(self, other: Usage) -> Usage:
        """Suma dos consumos campo a campo (útil para acumular varias llamadas)."""
        return Usage(
            llm_calls=self.llm_calls + other.llm_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )

    @property
    def total_tokens(self) -> int:
        """Tokens de entrada + salida (sin contar caché)."""
        return self.input_tokens + self.output_tokens


class TraceEvent(_Contract):
    """Evento de una etapa del pipeline.

    ``stage`` ∈ {input_guard, router, retrieval.bm25, retrieval.dense, retrieval.fusion,
    rerank, mmr, evidence_gate, permission_check, tool.<name>, generation, output_guard,
    respond, error}. ``data`` va sanitizado (sin PII) y resumido.
    """

    stage: str = Field(min_length=1)
    status: TraceStatus
    duration_ms: float = Field(ge=0.0)
    data: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(_Contract):
    """Respuesta completa del agente/API: contrato §11 + traza, uso y acción pendiente."""

    trace_id: str = Field(min_length=1)
    intent: Intent
    response: Answer
    pending_action: PendingAction | None = None
    blocked: bool = False
    block_reason: str | None = None
    latency_ms: float = Field(ge=0.0)
    usage: Usage = Field(default_factory=Usage)
    events: list[TraceEvent] = Field(default_factory=list)


def is_ticket_id(value: str) -> bool:
    """Indica si ``value`` tiene la forma ``TCK-\\d{5}``."""
    return re.fullmatch(TICKET_ID_PATTERN, value) is not None


__all__ = [
    "ABSTENTION_MESSAGE",
    "CANARY_TOKEN",
    "CHUNK_ID_PATTERN",
    "QUERY_MAX_CHARS",
    "QUERY_MIN_CHARS",
    "ROLE_LEVEL",
    "SESSION_ID_PATTERN",
    "SHA256_PATTERN",
    "TICKET_ID_PATTERN",
    "USER_ID_PATTERN",
    "Answer",
    "Chunk",
    "ChunkErpVersion",
    "DocStatus",
    "DocType",
    "ErpModule",
    "ErpVersion",
    "Intent",
    "Language",
    "PendingAction",
    "QueryRequest",
    "QueryResponse",
    "RetrieverName",
    "RiskLevel",
    "ScoredChunk",
    "SourceRef",
    "TicketCategory",
    "TicketDraft",
    "TicketPriority",
    "TraceEvent",
    "TraceStatus",
    "Usage",
    "UserRole",
    "is_ticket_id",
    "make_chunk_id",
    "normalize_text",
]
