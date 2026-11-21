"""RBAC simulado (``configs/roles.yaml``): lectura de chunks, uso de herramientas y
revalidación posterior a la recuperación.

Defensa en profundidad: la recuperación ya pre-filtra por rol, versión y vigencia
(SPEC §5.2), pero este módulo **vuelve a verificar** cada hit antes de que llegue al
generador y antes de aceptar una cita en el guard de salida. Si un error en el índice o
en un filtro dejara pasar un runbook ADM a un analista N1, aquí se detiene.

Además valida las llamadas a herramientas (rol, formato de argumentos, rangos, límites
por consulta y por sesión) y el contexto de un token de confirmación (dueño, sesión,
vigencia y un solo uso). La verificación criptográfica del token (HMAC) es del agente;
este módulo decide si, una vez decodificado, el token puede usarse por quien lo presenta.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from docuerp import paths
from docuerp.schemas import (
    QUERY_MAX_CHARS,
    ROLE_LEVEL,
    TICKET_ID_PATTERN,
    Chunk,
    ScoredChunk,
    UserRole,
)

log = logging.getLogger(__name__)

SUSPICIOUS_FLAG = "suspicious_instruction"
"""Bandera que el escáner de ingesta pone a chunks con instrucciones sospechosas."""

SEARCH_TOP_K_MAX = 8
"""Máximo ``top_k`` permitido a ``search_documents`` (SPEC §5.4)."""

CLASSIFY_TEXT_MAX = 2000
"""Máximo de caracteres para ``classify_ticket``."""

# ---------------------------------------------------------------------------
# Política (configs/roles.yaml)
# ---------------------------------------------------------------------------


class RolePolicy(BaseModel):
    """Política RBAC validada a partir de ``configs/roles.yaml``."""

    model_config = ConfigDict(extra="ignore")

    hierarchy: list[UserRole]
    tools: dict[str, list[UserRole]]
    ticket_scope: dict[UserRole, str] = Field(default_factory=dict)


_DEFAULT_POLICY: dict[str, Any] = {
    "hierarchy": ["portal_user", "n1", "n2", "admin"],
    "tools": {
        "search_documents": ["portal_user", "n1", "n2", "admin"],
        "get_ticket_status": ["portal_user", "n1", "n2", "admin"],
        "classify_ticket": ["n1", "n2", "admin"],
        "create_ticket": ["n1", "n2", "admin"],
        "create_ticket_escalated": ["portal_user"],
    },
    "ticket_scope": {
        "portal_user": "reported_by_self",
        "n1": "assigned_to_self",
        "n2": "portfolio_accounts",
        "admin": "all",
    },
}
"""Política equivalente a SPEC §3, usada solo si ``configs/roles.yaml`` no existe."""


def load_policy(path: Path | None = None) -> RolePolicy:
    """Lee y valida la política RBAC.

    Args:
        path: ruta del YAML; por defecto ``configs/roles.yaml``.

    Raises:
        ValueError: si el archivo existe pero su contenido no es una política válida
            (una política corrupta nunca debe degradar a "todo permitido").
    """
    source = path or paths.ROLES_FILE
    if not source.is_file():
        log.warning("No existe %s; se usa la política RBAC por defecto de SPEC §3", source)
        return RolePolicy.model_validate(_DEFAULT_POLICY)
    with source.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    try:
        return RolePolicy.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Política RBAC inválida en {source}: {exc}") from exc


@lru_cache(maxsize=1)
def get_policy() -> RolePolicy:
    """Política RBAC global (cacheada). Use ``get_policy.cache_clear()`` en pruebas."""
    return load_policy()


def _known_role(role: object) -> bool:
    """``True`` si ``role`` es uno de los cuatro roles válidos (entrada no confiable)."""
    return isinstance(role, str) and role in ROLE_LEVEL


def role_at_least(role: UserRole, minimum: UserRole) -> bool:
    """``True`` si ``role`` tiene un nivel jerárquico ≥ al de ``minimum`` (``ROLE_LEVEL``)."""
    if not (_known_role(role) and _known_role(minimum)):
        return False
    return ROLE_LEVEL[role] >= ROLE_LEVEL[minimum]


# ---------------------------------------------------------------------------
# Lectura de documentos
# ---------------------------------------------------------------------------


def can_read(role: UserRole, chunk: Chunk) -> bool:
    """``True`` si ``role`` está en ``chunk.allowed_roles``.

    Se usa pertenencia explícita (no la jerarquía) porque ``allowed_roles`` es la fuente de
    verdad que la ingesta copia del manifiesto; un rol desconocido nunca lee nada.
    """
    return _known_role(role) and role in chunk.allowed_roles


def filter_authorized(role: UserRole, hits: list[ScoredChunk]) -> tuple[list[ScoredChunk], int]:
    """Filtra hits no autorizados; devuelve ``(autorizados, descartados)``.

    Los hits autorizados se devuelven sin modificar (mismo ``rank`` y ``score``) para que
    la traza siga reflejando la posición original de recuperación.
    """
    authorized = [hit for hit in hits if can_read(role, hit.chunk)]
    return authorized, len(hits) - len(authorized)


class RevalidationResult(BaseModel):
    """Resultado de la revalidación completa de hits recuperados."""

    model_config = ConfigDict(extra="forbid")

    kept: list[ScoredChunk]
    dropped: dict[str, int] = Field(default_factory=dict)
    """Conteo de descartes por motivo: ``role``, ``version``, ``obsolete``, ``suspicious``."""
    dropped_ids: list[str] = Field(default_factory=list)

    @property
    def dropped_total(self) -> int:
        """Total de hits descartados por cualquier motivo."""
        return sum(self.dropped.values())


def revalidate_hits(
    role: UserRole,
    erp_version: str,
    hits: list[ScoredChunk],
    *,
    include_obsolete: bool = False,
    drop_suspicious: bool = True,
) -> RevalidationResult:
    """Revalida rol, versión, vigencia y banderas de cada hit (defensa en profundidad).

    Un hit se descarta (en este orden de precedencia) si el rol no lo puede leer, si su
    ``erp_version`` no es la pedida ni ``any``, si está obsoleto y no se pidieron obsoletos,
    o si la ingesta lo marcó con ``suspicious_instruction`` (posible inyección indirecta,
    p. ej. un runbook envenenado) y ``drop_suspicious`` está activo.
    """
    kept: list[ScoredChunk] = []
    dropped: dict[str, int] = {}
    dropped_ids: list[str] = []
    for hit in hits:
        chunk = hit.chunk
        reason: str | None = None
        if not can_read(role, chunk):
            reason = "role"
        elif chunk.erp_version not in (erp_version, "any"):
            reason = "version"
        elif chunk.status != "vigente" and not include_obsolete:
            reason = "obsolete"
        elif drop_suspicious and SUSPICIOUS_FLAG in chunk.flags:
            reason = "suspicious"
        if reason is None:
            kept.append(hit)
        else:
            dropped[reason] = dropped.get(reason, 0) + 1
            dropped_ids.append(chunk.chunk_id)
    return RevalidationResult(kept=kept, dropped=dropped, dropped_ids=dropped_ids)


# ---------------------------------------------------------------------------
# Herramientas
# ---------------------------------------------------------------------------


def can_use_tool(role: UserRole, tool: str) -> bool:
    """``True`` si el rol puede invocar ``tool`` según ``configs/roles.yaml``.

    Una herramienta no declarada en la política (p. ej. ``delete_ticket``) está prohibida
    para todos: la lista es de permitidos, nunca de prohibidos.
    """
    if not _known_role(role):
        return False
    return role in get_policy().tools.get(tool, [])


ToolDecisionKind = Literal["allow", "deny", "needs_confirmation", "escalate"]


class ToolDecision(BaseModel):
    """Decisión sobre una llamada a herramienta.

    * ``allow``: se puede ejecutar ya (herramientas de solo lectura).
    * ``needs_confirmation``: es escritura; solo se crea un borrador y se pide confirmación.
    * ``escalate``: el rol no puede escribir pero sí pedirlo (portal_user → borrador con
      ``needs_human_review``).
    * ``deny``: no se ejecuta; ``reason`` explica por qué en español.
    """

    model_config = ConfigDict(extra="forbid")

    decision: ToolDecisionKind
    tool: str
    reason: str
    violations: list[str] = Field(default_factory=list)


_TICKET_ID_RE = re.compile(TICKET_ID_PATTERN)
WRITE_TOOLS: frozenset[str] = frozenset({"create_ticket"})
"""Herramientas con efectos de escritura (requieren confirmación humana)."""


def _validate_tool_args(tool: str, args: dict[str, Any]) -> list[str]:
    """Valida formato y rango de los argumentos conocidos; devuelve violaciones."""
    violations: list[str] = []
    if tool == "get_ticket_status":
        ticket_id = args.get("ticket_id")
        if not isinstance(ticket_id, str) or not _TICKET_ID_RE.fullmatch(ticket_id):
            violations.append("ticket_id_invalido")
    elif tool == "search_documents":
        top_k = args.get("top_k", 5)
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            violations.append("top_k_no_entero")
        elif not 1 <= top_k <= SEARCH_TOP_K_MAX:
            violations.append("top_k_fuera_de_rango")
        query = args.get("query", "")
        if not isinstance(query, str) or not 3 <= len(query.strip()) <= QUERY_MAX_CHARS:
            violations.append("query_invalida")
    elif tool == "classify_ticket":
        text = args.get("text", "")
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= CLASSIFY_TEXT_MAX:
            violations.append("texto_fuera_de_rango")
    return violations


def authorize_tool_call(
    role: UserRole,
    tool: str,
    args: dict[str, Any] | None = None,
    *,
    calls_in_query: int = 0,
    max_calls_per_query: int = 3,
    tickets_created_in_session: int = 0,
    max_tickets_per_session: int = 3,
) -> ToolDecision:
    """Decide si una llamada a herramienta puede ejecutarse (revalidación de seguridad).

    Args:
        role: rol del usuario que origina la consulta.
        tool: nombre de la herramienta pedida por el agente/LLM.
        args: argumentos propuestos (entrada no confiable: puede venir del LLM).
        calls_in_query: llamadas a herramientas ya hechas en esta consulta.
        max_calls_per_query: límite de llamadas por consulta (SPEC: 3).
        tickets_created_in_session: borradores de ticket ya emitidos en la sesión.
        max_tickets_per_session: límite de tickets por sesión (SPEC: 3).
    """
    arguments = args or {}
    if tool not in get_policy().tools or tool == "create_ticket_escalated":
        return ToolDecision(
            decision="deny",
            tool=tool,
            reason=f"La herramienta '{tool}' no existe o no está expuesta al agente.",
            violations=["herramienta_desconocida"],
        )
    if calls_in_query >= max_calls_per_query:
        return ToolDecision(
            decision="deny",
            tool=tool,
            reason=f"Se alcanzó el límite de {max_calls_per_query} llamadas a herramientas.",
            violations=["limite_llamadas"],
        )
    violations = _validate_tool_args(tool, arguments)
    if violations:
        return ToolDecision(
            decision="deny",
            tool=tool,
            reason="Argumentos inválidos para la herramienta: " + ", ".join(violations) + ".",
            violations=violations,
        )
    if tool in WRITE_TOOLS:
        if tickets_created_in_session >= max_tickets_per_session:
            return ToolDecision(
                decision="deny",
                tool=tool,
                reason=f"Se alcanzó el límite de {max_tickets_per_session} tickets por sesión.",
                violations=["limite_sesion"],
            )
        if can_use_tool(role, tool):
            return ToolDecision(
                decision="needs_confirmation",
                tool=tool,
                reason="Acción de escritura: se crea un borrador que requiere confirmación.",
            )
        if can_use_tool(role, "create_ticket_escalated"):
            return ToolDecision(
                decision="escalate",
                tool=tool,
                reason="El rol no puede crear tickets: se genera un borrador para revisión.",
            )
        return ToolDecision(
            decision="deny",
            tool=tool,
            reason=f"El rol '{role}' no puede usar '{tool}'.",
            violations=["rol_no_autorizado"],
        )
    if not can_use_tool(role, tool):
        return ToolDecision(
            decision="deny",
            tool=tool,
            reason=f"El rol '{role}' no puede usar '{tool}'.",
            violations=["rol_no_autorizado"],
        )
    return ToolDecision(
        decision="allow", tool=tool, reason="Herramienta de solo lectura permitida."
    )


# ---------------------------------------------------------------------------
# Confirmación humana (HITL)
# ---------------------------------------------------------------------------


class ConfirmationClaim(BaseModel):
    """Contexto de un token de confirmación ya decodificado y con firma verificada."""

    model_config = ConfigDict(extra="forbid")

    owner_user_id: str
    session_id: str | None
    expires_at: datetime
    used: bool = False


class ConfirmationCheck(BaseModel):
    """Resultado de verificar si un token puede usarlo quien lo presenta."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    reason: str
    violations: list[str] = Field(default_factory=list)


def verify_confirmation(
    claim: ConfirmationClaim,
    *,
    user_id: str,
    session_id: str | None,
    now: datetime | None = None,
) -> ConfirmationCheck:
    """Verifica dueño, sesión, vigencia y un solo uso de un token de confirmación.

    Se acumulan todas las violaciones (no solo la primera) para que la auditoría muestre
    el cuadro completo de un intento de abuso.
    """
    current = now or datetime.now(UTC)
    expires = claim.expires_at if claim.expires_at.tzinfo else claim.expires_at.replace(tzinfo=UTC)
    violations: list[str] = []
    if claim.owner_user_id != user_id:
        violations.append("token_ajeno")
    if claim.session_id is not None and claim.session_id != session_id:
        violations.append("sesion_distinta")
    if claim.used:
        violations.append("token_reutilizado")
    if current >= expires:
        violations.append("token_expirado")
    if violations:
        return ConfirmationCheck(
            ok=False,
            reason="Confirmación rechazada: " + ", ".join(violations) + ".",
            violations=violations,
        )
    return ConfirmationCheck(ok=True, reason="Confirmación válida.")


__all__ = [
    "SEARCH_TOP_K_MAX",
    "SUSPICIOUS_FLAG",
    "WRITE_TOOLS",
    "ConfirmationCheck",
    "ConfirmationClaim",
    "RevalidationResult",
    "RolePolicy",
    "ToolDecision",
    "authorize_tool_call",
    "can_read",
    "can_use_tool",
    "filter_authorized",
    "get_policy",
    "load_policy",
    "revalidate_hits",
    "role_at_least",
    "verify_confirmation",
]
