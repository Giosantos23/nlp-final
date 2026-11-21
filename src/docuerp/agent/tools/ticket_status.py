"""Herramienta ``get_ticket_status`` (SQLite solo lectura ``file:...?mode=ro``, SQL
parametrizado, alcance por rol).

Alcance por rol (SPEC §3), traducido a una condición SQL fija por rol:

* ``portal_user`` → solo tickets que **reportó** (``reported_by``).
* ``n1`` → solo tickets **asignados** a él (``assigned_to``).
* ``n2`` → tickets de las cuentas de su **cartera** (tabla ``portfolios``).
* ``admin`` → todos.

Decisiones de seguridad:

* La conexión se abre en modo solo lectura (``mode=ro``): aunque hubiera un error de
  programación, SQLite rechaza cualquier escritura.
* El identificador del ticket y el usuario viajan como **parámetros** (``?``), nunca
  concatenados: no hay inyección SQL posible.
* "No existe" y "existe pero no es tuyo" producen la **misma** respuesta, para que nadie
  pueda enumerar tickets ajenos probando identificadores.
* La identidad simulada se verifica contra la tabla ``users``: el ``user_id`` debe existir y
  tener el rol declarado.
* Los textos libres (resumen y descripción) se enmascaran con el detector de PII antes de
  salir; si el enmascarador falla, se omiten (fallar cerrado), nunca se devuelven crudos.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from docuerp.agent.tools.base import ToolContext, ToolResult, ToolSpec, tool_roles
from docuerp.schemas import TICKET_ID_PATTERN, UserRole

if TYPE_CHECKING:
    from docuerp.config import Settings

log = logging.getLogger(__name__)

Masker = Callable[[str], tuple[str, list[str]]]
"""Firma del enmascarador de PII (``docuerp.security.pii.mask_pii``)."""

Clock = Callable[[], datetime]
"""Reloj inyectable (UTC) para pruebas deterministas del SLA."""

DESCRIPTION_MAX_CHARS = 300
"""La descripción se resume a este largo: la herramienta informa estado, no expedientes."""

CLOSED_STATUSES: frozenset[str] = frozenset({"resuelto", "cerrado"})
"""Estados en los que el SLA ya no corre."""

# Condición de alcance por rol. Son constantes del código (no entrada de usuario), por lo
# que interpolarlas en el SQL es seguro; los valores variables van como parámetros ``?``.
_SCOPE_SQL: dict[str, str] = {
    "admin": "1 = 1",
    "n2": "t.account_id IN (SELECT p.account_id FROM portfolios p WHERE p.user_id = ?)",
    "n1": "t.assigned_to = ?",
    "portal_user": "t.reported_by = ?",
}

_STATUS_SQL = """
SELECT t.ticket_id, t.account_id, t.status, t.priority, t.category, t.module,
       t.erp_version, t.created_at, t.updated_at, t.sla_due_at, t.needs_human_review,
       t.summary, t.description, t.assigned_to, a.name AS account_name, a.sla_tier
FROM tickets t
JOIN accounts a ON a.account_id = t.account_id
WHERE t.ticket_id = ? AND {scope}
"""


class TicketStatusInput(BaseModel):
    """Argumentos de ``get_ticket_status``."""

    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(pattern=TICKET_ID_PATTERN)

    @field_validator("ticket_id", mode="before")
    @classmethod
    def _upper(cls, value: Any) -> Any:
        """Acepta ``tck-00042`` y lo normaliza a ``TCK-00042``."""
        return value.strip().upper() if isinstance(value, str) else value


@dataclass(frozen=True)
class UserRecord:
    """Fila de la tabla ``users`` (sin datos personales: solo id, rol y cuenta)."""

    user_id: str
    role: str
    account_id: str | None


@contextmanager
def connect_readonly(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Abre la BD de tickets en modo **solo lectura** (``file:...?mode=ro``) y la cierra.

    Se usa como ``with connect_readonly(ruta) as conn:``. (El ``with`` nativo de
    ``sqlite3.Connection`` solo confirma transacciones; no cierra la conexión.)

    Raises:
        FileNotFoundError: si la BD no existe (SQLite en ``mode=ro`` no la crea).
    """
    if not db_path.is_file():
        raise FileNotFoundError(f"No existe la base de tickets en {db_path}")
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def default_masker(text: str) -> tuple[str, list[str]]:
    """Enmascarador real de la rama de seguridad (importación perezosa)."""
    from docuerp.security.pii import mask_pii

    return mask_pii(text)


def _parse_iso(value: str | None) -> datetime | None:
    """Convierte ISO-8601 a ``datetime`` UTC; ``None`` si falta o es inválido."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class TicketReader:
    """Lecturas de la BD de tickets con verificación de identidad y alcance por rol."""

    def __init__(
        self,
        db_path: Path | None = None,
        *,
        masker: Masker | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._db_path = db_path
        self._masker = masker or default_masker
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def db_path(self) -> Path:
        """Ruta de la BD (la de ``settings.paths.tickets_db`` si no se inyectó)."""
        if self._db_path is None:
            from docuerp.config import get_settings

            self._db_path = get_settings().paths.resolve("tickets_db")
        return self._db_path

    def get_user(self, user_id: str) -> UserRecord | None:
        """Busca el usuario simulado; ``None`` si no existe."""
        with connect_readonly(self.db_path) as conn:
            row = conn.execute(
                "SELECT user_id, role, account_id FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            return None
        return UserRecord(user_id=row["user_id"], role=row["role"], account_id=row["account_id"])

    def verify_identity(self, user_id: str, role: UserRole) -> bool:
        """``True`` si ``user_id`` existe en ``users`` con exactamente el rol ``role``."""
        user = self.get_user(user_id)
        return user is not None and user.role == role

    def account_exists(self, account_id: str) -> bool:
        """``True`` si la cuenta existe."""
        with connect_readonly(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM accounts WHERE account_id = ?", (account_id,)
            ).fetchone()
        return row is not None

    def portfolio(self, user_id: str) -> list[str]:
        """Cuentas de la cartera de un N2, ordenadas."""
        with connect_readonly(self.db_path) as conn:
            rows = conn.execute(
                "SELECT account_id FROM portfolios WHERE user_id = ? ORDER BY account_id",
                (user_id,),
            ).fetchall()
        return [row["account_id"] for row in rows]

    def _mask(self, text: str) -> str | None:
        """Enmascara PII; si el enmascarador falla devuelve ``None`` (se omite el texto)."""
        try:
            masked, _kinds = self._masker(text)
        except Exception:
            log.warning("Enmascarador de PII no disponible: se omite texto libre del ticket")
            return None
        return masked

    def read_status(self, ticket_id: str, ctx: ToolContext) -> dict[str, Any]:
        """Lee el ticket dentro del alcance del rol y devuelve datos ya sanitizados.

        Returns:
            ``{"ticket_id", "visible": False, ...}`` si no existe, está fuera de alcance o la
            identidad no se verifica; si es visible, sus campos de estado y SLA.
        """
        if not self.verify_identity(ctx.user_id, ctx.role):
            return {"ticket_id": ticket_id, "visible": False, "identity_verified": False}
        scope = _SCOPE_SQL[ctx.role]
        params: tuple[str, ...] = (ticket_id,) if ctx.role == "admin" else (ticket_id, ctx.user_id)
        with connect_readonly(self.db_path) as conn:
            row = conn.execute(_STATUS_SQL.format(scope=scope), params).fetchone()
        if row is None:
            return {"ticket_id": ticket_id, "visible": False, "identity_verified": True}
        return self._sanitize(row)

    def _sanitize(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convierte la fila en datos de salida: enmascara texto libre y calcula el SLA."""
        due = _parse_iso(row["sla_due_at"])
        breached: bool | None = None
        if due is not None:
            breached = row["status"] not in CLOSED_STATUSES and self._clock() > due
        description = self._mask(row["description"] or "")
        if description is not None and len(description) > DESCRIPTION_MAX_CHARS:
            description = description[: DESCRIPTION_MAX_CHARS - 1].rstrip() + "…"
        summary = self._mask(row["summary"])
        return {
            "ticket_id": row["ticket_id"],
            "visible": True,
            "identity_verified": True,
            "status": row["status"],
            "priority": row["priority"],
            "category": row["category"],
            "module": row["module"],
            "erp_version": row["erp_version"],
            "account_id": row["account_id"],
            "account_name": row["account_name"],
            "sla_tier": row["sla_tier"],
            "assigned_to": row["assigned_to"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "sla_due_at": row["sla_due_at"],
            "sla_breached": breached,
            "needs_human_review": bool(row["needs_human_review"]),
            "summary": summary,
            "description": description,
            "text_omitted": summary is None or description is None,
        }


async def get_ticket_status(
    args: TicketStatusInput,
    *,
    role: UserRole,
    user_id: str,
    reader: TicketReader | None = None,
) -> ToolResult:
    """Devuelve estado, prioridad, categoría y SLA del ticket si el rol lo alcanza.

    Un ticket inexistente o fuera de alcance **no** es un error de la herramienta: se
    devuelve ``ok=True`` con ``data["visible"] = False``. ``ok=False`` queda reservado a
    fallas reales (BD ausente o dañada), que el grafo trata como degradación.
    """
    active = reader or TicketReader()
    ctx = ToolContext(role=role, user_id=user_id)
    try:
        # sqlite3 es bloqueante: se ejecuta en un hilo para no congelar el event loop.
        data = await asyncio.to_thread(active.read_status, args.ticket_id, ctx)
    except (FileNotFoundError, sqlite3.Error) as exc:
        log.warning("BD de tickets no disponible: %s", type(exc).__name__)
        return ToolResult(
            tool="get_ticket_status",
            ok=False,
            error="La base de tickets no está disponible en este momento.",
            data={"unavailable": True, "error_type": type(exc).__name__},
        )
    return ToolResult(
        tool="get_ticket_status",
        ok=True,
        data=data,
        needs_human_review=bool(data.get("needs_human_review", False)),
    )


def build_ticket_status_spec(reader: TicketReader, *, settings: Settings) -> ToolSpec:
    """Especificación registrable de ``get_ticket_status`` ligada a ``reader``."""

    async def _handler(args: TicketStatusInput, ctx: ToolContext) -> ToolResult:
        return await get_ticket_status(args, role=ctx.role, user_id=ctx.user_id, reader=reader)

    return ToolSpec(
        name="get_ticket_status",
        description="Consulta el estado, prioridad y SLA de un ticket TCK-xxxxx dentro del "
        "alcance del rol (solo lectura).",
        input_model=TicketStatusInput,
        allowed_roles=tool_roles("get_ticket_status"),
        timeout_s=settings.agent.tool_timeout_s,
        max_calls_per_query=1,
        handler=_handler,
    )
