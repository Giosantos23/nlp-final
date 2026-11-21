"""Herramienta ``create_ticket(TicketDraft)``: borrador pendiente + token de confirmación.

Flujo con humano en el ciclo (HITL):

1. **Proponer** (``propose``): el agente arma un :class:`~docuerp.schemas.TicketDraft`, se
   valida la identidad y la cuenta, y se guarda el borrador **en memoria**. Se emite un
   token firmado con HMAC-SHA256. **No se escribe nada en la BD.**
2. **Confirmar** (``commit``): el usuario devuelve el token. Se verifica firma, vigencia
   (10 min), uso único y que el token sea del mismo usuario, rol y sesión. Solo entonces
   se abre una conexión de escritura y se inserta el ticket.

Garantías:

* Sin token válido no hay escritura (la escritura solo existe en ``commit``).
* Token expirado, reutilizado, alterado o de otro usuario/sesión → rechazo.
* Idempotencia: cada borrador tiene una clave (hash del usuario, la sesión y el
  contenido) que se guarda en la columna ``UNIQUE idempotency_key``. Proponer dos veces el
  mismo borrador devuelve el mismo token y confirmar algo ya escrito devuelve el ticket
  existente en lugar de duplicarlo.
* ``portal_user`` puede pedirlo, pero se escala: el borrador y el ticket quedan con
  ``needs_human_review``.
* El límite de 3 solicitudes por sesión lo aplica el registro (``max_calls_per_session``).

El secreto HMAC sale de ``DOCUERP_HMAC_SECRET``; si no está definido se genera uno
**efímero** por proceso (los tokens dejan de valer al reiniciar, lo cual es seguro).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field

from docuerp.agent.tools.base import ToolContext, ToolResult, ToolSpec, session_key_for, tool_roles
from docuerp.agent.tools.ticket_status import TicketReader, connect_readonly
from docuerp.schemas import ErpVersion, PendingAction, TicketDraft, TicketPriority, UserRole

if TYPE_CHECKING:
    from docuerp.config import Settings

log = logging.getLogger(__name__)

Clock = Callable[[], datetime]

TOKEN_VERSION = "v1"
"""Prefijo de versión del formato de token (permite rotar el formato a futuro)."""

ACCOUNT_ID_RE = re.compile(r"\bACC-\d{4}\b", re.IGNORECASE)
"""Identificador de cuenta que el usuario puede mencionar en la consulta."""

SLA_HOURS: dict[str, int] = {"critica": 4, "alta": 8, "media": 24, "baja": 72}
"""Plazo de SLA (horas) por prioridad para el ticket recién creado (simplificación)."""


class CreateTicketInput(TicketDraft):
    """Argumentos de ``create_ticket``: el borrador más una marca de revisión.

    ``flag_for_review`` la fija el agente (no el usuario) cuando la clasificación automática
    tuvo baja confianza, para que el ticket quede marcado para revisión humana.
    """

    flag_for_review: bool = Field(default=False)

    def to_draft(self) -> TicketDraft:
        """Devuelve el :class:`TicketDraft` sin la marca interna."""
        return TicketDraft.model_validate(self.model_dump(exclude={"flag_for_review"}))


class TokenError(ValueError):
    """Token de confirmación rechazado; el mensaje está en español y es apto para el usuario."""


@dataclass
class PendingRecord:
    """Borrador pendiente guardado en memoria hasta su confirmación o expiración."""

    jti: str
    token: str
    draft: TicketDraft
    role: UserRole
    user_id: str
    session_key: str
    erp_version: ErpVersion
    account_id: str
    idempotency_key: str
    expires_at: datetime
    needs_human_review: bool
    used: bool = False


def _b64encode(raw: bytes) -> str:
    """Base64 URL-safe sin relleno (apto para URLs y JSON)."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(text: str) -> bytes:
    """Inverso de :func:`_b64encode` (restaura el relleno)."""
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _resolve_secret(secret: bytes | str | None) -> bytes:
    """Secreto explícito, o ``DOCUERP_HMAC_SECRET``, o uno efímero aleatorio de 32 bytes."""
    if isinstance(secret, str):
        return secret.encode("utf-8")
    if secret is not None:
        return secret
    from docuerp.config import get_settings

    configured = get_settings().security.hmac_secret
    if configured is not None and configured.get_secret_value():
        return configured.get_secret_value().encode("utf-8")
    log.warning("DOCUERP_HMAC_SECRET no definido: se usa un secreto efímero por proceso")
    return secrets.token_bytes(32)


class TicketDraftService:
    """Propone borradores con token HMAC y escribe tickets solo tras la confirmación."""

    def __init__(
        self,
        db_path: Path | None = None,
        *,
        secret: bytes | str | None = None,
        ttl_min: int = 10,
        clock: Clock | None = None,
    ) -> None:
        self._db_path = db_path
        self._secret: bytes | None = _resolve_secret(secret) if secret is not None else None
        self.ttl_min = ttl_min
        self._clock = clock or (lambda: datetime.now(UTC))
        self._pending: dict[str, PendingRecord] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ utilidades
    @property
    def db_path(self) -> Path:
        """Ruta de la BD de tickets (``settings.paths.tickets_db`` si no se inyectó)."""
        if self._db_path is None:
            from docuerp.config import get_settings

            self._db_path = get_settings().paths.resolve("tickets_db")
        return self._db_path

    @property
    def secret(self) -> bytes:
        """Secreto HMAC (se resuelve en el primer uso)."""
        if self._secret is None:
            self._secret = _resolve_secret(None)
        return self._secret

    @property
    def reader(self) -> TicketReader:
        """Lector de solo lectura sobre la misma BD (identidad, cuentas, carteras)."""
        return TicketReader(self.db_path)

    def _sign(self, body: str) -> str:
        """Firma HMAC-SHA256 de ``body`` en base64 URL-safe."""
        digest = hmac.new(self.secret, body.encode("ascii"), hashlib.sha256).digest()
        return _b64encode(digest)

    # ------------------------------------------------------------------ tokens
    def issue_token(self, payload: dict[str, str], ttl_min: int | None = None) -> tuple[str, str]:
        """Firma ``payload`` (agrega ``exp``) y devuelve ``(token, expires_at_iso)``.

        Formato: ``v1.<payload base64>.<firma base64>``. El payload no lleva el borrador
        (vive en el servidor), solo identificadores, así el token es corto y no expone datos.
        """
        expires = self._clock() + timedelta(minutes=ttl_min or self.ttl_min)
        claims = {**payload, "exp": str(int(expires.timestamp()))}
        encoded = _b64encode(json.dumps(claims, sort_keys=True).encode("utf-8"))
        body = f"{TOKEN_VERSION}.{encoded}"
        return f"{body}.{self._sign(body)}", expires.isoformat()

    def verify_signature(self, token: str) -> dict[str, str]:
        """Verifica formato, firma y expiración (no el uso único).

        Raises:
            TokenError: token mal formado, alterado o expirado.
        """
        parts = token.split(".")
        if len(parts) != 3 or parts[0] != TOKEN_VERSION:
            raise TokenError("El token de confirmación no tiene un formato válido.")
        body = f"{parts[0]}.{parts[1]}"
        # compare_digest compara en tiempo constante (evita ataques de temporización).
        if not hmac.compare_digest(self._sign(body), parts[2]):
            raise TokenError("La firma del token de confirmación no es válida.")
        try:
            claims = json.loads(_b64decode(parts[1]))
            expires_ts = int(claims["exp"])
        except (ValueError, KeyError, TypeError) as exc:
            raise TokenError("El token de confirmación no tiene un formato válido.") from exc
        if self._clock().timestamp() >= expires_ts:
            raise TokenError("El token de confirmación expiró; solicita un nuevo borrador.")
        return {str(k): str(v) for k, v in claims.items()}

    def verify_token(self, token: str) -> dict[str, str]:
        """Verifica firma, expiración y uso único; devuelve el payload o lanza ``TokenError``."""
        claims = self.verify_signature(token)
        with self._lock:
            record = self._pending.get(claims.get("jti", ""))
            if record is None:
                raise TokenError("No hay un borrador pendiente asociado a este token.")
            if record.used:
                raise TokenError("Este token de confirmación ya fue utilizado.")
        return claims

    # ------------------------------------------------------------------ propuesta
    @staticmethod
    def idempotency_key(draft: TicketDraft, *, user_id: str, session_key: str) -> str:
        """Hash estable del usuario, la sesión y el contenido normalizado del borrador."""
        material = json.dumps(
            {
                "user_id": user_id,
                "session": session_key,
                "summary": draft.summary.casefold(),
                "description": draft.description.casefold(),
                "category": draft.category,
                "priority": draft.priority,
                "module": draft.module,
                "account_id": (draft.account_id or "").upper(),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _resolve_account(self, draft: TicketDraft, ctx: ToolContext) -> tuple[str | None, str]:
        """Determina la cuenta del ticket respetando el alcance del rol.

        Returns:
            ``(account_id, motivo)``; ``account_id`` es ``None`` si no se pudo resolver y
            ``motivo`` explica por qué (para responder al usuario).
        """
        reader = self.reader
        requested = draft.account_id.upper() if draft.account_id else None
        user = reader.get_user(ctx.user_id)
        own_account = user.account_id if user else None
        if requested is not None:
            if not reader.account_exists(requested):
                return None, "account_not_found"
            if ctx.role == "portal_user" and requested != own_account:
                return None, "account_out_of_scope"
            if ctx.role == "n2" and requested not in reader.portfolio(ctx.user_id):
                return None, "account_out_of_scope"
            return requested, "explicit"
        if ctx.role == "portal_user" and own_account:
            return own_account, "own_account"
        if ctx.role == "n2":
            accounts = reader.portfolio(ctx.user_id)
            if len(accounts) == 1:
                return accounts[0], "single_portfolio_account"
        return None, "account_required"

    def _existing_ticket(self, idempotency_key: str) -> str | None:
        """Ticket ya escrito con esta clave de idempotencia (lectura en modo ``ro``)."""
        with connect_readonly(self.db_path) as conn:
            row = conn.execute(
                "SELECT ticket_id FROM tickets WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
        return row["ticket_id"] if row else None

    def propose(
        self, draft: TicketDraft, ctx: ToolContext, *, flag_for_review: bool = False
    ) -> tuple[ToolResult, PendingAction | None]:
        """Crea el borrador pendiente y su token (no escribe en la BD)."""
        if not self.reader.verify_identity(ctx.user_id, ctx.role):
            return (
                ToolResult(
                    tool="create_ticket",
                    ok=True,
                    data={"created": False, "identity_verified": False},
                ),
                None,
            )
        account_id, reason = self._resolve_account(draft, ctx)
        if account_id is None:
            return (
                ToolResult(
                    tool="create_ticket",
                    ok=True,
                    data={"created": False, "identity_verified": True, "account_issue": reason},
                ),
                None,
            )
        resolved = draft.model_copy(update={"account_id": account_id})
        key = self.idempotency_key(resolved, user_id=ctx.user_id, session_key=ctx.session_key)
        existing = self._existing_ticket(key)
        if existing is not None:
            return (
                ToolResult(
                    tool="create_ticket",
                    ok=True,
                    data={"created": False, "duplicate_of": existing, "identity_verified": True},
                ),
                None,
            )
        review = flag_for_review or ctx.role == "portal_user"
        now = self._clock()
        with self._lock:
            reusable = next(
                (
                    rec
                    for rec in self._pending.values()
                    if rec.idempotency_key == key and not rec.used and rec.expires_at > now
                ),
                None,
            )
            if reusable is None:
                jti = secrets.token_hex(16)
                token, expires_iso = self.issue_token(
                    {
                        "jti": jti,
                        "uid": ctx.user_id,
                        "sid": ctx.session_key,
                        "role": ctx.role,
                        "act": "create_ticket",
                    }
                )
                reusable = PendingRecord(
                    jti=jti,
                    token=token,
                    draft=resolved,
                    role=ctx.role,
                    user_id=ctx.user_id,
                    session_key=ctx.session_key,
                    erp_version=ctx.erp_version,
                    account_id=account_id,
                    idempotency_key=key,
                    expires_at=datetime.fromisoformat(expires_iso),
                    needs_human_review=review,
                )
                self._pending[jti] = reusable
                reused = False
            else:
                reused = True
        pending = PendingAction(
            action="create_ticket",
            draft=reusable.draft,
            confirmation_token=reusable.token,
            expires_at=reusable.expires_at.isoformat(),
        )
        result = ToolResult(
            tool="create_ticket",
            ok=True,
            data={
                "created": False,
                "pending": True,
                "reused": reused,
                "identity_verified": True,
                "account_id": account_id,
                "category": resolved.category,
                "priority": resolved.priority,
                "module": resolved.module,
                "expires_at": pending.expires_at,
            },
            needs_human_review=review,
        )
        return result, pending

    # ------------------------------------------------------------------ confirmación
    def _claim(
        self, token: str, *, user_id: str, session_key: str, role: UserRole | None
    ) -> PendingRecord:
        """Valida el token y lo marca como usado de forma atómica (uso único)."""
        claims = self.verify_signature(token)
        with self._lock:
            record = self._pending.get(claims.get("jti", ""))
            if record is None:
                raise TokenError("No hay un borrador pendiente asociado a este token.")
            if record.used:
                raise TokenError("Este token de confirmación ya fue utilizado.")
            owner_ok = claims.get("uid") == user_id and claims.get("sid") == session_key
            if not owner_ok or (role is not None and claims.get("role") != role):
                raise TokenError("El token de confirmación pertenece a otro usuario o sesión.")
            record.used = True
            return record

    def _release(self, record: PendingRecord) -> None:
        """Devuelve el token a "no usado" si la escritura falló (permite reintentar)."""
        with self._lock:
            record.used = False

    def _write(self, record: PendingRecord) -> tuple[str, bool]:
        """Inserta el ticket (conexión de escritura aparte). Devuelve ``(ticket_id, duplicado)``."""
        now = self._clock()
        priority: TicketPriority = record.draft.priority
        due = now + timedelta(hours=SLA_HOURS[priority])
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        try:
            # BEGIN IMMEDIATE toma el candado de escritura antes de calcular el siguiente id,
            # así dos confirmaciones simultáneas no generan el mismo TCK-xxxxx.
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT ticket_id FROM tickets WHERE idempotency_key = ?",
                (record.idempotency_key,),
            ).fetchone()
            if row is not None:
                conn.rollback()
                return str(row[0]), True
            last = conn.execute("SELECT MAX(ticket_id) FROM tickets").fetchone()[0]
            next_number = int(str(last).split("-")[1]) + 1 if last else 1
            if next_number > 99999:
                raise sqlite3.IntegrityError("Se agotó el rango de identificadores TCK")
            ticket_id = f"TCK-{next_number:05d}"
            conn.execute(
                """
                INSERT INTO tickets (ticket_id, account_id, reported_by, assigned_to, summary,
                    description, category, priority, module, status, erp_version, created_at,
                    updated_at, sla_due_at, needs_human_review, idempotency_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'nuevo', ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    record.account_id,
                    record.user_id,
                    record.user_id if record.role == "n1" else None,
                    record.draft.summary,
                    record.draft.description,
                    record.draft.category,
                    priority,
                    record.draft.module,
                    record.erp_version,
                    now.isoformat(),
                    now.isoformat(),
                    due.isoformat(),
                    int(record.needs_human_review),
                    record.idempotency_key,
                ),
            )
            conn.commit()
            return ticket_id, False
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def commit(
        self,
        token: str,
        *,
        user_id: str,
        session_id: str | None,
        role: UserRole | None = None,
    ) -> ToolResult:
        """Escribe el ticket confirmado (idempotente) y devuelve su ``TCK-xxxxx``.

        Nunca lanza: un token rechazado devuelve ``ok=False`` con ``data["rejected"]``.
        """
        session_key = session_key_for(user_id, session_id)
        try:
            record = self._claim(token, user_id=user_id, session_key=session_key, role=role)
        except TokenError as exc:
            return ToolResult(
                tool="create_ticket", ok=False, error=str(exc), data={"rejected": True}
            )
        try:
            ticket_id, duplicate = self._write(record)
        except (sqlite3.Error, OSError) as exc:
            self._release(record)
            log.warning("No se pudo escribir el ticket: %s", type(exc).__name__)
            return ToolResult(
                tool="create_ticket",
                ok=False,
                error="No se pudo registrar el ticket; intenta confirmar de nuevo.",
                data={"unavailable": True, "error_type": type(exc).__name__},
            )
        return ToolResult(
            tool="create_ticket",
            ok=True,
            data={
                "created": not duplicate,
                "duplicate": duplicate,
                "ticket_id": ticket_id,
                "status": "nuevo",
                "account_id": record.account_id,
                "priority": record.draft.priority,
                "category": record.draft.category,
            },
            needs_human_review=record.needs_human_review,
        )


@lru_cache(maxsize=1)
def get_ticket_service() -> TicketDraftService:
    """Servicio por defecto del proceso (mismo almacén de borradores para proponer y
    confirmar). Usa ``settings.agent.confirmation_ttl_min``."""
    from docuerp.config import get_settings

    return TicketDraftService(ttl_min=get_settings().agent.confirmation_ttl_min)


async def propose_ticket(
    draft: TicketDraft,
    *,
    role: UserRole,
    user_id: str,
    session_id: str | None,
    erp_version: ErpVersion = "18.0",
    service: TicketDraftService | None = None,
) -> tuple[ToolResult, PendingAction | None]:
    """Crea el borrador pendiente y su token (no escribe en la BD)."""
    active = service or get_ticket_service()
    ctx = ToolContext(role=role, user_id=user_id, session_id=session_id, erp_version=erp_version)
    return await asyncio.to_thread(active.propose, draft, ctx)


def issue_token(payload: dict[str, str], ttl_min: int = 10) -> tuple[str, str]:
    """Firma ``payload`` y devuelve ``(token, expires_at_iso)``."""
    return get_ticket_service().issue_token(payload, ttl_min)


def verify_token(token: str) -> dict[str, str]:
    """Verifica firma, expiración y uso único; devuelve el payload o lanza ``ValueError``."""
    return get_ticket_service().verify_token(token)


async def commit_ticket(
    token: str,
    *,
    user_id: str,
    session_id: str | None,
    role: UserRole | None = None,
    service: TicketDraftService | None = None,
) -> ToolResult:
    """Escribe el ticket confirmado (idempotente) y devuelve su ``TCK-xxxxx``."""
    active = service or get_ticket_service()
    return await asyncio.to_thread(
        active.commit, token, user_id=user_id, session_id=session_id, role=role
    )


def build_create_ticket_spec(service: TicketDraftService, *, settings: Settings) -> ToolSpec:
    """Especificación registrable de ``create_ticket`` ligada a ``service``."""

    async def _handler(args: CreateTicketInput, ctx: ToolContext) -> ToolResult:
        result, pending = await asyncio.to_thread(
            service.propose, args.to_draft(), ctx, flag_for_review=args.flag_for_review
        )
        if pending is not None:
            # El PendingAction viaja en ``data`` para que el grafo lo publique en la respuesta.
            result.data["pending_action"] = pending.model_dump(mode="json")
        return result

    return ToolSpec(
        name="create_ticket",
        description="Propone un borrador de ticket; se escribe solo tras confirmación humana "
        "con token HMAC (10 min, un solo uso).",
        input_model=CreateTicketInput,
        allowed_roles=tool_roles("create_ticket"),
        writes=True,
        requires_confirmation=True,
        timeout_s=settings.agent.tool_timeout_s,
        max_calls_per_query=1,
        max_calls_per_session=settings.agent.create_ticket_per_session,
        handler=_handler,
    )


def extract_account_id(text: str) -> str | None:
    """Primer ``ACC-xxxx`` mencionado en ``text`` (en mayúsculas) o ``None``."""
    match = ACCOUNT_ID_RE.search(text)
    return match.group(0).upper() if match else None


__all__: list[str] = [
    "CreateTicketInput",
    "PendingRecord",
    "TicketDraftService",
    "TokenError",
    "build_create_ticket_spec",
    "commit_ticket",
    "extract_account_id",
    "get_ticket_service",
    "issue_token",
    "propose_ticket",
    "verify_token",
]
