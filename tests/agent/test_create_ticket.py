"""``create_ticket`` con humano en el ciclo: sin token no se escribe; token expirado,
reutilizado, alterado o ajeno se rechaza; idempotencia y escalamiento de portal_user."""

from __future__ import annotations

import sqlite3
import types
from pathlib import Path

import pytest

from docuerp.agent.tools.base import ToolContext
from docuerp.agent.tools.create_ticket import (
    TicketDraftService,
    TokenError,
    commit_ticket,
    extract_account_id,
    propose_ticket,
)
from docuerp.schemas import TicketDraft

DRAFT = TicketDraft(
    summary="No puedo validar la recepción parcial",
    description="Al validar aparece ERR-INV-204.",
    category="error_funcional",
    priority="alta",
    module="inventario",
    account_id="ACC-0001",
)
N1 = ToolContext(role="n1", user_id="USR-0101", session_id="sesion-0001", erp_version="18.0")


def _row(db: Path, ticket_id: str) -> sqlite3.Row:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
    finally:
        conn.close()


def test_propose_does_not_write(
    ticket_service: TicketDraftService, tickets_db: Path, kit: types.SimpleNamespace
) -> None:
    before = kit.count_tickets(tickets_db)
    result, pending = ticket_service.propose(DRAFT, N1)
    assert result.ok
    assert pending is not None
    assert pending.action == "create_ticket"
    assert pending.draft.account_id == "ACC-0001"
    assert len(pending.confirmation_token) <= 512
    assert kit.count_tickets(tickets_db) == before  # nada escrito sin confirmación


def test_commit_writes_once_and_rejects_reuse(
    ticket_service: TicketDraftService, tickets_db: Path, kit: types.SimpleNamespace
) -> None:
    _, pending = ticket_service.propose(DRAFT, N1)
    assert pending is not None
    token = pending.confirmation_token
    first = ticket_service.commit(token, user_id="USR-0101", session_id="sesion-0001", role="n1")
    assert first.ok
    assert first.data["created"] is True
    assert first.data["ticket_id"] == "TCK-00004"
    row = _row(tickets_db, "TCK-00004")
    assert row["status"] == "nuevo"
    assert row["reported_by"] == "USR-0101"
    assert row["assigned_to"] == "USR-0101"  # un N1 se autoasigna
    assert row["needs_human_review"] == 0
    assert row["idempotency_key"]
    again = ticket_service.commit(token, user_id="USR-0101", session_id="sesion-0001", role="n1")
    assert not again.ok
    assert again.data["rejected"] is True
    assert "ya fue utilizado" in (again.error or "")
    assert kit.count_tickets(tickets_db) == 4


def test_expired_token_rejected(
    ticket_service: TicketDraftService, clock: object, tickets_db: Path, kit: types.SimpleNamespace
) -> None:
    _, pending = ticket_service.propose(DRAFT, N1)
    assert pending is not None
    clock.advance(minutes=10, seconds=1)  # type: ignore[attr-defined]
    result = ticket_service.commit(
        pending.confirmation_token, user_id="USR-0101", session_id="sesion-0001"
    )
    assert not result.ok
    assert "expiró" in (result.error or "")
    assert kit.count_tickets(tickets_db) == 3


@pytest.mark.parametrize(
    ("user_id", "session_id", "role"),
    [
        ("USR-0102", "sesion-0001", "n1"),  # otro usuario
        ("USR-0101", "otra-sesion-99", "n1"),  # otra sesión
        ("USR-0101", None, "n1"),  # sin sesión
        ("USR-0101", "sesion-0001", "admin"),  # rol distinto
    ],
)
def test_token_of_someone_else_rejected(
    ticket_service: TicketDraftService,
    tickets_db: Path,
    kit: types.SimpleNamespace,
    user_id: str,
    session_id: str | None,
    role: str,
) -> None:
    _, pending = ticket_service.propose(DRAFT, N1)
    assert pending is not None
    result = ticket_service.commit(
        pending.confirmation_token,
        user_id=user_id,
        session_id=session_id,
        role=role,  # type: ignore[arg-type]
    )
    assert not result.ok
    assert result.data["rejected"] is True
    assert kit.count_tickets(tickets_db) == 3
    # El dueño legítimo todavía puede confirmar (el intento ajeno no consumió el token).
    owner = ticket_service.commit(
        pending.confirmation_token, user_id="USR-0101", session_id="sesion-0001", role="n1"
    )
    assert owner.ok


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t[:-2] + ("AA" if not t.endswith("AA") else "BB"),  # firma alterada
        lambda t: "v2" + t[2:],  # versión desconocida
        lambda t: t.split(".")[0],  # mal formado
        lambda t: "v1.eyJqdGkiOiAiYSJ9." + t.split(".")[2],  # payload cambiado
    ],
)
def test_tampered_token_rejected(ticket_service: TicketDraftService, mutate: object) -> None:
    _, pending = ticket_service.propose(DRAFT, N1)
    assert pending is not None
    bad = mutate(pending.confirmation_token)  # type: ignore[operator]
    with pytest.raises(TokenError):
        ticket_service.verify_token(bad)
    result = ticket_service.commit(bad, user_id="USR-0101", session_id="sesion-0001")
    assert not result.ok


def test_token_signed_with_other_secret_rejected(
    ticket_service: TicketDraftService, tickets_db: Path, clock: object
) -> None:
    other = TicketDraftService(tickets_db, secret="otro-secreto", clock=clock)  # type: ignore[arg-type]
    _, pending = other.propose(DRAFT, N1)
    assert pending is not None
    with pytest.raises(TokenError, match="firma"):
        ticket_service.verify_token(pending.confirmation_token)


def test_verify_token_checks_single_use(ticket_service: TicketDraftService) -> None:
    _, pending = ticket_service.propose(DRAFT, N1)
    assert pending is not None
    claims = ticket_service.verify_token(pending.confirmation_token)
    assert claims["uid"] == "USR-0101"
    assert claims["act"] == "create_ticket"
    ticket_service.commit(pending.confirmation_token, user_id="USR-0101", session_id="sesion-0001")
    with pytest.raises(TokenError, match="utilizado"):
        ticket_service.verify_token(pending.confirmation_token)
    # Un token bien firmado pero sin borrador en memoria (p. ej. tras reiniciar) se rechaza.
    orphan, _ = ticket_service.issue_token({"jti": "inexistente"})
    with pytest.raises(TokenError, match="borrador"):
        ticket_service.verify_token(orphan)


def test_idempotency_same_draft_same_token_and_no_duplicate(
    ticket_service: TicketDraftService, tickets_db: Path, kit: types.SimpleNamespace
) -> None:
    first, pending_a = ticket_service.propose(DRAFT, N1)
    second, pending_b = ticket_service.propose(DRAFT, N1)
    assert pending_a is not None
    assert pending_b is not None
    assert pending_a.confirmation_token == pending_b.confirmation_token
    assert second.data["reused"] is True
    assert first.data["reused"] is False
    ticket_service.commit(
        pending_a.confirmation_token, user_id="USR-0101", session_id="sesion-0001"
    )
    after, pending_c = ticket_service.propose(DRAFT, N1)
    assert pending_c is None
    assert after.data["duplicate_of"] == "TCK-00004"
    assert kit.count_tickets(tickets_db) == 4


def test_commit_is_idempotent_at_db_level(
    ticket_service: TicketDraftService, tickets_db: Path, clock: object, kit: types.SimpleNamespace
) -> None:
    _, pending = ticket_service.propose(DRAFT, N1)
    assert pending is not None
    # Simula dos borradores con la misma clave (p. ej. el primero expiró y se volvió a pedir).
    clock.advance(minutes=11)  # type: ignore[attr-defined]
    _, renewed = ticket_service.propose(DRAFT, N1)
    assert renewed is not None
    assert renewed.confirmation_token != pending.confirmation_token
    ok = ticket_service.commit(
        renewed.confirmation_token, user_id="USR-0101", session_id="sesion-0001"
    )
    assert ok.ok
    assert ok.data["created"]
    record = ticket_service._pending[
        ticket_service.verify_signature(renewed.confirmation_token)["jti"]
    ]
    record.used = False  # fuerza un segundo intento con la misma clave de idempotencia
    dup = ticket_service.commit(
        renewed.confirmation_token, user_id="USR-0101", session_id="sesion-0001"
    )
    assert dup.ok
    assert dup.data["duplicate"] is True
    assert dup.data["ticket_id"] == ok.data["ticket_id"]
    assert kit.count_tickets(tickets_db) == 4


def test_portal_user_is_escalated(ticket_service: TicketDraftService, tickets_db: Path) -> None:
    ctx = ToolContext(role="portal_user", user_id="USR-0001", session_id="portal-sesion-1")
    draft = DRAFT.model_copy(update={"account_id": None})
    result, pending = ticket_service.propose(draft, ctx)
    assert pending is not None
    assert result.needs_human_review
    assert pending.draft.account_id == "ACC-0001"  # su propia cuenta
    done = ticket_service.commit(
        pending.confirmation_token, user_id="USR-0001", session_id="portal-sesion-1"
    )
    assert done.ok
    assert done.needs_human_review
    row = _row(tickets_db, done.data["ticket_id"])
    assert row["needs_human_review"] == 1
    assert row["assigned_to"] is None


@pytest.mark.parametrize(
    ("ctx", "account", "issue"),
    [
        (ToolContext(role="portal_user", user_id="USR-0001"), "ACC-0002", "account_out_of_scope"),
        (ToolContext(role="n2", user_id="USR-0202"), "ACC-0001", "account_out_of_scope"),
        (ToolContext(role="n1", user_id="USR-0101"), "ACC-0099", "account_not_found"),
        (ToolContext(role="n1", user_id="USR-0101"), None, "account_required"),
        (ToolContext(role="n2", user_id="USR-0201"), None, "account_required"),  # 2 cuentas
    ],
)
def test_account_resolution_respects_scope(
    ticket_service: TicketDraftService, ctx: ToolContext, account: str | None, issue: str
) -> None:
    result, pending = ticket_service.propose(DRAFT.model_copy(update={"account_id": account}), ctx)
    assert pending is None
    assert result.data["account_issue"] == issue


def test_n2_single_portfolio_account_is_inferred(ticket_service: TicketDraftService) -> None:
    ctx = ToolContext(role="n2", user_id="USR-0202")
    _, pending = ticket_service.propose(DRAFT.model_copy(update={"account_id": None}), ctx)
    assert pending is not None
    assert pending.draft.account_id == "ACC-0003"


def test_unknown_identity_gets_no_token(ticket_service: TicketDraftService) -> None:
    result, pending = ticket_service.propose(DRAFT, ToolContext(role="admin", user_id="USR-0101"))
    assert pending is None
    assert result.data["identity_verified"] is False


def test_write_failure_releases_token(ticket_service: TicketDraftService, tickets_db: Path) -> None:
    _, pending = ticket_service.propose(DRAFT, N1)
    assert pending is not None
    conn = sqlite3.connect(tickets_db)
    conn.execute("DROP TABLE tickets")
    conn.commit()
    conn.close()
    failed = ticket_service.commit(
        pending.confirmation_token, user_id="USR-0101", session_id="sesion-0001"
    )
    assert not failed.ok
    assert failed.data["unavailable"] is True
    # El token no quedó consumido: se puede reintentar.
    assert ticket_service.verify_token(pending.confirmation_token)


async def test_module_level_wrappers(ticket_service: TicketDraftService) -> None:
    result, pending = await propose_ticket(
        DRAFT, role="n1", user_id="USR-0101", session_id="sesion-0001", service=ticket_service
    )
    assert result.ok
    assert pending is not None
    done = await commit_ticket(
        pending.confirmation_token,
        user_id="USR-0101",
        session_id="sesion-0001",
        service=ticket_service,
    )
    assert done.ok


def test_ephemeral_secret_when_not_configured(
    tickets_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from docuerp.config import get_settings

    monkeypatch.delenv("DOCUERP_HMAC_SECRET", raising=False)
    monkeypatch.setenv("DOCUERP_SECURITY__HMAC_SECRET", "")
    get_settings.cache_clear()
    try:
        a = TicketDraftService(tickets_db)
        b = TicketDraftService(tickets_db)
        assert len(a.secret) == 32
        assert a.secret != b.secret
    finally:
        get_settings.cache_clear()


def test_extract_account_id() -> None:
    assert extract_account_id("para la cuenta acc-0003 urgente") == "ACC-0003"
    assert extract_account_id("sin cuenta") is None
