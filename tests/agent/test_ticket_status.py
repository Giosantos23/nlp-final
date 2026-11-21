"""``get_ticket_status``: solo lectura, SQL parametrizado, alcance por rol y PII enmascarada."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from docuerp.agent.tools.base import ToolContext
from docuerp.agent.tools.ticket_status import (
    TicketReader,
    TicketStatusInput,
    connect_readonly,
    get_ticket_status,
)
from docuerp.schemas import UserRole


async def _status(reader: TicketReader, ticket: str, role: UserRole, user: str) -> dict:
    result = await get_ticket_status(
        TicketStatusInput(ticket_id=ticket), role=role, user_id=user, reader=reader
    )
    assert result.ok
    return result.data


@pytest.mark.parametrize(
    ("role", "user", "visible"),
    [
        ("portal_user", "USR-0001", {"TCK-00001"}),  # solo lo que reportó
        ("portal_user", "USR-0002", {"TCK-00002"}),
        ("n1", "USR-0101", {"TCK-00001"}),  # solo lo asignado
        ("n1", "USR-0102", {"TCK-00002"}),
        ("n2", "USR-0201", {"TCK-00001", "TCK-00002"}),  # cartera ACC-0001/0002
        ("n2", "USR-0202", {"TCK-00003"}),
        ("admin", "USR-0301", {"TCK-00001", "TCK-00002", "TCK-00003"}),
    ],
)
async def test_scope_by_role(
    reader: TicketReader, role: UserRole, user: str, visible: set[str]
) -> None:
    seen = set()
    for ticket in ("TCK-00001", "TCK-00002", "TCK-00003"):
        if (await _status(reader, ticket, role, user))["visible"]:
            seen.add(ticket)
    assert seen == visible


async def test_visible_ticket_fields_and_sla(reader: TicketReader) -> None:
    data = await _status(reader, "TCK-00001", "n1", "USR-0101")
    assert data["status"] == "en_progreso"
    assert data["priority"] == "alta"
    assert data["sla_breached"] is True  # venció hace 2 horas y sigue abierto
    assert data["account_name"] == "Comercial Los Volcanes"
    resolved = await _status(reader, "TCK-00002", "admin", "USR-0301")
    assert resolved["sla_breached"] is False  # resuelto: el SLA ya no corre
    future = await _status(reader, "TCK-00003", "admin", "USR-0301")
    assert future["sla_breached"] is False


async def test_pii_is_masked_never_raw(reader: TicketReader) -> None:
    data = await _status(reader, "TCK-00001", "admin", "USR-0301")
    assert "juan.perez@volcanes.com" not in str(data)
    assert "[EMAIL]" in data["description"]
    assert "display_name" not in data
    assert "Ana Portal" not in str(data)


async def test_masker_failure_omits_free_text(tickets_db: Path) -> None:
    def broken(text: str) -> tuple[str, list[str]]:
        raise NotImplementedError

    reader = TicketReader(tickets_db, masker=broken)
    data = await _status(reader, "TCK-00001", "admin", "USR-0301")
    assert data["description"] is None
    assert data["summary"] is None
    assert data["text_omitted"] is True
    assert "volcanes.com" not in str(data)


async def test_missing_and_out_of_scope_are_indistinguishable(reader: TicketReader) -> None:
    missing = await _status(reader, "TCK-09999", "n1", "USR-0101")
    foreign = await _status(reader, "TCK-00002", "n1", "USR-0101")
    assert missing.keys() == foreign.keys()
    assert missing["visible"] is foreign["visible"] is False


async def test_identity_must_match_role(reader: TicketReader) -> None:
    # USR-0101 es n1: declararse admin no da acceso.
    spoof = await _status(reader, "TCK-00003", "admin", "USR-0101")
    assert spoof == {"ticket_id": "TCK-00003", "visible": False, "identity_verified": False}
    ghost = await _status(reader, "TCK-00001", "admin", "USR-9999")
    assert ghost["identity_verified"] is False


def test_input_contract() -> None:
    assert TicketStatusInput(ticket_id=" tck-00042 ").ticket_id == "TCK-00042"
    for bad in ("TCK-42", "TCK-000420", "TCK-00001' OR '1'='1", "00001"):
        with pytest.raises(ValidationError):
            TicketStatusInput(ticket_id=bad)


def test_connection_is_read_only(tickets_db: Path) -> None:
    with connect_readonly(tickets_db) as conn, pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM tickets")


async def test_missing_db_degrades(tmp_path: Path) -> None:
    reader = TicketReader(tmp_path / "no-existe.db")
    result = await get_ticket_status(
        TicketStatusInput(ticket_id="TCK-00001"), role="admin", user_id="USR-0301", reader=reader
    )
    assert not result.ok
    assert result.data["unavailable"] is True
    assert not (tmp_path / "no-existe.db").exists()  # mode=ro nunca crea la BD


def test_reader_helpers(reader: TicketReader) -> None:
    assert reader.portfolio("USR-0201") == ["ACC-0001", "ACC-0002"]
    assert reader.account_exists("ACC-0003")
    assert not reader.account_exists("ACC-0099")
    user = reader.get_user("USR-0001")
    assert user is not None
    assert user.account_id == "ACC-0001"
    assert reader.get_user("USR-0404") is None
    assert reader.read_status("TCK-00001", ToolContext(role="n1", user_id="USR-0101"))["visible"]
