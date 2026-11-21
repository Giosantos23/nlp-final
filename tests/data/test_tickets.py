"""Pruebas de la BD de tickets sintéticos: esquema, usuarios demo, determinismo y datos."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from docuerp.data import tickets as tk
from docuerp.data.ticket_templates import TEMPLATES

# Columnas del contrato original (SPEC §9): no pueden renombrarse ni reordenarse.
CONTRACT_TICKET_COLUMNS = [
    "ticket_id",
    "account_id",
    "reported_by",
    "assigned_to",
    "summary",
    "description",
    "category",
    "priority",
    "module",
    "status",
    "erp_version",
    "created_at",
    "updated_at",
    "sla_due_at",
    "needs_human_review",
    "idempotency_key",
]
CATEGORIES = {
    "configuracion",
    "error_funcional",
    "acceso_permisos",
    "rendimiento",
    "migracion",
    "capacitacion",
    "integracion",
}
MODULES = {"inventario", "compras", "ventas", "facturacion", "accesos", "actualizacion", "general"}

EMAIL_RE = re.compile(r"[\w.]+@[\w.]+\.[a-z]{2,}")
PHONE_RE = re.compile(r"\+502 \d{4}-\d{4}")
DPI_RE = re.compile(r"\b\d{13}\b|\b\d{4} \d{5} \d{4}\b")


@pytest.fixture(scope="module")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """BD sembrada una sola vez con los valores por defecto (300 tickets, semilla 42)."""
    return tk.seed_tickets(tmp_path_factory.mktemp("tickets") / "tickets.db")


@pytest.fixture
def conn(db_path: Path):
    """Conexión de solo lectura, como la que usará el agente."""
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    yield connection
    connection.close()


def dump(path: Path) -> list[str]:
    """Volcado SQL completo de una BD (para comparar bases byte a byte en contenido)."""
    connection = sqlite3.connect(path)
    try:
        return list(connection.iterdump())
    finally:
        connection.close()


# --------------------------------------------------------------------------- esquema


def test_schema_tables_and_contract_columns(conn: sqlite3.Connection) -> None:
    tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"accounts", "users", "portfolios", "tickets"} <= tables  # contrato original
    assert {"ticket_events", "sla_policies"} <= tables  # agregadas por la rama de corpus
    columns = [r["name"] for r in conn.execute("PRAGMA table_info(tickets)")]
    assert columns[: len(CONTRACT_TICKET_COLUMNS)] == CONTRACT_TICKET_COLUMNS
    assert columns[len(CONTRACT_TICKET_COLUMNS) :] == [
        "resolved_at",
        "first_response_at",
        "channel",
    ]


def test_ddl_is_idempotent_and_keeps_contract_tables() -> None:
    connection = sqlite3.connect(":memory:")
    connection.executescript(tk.TICKETS_DDL)
    connection.executescript(tk.TICKETS_DDL)  # CREATE ... IF NOT EXISTS: no falla al repetir
    names = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"accounts", "users", "portfolios", "tickets"} <= names
    connection.close()


def test_agent_style_insert_with_only_contract_columns_works(conn: sqlite3.Connection) -> None:
    """``create_ticket`` solo conoce las columnas del contrato: las nuevas deben ser opcionales."""
    memory = sqlite3.connect(":memory:")
    memory.executescript(tk.TICKETS_DDL)
    memory.execute("INSERT INTO accounts VALUES ('ACC-0001', 'X', '18.0', 'basico')")
    memory.execute("INSERT INTO users VALUES ('USR-0001', 'U', 'portal_user', 'ACC-0001')")
    memory.execute(
        "INSERT INTO tickets (ticket_id, account_id, reported_by, summary, description, category, "
        "priority, module, status, erp_version, created_at, updated_at, needs_human_review, "
        "idempotency_key) VALUES ('TCK-90001', 'ACC-0001', 'USR-0001', 's', 'd', 'configuracion', "
        "'baja', 'general', 'nuevo', '18.0', 't', 't', 1, 'key-1')"
    )
    row = memory.execute("SELECT channel, resolved_at FROM tickets").fetchone()
    assert row == ("portal", None)
    memory.close()


def test_constraints_reject_invalid_rows() -> None:
    memory = sqlite3.connect(":memory:")
    memory.executescript(tk.TICKETS_DDL)
    with pytest.raises(sqlite3.IntegrityError):
        memory.execute(
            "INSERT INTO users VALUES ('USR-1', 'X', 'portal_user', NULL)"
        )  # id inválido
    with pytest.raises(sqlite3.IntegrityError):
        memory.execute("INSERT INTO users VALUES ('USR-0002', 'X', 'jefe', NULL)")  # rol inválido
    memory.close()


# --------------------------------------------------------------------------- contenido


def test_counts(conn: sqlite3.Connection) -> None:
    def count(sql: str) -> int:
        return int(conn.execute(sql).fetchone()[0])

    assert count("SELECT COUNT(*) FROM accounts") == 12
    assert count("SELECT COUNT(*) FROM tickets") == 300
    assert count("SELECT COUNT(*) FROM users WHERE role='n1'") == 10
    assert count("SELECT COUNT(*) FROM users WHERE role='n2'") == 5
    assert count("SELECT COUNT(*) FROM users WHERE role='admin'") == 2
    assert count("SELECT COUNT(*) FROM users WHERE role='portal_user'") >= 12
    assert count("SELECT COUNT(*) FROM sla_policies") == 4
    assert count("SELECT COUNT(*) FROM ticket_events") > 300


def test_demo_users_are_fixed(conn: sqlite3.Connection) -> None:
    rows = {r["user_id"]: r for r in conn.execute("SELECT * FROM users")}
    assert rows["USR-0001"]["role"] == "portal_user"
    assert rows["USR-0101"]["role"] == "n1"
    assert rows["USR-0201"]["role"] == "n2"
    assert rows["USR-0301"]["role"] == "admin"
    account = conn.execute(
        "SELECT * FROM accounts WHERE account_id = ?", (rows["USR-0001"]["account_id"],)
    ).fetchone()
    assert account["name"] == "Comercial Los Volcanes"
    assert dict(tk.DEMO_USERS) == {
        "portal_user": "USR-0001",
        "n1": "USR-0101",
        "n2": "USR-0201",
        "admin": "USR-0301",
    }


def test_user_ids_match_spec_pattern_and_only_portal_users_have_account(
    conn: sqlite3.Connection,
) -> None:
    for row in conn.execute("SELECT * FROM users"):
        assert re.fullmatch(r"USR-\d{4}", row["user_id"])
        assert (row["account_id"] is not None) == (row["role"] == "portal_user")


def test_n2_portfolios_cover_all_accounts_and_demo_n2_has_los_volcanes(
    conn: sqlite3.Connection,
) -> None:
    owned = [r["account_id"] for r in conn.execute("SELECT account_id FROM portfolios")]
    assert sorted(owned) == sorted(r["account_id"] for r in conn.execute("SELECT * FROM accounts"))
    demo = {
        r["account_id"]
        for r in conn.execute("SELECT account_id FROM portfolios WHERE user_id='USR-0201'")
    }
    assert "ACC-0001" in demo
    assert len(demo) >= 2


def test_foreign_keys_are_consistent(conn: sqlite3.Connection) -> None:
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_reporter_belongs_to_ticket_account_when_portal_user(conn: sqlite3.Connection) -> None:
    query = (
        "SELECT t.ticket_id FROM tickets t JOIN users u ON u.user_id = t.reported_by "
        "WHERE u.role = 'portal_user' AND u.account_id != t.account_id"
    )
    assert conn.execute(query).fetchall() == []


def test_assignees_are_staff_and_n2_assignments_respect_portfolios(
    conn: sqlite3.Connection,
) -> None:
    rows = conn.execute(
        "SELECT t.ticket_id, t.account_id, u.role, u.user_id FROM tickets t "
        "JOIN users u ON u.user_id = t.assigned_to"
    ).fetchall()
    assert rows
    for row in rows:
        assert row["role"] in {"n1", "n2"}
        if row["role"] == "n2":
            owner = conn.execute(
                "SELECT 1 FROM portfolios WHERE user_id = ? AND account_id = ?",
                (row["user_id"], row["account_id"]),
            ).fetchone()
            assert owner is not None, row["ticket_id"]


def test_enumerations_and_variety(conn: sqlite3.Connection) -> None:
    def distinct(column: str) -> set[str]:
        return {r[0] for r in conn.execute(f"SELECT DISTINCT {column} FROM tickets")}

    assert distinct("category") == CATEGORIES
    assert distinct("module") <= MODULES
    assert distinct("priority") == {"baja", "media", "alta", "critica"}
    assert distinct("status") == {"nuevo", "en_progreso", "en_espera", "resuelto", "cerrado"}
    assert distinct("channel") <= {"portal", "correo", "telefono", "chat"}
    for category in CATEGORIES:
        n = conn.execute("SELECT COUNT(*) FROM tickets WHERE category=?", (category,)).fetchone()[0]
        assert n >= 15, category  # cada clase tiene ejemplos suficientes para el clasificador


def test_basic_tier_accounts_only_use_portal_channel(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT t.channel, t.reported_by FROM tickets t JOIN accounts a USING (account_id) "
        "JOIN users u ON u.user_id = t.reported_by "
        "WHERE a.sla_tier = 'basico' AND u.role = 'portal_user'"
    ).fetchall()
    assert rows
    assert {r["channel"] for r in rows} == {"portal"}


def test_text_quality(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT ticket_id, summary, description FROM tickets").fetchall()
    for row in rows:
        assert 10 <= len(row["summary"]) <= 200, row["ticket_id"]
        assert 20 <= len(row["description"]) <= 2000, row["ticket_id"]
        assert "{" not in row["summary"] + row["description"]  # ningún marcador sin rellenar
    distinct_texts = {(r["summary"], r["description"]) for r in rows}
    assert len(distinct_texts) >= 270  # alta variedad, casi sin duplicados


def test_pii_is_present_in_some_descriptions(conn: sqlite3.Connection) -> None:
    descriptions = [r[0] for r in conn.execute("SELECT description FROM tickets")]
    assert sum(bool(EMAIL_RE.search(d)) for d in descriptions) >= 5
    assert sum(bool(PHONE_RE.search(d)) for d in descriptions) >= 5
    assert sum(bool(DPI_RE.search(d)) for d in descriptions) >= 3
    with_pii = sum(
        bool(EMAIL_RE.search(d) or PHONE_RE.search(d) or DPI_RE.search(d)) for d in descriptions
    )
    assert 20 <= with_pii <= 90  # una fracción minoritaria de los tickets


def test_sla_and_timestamps_are_consistent(conn: sqlite3.Connection) -> None:
    now = tk.REFERENCE_NOW.strftime("%Y-%m-%dT%H:%M:%S")
    for row in conn.execute("SELECT * FROM tickets"):
        assert row["created_at"] <= row["updated_at"] <= now, row["ticket_id"]
        assert row["sla_due_at"] > row["created_at"]
        assert (row["resolved_at"] is not None) == (row["status"] in ("resuelto", "cerrado"))
        if row["resolved_at"]:
            assert row["created_at"] < row["resolved_at"] <= row["updated_at"]
        if row["first_response_at"]:
            assert row["first_response_at"] > row["created_at"]


def test_sla_due_matches_policy(conn: sqlite3.Connection) -> None:
    policies = {r["priority"]: r for r in conn.execute("SELECT * FROM sla_policies")}
    assert (
        policies["critica"]["first_response_hours"],
        policies["critica"]["resolution_value"],
    ) == (1, 8)
    assert (policies["alta"]["first_response_hours"], policies["alta"]["resolution_value"]) == (
        4,
        24,
    )
    assert policies["media"]["resolution_unit"] == "business_days"
    assert (policies["media"]["first_response_hours"], policies["media"]["resolution_value"]) == (
        8,
        3,
    )
    assert (policies["baja"]["first_response_hours"], policies["baja"]["resolution_value"]) == (
        24,
        5,
    )
    for row in conn.execute("SELECT created_at, priority, sla_due_at FROM tickets"):
        created = datetime.fromisoformat(row["created_at"])
        assert tk.sla_due_at(created, row["priority"]).isoformat() == row["sla_due_at"]


def test_events_are_chronological_and_end_with_ticket_state(conn: sqlite3.Connection) -> None:
    for ticket in conn.execute("SELECT * FROM tickets"):
        events = conn.execute(
            "SELECT * FROM ticket_events WHERE ticket_id = ? ORDER BY event_id",
            (ticket["ticket_id"],),
        ).fetchall()
        assert events[0]["event_type"] == "creado"
        assert events[0]["created_at"] == ticket["created_at"]
        stamps = [e["created_at"] for e in events]
        assert stamps == sorted(stamps), ticket["ticket_id"]
        assert stamps[-1] == ticket["updated_at"]
        state_changes = [
            e["to_value"]
            for e in events
            if e["event_type"] in ("cambio_estado", "resuelto", "cerrado")
        ]
        expected_last = ticket["status"] if ticket["status"] != "nuevo" else None
        assert (state_changes[-1] if state_changes else None) == expected_last


# --------------------------------------------------------------------------- tickets ancla y RBAC


def test_anchor_tickets(conn: sqlite3.Connection) -> None:
    for number, anchor in tk.ANCHOR_TICKETS.items():
        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (f"TCK-{number:05d}",)
        ).fetchone()
        assert row["account_id"] == anchor["account_id"]
        assert row["reported_by"] == anchor["reported_by"]
        assert row["assigned_to"] == anchor["assigned_to"]
        assert row["status"] == anchor["status"]
        assert row["summary"] == anchor["summary"]
    ticket = conn.execute("SELECT * FROM tickets WHERE ticket_id = 'TCK-00042'").fetchone()
    assert ticket["priority"] == "alta"
    assert "ERR-INV-204" in ticket["summary"]


def test_rbac_scopes_for_demo_users(conn: sqlite3.Connection) -> None:
    """Las consultas de alcance de SPEC §3 devuelven lo esperado para los usuarios demo."""

    def visible(sql: str, *params: str) -> list[str]:
        return [r[0] for r in conn.execute(sql, params)]

    portal = visible("SELECT ticket_id FROM tickets WHERE reported_by = ?", "USR-0001")
    n1 = visible("SELECT ticket_id FROM tickets WHERE assigned_to = ?", "USR-0101")
    n2 = visible(
        "SELECT t.ticket_id FROM tickets t JOIN portfolios p ON p.account_id = t.account_id "
        "WHERE p.user_id = ?",
        "USR-0201",
    )
    admin = visible("SELECT ticket_id FROM tickets")
    assert "TCK-00042" in portal
    assert "TCK-00042" in n1
    assert "TCK-00042" in n2
    assert len(admin) == 300
    assert len(portal) >= 20
    assert len(n1) >= 20
    assert len(n2) >= 40
    # Aislamiento: tickets de otras cuentas/asignados a otros no son visibles.
    assert "TCK-00100" not in portal
    assert "TCK-00100" not in n1
    assert "TCK-00100" not in n2
    assert "TCK-00300" not in portal
    assert "TCK-00300" not in n2
    assert "TCK-00175" in n2
    assert "TCK-00175" not in n1
    assert "TCK-00175" not in portal


# --------------------------------------------------------------------------- determinismo


def test_seed_is_deterministic(tmp_path: Path) -> None:
    first = tk.seed_tickets(tmp_path / "a.db", n_tickets=60, seed=42)
    second = tk.seed_tickets(tmp_path / "b.db", n_tickets=60, seed=42)
    assert dump(first) == dump(second)


def test_different_seed_changes_content(tmp_path: Path) -> None:
    first = tk.seed_tickets(tmp_path / "a.db", n_tickets=60, seed=42)
    other = tk.seed_tickets(tmp_path / "b.db", n_tickets=60, seed=7)
    assert dump(first) != dump(other)
    # Las cuentas, usuarios demo y tickets ancla no dependen de la semilla.
    connection = sqlite3.connect(other)
    assert connection.execute("SELECT display_name FROM users WHERE user_id='USR-0001'").fetchone()[
        0
    ] == ("Brenda Cifuentes")
    assert (
        connection.execute("SELECT status FROM tickets WHERE ticket_id='TCK-00042'").fetchone()[0]
        == "en_progreso"
    )
    connection.close()


def test_reseeding_recreates_database(tmp_path: Path) -> None:
    target = tmp_path / "t.db"
    tk.seed_tickets(target, n_tickets=20)
    tk.seed_tickets(target, n_tickets=35)
    connection = sqlite3.connect(target)
    assert connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 35
    connection.close()


def test_invalid_ticket_count_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n_tickets"):
        tk.seed_tickets(tmp_path / "x.db", n_tickets=0)


def test_small_database_skips_anchors_that_do_not_fit(tmp_path: Path) -> None:
    target = tk.seed_tickets(tmp_path / "small.db", n_tickets=10)
    connection = sqlite3.connect(target)
    ids = [r[0] for r in connection.execute("SELECT ticket_id FROM tickets ORDER BY 1")]
    assert ids[0] == "TCK-00001"
    assert ids[-1] == "TCK-00010"
    assert (
        connection.execute("SELECT 1 FROM tickets WHERE ticket_id='TCK-00042'").fetchone() is None
    )
    assert (
        connection.execute("SELECT status FROM tickets WHERE ticket_id='TCK-00007'").fetchone()[0]
        == "resuelto"
    )
    connection.close()


# --------------------------------------------------------------------------- utilidades


def test_business_days_skip_weekends_and_holidays() -> None:
    friday = datetime(2026, 11, 13, 10, 0)  # viernes
    assert tk.add_business_days(friday, 1) == datetime(2026, 11, 16, 10, 0)  # lunes
    assert tk.add_business_days(friday, 3) == datetime(2026, 11, 18, 10, 0)
    before_may_day = datetime(2026, 4, 30, 9, 0)  # jueves; el 1 de mayo es feriado
    assert tk.add_business_days(before_may_day, 1) == datetime(2026, 5, 4, 9, 0)
    assert tk.sla_due_at(friday, "critica") == datetime(2026, 11, 13, 18, 0)
    assert tk.sla_due_at(friday, "alta") == datetime(2026, 11, 14, 10, 0)
    with pytest.raises(ValueError):
        tk.sla_due_at(friday, "urgente")


def test_templates_are_well_formed() -> None:
    slots = dict.fromkeys(
        [
            "producto", "cantidad", "faltante", "po", "recepcion", "entrega", "so", "factura",
            "proveedor", "monto", "usuario", "almacen", "ubicacion", "lote", "modulo_custom",
            "minutos", "afectados", "segundos", "fecha", "erp", "err",
        ],
        "X",
    )  # fmt: skip
    for template in TEMPLATES:
        assert template.category in CATEGORIES
        assert template.module in MODULES
        assert template.summaries
        assert template.descriptions
        for text in (*template.summaries, *template.descriptions):
            text.format(**slots)  # lanza KeyError si usa un marcador inexistente
        if not template.errors:
            assert not any(
                "{err}" in text for text in (*template.summaries, *template.descriptions)
            )
        assert template.erp_only in (None, "17.0", "18.0")
    migration_17 = [t for t in TEMPLATES if t.category == "migracion" and t.erp_only == "17.0"]
    migration_18 = [t for t in TEMPLATES if t.category == "migracion" and t.erp_only == "18.0"]
    assert migration_17
    assert migration_18


def test_write_tickets_readme(tmp_path: Path) -> None:
    readme = tk.write_tickets_readme(tmp_path / "README.md")
    text = readme.read_text(encoding="utf-8")
    for table in ("accounts", "users", "portfolios", "tickets", "ticket_events", "sla_policies"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in text
    assert "Comercial Los Volcanes" in text
    assert "USR-0201" in text
    assert "TCK-00042" in text


def test_summarize_db(db_path: Path) -> None:
    summary = tk.summarize_db(db_path)
    assert summary["tickets"] == 300
    assert summary["accounts"] == 12
    assert summary["sla_policies"] == 4
