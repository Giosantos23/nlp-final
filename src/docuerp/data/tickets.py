"""Base SQLite de tickets sintéticos (~300) con usuarios, cuentas y carteras.

``TICKETS_DDL`` es el **contrato compartido** con ``agent/tools/ticket_status.py`` y
``create_ticket.py``: la rama de corpus lo implementa y la de agente lo consume. Respecto al
contrato original, esta rama solo **agregó** (sin renombrar nada): las columnas
``resolved_at``, ``first_response_at`` y ``channel`` al final de ``tickets`` y las tablas
``sla_policies`` y ``ticket_events``.

Todo es ficticio y determinista: con la misma semilla se obtiene exactamente la misma base
(incluidos ids, textos, fechas y PII sintética). El "ahora" de la simulación es fijo
(:data:`REFERENCE_NOW`) para que no dependa del reloj de la máquina.
"""

from __future__ import annotations

import random
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from faker import Faker

from docuerp import paths
from docuerp.data.ticket_templates import (
    CLIENT_NOTES,
    CLOSINGS,
    COMMENT_NOTES,
    OPENINGS,
    TEMPLATES,
    URGENCY_PHRASES,
    Template,
)

TICKETS_DDL = """
CREATE TABLE IF NOT EXISTS accounts (
    account_id   TEXT PRIMARY KEY,          -- ACC-0001
    name         TEXT NOT NULL,             -- "Comercial Los Volcanes"
    erp_version  TEXT NOT NULL CHECK (erp_version IN ('17.0','18.0')),
    sla_tier     TEXT NOT NULL              -- basico | estandar | premium
);
CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT PRIMARY KEY CHECK (user_id GLOB 'USR-[0-9][0-9][0-9][0-9]'),
    display_name TEXT NOT NULL,
    role         TEXT NOT NULL CHECK (role IN ('portal_user','n1','n2','admin')),
    account_id   TEXT REFERENCES accounts(account_id)  -- solo portal_user
);
CREATE TABLE IF NOT EXISTS portfolios (      -- cartera de cuentas de cada N2
    user_id      TEXT NOT NULL REFERENCES users(user_id),
    account_id   TEXT NOT NULL REFERENCES accounts(account_id),
    PRIMARY KEY (user_id, account_id)
);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id    TEXT PRIMARY KEY CHECK (ticket_id GLOB 'TCK-[0-9][0-9][0-9][0-9][0-9]'),
    account_id   TEXT NOT NULL REFERENCES accounts(account_id),
    reported_by  TEXT NOT NULL REFERENCES users(user_id),
    assigned_to  TEXT REFERENCES users(user_id),
    summary      TEXT NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    category     TEXT NOT NULL,
    priority     TEXT NOT NULL CHECK (priority IN ('baja','media','alta','critica')),
    module       TEXT NOT NULL,
    status       TEXT NOT NULL
                 CHECK (status IN ('nuevo','en_progreso','en_espera','resuelto','cerrado')),
    erp_version  TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    sla_due_at   TEXT,
    needs_human_review INTEGER NOT NULL DEFAULT 0,
    idempotency_key TEXT UNIQUE,
    -- Columnas agregadas por la rama de corpus (opcionales, siempre al final):
    resolved_at       TEXT,                                  -- NULL si sigue abierto
    first_response_at TEXT,                                  -- primera respuesta humana
    channel           TEXT NOT NULL DEFAULT 'portal'
                      CHECK (channel IN ('portal','correo','telefono','chat'))
);
CREATE INDEX IF NOT EXISTS idx_tickets_account ON tickets(account_id);
CREATE INDEX IF NOT EXISTS idx_tickets_assigned ON tickets(assigned_to);
CREATE INDEX IF NOT EXISTS idx_tickets_reported ON tickets(reported_by);
-- Tablas agregadas por la rama de corpus:
CREATE TABLE IF NOT EXISTS sla_policies (    -- POL-SLA-001 §3
    priority            TEXT PRIMARY KEY CHECK (priority IN ('baja','media','alta','critica')),
    first_response_hours INTEGER NOT NULL,
    resolution_unit     TEXT NOT NULL CHECK (resolution_unit IN ('hours','business_days')),
    resolution_value    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS ticket_events (   -- historial de cada ticket
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id   TEXT NOT NULL REFERENCES tickets(ticket_id),
    event_type  TEXT NOT NULL CHECK (event_type IN (
                    'creado','asignado','escalado','cambio_estado','cambio_prioridad',
                    'comentario','resuelto','cerrado')),
    actor_id    TEXT REFERENCES users(user_id),
    from_value  TEXT,
    to_value    TEXT,
    note        TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ticket ON ticket_events(ticket_id);
"""
"""Esquema de la BD de tickets (``data/tickets/tickets.db``)."""

REFERENCE_NOW = datetime(2026, 11, 16, 17, 0, 0)
"""'Ahora' fijo de la simulación (hora de Guatemala). Los tickets nacen antes de esta fecha."""

DEMO_USERS: dict[str, str] = {
    "portal_user": "USR-0001",
    "n1": "USR-0101",
    "n2": "USR-0201",
    "admin": "USR-0301",
}
"""Usuarios demo fijos por rol (SPEC §3)."""

# (id, nombre, versión de Odoo, nivel de SLA, dominio de correo ficticio)
_ACCOUNTS: tuple[tuple[str, str, str, str, str], ...] = (
    ("ACC-0001", "Comercial Los Volcanes", "18.0", "estandar", "comercialvolcanes.com.gt"),
    ("ACC-0002", "Distribuidora La Antigua", "17.0", "premium", "distlaantigua.com.gt"),
    ("ACC-0003", "Agroexportadora Pacaya", "18.0", "premium", "agropacaya.com.gt"),
    ("ACC-0004", "Ferretería El Quetzal", "17.0", "basico", "ferreteriaquetzal.com.gt"),
    ("ACC-0005", "Textiles Atitlán", "18.0", "estandar", "textilesatitlan.com.gt"),
    ("ACC-0006", "Café Tikal Exportaciones", "18.0", "premium", "cafetikal.com.gt"),
    ("ACC-0007", "Transportes Petén", "17.0", "estandar", "transportespeten.com.gt"),
    ("ACC-0008", "Industrias Chichicastenango", "17.0", "estandar", "industriaschichi.com.gt"),
    ("ACC-0009", "Constructora Los Cerros", "18.0", "basico", "constructoracerros.com.gt"),
    ("ACC-0010", "Muebles Xelajú", "17.0", "basico", "mueblesxelaju.com.gt"),
    ("ACC-0011", "Lácteos del Motagua", "18.0", "estandar", "lacteosmotagua.com.gt"),
    ("ACC-0012", "Farmacias Santa Lucía", "18.0", "premium", "farmaciassantalucia.com.gt"),
)
_ACCOUNT_WEIGHTS: tuple[float, ...] = (3.0, 1.6, 1.4, 1.0, 1.2, 1.4, 0.9, 1.0, 0.8, 0.7, 1.0, 1.3)

_PORTFOLIOS: dict[str, tuple[str, ...]] = {
    "USR-0201": ("ACC-0001", "ACC-0002", "ACC-0003"),
    "USR-0202": ("ACC-0004", "ACC-0005", "ACC-0006"),
    "USR-0203": ("ACC-0007", "ACC-0008"),
    "USR-0204": ("ACC-0009", "ACC-0010"),
    "USR-0205": ("ACC-0011", "ACC-0012"),
}

_FIRST_NAMES = (
    "Mynor", "Brenda", "Edwin", "Lucrecia", "Byron", "Ingrid", "Sandra", "Walter", "Karla",
    "Gerson", "Heidy", "Josué", "Mayra", "Otto", "Silvia", "Rolando", "Ana Lucía", "Wilfredo",
    "Claudia", "Estuardo", "Rosa", "Marvin", "Dulce", "Fredy", "Gladys", "Hugo", "Jessica",
    "Leonel", "Mónica", "Néstor", "Patricia", "Rudy", "Sonia", "Tomás", "Verónica", "Julio",
)  # fmt: skip
_LAST_NAMES = (
    "López", "Pérez", "García", "Hernández", "Morales", "Ajú", "Tzoc", "Xicay", "Chocooj",
    "Batz", "Cifuentes", "Barrios", "Orellana", "Barillas", "Estrada", "Gómez", "Ramírez",
    "Méndez", "Castillo", "Cabrera", "Monzón", "Sandoval", "Velásquez", "Coc", "Ixcot",
    "Tuy", "Álvarez", "Menchú", "Pop", "Quiñónez",
)  # fmt: skip
_DEMO_NAMES: dict[str, str] = {
    "USR-0001": "Brenda Cifuentes",
    "USR-0101": "Edwin Barrios",
    "USR-0201": "Karla Orellana",
    "USR-0301": "Walter Estrada",
}

_PRODUCTS = (
    "Papel bond carta (resma)", "Cemento UGC 4000 PSI (saco)", "Café pergamino (quintal)",
    "Varilla corrugada 3/8\"", "Tela popelina azul (yarda)", "Frijol negro (quintal)",
    "Pintura látex blanca (galón)", "Alimento balanceado para pollo (saco)",
    "Paracetamol 500 mg (caja x100)", "Silla ergonómica E-200", "Escritorio ejecutivo EJ-45",
    "Leche pasteurizada (litro)", "Queso fresco (libra)", "Tornillo hexagonal 1/2\" (caja)",
    "Hilo de algodón (cono)", "Cable THHN calibre 12 (rollo)", "Llanta 295/80 R22.5",
    "Aceite de motor 15W-40 (galón)", "Ibuprofeno 400 mg (caja x50)", "Azúcar (quintal)",
)  # fmt: skip
_SUPPLIERS = (
    "Distribuidora Quetzal", "Importadora Centroamericana", "Suministros del Norte",
    "Papelera Guatemalteca", "Agroinsumos Pacífico", "Ferreteros Unidos", "Logística Mixco",
    "Comercial Sacatepéquez", "Químicos del Istmo", "Empaques Escuintla",
)  # fmt: skip
_WAREHOUSES = (
    "WH/Stock",
    "Bodega Zona 12",
    "Bodega Mixco",
    "Bodega Escuintla",
    "Bodega Villa Nueva",
)
_CUSTOM_MODULES = ("x_comisiones_vendedor", "x_rutas_reparto", "x_fel_alterno", "x_cuotas_clientes")

_CATEGORY_WEIGHTS: dict[str, float] = {
    "error_funcional": 0.22,
    "acceso_permisos": 0.18,
    "configuracion": 0.17,
    "integracion": 0.15,
    "capacitacion": 0.11,
    "rendimiento": 0.09,
    "migracion": 0.08,
}
# Pesos de prioridad (baja, media, alta, critica) por categoría.
_PRIORITIES = ("baja", "media", "alta", "critica")
_PRIORITY_WEIGHTS: dict[str, tuple[float, float, float, float]] = {
    "error_funcional": (0.10, 0.40, 0.35, 0.15),
    "configuracion": (0.30, 0.50, 0.17, 0.03),
    "acceso_permisos": (0.15, 0.50, 0.28, 0.07),
    "rendimiento": (0.05, 0.30, 0.45, 0.20),
    "migracion": (0.05, 0.30, 0.45, 0.20),
    "capacitacion": (0.70, 0.27, 0.03, 0.00),
    "integracion": (0.10, 0.40, 0.38, 0.12),
}
# Tope de prioridad por categoría cuando la plantilla no define el suyo.
_CATEGORY_MAX_PRIORITY = {"configuracion": "alta", "capacitacion": "media"}
_STATUSES = ("nuevo", "en_progreso", "en_espera", "resuelto", "cerrado")
# Pesos de estado según la antigüedad del ticket (días): (hasta, pesos).
_STATUS_BY_AGE: tuple[tuple[int, tuple[float, ...]], ...] = (
    (7, (0.30, 0.40, 0.12, 0.18, 0.00)),
    (30, (0.05, 0.25, 0.10, 0.35, 0.25)),
    (10_000, (0.00, 0.09, 0.06, 0.30, 0.55)),
)
_CHANNELS_BY_TIER: dict[str, tuple[str, ...]] = {
    "basico": ("portal",),
    "estandar": ("portal", "portal", "portal", "correo"),
    "premium": ("portal", "portal", "correo", "correo", "telefono", "chat"),
}

SLA_POLICIES: tuple[tuple[str, int, str, int], ...] = (
    ("critica", 1, "hours", 8),
    ("alta", 4, "hours", 24),
    ("media", 8, "business_days", 3),
    ("baja", 24, "business_days", 5),
)
"""Política de SLA (POL-SLA-001 §3): prioridad, horas de 1.ª respuesta, unidad y valor."""

_HOLIDAYS_MM_DD = frozenset({"01-01", "05-01", "06-30", "09-15", "10-20", "11-01", "12-25"})


# ---------------------------------------------------------------------------
# Utilidades de tiempo y SLA
# ---------------------------------------------------------------------------


def _iso(moment: datetime) -> str:
    """Formato ISO-8601 sin zona (hora de Guatemala), a segundos."""
    return moment.strftime("%Y-%m-%dT%H:%M:%S")


def add_business_days(start: datetime, days: int) -> datetime:
    """Suma ``days`` días hábiles (lunes a viernes, sin feriados fijos) conservando la hora."""
    current = start
    remaining = days
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5 and current.strftime("%m-%d") not in _HOLIDAYS_MM_DD:
            remaining -= 1
    return current


def sla_due_at(created: datetime, priority: str) -> datetime:
    """Fecha límite de resolución según POL-SLA-001 (horas corridas o días hábiles)."""
    for name, _response, unit, value in SLA_POLICIES:
        if name == priority:
            if unit == "hours":
                return created + timedelta(hours=value)
            return add_business_days(created, value)
    raise ValueError(f"Prioridad desconocida: {priority!r}")


def _resolution_hours(priority: str) -> float:
    """Horas aproximadas de resolución comprometida (para simular tiempos reales)."""
    for name, _response, unit, value in SLA_POLICIES:
        if name == priority:
            return float(value if unit == "hours" else value * 24)
    raise ValueError(f"Prioridad desconocida: {priority!r}")


# ---------------------------------------------------------------------------
# Datos maestros: cuentas, usuarios, carteras
# ---------------------------------------------------------------------------


def _slug(text: str) -> str:
    """Minúsculas sin tildes ni espacios, apto para un correo (``Ana Lucía`` → ``analucia``)."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", plain.lower())


@dataclass(frozen=True)
class SeedUser:
    """Usuario de la simulación."""

    user_id: str
    display_name: str
    role: str
    account_id: str | None = None

    @property
    def email(self) -> str:
        """Correo ficticio: ``nombre.apellido@dominio-de-la-cuenta`` (o de Altamira)."""
        first, _, last = self.display_name.partition(" ")
        domain = "altamira-erp.com.gt"
        for acc_id, _name, _erp, _tier, acc_domain in _ACCOUNTS:
            if acc_id == self.account_id:
                domain = acc_domain
        return f"{_slug(first)}.{_slug(last.split(' ')[0] if last else '')}@{domain}"


def build_users(rng: random.Random) -> list[SeedUser]:
    """Crea los usuarios: portal por cuenta (+6 contactos extra), 10 N1, 5 N2 y 2 admin."""
    used: set[str] = set(_DEMO_NAMES.values())

    def fresh_name() -> str:
        while True:
            name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
            if name not in used:
                used.add(name)
                return name

    def name_for(user_id: str) -> str:
        return _DEMO_NAMES.get(user_id) or fresh_name()

    users: list[SeedUser] = []
    for index, (acc_id, *_rest) in enumerate(_ACCOUNTS, start=1):
        uid = f"USR-{index:04d}"
        users.append(SeedUser(uid, name_for(uid), "portal_user", acc_id))
    for index in range(1, 7):  # segundo contacto para las seis primeras cuentas
        uid = f"USR-{12 + index:04d}"
        users.append(SeedUser(uid, name_for(uid), "portal_user", _ACCOUNTS[index - 1][0]))
    for index in range(1, 11):
        uid = f"USR-{100 + index:04d}"
        users.append(SeedUser(uid, name_for(uid), "n1"))
    for index in range(1, 6):
        uid = f"USR-{200 + index:04d}"
        users.append(SeedUser(uid, name_for(uid), "n2"))
    for index in range(1, 3):
        uid = f"USR-{300 + index:04d}"
        users.append(SeedUser(uid, name_for(uid), "admin"))
    return users


# ---------------------------------------------------------------------------
# Generación de textos
# ---------------------------------------------------------------------------


def _gt_phone(rng: random.Random) -> str:
    """Teléfono ficticio con formato guatemalteco: ``+502 5512-3456``."""
    return f"+502 {rng.choice((3, 4, 5))}{rng.randint(0, 999):03d}-{rng.randint(0, 9999):04d}"


def _dpi(rng: random.Random) -> str:
    """DPI ficticio de 13 dígitos (8 de número + 2 de departamento + 2 de municipio)."""
    number = f"{rng.randint(1, 9)}{rng.randint(0, 10**8 - 1):08d}"
    digits = f"{number}{rng.randint(1, 22):02d}{rng.randint(1, 17):02d}"
    if rng.random() < 0.3:
        return f"{digits[:4]} {digits[4:9]} {digits[9:]}"
    return digits


def _nit(rng: random.Random) -> str:
    """NIT ficticio: 6-7 dígitos, guion y dígito verificador (0-9 o K)."""
    return f"{rng.randint(100000, 9999999)}-{rng.choice('0123456789K')}"


def _pii_sentence(rng: random.Random, reporter: SeedUser) -> str:
    """Frase de contacto con PII ficticia (correo, teléfono, DPI o NIT) para el enmascarado."""
    options = [
        f"Pueden escribirme a {reporter.email} para coordinar.",
        f"Mi teléfono directo es {_gt_phone(rng)}.",
        f"Contacto: {reporter.email}, tel. {_gt_phone(rng)}.",
        f"Para validar mi identidad, mi DPI es {_dpi(rng)}.",
        f"El NIT de la empresa es {_nit(rng)}; llámenme al {_gt_phone(rng)}.",
        f"Pueden llamarme al {_gt_phone(rng)} (DPI {_dpi(rng)}).",
    ]
    return rng.choice(options)


@dataclass
class _TicketSpec:
    """Ticket en construcción antes de insertarse en la base."""

    number: int
    account_id: str
    reported_by: str
    assigned_to: str | None
    summary: str
    description: str
    category: str
    priority: str
    module: str
    status: str
    erp_version: str
    created: datetime
    needs_human_review: bool
    channel: str
    contains_pii: bool = False
    events: list[tuple[str, str | None, str | None, str | None, str | None, datetime]] = field(
        default_factory=list
    )

    @property
    def ticket_id(self) -> str:
        return f"TCK-{self.number:05d}"


def _weighted(rng: random.Random, options: list[Any] | tuple[Any, ...], weights: Any) -> Any:
    """Elección ponderada determinista."""
    return rng.choices(list(options), weights=list(weights), k=1)[0]


def _make_slots(
    rng: random.Random, fake: Faker, template: Template, contact: str
) -> dict[str, str]:
    """Valores ficticios para los marcadores de una plantilla."""
    quantity = rng.randint(20, 500)
    shortfall = max(2, int(quantity * rng.uniform(0.04, 0.25)))
    return {
        "producto": rng.choice(_PRODUCTS),
        "cantidad": str(quantity),
        "faltante": str(shortfall),
        "po": f"P{rng.randint(100, 2400):05d}",
        "recepcion": f"WH/IN/{rng.randint(10, 900):05d}",
        "entrega": f"WH/OUT/{rng.randint(10, 900):05d}",
        "so": f"S{rng.randint(100, 3200):05d}",
        "factura": f"INV/2026/{rng.randint(100, 4000):05d}",
        "proveedor": rng.choice((*_SUPPLIERS, fake.company())),
        "monto": f"Q{rng.randint(300, 45000):,}.{rng.choice(('00', '50', '75', '25'))}",
        "usuario": contact,
        "almacen": rng.choice(_WAREHOUSES),
        "ubicacion": f"WH/Stock/{rng.choice('ABCD')}-{rng.randint(1, 12):02d}",
        "lote": fake.bothify("L-####-??").upper(),
        "modulo_custom": rng.choice(_CUSTOM_MODULES),
        "minutos": str(rng.randint(10, 60)),
        "afectados": str(rng.randint(3, 25)),
        "segundos": str(rng.randint(12, 90)),
        "fecha": f"{rng.randint(1, 28)}/{rng.randint(1, 11)}/2026",
        "erp": "",
        "err": rng.choice(template.errors) if template.errors else "ERR-GEN-000",
    }


def _compose_texts(
    rng: random.Random,
    template: Template,
    slots: dict[str, str],
    priority: str,
    reporter: SeedUser,
    with_pii: bool,
) -> tuple[str, str]:
    """Arma resumen y descripción: saludo + cuerpo + urgencia + (PII) + cierre."""
    # El resumen y la descripción se emparejan por índice para que hablen de lo mismo.
    index = rng.randrange(len(template.summaries))
    summary = template.summaries[index].format(**slots)
    body = template.descriptions[index % len(template.descriptions)].format(**slots)
    parts = [rng.choice(OPENINGS) + body]
    urgency = rng.choice(URGENCY_PHRASES[priority])
    if urgency:
        parts.append(urgency)
    if with_pii:
        parts.append(_pii_sentence(rng, reporter))
    description = " ".join(parts).strip() + rng.choice(CLOSINGS)
    return summary[:200], description[:2000]


# ---------------------------------------------------------------------------
# Historial (eventos) y tiempos
# ---------------------------------------------------------------------------


def _build_timeline(
    rng: random.Random,
    spec: _TicketSpec,
    n1_ids: list[str],
    admin_ids: list[str],
    n2_by_account: dict[str, str],
) -> tuple[datetime, datetime | None, datetime | None]:
    """Genera los eventos del ticket y devuelve ``(updated_at, resolved_at, first_response_at)``.

    Los tiempos respetan la antigüedad del ticket: nada ocurre después de ``REFERENCE_NOW``.
    """
    created = spec.created
    events = spec.events
    events.append(("creado", spec.reported_by, None, None, "Ticket creado", created))
    span_hours = max(1.0, (REFERENCE_NOW - created).total_seconds() / 3600)
    last = created
    first_response: datetime | None = None

    def clip(moment: datetime) -> datetime:
        """No permite eventos futuros respecto del 'ahora' de la simulación."""
        return min(moment, REFERENCE_NOW - timedelta(minutes=1))

    response_hours = {"critica": 1.2, "alta": 5.0, "media": 10.0, "baja": 30.0}[spec.priority]
    if spec.assigned_to is not None:
        assigned_at = clip(created + timedelta(minutes=rng.randint(5, 90)))
        assignee_is_n2 = spec.assigned_to in n2_by_account.values()
        if assignee_is_n2:
            n1_first = rng.choice(n1_ids)
            events.append(("asignado", rng.choice(admin_ids), None, n1_first, None, assigned_at))
            escalated_at = clip(assigned_at + timedelta(minutes=rng.randint(30, 240)))
            events.append(
                (
                    "escalado",
                    n1_first,
                    n1_first,
                    spec.assigned_to,
                    "Escalado a N2 por criterio de runbook",
                    escalated_at,
                )
            )
            last = escalated_at
        else:
            actor = rng.choice((*admin_ids, spec.assigned_to))
            events.append(("asignado", actor, None, spec.assigned_to, None, assigned_at))
            last = assigned_at
        first_response = clip(created + timedelta(hours=rng.uniform(0.15, 1.0) * response_hours))
        first_response = max(first_response, created + timedelta(minutes=3))
        events.append(
            (
                "comentario",
                spec.assigned_to,
                None,
                None,
                "Primera respuesta al cliente",
                first_response,
            )
        )
        last = max(last, first_response)

    if rng.random() < 0.08 and spec.priority != "baja":
        events.append(
            (
                "cambio_prioridad",
                rng.choice(n1_ids),
                "media" if spec.priority != "media" else "baja",
                spec.priority,
                "Prioridad ajustada tras evaluar impacto",
                clip(created + timedelta(minutes=rng.randint(10, 180))),
            )
        )

    resolved_at: datetime | None = None
    status = spec.status
    if status != "nuevo" and spec.assigned_to is None:
        raise ValueError(f"{spec.ticket_id}: un ticket en curso debe tener responsable")

    if status != "nuevo":
        start_work = clip(last + timedelta(minutes=rng.randint(10, 120)))
        events.append(("cambio_estado", spec.assigned_to, "nuevo", "en_progreso", None, start_work))
        last = start_work
        horizon = min(_resolution_hours(spec.priority) * rng.uniform(0.2, 1.3), span_hours * 0.85)
        if status == "en_espera":
            wait_at = clip(last + timedelta(hours=max(0.5, horizon * rng.uniform(0.2, 0.6))))
            events.append(
                (
                    "cambio_estado",
                    spec.assigned_to,
                    "en_progreso",
                    "en_espera",
                    "A la espera de información del cliente",
                    wait_at,
                )
            )
            last = wait_at
        elif status in ("resuelto", "cerrado"):
            if rng.random() < 0.25:  # algunos casos pasan por espera antes de resolverse
                wait_at = clip(last + timedelta(hours=max(0.5, horizon * 0.2)))
                back_at = clip(wait_at + timedelta(hours=max(0.5, horizon * 0.2)))
                events.append(
                    (
                        "cambio_estado",
                        spec.assigned_to,
                        "en_progreso",
                        "en_espera",
                        "Esperando respuesta del cliente",
                        wait_at,
                    )
                )
                events.append(
                    (
                        "cambio_estado",
                        spec.assigned_to,
                        "en_espera",
                        "en_progreso",
                        "Cliente respondió",
                        back_at,
                    )
                )
                last = back_at
            resolved_at = clip(last + timedelta(hours=max(0.5, horizon * rng.uniform(0.3, 0.8))))
            events.append(
                (
                    "resuelto",
                    spec.assigned_to,
                    "en_progreso",
                    "resuelto",
                    "Solución confirmada",
                    resolved_at,
                )
            )
            last = resolved_at
            if status == "cerrado":
                closed_at = clip(resolved_at + timedelta(hours=rng.uniform(4, 96)))
                events.append(
                    (
                        "cerrado",
                        spec.assigned_to,
                        "resuelto",
                        "cerrado",
                        "Cierre tras confirmación del cliente",
                        closed_at,
                    )
                )
                last = closed_at

    # Comentarios intermedios: siempre después de la primera respuesta y antes del último evento.
    if spec.assigned_to is not None and first_response is not None:
        for _ in range(rng.randint(0, 3)):
            moment = first_response + (last - first_response) * rng.uniform(0.2, 0.9)
            by_client = rng.random() < 0.4
            actor = spec.reported_by if by_client else spec.assigned_to
            note = rng.choice(CLIENT_NOTES if by_client else COMMENT_NOTES)
            events.append(("comentario", actor, None, None, note, moment.replace(microsecond=0)))

    events.sort(key=lambda event: event[5])
    updated = events[-1][5]
    return updated, resolved_at, first_response


# ---------------------------------------------------------------------------
# Tickets ancla: casos fijos para demos, pruebas y banco de evaluación
# ---------------------------------------------------------------------------

ANCHOR_TICKETS: dict[int, dict[str, Any]] = {
    7: {
        "account_id": "ACC-0001",
        "reported_by": "USR-0001",
        "assigned_to": "USR-0101",
        "status": "resuelto",
        "priority": "media",
        "category": "acceso_permisos",
        "module": "accesos",
        "created": datetime(2026, 9, 2, 10, 30),
        "channel": "portal",
        "summary": "Usuario del portal no puede restablecer su contraseña (ERR-ACC-140)",
        "description": (
            "Buen día. No puedo ingresar al portal de Comercial Los Volcanes. El enlace del "
            "correo de restablecimiento dice que ya no es válido. Pueden contactarme al correo "
            "brenda.cifuentes@comercialvolcanes.com.gt o al +502 5512-3456."
        ),
    },
    42: {
        "account_id": "ACC-0001",
        "reported_by": "USR-0001",
        "assigned_to": "USR-0101",
        "status": "en_progreso",
        "priority": "alta",
        "category": "error_funcional",
        "module": "inventario",
        "created": datetime(2026, 11, 10, 9, 15),
        "channel": "portal",
        "summary": "Recepción de la orden P00418 llegó con 20 cajas menos (ERR-INV-204)",
        "description": (
            "La orden de compra P00418 era de 200 cajas de papel bond a Q185.00 y llegaron 180. "
            "La recepción WH/IN/00045 muestra diferencia de cantidad (ERR-INV-204). El proveedor "
            "dice que enviará el saldo el viernes. No sabemos si validar con orden pendiente."
        ),
    },
    100: {
        "account_id": "ACC-0005",
        "reported_by": "USR-0005",
        "assigned_to": "USR-0102",
        "status": "en_espera",
        "priority": "media",
        "category": "configuracion",
        "module": "facturacion",
        "created": datetime(2026, 10, 20, 14, 0),
        "channel": "correo",
        "summary": "Nueva serie y secuencia de facturas para sucursal Quetzaltenango",
        "description": (
            "Abrimos una sucursal en Quetzaltenango y necesitamos una serie de facturación "
            "independiente. Para el trámite del certificador adjunto el DPI del representante "
            "legal: 2412345670101."
        ),
    },
    175: {
        "account_id": "ACC-0002",
        "reported_by": "USR-0002",
        "assigned_to": "USR-0103",
        "status": "en_progreso",
        "priority": "alta",
        "category": "rendimiento",
        "module": "general",
        "created": datetime(2026, 11, 12, 8, 40),
        "channel": "telefono",
        "summary": "Odoo tarda más de 40 segundos en abrir cualquier pantalla",
        "description": (
            "Desde las 8:00 el sistema responde con mucha lentitud. Somos 12 usuarios afectados "
            "y estamos atrasados con pedidos. Nos importa resolverlo hoy."
        ),
    },
    210: {
        "account_id": "ACC-0001",
        "reported_by": "USR-0001",
        "assigned_to": "USR-0101",
        "status": "nuevo",
        "priority": "critica",
        "category": "integracion",
        "module": "facturacion",
        "created": datetime(2026, 11, 16, 8, 5),
        "channel": "portal",
        "summary": "Facturas FEL rechazadas por el certificador",
        "description": (
            "Desde esta mañana el certificador rechaza todas las facturas. El mensaje indica "
            "NIT inválido del receptor. Es crítico: no podemos facturar a 9 clientes hoy. "
            "NIT de la empresa 4521897-K."
        ),
    },
    300: {
        "account_id": "ACC-0006",
        "reported_by": "USR-0006",
        "assigned_to": "USR-0202",
        "status": "en_progreso",
        "priority": "alta",
        "category": "rendimiento",
        "module": "inventario",
        "created": datetime(2026, 11, 13, 11, 20),
        "channel": "correo",
        "summary": "Reporte de valoración de inventario no termina de cargar (ERR-PERF-510)",
        "description": (
            "El reporte de valoración se queda cargando más de 20 minutos y el navegador marca "
            "que la página no responde. Lo necesitamos para el cierre del mes."
        ),
    },
}
"""Tickets fijos (por número) con propietario, estado y texto conocidos.

Sirven para demos y pruebas de RBAC: ``TCK-00042`` pertenece a USR-0001, está asignado a
USR-0101 y la cuenta está en la cartera de USR-0201; ``TCK-00100`` y ``TCK-00300`` son de
cuentas ajenas a USR-0001/USR-0201; ``TCK-00175`` está en la cartera de USR-0201 pero asignado
a otro N1.
"""


def _apply_anchor(spec: _TicketSpec, anchor: dict[str, Any]) -> None:
    """Sobrescribe un ticket generado con los valores fijos de su ancla."""
    for key, value in anchor.items():
        setattr(spec, key, value)
    spec.erp_version = next(erp for acc, _n, erp, _t, _d in _ACCOUNTS if acc == spec.account_id)
    spec.needs_human_review = False


# ---------------------------------------------------------------------------
# Siembra de la base
# ---------------------------------------------------------------------------


def _generate_specs(
    rng: random.Random, fake: Faker, users: list[SeedUser], n_tickets: int
) -> list[_TicketSpec]:
    """Genera los tickets (sin eventos) con categorías, prioridades y estados realistas."""
    by_role: dict[str, list[SeedUser]] = {}
    for user in users:
        by_role.setdefault(user.role, []).append(user)
    portal_by_account: dict[str, list[SeedUser]] = {}
    for user in by_role["portal_user"]:
        portal_by_account.setdefault(str(user.account_id), []).append(user)
    n1_ids = [u.user_id for u in by_role["n1"]]
    n1_weights = [2.2 if uid == DEMO_USERS["n1"] else 1.0 for uid in n1_ids]
    n2_by_account = {acc: n2 for n2, accounts in _PORTFOLIOS.items() for acc in accounts}
    categories = list(_CATEGORY_WEIGHTS)
    by_category: dict[str, list[Template]] = {c: [] for c in categories}
    for template in TEMPLATES:
        by_category[template.category].append(template)

    specs: list[_TicketSpec] = []
    for number in range(1, n_tickets + 1):
        account = _weighted(rng, _ACCOUNTS, _ACCOUNT_WEIGHTS)
        account_id, _name, erp_version, tier, _domain = account
        category = _weighted(rng, categories, [_CATEGORY_WEIGHTS[c] for c in categories])
        candidates = [t for t in by_category[category] if t.erp_only in (None, erp_version)]
        template = _weighted(rng, candidates, [t.weight for t in candidates])
        priority = _weighted(rng, _PRIORITIES, _PRIORITY_WEIGHTS[category])
        cap = template.max_priority or _CATEGORY_MAX_PRIORITY.get(category)
        if cap is not None and _PRIORITIES.index(priority) > _PRIORITIES.index(cap):
            priority = cap  # una solicitud de configuración o capacitación no es crítica

        reporter_is_staff = rng.random() < 0.10  # un N1 registra la llamada del cliente
        contacts = portal_by_account[account_id]
        contact = rng.choice(contacts)
        reporter = rng.choice(by_role["n1"]) if reporter_is_staff else contact
        colleague = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
        slots = _make_slots(rng, fake, template, colleague)
        with_pii = rng.random() < 0.14
        summary, description = _compose_texts(rng, template, slots, priority, reporter, with_pii)

        # u**2 concentra los tickets en fechas recientes (hay más abiertos que antiguos).
        age_days = 0.05 + 330 * rng.random() ** 2
        created = REFERENCE_NOW - timedelta(days=age_days)
        created = created.replace(microsecond=0)
        weights = next(w for limit, w in _STATUS_BY_AGE if age_days <= limit)
        status = _weighted(rng, _STATUSES, weights)

        assigned: str | None = None
        if not (status == "nuevo" and rng.random() < 0.5):
            high_stakes = priority in ("alta", "critica")
            if high_stakes and (category in ("migracion", "rendimiento") or rng.random() < 0.25):
                assigned = n2_by_account[account_id]
            else:
                assigned = _weighted(rng, n1_ids, n1_weights)
        channel = "telefono" if reporter_is_staff else rng.choice(_CHANNELS_BY_TIER[tier])
        specs.append(
            _TicketSpec(
                number=number,
                account_id=account_id,
                reported_by=reporter.user_id,
                assigned_to=assigned,
                summary=summary,
                description=description,
                category=category,
                priority=priority,
                module=template.module,
                status=status,
                erp_version=erp_version,
                created=created,
                needs_human_review=template.ambiguous or rng.random() < 0.03,
                channel=channel,
                contains_pii=with_pii,
            )
        )
    for number, anchor in ANCHOR_TICKETS.items():
        if number <= n_tickets:
            _apply_anchor(specs[number - 1], anchor)
    return specs


def seed_tickets(db_path: Path | None = None, *, n_tickets: int = 300, seed: int = 42) -> Path:
    """Crea la BD con usuarios demo fijos (USR-0001/0101/0201/0301) y tickets sintéticos.

    La base se **recrea desde cero**: si ``db_path`` existe, se borra primero. Con la misma
    ``seed`` y ``n_tickets`` el resultado es idéntico byte a byte en contenido (ids, textos,
    fechas, eventos).

    Args:
        db_path: destino (por defecto ``data/tickets/tickets.db``).
        n_tickets: número de tickets a generar (≥ 1; los tickets ancla aplican si caben).
        seed: semilla del generador pseudoaleatorio.

    Returns:
        Ruta de la base creada.

    Raises:
        ValueError: si ``n_tickets`` es menor que 1.
    """
    if n_tickets < 1:
        raise ValueError("n_tickets debe ser al menos 1")
    target = db_path or paths.TICKETS_DB
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()

    rng = random.Random(seed)
    fake = Faker("es_ES")
    fake.seed_instance(seed)
    users = build_users(rng)
    specs = _generate_specs(rng, fake, users, n_tickets)

    n1_ids = [u.user_id for u in users if u.role == "n1"]
    admin_ids = [u.user_id for u in users if u.role == "admin"]
    n2_by_account = {acc: n2 for n2, accounts in _PORTFOLIOS.items() for acc in accounts}

    connection = sqlite3.connect(target)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(TICKETS_DDL)
        with connection:
            connection.executemany(
                "INSERT INTO accounts (account_id, name, erp_version, sla_tier) "
                "VALUES (?, ?, ?, ?)",
                [(a, n, e, t) for a, n, e, t, _d in _ACCOUNTS],
            )
            connection.executemany(
                "INSERT INTO users (user_id, display_name, role, account_id) VALUES (?, ?, ?, ?)",
                [(u.user_id, u.display_name, u.role, u.account_id) for u in users],
            )
            connection.executemany(
                "INSERT INTO portfolios (user_id, account_id) VALUES (?, ?)",
                [(n2, acc) for n2, accounts in _PORTFOLIOS.items() for acc in accounts],
            )
            connection.executemany(
                "INSERT INTO sla_policies (priority, first_response_hours, resolution_unit, "
                "resolution_value) VALUES (?, ?, ?, ?)",
                SLA_POLICIES,
            )
            for spec in specs:
                updated, resolved, first_response = _build_timeline(
                    rng, spec, n1_ids, admin_ids, n2_by_account
                )
                connection.execute(
                    "INSERT INTO tickets (ticket_id, account_id, reported_by, assigned_to, "
                    "summary, "
                    "description, category, priority, module, status, erp_version, created_at, "
                    "updated_at, sla_due_at, needs_human_review, idempotency_key, resolved_at, "
                    "first_response_at, channel) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)",
                    (
                        spec.ticket_id,
                        spec.account_id,
                        spec.reported_by,
                        spec.assigned_to,
                        spec.summary,
                        spec.description,
                        spec.category,
                        spec.priority,
                        spec.module,
                        spec.status,
                        spec.erp_version,
                        _iso(spec.created),
                        _iso(updated),
                        _iso(sla_due_at(spec.created, spec.priority)),
                        int(spec.needs_human_review),
                        _iso(resolved) if resolved else None,
                        _iso(first_response) if first_response else None,
                        spec.channel,
                    ),
                )
                connection.executemany(
                    "INSERT INTO ticket_events (ticket_id, event_type, actor_id, from_value, "
                    "to_value, note, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        (spec.ticket_id, kind, actor, old, new, note, _iso(moment))
                        for kind, actor, old, new, note, moment in spec.events
                    ],
                )
    finally:
        connection.close()
    return target


def summarize_db(db_path: Path | None = None) -> dict[str, int]:
    """Cuenta los registros de cada tabla (en modo solo lectura)."""
    target = db_path or paths.TICKETS_DB
    connection = sqlite3.connect(f"file:{target}?mode=ro", uri=True)
    try:
        tables = ("accounts", "users", "portfolios", "tickets", "ticket_events", "sla_policies")
        return {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# README del esquema
# ---------------------------------------------------------------------------


def write_tickets_readme(readme_path: Path | None = None) -> Path:
    """Escribe ``data/tickets/README.md`` con el esquema, usuarios demo y consultas de ejemplo.

    El texto se arma a partir de las constantes del módulo (DDL, cuentas, anclas) para que
    la documentación no se desfase del código.
    """
    target = readme_path or (paths.TICKETS_DIR / "README.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    accounts = "\n".join(
        f"| {acc} | {name} | {erp} | {tier} |" for acc, name, erp, tier, _d in _ACCOUNTS
    )
    portfolios = "\n".join(
        f"| {n2} | {', '.join(accounts_)} |" for n2, accounts_ in _PORTFOLIOS.items()
    )
    anchors = "\n".join(
        f"| TCK-{number:05d} | {a['account_id']} | {a['reported_by']} | {a['assigned_to']} "
        f"| {a['status']} | {a['priority']} | {a['summary']} |"
        for number, a in ANCHOR_TICKETS.items()
    )
    sla = "\n".join(
        f"| {p} | {r} h | {v} {'horas' if u == 'hours' else 'días hábiles'} |"
        for p, r, u, v in SLA_POLICIES
    )
    text = f"""# Base de tickets sintéticos (`tickets.db`)

Base SQLite **ficticia y determinista** de Altamira Soluciones ERP. Se genera con:

```bash
uv run docuerp data seed-tickets          # 300 tickets, semilla 42
uv run docuerp data seed-tickets --n 500 --seed 7
```

`data/tickets/tickets.db` está en `.gitignore` (es reproducible); este README sí se versiona.
Todos los nombres, correos, teléfonos (+502), DPI (13 dígitos) y NIT son inventados; cerca del
14 % de las descripciones incluyen PII ficticia para probar el enmascarado.
El "ahora" de la simulación es fijo (`{_iso(REFERENCE_NOW)}`, hora de Guatemala).

## Esquema

La fuente de verdad es `TICKETS_DDL` en `src/docuerp/data/tickets.py`. Las lecturas del agente
usan `sqlite3.connect("file:...?mode=ro", uri=True)` y SQL parametrizado; solo `create_ticket`
(confirmado por un humano) escribe, con otra conexión.

```sql
{TICKETS_DDL.strip()}
```

Valores permitidos: estados `nuevo | en_progreso | en_espera | resuelto | cerrado`; prioridades
`baja | media | alta | critica`; categorías `configuracion | error_funcional | acceso_permisos |
rendimiento | migracion | capacitacion | integracion`; módulos `inventario | compras | ventas |
facturacion | accesos | actualizacion | general`.

## Cuentas cliente

| account_id | Nombre | Odoo | Nivel SLA |
|---|---|---|---|
{accounts}

## Usuarios

- **Portal** (`portal_user`): `USR-0001` a `USR-0012` (uno por cuenta, en el mismo orden) y
  `USR-0013` a `USR-0018` (segundo contacto de las cuentas 1 a 6).
- **N1**: `USR-0101` a `USR-0110`. **N2**: `USR-0201` a `USR-0205`.
  **Admin**: `USR-0301` y `USR-0302`.
- Usuarios demo fijos: `USR-0001` (portal, Comercial Los Volcanes), `USR-0101` (N1),
  `USR-0201` (N2) y `USR-0301` (admin).

Carteras de los N2 (tabla `portfolios`):

| N2 | Cuentas |
|---|---|
{portfolios}

## Política de SLA (`sla_policies`, POL-SLA-001)

| Prioridad | Primera respuesta | Resolución |
|---|---|---|
{sla}

`sla_due_at` de cada ticket se calcula con esta tabla (horas corridas en crítica y alta; días
hábiles de lunes a viernes, sin feriados fijos, en media y baja).

## Tickets ancla

Números fijos con propietario y estado conocidos, útiles para demos y pruebas de RBAC:

| Ticket | Cuenta | Reportó | Asignado | Estado | Prioridad | Resumen |
|---|---|---|---|---|---|---|
{anchors}

## Alcance por rol (SPEC §3)

```sql
-- portal_user: solo lo que reportó
SELECT * FROM tickets WHERE ticket_id = ? AND reported_by = ?;
-- n1: solo lo asignado a él
SELECT * FROM tickets WHERE ticket_id = ? AND assigned_to = ?;
-- n2: tickets de las cuentas de su cartera
SELECT t.* FROM tickets t JOIN portfolios p ON p.account_id = t.account_id
 WHERE t.ticket_id = ? AND p.user_id = ?;
-- admin: todos
SELECT * FROM tickets WHERE ticket_id = ?;
```

## Tablas agregadas por la rama de corpus

- `ticket_events`: historial (`creado`, `asignado`, `escalado`, `cambio_estado`,
  `cambio_prioridad`, `comentario`, `resuelto`, `cerrado`) con actor, valores previo/nuevo y nota.
- `sla_policies`: tiempos comprometidos por prioridad.
- Columnas nuevas en `tickets` (al final): `resolved_at`, `first_response_at`, `channel`
  (`portal | correo | telefono | chat`; las cuentas de nivel básico solo usan portal).
"""
    target.write_text(text, encoding="utf-8")
    return target
