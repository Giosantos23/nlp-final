"""Fixtures del agente: BD SQLite temporal con ``TICKETS_DDL``, *fakes* de recuperador,
LLM, guards y auditoría, y fábrica de dependencias inyectables."""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from docuerp.agent.deps import AgentDeps
from docuerp.agent.router import RouteDecision, rule_intent
from docuerp.agent.service import AgentService
from docuerp.agent.tools.base import default_registry
from docuerp.agent.tools.classify import TicketClassifier
from docuerp.agent.tools.create_ticket import TicketDraftService
from docuerp.agent.tools.ticket_status import TicketReader
from docuerp.config import AgentSettings, LLMSettings, Settings
from docuerp.data.tickets import TICKETS_DDL
from docuerp.llm.contracts import LLMResult
from docuerp.retrieval.contracts import RetrievalQuery, RetrievalResult
from docuerp.schemas import (
    ABSTENTION_MESSAGE,
    CANARY_TOKEN,
    Answer,
    Chunk,
    Intent,
    QueryRequest,
    ScoredChunk,
    SourceRef,
    UserRole,
    make_chunk_id,
)
from docuerp.security.input_guard import GuardVerdict
from docuerp.security.output_guard import OutputVerdict

NOW = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)
SECRET = "secreto-de-prueba-hmac-123"

# ---------------------------------------------------------------------------
# Base de tickets temporal
# ---------------------------------------------------------------------------

ACCOUNTS = [
    ("ACC-0001", "Comercial Los Volcanes", "18.0", "premium"),
    ("ACC-0002", "Distribuidora Atitlán", "17.0", "estandar"),
    ("ACC-0003", "Ferretería El Quetzal", "18.0", "basico"),
]
USERS = [
    ("USR-0001", "Ana Portal", "portal_user", "ACC-0001"),
    ("USR-0002", "Beto Portal", "portal_user", "ACC-0002"),
    ("USR-0101", "Carla N1", "n1", None),
    ("USR-0102", "Diego N1", "n1", None),
    ("USR-0201", "Elena N2", "n2", None),
    ("USR-0202", "Fabio N2", "n2", None),
    ("USR-0301", "Gabi Admin", "admin", None),
]
PORTFOLIOS = [("USR-0201", "ACC-0001"), ("USR-0201", "ACC-0002"), ("USR-0202", "ACC-0003")]
TICKETS = [
    # ticket, cuenta, reportó, asignado, resumen, descripción, cat, prio, módulo, estado, sla
    (
        "TCK-00001",
        "ACC-0001",
        "USR-0001",
        "USR-0101",
        "No puedo validar la recepción parcial",
        "Escribir a juan.perez@volcanes.com con el detalle del error ERR-INV-204.",
        "error_funcional",
        "alta",
        "inventario",
        "en_progreso",
        (NOW - timedelta(hours=2)).isoformat(),
    ),
    (
        "TCK-00002",
        "ACC-0002",
        "USR-0002",
        "USR-0102",
        "Factura duplicada al confirmar pedido",
        "Se generan dos facturas por pedido.",
        "error_funcional",
        "media",
        "facturacion",
        "resuelto",
        (NOW - timedelta(days=1)).isoformat(),
    ),
    (
        "TCK-00003",
        "ACC-0003",
        "USR-0301",
        None,
        "Actualizar a Odoo 18.0",
        "Planificar la migración.",
        "migracion",
        "baja",
        "actualizacion",
        "nuevo",
        (NOW + timedelta(days=3)).isoformat(),
    ),
]


def create_tickets_db(path: Path) -> Path:
    """Crea la BD con ``TICKETS_DDL`` y datos mínimos deterministas."""
    conn = sqlite3.connect(path)
    try:
        conn.executescript(TICKETS_DDL)
        conn.executemany("INSERT INTO accounts VALUES (?, ?, ?, ?)", ACCOUNTS)
        conn.executemany("INSERT INTO users VALUES (?, ?, ?, ?)", USERS)
        conn.executemany("INSERT INTO portfolios VALUES (?, ?)", PORTFOLIOS)
        for row in TICKETS:
            ticket_id, account, reporter, assignee, summary, desc, cat, prio, mod, status, sla = row
            conn.execute(
                """INSERT INTO tickets (ticket_id, account_id, reported_by, assigned_to, summary,
                description, category, priority, module, status, erp_version, created_at,
                updated_at, sla_due_at, needs_human_review, idempotency_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '18.0', ?, ?, ?, 0, NULL)""",
                (
                    ticket_id,
                    account,
                    reporter,
                    assignee,
                    summary,
                    desc,
                    cat,
                    prio,
                    mod,
                    status,
                    (NOW - timedelta(days=5)).isoformat(),
                    (NOW - timedelta(days=1)).isoformat(),
                    sla,
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return path


def count_tickets(path: Path) -> int:
    """Número de filas en ``tickets``."""
    conn = sqlite3.connect(path)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0])
    finally:
        conn.close()


def fake_masker(text: str) -> tuple[str, list[str]]:
    """Enmascarador mínimo de correos para las pruebas."""
    masked, n = re.subn(r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]", text)
    return masked, ["email"] if n else []


class MutableClock:
    """Reloj controlable para probar expiraciones."""

    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now = self.now + timedelta(**delta)


@pytest.fixture
def tickets_db(tmp_path: Path) -> Path:
    return create_tickets_db(tmp_path / "tickets.db")


@pytest.fixture
def clock() -> MutableClock:
    return MutableClock()


@pytest.fixture
def reader(tickets_db: Path, clock: MutableClock) -> TicketReader:
    return TicketReader(tickets_db, masker=fake_masker, clock=clock)


@pytest.fixture
def ticket_service(tickets_db: Path, clock: MutableClock) -> TicketDraftService:
    return TicketDraftService(tickets_db, secret=SECRET, ttl_min=10, clock=clock)


# ---------------------------------------------------------------------------
# Datos de entrenamiento pequeños (mismo formato que SPEC §10)
# ---------------------------------------------------------------------------

CLASSIFIER_ROWS: list[tuple[str, str, str, str]] = [
    (
        "no puedo validar la recepcion en inventario error al validar",
        "error_funcional",
        "inventario",
        "alta",
    ),
    (
        "error funcional al validar la entrega de inventario",
        "error_funcional",
        "inventario",
        "alta",
    ),
    ("aparece un error al confirmar la orden de compra", "error_funcional", "compras", "alta"),
    ("error funcional al publicar la factura de cliente", "error_funcional", "facturacion", "alta"),
    (
        "el usuario no tiene permisos de acceso al menu de ventas",
        "acceso_permisos",
        "accesos",
        "media",
    ),
    ("necesito acceso y permisos para el usuario nuevo", "acceso_permisos", "accesos", "media"),
    (
        "permisos denegados acceso restringido al modulo de compras",
        "acceso_permisos",
        "accesos",
        "media",
    ),
    ("el usuario perdio acceso permisos de grupo", "acceso_permisos", "accesos", "media"),
    (
        "el sistema esta muy lento rendimiento bajo al cargar reportes",
        "rendimiento",
        "general",
        "critica",
    ),
    ("rendimiento lento la pantalla tarda demasiado", "rendimiento", "general", "critica"),
    (
        "lentitud y bajo rendimiento en la carga de inventario",
        "rendimiento",
        "inventario",
        "critica",
    ),
    ("todo esta lento rendimiento del servidor degradado", "rendimiento", "general", "critica"),
]
CLASSIFIER_TEST: list[tuple[str, str, str, str]] = [
    ("error al validar la recepcion de inventario", "error_funcional", "inventario", "alta"),
    ("el usuario no tiene acceso permisos", "acceso_permisos", "accesos", "media"),
    ("rendimiento muy lento al cargar", "rendimiento", "general", "critica"),
]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return path


@pytest.fixture
def classifier_data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "classifier"
    for name, rows in (("tickets_train", CLASSIFIER_ROWS), ("tickets_test", CLASSIFIER_TEST)):
        _write_jsonl(
            data_dir / f"{name}.jsonl",
            [
                {"text": t, "category": c, "module": m, "priority": p, "ticket_id": f"TCK-{i:05d}"}
                for i, (t, c, m, p) in enumerate(rows, start=1)
            ],
        )
    return data_dir


@pytest.fixture
def tfidf_classifier(classifier_data: Path, tmp_path: Path) -> TicketClassifier:
    return TicketClassifier(
        "tfidf", model_dir=tmp_path / "models", data_dir=classifier_data, min_confidence=0.34
    )


INTENT_ROWS: list[tuple[str, str, str, bool]] = [
    ("como valido una recepcion parcial en inventario", "doc_query", "train", False),
    ("pasos para configurar rutas de inventario en odoo", "doc_query", "train", False),
    ("donde configuro las listas de precios de ventas en odoo", "doc_query", "train", False),
    ("procedimiento para publicar una factura en odoo", "doc_query", "train", False),
    ("como configuro los permisos de usuario en odoo", "doc_query", "train", False),
    ("en que va mi ticket de soporte", "ticket_status", "train", False),
    ("quiero saber como va mi solicitud de soporte abierta", "ticket_status", "train", False),
    ("cual es el avance de mi solicitud abierta", "ticket_status", "train", False),
    ("necesito reportar una falla nueva al equipo de soporte", "ticket_create", "train", False),
    ("quiero dejar registrada una falla nueva para soporte", "ticket_create", "train", False),
    (
        "que tipo de problema es este para soporte asignar categoria",
        "ticket_classify",
        "train",
        False,
    ),
    ("asigna categoria y prioridad a este problema", "ticket_classify", "train", False),
    ("receta de pastel de chocolate con fresas", "out_of_domain", "train", False),
    ("quien gano el partido de futbol ayer", "out_of_domain", "train", False),
    ("escribe un poema sobre el mar", "out_of_domain", "train", False),
    ("como valido una recepcion en inventario odoo", "doc_query", "test", False),
    ("en que va mi ticket", "ticket_status", "test", True),
]


@pytest.fixture
def intents_path(tmp_path: Path) -> Path:
    return _write_jsonl(
        tmp_path / "router" / "intents.jsonl",
        [{"text": t, "intent": i, "split": s, "hard": h} for t, i, s, h in INTENT_ROWS],
    )


# ---------------------------------------------------------------------------
# Corpus y recuperador falsos
# ---------------------------------------------------------------------------


def make_chunk(
    doc_id: str = "odoo-18.0-inventario",
    page: int = 12,
    idx: int = 0,
    *,
    roles: list[UserRole] | None = None,
    text: str = "Para validar una recepción parcial, abra la transferencia y pulse Validar.",
) -> Chunk:
    return Chunk(
        chunk_id=make_chunk_id(doc_id, page, idx),
        doc_id=doc_id,
        text=text,
        source=f"{doc_id}.pdf",
        page=page,
        page_end=page,
        version="18.0@a1b2c3d",
        erp_version="18.0",
        status="vigente",
        section_path="Inventario > Operaciones > Recepciones",
        module="inventario",
        doc_type="odoo_doc",
        language="es",
        allowed_roles=roles or ["portal_user", "n1", "n2", "admin"],
        content_hash="a" * 64,
        ingested_at="2026-11-17T08:00:00Z",
        token_count=40,
    )


def scored(chunk: Chunk, rank: int = 1, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score, rank=rank, retriever="rerank")


@dataclass
class FakeRetriever:
    """Recuperador programable: devuelve ``hits`` y registra las consultas."""

    hits: list[ScoredChunk] = field(default_factory=lambda: [scored(make_chunk())])
    evidence_ok: bool = True
    fail: bool = False
    delay_s: float = 0.0
    queries: list[RetrievalQuery] = field(default_factory=list)

    def retrieve(self, q: RetrievalQuery) -> RetrievalResult:
        self.queries.append(q)
        if self.delay_s:
            import time

            time.sleep(self.delay_s)
        if self.fail:
            raise RuntimeError("índice no disponible")
        top = self.hits[0].score if self.hits else None
        return RetrievalResult(
            query=q,
            hits=self.hits[: q.top_k],
            stages={"bm25": self.hits, "rerank": self.hits},
            filtered_out=0,
            evidence_ok=self.evidence_ok,
            top_score=top,
            timings_ms={"bm25": 1.5, "dense": 2.0, "fusion": 0.2, "rerank": 5.0},
        )


# ---------------------------------------------------------------------------
# LLM, generador, guards y auditoría falsos
# ---------------------------------------------------------------------------


class ScriptedProvider:
    """Proveedor LLM falso: ``ok`` responde, ``slow`` duerme, ``raise`` falla."""

    def __init__(self, name: str = "fake", behavior: str = "ok", text: str = "") -> None:
        self.name = name
        self.behavior = behavior
        self.text = text or "Abra la recepción y pulse Validar [chunk 1]."
        self.calls = 0

    async def complete(self, system: str, user: str, **kwargs: Any) -> LLMResult:
        self.calls += 1
        if self.behavior == "slow":
            await asyncio.sleep(5)
        if self.behavior == "raise":
            raise ConnectionError("LLM caído")
        return LLMResult(
            text=self.text,
            model=kwargs.get("model") or "claude-haiku-5-5",
            provider=self.name,
            input_tokens=120,
            output_tokens=30,
            cost_usd=0.0005,
            latency_ms=10.0,
        )


async def fake_generate(
    question: str,
    hits: list[ScoredChunk],
    *,
    intent: Intent,
    tool_result: Any | None = None,
    provider: Any,
    model: str | None = None,
    erp_version: str | None = None,
) -> tuple[Answer, LLMResult]:
    """Generador falso que delega en el proveedor y cita el primer hit."""
    llm = await provider.complete("sistema", question, model=model)
    sources = [
        SourceRef(source=h.chunk.source, page=h.chunk.page, chunk_id=h.chunk.chunk_id)
        for h in hits[:1]
    ]
    answer = Answer(
        answer=llm.text, sources=sources, abstained=False, tool_used=None, needs_human_review=False
    )
    return answer, llm


def fake_input_guard(req: QueryRequest) -> GuardVerdict:
    """Bloquea frases de inyección obvias; si no, deja pasar el texto."""
    if re.search(r"ignora (todas )?las instrucciones", req.query, re.IGNORECASE):
        return GuardVerdict(
            allowed=False,
            sanitized_text=req.query,
            flags=["prompt_injection"],
            risk_score=0.95,
            reason="intento de inyección de instrucciones",
        )
    return GuardVerdict(allowed=True, sanitized_text=req.query, flags=[], risk_score=0.05)


def fake_output_guard(
    answer: Answer,
    retrieved: list[ScoredChunk],
    *,
    tool_result: Any | None = None,
    role: Any | None = None,
) -> OutputVerdict:
    """Invalida respuestas con el canario o con citas fuera de lo recuperado."""
    ids = {h.chunk.chunk_id for h in retrieved}
    errors = []
    if CANARY_TOKEN in answer.answer:
        errors.append("canary_leak")
    if any(ref.chunk_id not in ids for ref in answer.sources):
        errors.append("citation_not_retrieved")
    if errors:
        safe = Answer(
            answer=ABSTENTION_MESSAGE,
            sources=[],
            abstained=True,
            tool_used=answer.tool_used,
            needs_human_review=True,
        )
        return OutputVerdict(valid=False, errors=errors, answer=safe)
    return OutputVerdict(valid=True, errors=[], answer=answer)


def fake_filter_authorized(
    role: UserRole, hits: list[ScoredChunk]
) -> tuple[list[ScoredChunk], int]:
    kept = [h for h in hits if role in h.chunk.allowed_roles]
    return kept, len(hits) - len(kept)


class RecordingAudit:
    """Auditoría en memoria."""

    def __init__(self, fail: bool = False) -> None:
        self.records: list[Any] = []
        self.requests: list[QueryRequest | None] = []
        self.fail = fail

    def log(self, trace: Any, *, request: QueryRequest | None = None) -> None:
        if self.fail:
            raise OSError("disco lleno")
        self.records.append(trace)
        self.requests.append(request)


class RuleRouter:
    """Router falso: reglas reales y, si no aplican, una intención fija."""

    def __init__(self, default: Intent = "doc_query") -> None:
        self.default = default

    def decide(self, text: str) -> RouteDecision:
        rule = rule_intent(text)
        if rule is not None:
            return RouteDecision(intent=rule[0], confidence=0.95, rationale=rule[1])
        return RouteDecision(
            intent=self.default, confidence=0.8, rationale="fake", method="classifier"
        )


def make_settings(**agent_overrides: Any) -> Settings:
    llm_timeout = agent_overrides.pop("llm_timeout_s", 0.3)
    return Settings(
        agent=AgentSettings(**agent_overrides),
        llm=LLMSettings(timeout_s=llm_timeout),
    )


@pytest.fixture
def make_deps(
    reader: TicketReader,
    ticket_service: TicketDraftService,
    tfidf_classifier: TicketClassifier,
) -> Callable[..., AgentDeps]:
    """Fábrica de ``AgentDeps`` con *fakes*; cada parámetro se puede sustituir."""

    def _factory(**overrides: Any) -> AgentDeps:
        settings = overrides.pop("settings", None) or make_settings()
        retriever = overrides.pop("retriever", None) or FakeRetriever()
        classifier = overrides.pop("classifier", None) or tfidf_classifier
        registry = overrides.pop("registry", None) or default_registry(
            retriever=retriever,
            ticket_reader=reader,
            classifier=classifier,
            ticket_service=ticket_service,
            settings=settings,
        )
        values: dict[str, Any] = {
            "settings": settings,
            "router": RuleRouter(),
            "registry": registry,
            "ticket_service": ticket_service,
            "input_guard": fake_input_guard,
            "output_guard": fake_output_guard,
            "filter_authorized": fake_filter_authorized,
            "generator": fake_generate,
            "provider": ScriptedProvider("agent_sdk"),
            "fallback_provider": ScriptedProvider("offline", text="Extracto offline."),
            "audit": RecordingAudit(),
        }
        values.update(overrides)
        return AgentDeps(**values)

    return _factory


@pytest.fixture
def make_service(make_deps: Callable[..., AgentDeps]) -> Callable[..., AgentService]:
    def _factory(**overrides: Any) -> AgentService:
        return AgentService(make_deps(**overrides))

    return _factory


def request(
    query: str, role: UserRole = "n1", user_id: str = "USR-0101", **kw: Any
) -> QueryRequest:
    """Atajo para construir un ``QueryRequest`` válido."""
    return QueryRequest(query=query, role=role, user_id=user_id, **kw)


@pytest.fixture
def kit() -> types.SimpleNamespace:
    """Utilidades de prueba (con ``--import-mode=importlib`` el conftest no es importable)."""
    return types.SimpleNamespace(
        NOW=NOW,
        SECRET=SECRET,
        request=request,
        make_chunk=make_chunk,
        scored=scored,
        count_tickets=count_tickets,
        make_settings=make_settings,
        FakeRetriever=FakeRetriever,
        ScriptedProvider=ScriptedProvider,
        RecordingAudit=RecordingAudit,
        RuleRouter=RuleRouter,
        MutableClock=MutableClock,
        fake_masker=fake_masker,
        create_tickets_db=create_tickets_db,
        write_jsonl=_write_jsonl,
    )
