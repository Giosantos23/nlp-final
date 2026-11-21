"""Modo mock de la API (``DOCUERP_API_MOCK=1``): respuestas realistas sin modelos ni índices.

El frontend se desarrolla contra este modo. Los fixtures viven en ``mock_data/*.json`` y
imitan *exactamente* ``QueryResponse``/``TraceEvent`` (las pruebas los validan contra
``schemas.py``). Escenarios de demostración y la consulta que los activa:

=====================  ==============================================================
Escenario              Disparador (sobre la consulta en minúsculas)
=====================  ==============================================================
``injection_blocked``  "ignora tus instrucciones", "prompt de sistema", "system prompt"…
``ticket_create``      verbo de creación + "ticket" (devuelve borrador + token HITL)
``ticket_classify``    "clasifica"
``ticket_status``      identificador ``TCK-\\d{5}`` o "estado del ticket"
``denied_role``        "rb-adm"/"contraseña maestra" con un rol distinto de admin
``version_17/18``      la consulta menciona "17" y "18" (se elige según ``erp_version``)
``abstention``         temas fuera del corpus (SAP, blockchain, clima…)
``nominal``            cualquier otra consulta (3 citas)
=====================  ==============================================================

El mock también genera PDF/DOCX mínimos en memoria para que las citas del frontend puedan
abrirse sin tener el corpus real.
"""

from __future__ import annotations

import asyncio
import copy
import io
import json
import re
import textwrap
import uuid
import zipfile
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any

from docuerp.api.datasources import (
    FALLBACK_USERS,
    MEDIA_TYPES,
    RUN_ID_RE,
    SOURCE_NAME_RE,
    DocumentFile,
    role_allowed,
)
from docuerp.api.errors import ApiError
from docuerp.api.service import ConfirmationRejectedError
from docuerp.schemas import QueryRequest, QueryResponse, TraceEvent, UserRole

MOCK_DIR = Path(__file__).parent / "mock_data"
"""Directorio con los fixtures JSON."""

_INJECTION = re.compile(
    r"ignora\w*\s+(todas\s+)?(tus|las|mis)\s+instrucciones|prompt\s+de\s+sistema|system\s+prompt|"
    r"ignore\s+(all\s+)?(previous|your)\s+instructions",
    re.IGNORECASE,
)
_CREATE = re.compile(r"\b(crea|crear|abre|abrir|genera|generar|levanta|registra)\b.*\bticket", re.I)
_TICKET_ID = re.compile(r"\bTCK-\d{5}\b", re.IGNORECASE)
_DENIED = re.compile(r"rb-adm|contraseña maestra|master password", re.IGNORECASE)
_OUT_OF_CORPUS = re.compile(r"\bsap\b|blockchain|clima|bitcoin|receta|futbol|fútbol", re.IGNORECASE)


@cache
def load_fixture(name: str) -> Any:
    """Carga (y cachea) un fixture JSON de ``mock_data``."""
    return json.loads((MOCK_DIR / name).read_text(encoding="utf-8"))


def pick_scenario(req: QueryRequest) -> str:
    """Elige el escenario de demostración según la consulta, el rol y la versión."""
    text = req.query
    if _INJECTION.search(text):
        return "injection_blocked"
    if _CREATE.search(text):
        return "ticket_create"
    if "clasifica" in text.lower():
        return "ticket_classify"
    if _TICKET_ID.search(text) or "estado del ticket" in text.lower():
        return "ticket_status"
    if _DENIED.search(text) and req.role != "admin":
        return "denied_role"
    if "17" in text and "18" in text:
        return "version_17" if req.erp_version == "17.0" else "version_18"
    if _OUT_OF_CORPUS.search(text):
        return "abstention"
    return "nominal"


class MockService:
    """Implementa ``AgentService`` con respuestas precargadas y retardos realistas."""

    def __init__(self, delay_scale: float = 1.0, max_stage_delay_s: float = 1.5) -> None:
        """``delay_scale`` multiplica la duración de cada etapa (0 = sin esperas)."""
        self.delay_scale = delay_scale
        self.max_stage_delay_s = max_stage_delay_s
        self._pending: dict[str, tuple[str, datetime]] = {}

    def _delay(self, event: TraceEvent | dict[str, Any]) -> float:
        """Segundos de espera simulados para una etapa."""
        ms = event["duration_ms"] if isinstance(event, dict) else event.duration_ms
        return min(ms / 1000.0 * self.delay_scale, self.max_stage_delay_s)

    def _build(self, scenario: str, req: QueryRequest) -> QueryResponse:
        """Copia el fixture del escenario con un ``trace_id`` nuevo y datos dinámicos."""
        raw = copy.deepcopy(load_fixture("scenarios.json")[scenario])
        raw["trace_id"] = f"mock-{uuid.uuid4().hex[:12]}"
        if scenario == "ticket_status":
            found = _TICKET_ID.search(req.query)
            if found:
                raw = json.loads(
                    json.dumps(raw, ensure_ascii=False).replace("TCK-00042", found.group(0).upper())
                )
        if scenario == "ticket_create":
            token = f"mock-{uuid.uuid4().hex}"
            expires = datetime.now(UTC) + timedelta(minutes=10)
            raw["pending_action"]["confirmation_token"] = token
            raw["pending_action"]["expires_at"] = expires.strftime("%Y-%m-%dT%H:%M:%SZ")
            self._pending[token] = (req.user_id, expires)
        return QueryResponse.model_validate(raw)

    async def answer_query(self, req: QueryRequest) -> QueryResponse:
        """Devuelve la respuesta del escenario tras esperar la suma de los retardos."""
        response = self._build(pick_scenario(req), req)
        await asyncio.sleep(sum(self._delay(e) for e in response.events))
        return response

    async def answer_query_stream(
        self, req: QueryRequest
    ) -> AsyncIterator[TraceEvent | QueryResponse]:
        """Emite cada evento tras su retardo y, al final, la respuesta completa."""
        response = self._build(pick_scenario(req), req)
        for event in response.events:
            await asyncio.sleep(self._delay(event))
            yield event
        yield response

    async def confirm_action(self, token: str, req: QueryRequest) -> QueryResponse:
        """Confirma un borrador emitido por este mock (un solo uso, del mismo usuario)."""
        entry = self._pending.get(token)
        if entry is None or entry[0] != req.user_id or entry[1] < datetime.now(UTC):
            raise ConfirmationRejectedError("Token de confirmación inválido, vencido o ya usado.")
        del self._pending[token]
        response = self._build("ticket_create_confirmed", req)
        await asyncio.sleep(sum(self._delay(e) for e in response.events))
        return response


# ---------------------------------------------------------------------------------------
# Documentos simulados
# ---------------------------------------------------------------------------------------


def _pdf_escape(text: str) -> bytes:
    """Escapa un texto para una cadena literal PDF (codificación Latin-1 / WinAnsi)."""
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return safe.encode("latin-1", errors="replace")


def build_pdf(pages: list[list[str]]) -> bytes:
    """Construye un PDF mínimo válido (Helvetica, una lista de líneas por página)."""
    count = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(count))
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    for i, lines in enumerate(pages):
        body = (
            b"BT /F1 11 Tf 14 TL 56 780 Td "
            + b" ".join(b"(" + _pdf_escape(line) + b") Tj T*" for line in lines)
            + b" ET"
        )
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>"
            ).encode()
        )
        objects.append(f"<< /Length {len(body)} >>\nstream\n".encode() + body + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)


def build_docx(paragraphs: list[str]) -> bytes:
    """Construye un DOCX mínimo (un párrafo por línea) sin dependencias externas."""
    from xml.sax.saxutils import escape

    body = "".join(f"<w:p><w:r><w:t>{escape(p)}</w:t></w:r></w:p>" for p in paragraphs)
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-'
        'package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-'
        'officedocument.wordprocessingml.document.main+xml"/></Types>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        'relationships/officeDocument" Target="word/document.xml"/></Relationships>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document)
    return buffer.getvalue()


class MockDataSource:
    """``DataSource`` respaldado por los fixtures de ``mock_data``."""

    def documents(self, role: UserRole) -> list[dict[str, Any]]:
        """Manifiesto simulado filtrado por rol."""
        return [
            {**entry, "source": entry["source_file"], "file_available": True}
            for entry in load_fixture("documents.json")
            if role_allowed(entry, role)
        ]

    def corpus_stats(self, documents: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Sin reporte de ingesta en el mock: el frontend calcula lo mínimo."""
        return None

    def document_file(self, source: str, role: UserRole, *, original: bool = False) -> DocumentFile:
        """Genera en memoria un PDF/DOCX que contiene los chunks citados de ese documento.

        Igual que la API real, un DOCX se entrega como PDF salvo que se pida el original.
        """
        if not SOURCE_NAME_RE.fullmatch(source):
            raise ApiError(400, "invalid_source", "Nombre de documento no válido.")
        entry = next(
            (e for e in load_fixture("documents.json") if e["source_file"] == source), None
        )
        if entry is None:
            raise ApiError(404, "document_not_found", "Documento no encontrado.")
        if not role_allowed(entry, role):
            raise ApiError(403, "forbidden", "Tu rol no tiene acceso a este documento.")
        chunks = [c for c in load_fixture("chunks.json") if c["source"] == source]
        header = [f"DocuERP (modo mock) - {entry['title']}", f"Version: {entry['version']}", ""]
        if source.endswith(".docx") and original:
            body = [line for c in chunks for line in c["text"].splitlines()]
            return DocumentFile(
                filename=source,
                media_type=MEDIA_TYPES[".docx"],
                content=build_docx(header + body),
            )
        pages: list[list[str]] = []
        for number in range(1, int(entry.get("pages", 10)) + 1):
            lines = [*header, f"Pagina {number} de {entry['pages']}", ""]
            on_page = [c for c in chunks if c["page"] <= number <= c["page_end"]]
            for chunk in on_page:
                lines.append(f"[{chunk['chunk_id']}]")
                for raw in chunk["text"].splitlines():
                    lines.extend(textwrap.wrap(raw, 88) or [""])
                lines.append("")
            if not on_page:
                lines.append("(Contenido de ejemplo sin fragmentos citados en esta pagina.)")
            pages.append(lines[:50])
        is_docx = source.endswith(".docx")
        return DocumentFile(
            filename=f"{source[:-5]}.pdf" if is_docx else source,
            media_type=MEDIA_TYPES[".pdf"],
            content=build_pdf(pages),
            rendition_of=source if is_docx else None,
        )

    def chunk(self, chunk_id: str, role: UserRole) -> dict[str, Any]:
        """Chunk simulado si el rol puede leerlo."""
        record = next((c for c in load_fixture("chunks.json") if c["chunk_id"] == chunk_id), None)
        if record is None:
            raise ApiError(404, "chunk_not_found", "Chunk no encontrado.")
        if not role_allowed(record, role):
            raise ApiError(403, "forbidden", "Tu rol no tiene acceso a este chunk.")
        return dict(record)

    def eval_summary(self, run_id: str | None) -> dict[str, Any]:
        """Resumen simulado (la corrida final o la pedida)."""
        if run_id is None:
            summary: dict[str, Any] = load_fixture("eval_summary.json")
            return summary
        if not RUN_ID_RE.fullmatch(run_id):
            raise ApiError(400, "invalid_run_id", "Identificador de corrida no válido.")
        run = next(
            (r for r in load_fixture("eval_runs.json")["runs"] if r["run_id"] == run_id), None
        )
        if run is None:
            raise ApiError(404, "eval_not_found", "No hay resultados de evaluación disponibles.")
        return dict(run)

    def eval_runs(self) -> dict[str, Any]:
        """Corridas simuladas."""
        runs: dict[str, Any] = load_fixture("eval_runs.json")
        return runs

    def security_summary(self) -> dict[str, Any]:
        """Matriz red-team simulada."""
        summary: dict[str, Any] = load_fixture("security_summary.json")
        return summary

    def users(self) -> list[dict[str, Any]]:
        """Usuarios demo fijos."""
        return [dict(u) for u in FALLBACK_USERS]

    def readiness(self) -> dict[str, bool]:
        """En modo mock todo está 'listo'."""
        return {"index_ready": True, "tickets_db_ready": True}

    def audit_record(self, trace_id: str) -> dict[str, Any] | None:
        """El mock no tiene bitácora de auditoría."""
        return None
