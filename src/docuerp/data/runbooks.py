"""Runbooks sintéticos de Altamira Soluciones ERP: markdown versionado → PDF/DOCX.

Flujo de ``docuerp data build-runbooks``:

1. Lee cada ``data/runbooks/src/*.md``: un bloque YAML de metadatos (front matter) y el
   cuerpo en Markdown.
2. Según el campo ``format`` del front matter, renderiza un **PDF** (Markdown → HTML →
   Chromium con Playwright) en ``data/runbooks/pdf/`` o un **DOCX** (python-docx) en
   ``data/runbooks/docx/``.
3. Regenera ``data/manifest/internal.yaml`` con una entrada por documento (incluida la
   versión obsoleta) y el ``sha256`` del archivo generado.

Los archivos se normalizan (fechas de creación fijas, ZIP/PDF sin marcas de tiempo) para que
reconstruirlos produzca exactamente los mismos bytes en la misma máquina.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from docuerp import paths
from docuerp.data.markdown_ast import parse_blocks, split_front_matter
from docuerp.data.runbook_render import (
    DISCLAIMER,
    CoverRows,
    DocxPlan,
    build_docx,
    build_html,
    footer_template,
)
from docuerp.schemas import ChunkErpVersion, DocStatus, DocType, ErpModule, UserRole

logger = logging.getLogger(__name__)

ORIGIN = "Equipo DocuERP (autoría propia)"
LICENSE = "Uso académico interno (sintético)"

_ACCESS_LABELS: dict[str, str] = {
    "N1": "N1 (analistas N1, N2 y administradores)",
    "N1+": "N1+ (analistas N1, N2 y administradores)",
    "N2": "N2 (analistas N2 y administradores)",
    "ADMIN": "ADMIN — restringido a administradores",
}


class RunbookMeta(BaseModel):
    """Metadatos de un runbook (front matter YAML del archivo fuente)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    code: str
    title: str
    version: str
    status: DocStatus
    access_level: Literal["N1", "N1+", "N2", "ADMIN"]
    erp_version: ChunkErpVersion
    module: ErpModule
    doc_type: DocType
    allowed_roles: list[UserRole]
    sensitivity: Literal["public", "internal", "restricted"]
    date: str
    update_frequency: str
    format: Literal["pdf", "docx"]
    owner: str
    poisoned: bool = False
    supersedes: str | None = None
    superseded_by: str | None = None

    @field_validator("date", mode="before")
    @classmethod
    def _date_to_iso(cls, value: object) -> object:
        """YAML interpreta ``2026-09-15`` como ``date``; se guarda como texto ISO."""
        return value.isoformat() if isinstance(value, date) else value

    @property
    def filename(self) -> str:
        """Nombre citable del archivo generado, p. ej. ``RB-N2-003_v1.1.pdf``."""
        return f"{self.code}_{self.version}.{self.format}"


class Runbook(BaseModel):
    """Runbook cargado: metadatos + cuerpo Markdown (sin front matter)."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    meta: RunbookMeta
    body: str
    source_path: Path


def load_runbook(path: Path) -> Runbook:
    """Lee un runbook Markdown y valida su front matter.

    Raises:
        ValueError: si faltan metadatos, sobran campos o el nombre del archivo no
            coincide con ``<code>_<version>.md``.
    """
    front, body = split_front_matter(path.read_text(encoding="utf-8"))
    meta = RunbookMeta.model_validate(front)
    expected = f"{meta.code}_{meta.version}.md"
    if path.name != expected:
        raise ValueError(f"{path.name}: el nombre debe ser {expected} (código_versión.md)")
    if meta.id != f"{meta.code}@{meta.version}":
        raise ValueError(f"{path.name}: id {meta.id!r} debe ser '{meta.code}@{meta.version}'")
    return Runbook(meta=meta, body=body, source_path=path)


def load_runbooks(src_dir: Path | None = None) -> list[Runbook]:
    """Carga todos los ``*.md`` de ``src_dir`` ordenados por código y versión."""
    directory = src_dir or paths.RUNBOOKS / "src"
    runbooks = [load_runbook(p) for p in sorted(directory.glob("*.md"))]
    if not runbooks:
        raise FileNotFoundError(f"No hay runbooks Markdown en {directory}")
    ids = [rb.meta.id for rb in runbooks]
    if len(ids) != len(set(ids)):
        raise ValueError("Hay runbooks con id duplicado")
    return runbooks


def cover_rows(meta: RunbookMeta) -> CoverRows:
    """Filas de la portada: código, versión, estado, nivel de acceso, versión de Odoo y fecha."""
    erp = "Odoo 17.0 y 18.0" if meta.erp_version == "any" else f"Odoo {meta.erp_version}"
    return [
        ("Código", meta.code),
        ("Versión", meta.version),
        ("Estado", meta.status),
        ("Nivel de acceso", _ACCESS_LABELS[meta.access_level]),
        ("Versión de Odoo", erp),
        ("Fecha", meta.date),
        ("Propietario", meta.owner),
    ]


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def _normalize_pdf(data: bytes, meta: RunbookMeta) -> bytes:
    """Fija metadatos y quita el ``/ID`` aleatorio de Chromium (PDF reproducible)."""
    import pymupdf

    stamp = "D:" + meta.date.replace("-", "") + "000000Z"
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        doc.set_metadata(
            {
                "title": f"{meta.code} {meta.version} — {meta.title}",
                "author": "Equipo DocuERP",
                "subject": DISCLAIMER,
                "creator": "DocuERP build_runbooks",
                "producer": "Chromium (Playwright) + PyMuPDF",
                "creationDate": stamp,
                "modDate": stamp,
                "keywords": "",
            }
        )
        doc.xref_set_key(-1, "ID", "null")
        return doc.tobytes(garbage=4, deflate=True, no_new_id=True)
    finally:
        doc.close()


def render_pdfs(jobs: list[tuple[Runbook, Path]]) -> None:
    """Renderiza cada ``(runbook, destino)`` a PDF con una sola instancia de Chromium."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            for runbook, dest in jobs:
                meta = runbook.meta
                html_doc = build_html(
                    meta.title, meta.code, meta.status, cover_rows(meta), parse_blocks(runbook.body)
                )
                page = browser.new_page()
                try:
                    page.set_content(html_doc, wait_until="load")
                    raw = page.pdf(
                        format="Letter",
                        print_background=True,
                        prefer_css_page_size=True,
                        display_header_footer=True,
                        header_template="<div></div>",
                        footer_template=footer_template(meta.filename),
                    )
                finally:
                    page.close()
                dest.write_bytes(_normalize_pdf(raw, meta))
                logger.info("PDF generado: %s", dest)
        finally:
            browser.close()


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------


def render_docx(runbook: Runbook) -> DocxPlan:
    """Construye el DOCX de un runbook (ver :func:`runbook_render.build_docx`)."""
    meta = runbook.meta
    return build_docx(
        title=meta.title,
        code=meta.code,
        filename=meta.filename,
        cover_rows=cover_rows(meta),
        body_blocks=parse_blocks(runbook.body),
        created=datetime.fromisoformat(f"{meta.date}T00:00:00"),
    )


# ---------------------------------------------------------------------------
# Manifiesto
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Hash SHA-256 (hex) del contenido de un archivo."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_entry(meta: RunbookMeta, file_path: Path) -> dict[str, Any]:
    """Entrada ``SourceEntry`` del manifiesto para un runbook ya generado."""
    entry: dict[str, Any] = {
        "id": meta.id,
        "title": f"{meta.code} {meta.title} ({meta.version})",
        "source_file": meta.filename,
        "origin": ORIGIN,
        "license": LICENSE,
        "owner": meta.owner,
        "format": meta.format,
        "version": meta.version,
        "erp_version": meta.erp_version,
        "status": meta.status,
        "date": meta.date,
        "update_frequency": meta.update_frequency,
        "allowed_roles": list(meta.allowed_roles),
        "sensitivity": meta.sensitivity,
        "module": meta.module,
        "doc_type": meta.doc_type,
        "language": "es",
        "sha256": sha256_file(file_path),
    }
    if meta.poisoned:
        # Documento con inyección indirecta deliberada: debe indexarse para probar defensas.
        entry["poisoned"] = True
    return entry


def tickets_dataset_entry(n_tickets: int = 300, seed: int = 42) -> dict[str, Any]:
    """Entrada de gobernanza de la BD de tickets (no es un documento ingerible)."""
    return {
        "id": "tickets-db",
        "title": "Base de tickets sintéticos de soporte (SQLite)",
        "path": "data/tickets/tickets.db",
        "origin": ORIGIN,
        "license": LICENSE,
        "owner": "Equipo DocuERP",
        "format": "sqlite",
        "version": f"seed-{seed}",
        "date": "2026-11-16",
        "update_frequency": "bajo demanda (docuerp data seed-tickets)",
        "sensitivity": "internal",
        "language": "es",
        "generator": "docuerp data seed-tickets",
        "records": {"tickets": n_tickets},
        "contains_synthetic_pii": True,
        "access_policy": (
            "Lectura por rol vía get_ticket_status: portal_user = tickets que reportó; "
            "n1 = asignados; n2 = cuentas de su cartera; admin = todos."
        ),
    }


_MANIFEST_HEADER = """\
# Manifiesto del corpus interno sintético de Altamira Soluciones ERP.
# GENERADO por `docuerp data build-runbooks` (scripts/build_runbooks.py): no editar a mano.
# - `documents`: una entrada por runbook/política (SourceEntry); incluye la versión obsoleta
#   de RB-N2-003 para probar vigencia. RB-N1-008 lleva `poisoned: true` (inyección indirecta
#   deliberada en una nota al pie) y está vigente para que entre al índice.
# - `datasets`: fuentes de datos que no se ingieren como documentos (BD de tickets).
"""


def write_manifest(entries: list[dict[str, Any]], manifest_path: Path) -> Path:
    """Escribe ``internal.yaml`` con documentos y datasets."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"documents": entries, "datasets": [tickets_dataset_entry()]}
    body = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100)
    manifest_path.write_text(_MANIFEST_HEADER + body, encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------


def build_runbooks(
    out_dir: Path | None = None,
    *,
    src_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> list[Path]:
    """Renderiza ``data/runbooks/src/*.md`` a PDF/DOCX y devuelve las rutas generadas.

    Args:
        out_dir: raíz de salida; se crean ``pdf/`` y ``docx/`` dentro. Por defecto
            ``data/runbooks``.
        src_dir: carpeta con los ``.md`` (por defecto ``data/runbooks/src``).
        manifest_path: dónde escribir ``internal.yaml``. Por defecto se escribe en
            ``data/manifest/internal.yaml`` solo cuando ``out_dir`` es el predeterminado,
            para que construir en una carpeta temporal no toque el repositorio.
    """
    use_defaults = out_dir is None
    root = out_dir or paths.RUNBOOKS
    runbooks = load_runbooks(src_dir)
    pdf_dir, docx_dir = root / "pdf", root / "docx"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    docx_dir.mkdir(parents=True, exist_ok=True)

    pdf_jobs = [(rb, pdf_dir / rb.meta.filename) for rb in runbooks if rb.meta.format == "pdf"]
    if pdf_jobs:
        render_pdfs(pdf_jobs)
    generated: dict[str, Path] = {rb.meta.id: dest for rb, dest in pdf_jobs}
    for runbook in runbooks:
        if runbook.meta.format == "docx":
            dest = docx_dir / runbook.meta.filename
            dest.write_bytes(render_docx(runbook).data)
            generated[runbook.meta.id] = dest
            logger.info("DOCX generado: %s", dest)

    target = manifest_path or (paths.MANIFEST / "internal.yaml" if use_defaults else None)
    if target is not None:
        entries = [manifest_entry(rb.meta, generated[rb.meta.id]) for rb in runbooks]
        write_manifest(entries, target)
        logger.info("Manifiesto escrito: %s", target)
    return [generated[rb.meta.id] for rb in runbooks]
