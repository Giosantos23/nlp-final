"""Versión PDF de los runbooks DOCX, con la misma paginación que el DOCX citado.

Problema: el visor del frontend usa pdf.js y no puede mostrar un DOCX. Las citas de los tres
runbooks DOCX (``RB-ADM-007``, ``RB-N1-008``, ``RB-N2-004``) apuntan a una página del DOCX,
cuya paginación es determinista por construcción (``docuerp.data.runbook_render``): página 1
= portada y luego **una página por sección principal** (``##``), cada una cerrada con un
salto de página explícito.

Decisión: cuando se pide el archivo de una fuente ``.docx`` la API sirve un PDF generado desde
la **misma fuente Markdown** (``data/runbooks/src``), con el mismo renderizador HTML que usan
los runbooks PDF y un salto de página forzado después de la portada y de cada sección. Así la
página N del PDF contiene exactamente el texto de la página N del DOCX y la cita abre donde
debe. El DOCX original sigue disponible con ``?format=original``.

Por qué no convertir el DOCX: no hay LibreOffice en el entorno y agregarlo sería una
dependencia pesada; la fuente Markdown es la verdad de ambos formatos, así que regenerar desde
ella es más fiel que interpretar el DOCX.

El PDF se genera una sola vez con Chromium (Playwright, ya usado por la ingesta) y se cachea
en ``.cache/docx_pdf/`` con el hash de la fuente en el nombre: si cambia el Markdown se genera
de nuevo. Se valida que el PDF tenga ``1 + secciones`` páginas; si una sección no cupiera en
una página la paginación dejaría de coincidir y se registra una advertencia.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from docuerp import paths

logger = logging.getLogger("docuerp.api")

RENDER_VERSION = "1"
"""Se incluye en la clave de caché: subirlo invalida los PDF ya generados."""

PAGE_CSS = (
    "<style>"
    "section.docx-page { break-after: page; }"
    "section.docx-page:last-of-type { break-after: auto; }"
    "</style>"
)
"""Reglas extra: cada sección principal ocupa su propia página, igual que en el DOCX."""

_LOCK = threading.Lock()


@dataclass(frozen=True)
class DocxRendition:
    """PDF generado para un DOCX y su verificación de paginación."""

    source: str
    path: Path
    pages: int
    expected_pages: int

    @property
    def pagination_ok(self) -> bool:
        """``True`` si el PDF tiene tantas páginas como el DOCX (portada + secciones)."""
        return self.pages == self.expected_pages


def cache_dir() -> Path:
    """Directorio de caché de las versiones PDF (gitignored)."""
    return paths.CACHE / "docx_pdf"


def _runbook_for(source: str) -> object | None:
    """Runbook Markdown cuyo archivo citable es ``source`` (``None`` si no hay)."""
    from docuerp.data.runbooks import load_runbooks

    try:
        runbooks = load_runbooks(paths.RUNBOOKS / "src")
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("no se pudieron cargar los runbooks fuente: %s", type(exc).__name__)
        return None
    return next((rb for rb in runbooks if rb.meta.filename == source), None)


def build_paged_html(runbook: object) -> tuple[str, int]:
    """HTML del runbook con una página por sección principal y número esperado de páginas."""
    from docuerp.data.markdown_ast import parse_blocks
    from docuerp.data.runbook_render import blocks_to_html, build_html, split_sections
    from docuerp.data.runbooks import cover_rows

    meta = runbook.meta  # type: ignore[attr-defined]
    sections = split_sections(parse_blocks(runbook.body))  # type: ignore[attr-defined]
    shell = build_html(meta.title, meta.code, meta.status, cover_rows(meta), [])
    body = "".join(f'<section class="docx-page">{blocks_to_html(s)}</section>' for s in sections)
    html_doc = shell.replace("</head>", f"{PAGE_CSS}</head>", 1).replace(
        "</body>", f"{body}</body>", 1
    )
    return html_doc, len(sections) + 1


def _render(html_doc: str, footer_name: str) -> bytes:
    """Imprime el HTML a PDF tamaño carta con Chromium (mismo formato que los runbooks PDF)."""
    from playwright.sync_api import sync_playwright

    from docuerp.data.runbook_render import footer_template

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            page.set_content(html_doc, wait_until="load")
            return page.pdf(
                format="Letter",
                print_background=True,
                prefer_css_page_size=True,
                display_header_footer=True,
                header_template="<div></div>",
                footer_template=footer_template(footer_name),
            )
        finally:
            browser.close()


def _page_count(path: Path) -> int:
    import pymupdf

    with pymupdf.open(path) as doc:
        return int(doc.page_count)


def docx_rendition(source: str, *, force: bool = False) -> DocxRendition | None:
    """Devuelve (generándolo si hace falta) el PDF equivalente al DOCX ``source``.

    Returns:
        ``None`` si ``source`` no corresponde a un runbook DOCX con fuente Markdown.

    Raises:
        RuntimeError: si Chromium no pudo generar el PDF.
    """
    if not source.lower().endswith(".docx"):
        return None
    runbook = _runbook_for(source)
    if runbook is None:
        return None
    html_doc, expected = build_paged_html(runbook)
    digest = hashlib.sha256(f"{RENDER_VERSION}\n{html_doc}".encode()).hexdigest()[:12]
    stem = Path(source).stem
    target = cache_dir() / f"{stem}.{digest}.pdf"
    with _LOCK:
        if force or not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = _render(html_doc, source)
            except Exception as exc:  # Chromium ausente o caído
                raise RuntimeError(f"no se pudo generar el PDF de {source}") from exc
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(target)
            for stale in target.parent.glob(f"{stem}.*.pdf"):
                if stale != target:
                    stale.unlink(missing_ok=True)
            logger.info("PDF equivalente generado para %s", source)
    rendition = DocxRendition(
        source=source, path=target, pages=_page_count(target), expected_pages=expected
    )
    if not rendition.pagination_ok:
        logger.warning(
            "paginación distinta en %s: PDF %d págs., DOCX %d",
            source,
            rendition.pages,
            rendition.expected_pages,
        )
    return rendition
