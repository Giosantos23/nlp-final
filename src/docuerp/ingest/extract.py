"""Extracción de texto estructurado desde PDF (PyMuPDF) y DOCX (python-docx).

Salida común: :class:`ExtractedDocument` = lista de páginas, cada una con bloques tipados
(``heading``, ``paragraph``, ``list_item``, ``table``, ``code``, ``figure``, ``breadcrumb``).

PDF
    Se lee cada página con ``page.get_text("dict")`` (bloques → líneas → *spans* con fuente,
    tamaño y estilo). Como todos los PDF del proyecto usan la misma hoja de estilo
    (``odoo_pdf.PDF_BASE_CSS``), el **tamaño de fuente identifica el tipo de texto**:
    h1 22 pt, h2 17 pt, h3 14 pt, h4 12 pt, cuerpo 10.5 pt, tablas 9.5 pt, código 9 pt
    monoespaciado, miga de pan 9 pt y pie 8 pt. Se omiten la portada (página 1 con el
    título de 28 pt) y el pie ``archivo · pág N``. Las tablas se reconstruyen fila a fila
    con los bordes finos que dibuja Chromium (filas entre bordes horizontales, celdas entre
    verticales) y se emiten como un solo bloque ``celda | celda``.

DOCX
    Convención de los runbooks: cada página termina con un salto de página explícito, de
    modo que ``página = 1 + saltos previos``. Los encabezados salen de los estilos
    ``Heading 1``–``Heading 3``; las listas, de los estilos ``List Number``/``List Bullet``.
"""

from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from docuerp.ingest.odoo_pdf import (
    BREADCRUMB_SIZE_PT,
    CODE_SIZE_PT,
    COVER_TITLE_SIZE_PT,
    FIGURE_PREFIX,
    HEADING_SIZES_PT,
    TABLE_SIZE_PT,
)

log = logging.getLogger(__name__)

BlockKind = Literal["heading", "paragraph", "list_item", "table", "code", "figure", "breadcrumb"]

SIZE_TOLERANCE_PT = 0.35
"""Diferencia máxima (pt) para considerar que un *span* tiene un tamaño del contrato."""

FOOTER_BAND_PT = 34.0
"""Altura (pt) de la franja inferior donde vive el pie de página."""

FOOTER_RE = re.compile(r"(·|\|)\s*p[áa]g(?:ina)?\.?\s*\d+\s*$|^\s*\d+\s*$", re.IGNORECASE)
"""Pie ``archivo · pág N`` (o solo el número de página)."""

NUMBERED_MARKER_RE = re.compile(r"^(?:\d{1,3}|[a-z]|[ivxl]{1,5})[.)]\s+", re.IGNORECASE)
"""Marcador de paso numerado escrito como texto (``1. ``, ``a. ``, ``iv. ``)."""

BULLET_MARKER_RE = re.compile(r"^[•◦▪‣\-–]\s+")
"""Marcador de viñeta escrito como texto."""

LIST_MARKER_RE = re.compile(r"^(?:(?:\d{1,3}|[a-z]|[ivxl]{1,5})[.)]|[•◦▪‣\-–])\s+", re.IGNORECASE)
"""Cualquier marcador de lista (numerado o viñeta) al inicio del texto."""


class TextBlock(BaseModel):
    """Bloque de texto de una página con su tipo y nivel de encabezado (si aplica)."""

    model_config = ConfigDict(extra="forbid")

    text: str
    kind: BlockKind = "paragraph"
    level: int | None = Field(default=None, description="Nivel de encabezado (1 = H1)")
    font_size: float | None = None
    page: int = Field(ge=1)
    indent: float = Field(
        default=0.0, ge=0.0, description="Sangría (pt) respecto al margen izquierdo del cuerpo"
    )


class PageContent(BaseModel):
    """Contenido de una página (1-indexed)."""

    model_config = ConfigDict(extra="forbid")

    page: int = Field(ge=1)
    blocks: list[TextBlock] = Field(default_factory=list)

    @property
    def text(self) -> str:
        """Texto plano de la página (bloques unidos por saltos de línea)."""
        return "\n".join(b.text for b in self.blocks)


class ExtractedDocument(BaseModel):
    """Documento extraído listo para limpieza y chunking."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source: str
    pages: list[PageContent]
    flags: list[str] = Field(default_factory=list)
    page_count: int = Field(default=0, ge=0, description="Páginas físicas del archivo")
    skipped_pages: list[int] = Field(
        default_factory=list, description="Páginas omitidas (portada) en numeración física"
    )


class ExtractionError(RuntimeError):
    """El archivo no se pudo abrir o no tiene un formato soportado."""


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


@dataclass
class _Line:
    """Línea de texto de un PDF con sus rasgos tipográficos dominantes."""

    text: str
    size: float
    bold: bool
    mono: bool
    x0: float
    y0: float
    y1: float
    block_no: int


def _near(value: float, target: float) -> bool:
    """``True`` si ``value`` está a menos de ``SIZE_TOLERANCE_PT`` de ``target``."""
    return abs(value - target) <= SIZE_TOLERANCE_PT


def heading_level_for(size: float, bold: bool) -> int | None:
    """Nivel de encabezado para un tamaño de fuente (``None`` si no es encabezado)."""
    if not bold:
        return None
    for level, target in HEADING_SIZES_PT.items():
        if _near(size, target):
            return level
    return None


def _is_mono(span: dict[str, Any]) -> bool:
    """Indica si un *span* usa fuente monoespaciada (bit 3 de ``flags`` o nombre de fuente)."""
    font = str(span.get("font", "")).lower()
    return bool(int(span.get("flags", 0)) & 8) or any(
        name in font for name in ("mono", "courier", "menlo", "consolas")
    )


def _line_from_spans(spans: list[dict[str, Any]], block_no: int) -> _Line | None:
    """Combina los *spans* de una línea; el tamaño dominante es el de más caracteres."""
    text = "".join(str(s.get("text", "")) for s in spans)
    if not text.strip():
        return None
    weights: dict[float, int] = {}
    for span in spans:
        size = round(float(span.get("size", 0.0)), 2)
        weights[size] = weights.get(size, 0) + len(str(span.get("text", "")).strip())
    size = max(weights, key=lambda k: weights[k])
    visible = [s for s in spans if str(s.get("text", "")).strip()]
    bold = all(
        int(s.get("flags", 0)) & 16 or "bold" in str(s.get("font", "")).lower() for s in visible
    )
    mono = all(_is_mono(s) for s in visible)
    x0 = min(float(s["bbox"][0]) for s in visible)
    y0 = min(float(s["bbox"][1]) for s in visible)
    y1 = max(float(s["bbox"][3]) for s in visible)
    return _Line(
        text=text, size=size, bold=bool(bold), mono=mono, x0=x0, y0=y0, y1=y1, block_no=block_no
    )


def _page_lines(page: Any) -> list[_Line]:
    """Líneas de texto de una página en orden de lectura de PyMuPDF."""
    lines: list[_Line] = []
    data = page.get_text("dict", sort=False)
    for block_no, block in enumerate(data.get("blocks", [])):
        if block.get("type") != 0:
            continue
        for raw_line in block.get("lines", []):
            line = _line_from_spans(raw_line.get("spans", []), block_no)
            if line is not None:
                lines.append(line)
    return lines


def is_cover_page(page: Any) -> bool:
    """Una portada contiene el título de 28 pt definido en el contrato tipográfico."""
    data = page.get_text("dict", sort=False)
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if str(span.get("text", "")).strip() and _near(
                    float(span.get("size", 0.0)), COVER_TITLE_SIZE_PT
                ):
                    return True
    return False


def _is_footer(line: _Line, page_height: float) -> bool:
    """Pie de página: franja inferior y patrón ``· pág N`` (o número suelto)."""
    return line.y0 >= page_height - FOOTER_BAND_PT and bool(FOOTER_RE.search(line.text.strip()))


def _line_kind(line: _Line) -> tuple[BlockKind, int | None]:
    """Clasifica una línea por su tipografía (contrato de ``odoo_pdf.PDF_BASE_CSS``)."""
    level = heading_level_for(line.size, line.bold)
    if level is not None:
        return "heading", level
    if line.mono and _near(line.size, CODE_SIZE_PT):
        return "code", None
    if _near(line.size, BREADCRUMB_SIZE_PT) and not line.bold:
        return "breadcrumb", None
    if _near(line.size, TABLE_SIZE_PT):
        return "table", None
    return "paragraph", None


def join_lines(texts: list[str]) -> str:
    """Une líneas de un párrafo; si una termina en guion (``auto-``) no agrega espacio.

    Chromium no hace guionado automático: un guion al final de línea es parte de la
    palabra compuesta (``auto-populate``), así que se conserva y se pega a la siguiente.
    """
    out = ""
    for raw in texts:
        text = raw.strip()
        if not text:
            continue
        if not out:
            out = text
        elif out.endswith("-") and not out.endswith(" -"):
            out += text
        else:
            out += " " + text
    return out


def _cell_text(cell: str | None) -> str:
    """Texto de una celda de tabla en una sola línea."""
    return " ".join((cell or "").split())


def table_rows_to_text(rows: list[list[str | None]]) -> str:
    """Serializa filas de tabla como ``celda | celda | celda`` (una fila por línea).

    Se descartan las filas vacías y las columnas vacías en todas las filas (aparecen
    cuando la detección geométrica une dos tablas cercanas con bordes distintos).
    """
    texts = [[_cell_text(c) for c in row] for row in rows]
    width = max((len(r) for r in texts), default=0)
    keep = [i for i in range(width) if any(i < len(r) and r[i] for r in texts)]
    out: list[str] = []
    for row in texts:
        cells = [row[i] if i < len(row) else "" for i in keep]
        if any(cells):
            out.append(" | ".join(cells))
    return "\n".join(out)


def _fallback_table_text(lines: list[_Line]) -> str:
    """Reconstruye filas agrupando líneas de tabla por su coordenada vertical (sin bordes)."""
    rows: list[list[_Line]] = []
    for line in sorted(lines, key=lambda ln: (round(ln.y0), ln.x0)):
        if rows and abs(rows[-1][0].y0 - line.y0) < 2.0:
            rows[-1].append(line)
        else:
            rows.append([line])
    return "\n".join(
        " | ".join(ln.text.strip() for ln in sorted(row, key=lambda x: x.x0)) for row in rows
    )


RULE_MAX_THICKNESS_PT = 1.6
"""Grosor máximo (pt) de un rectángulo para considerarlo un borde de celda."""


@dataclass
class _Rules:
    """Bordes de tabla de una página: líneas horizontales (y) y verticales (x, y0, y1)."""

    horizontal: list[tuple[float, float, float]]  # (y, x0, x1)
    vertical: list[tuple[float, float, float]]  # (x, y0, y1)


def _page_rules(page: Any) -> _Rules:
    """Bordes finos dibujados por Chromium (rectángulos rellenos de ≤ 1.6 pt de grosor)."""
    horizontal: list[tuple[float, float, float]] = []
    vertical: list[tuple[float, float, float]] = []
    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is None:
            continue
        width, height = float(rect.width), float(rect.height)
        if height <= RULE_MAX_THICKNESS_PT and width >= 8.0:
            horizontal.append(
                ((float(rect.y0) + float(rect.y1)) / 2, float(rect.x0), float(rect.x1))
            )
        elif width <= RULE_MAX_THICKNESS_PT and height >= 4.0:
            vertical.append(((float(rect.x0) + float(rect.x1)) / 2, float(rect.y0), float(rect.y1)))
    return _Rules(horizontal=horizontal, vertical=vertical)


def _dedupe_positions(values: list[float], tolerance: float = 2.0) -> list[float]:
    """Ordena y une posiciones a menos de ``tolerance`` pt (bordes dobles de celdas vecinas)."""
    out: list[float] = []
    for value in sorted(values):
        if not out or value - out[-1] > tolerance:
            out.append(value)
    return out


def _ruled_table_text(lines: list[_Line], rules: _Rules) -> str | None:
    """Reconstruye una tabla usando sus bordes: filas entre líneas horizontales y celdas
    entre las líneas verticales que cruzan cada fila (admite celdas combinadas).

    Devuelve ``None`` si no hay bordes suficientes (se usa entonces el método por
    coordenadas de :func:`_fallback_table_text`).
    """
    # Margen vertical: el borde superior/inferior queda a ~padding + interlineado del texto.
    top = min(ln.y0 for ln in lines) - 10.0
    bottom = max(ln.y1 for ln in lines) + 10.0
    left = min(ln.x0 for ln in lines) - 8.0
    right = max(ln.x0 for ln in lines) + 8.0
    ys = _dedupe_positions(
        [y for y, x0, x1 in rules.horizontal if top <= y <= bottom and x1 > left and x0 < right]
    )
    if len(ys) < 2:
        return None
    rows: list[list[str | None]] = []
    for band_top, band_bottom in itertools.pairwise(ys):
        in_band = [ln for ln in lines if band_top <= (ln.y0 + ln.y1) / 2 <= band_bottom]
        if not in_band:
            continue
        xs = _dedupe_positions(
            [x for x, y0, y1 in rules.vertical if y0 < band_bottom - 1 and y1 > band_top + 1]
        )
        if len(xs) < 2:
            rows.append([join_lines([ln.text for ln in in_band])])
            continue
        cells: list[str | None] = []
        for cell_left, cell_right in itertools.pairwise(xs):
            cell_lines = sorted(
                (ln for ln in in_band if cell_left - 1 <= ln.x0 < cell_right),
                key=lambda ln: (ln.y0, ln.x0),
            )
            cells.append(join_lines([ln.text for ln in cell_lines]))
        rows.append(cells)
    text = table_rows_to_text(rows)
    return text or None


def _page_blocks(page: Any, page_no: int, body_left: float) -> list[TextBlock]:
    """Convierte una página PDF en bloques tipados.

    Reglas de agrupación de líneas en bloques:

    - si cambia el tipo/nivel tipográfico, empieza un bloque nuevo;
    - PyMuPDF ya separa bloques por espacio vertical (párrafos, elementos de lista): se
      respeta, salvo en código, donde todas las líneas consecutivas forman un bloque;
    - una línea que empieza con marcador de lista (``1.``, ``•``) abre un elemento nuevo;
    - las líneas consecutivas con la fuente de tabla (9.5 pt) forman **un** bloque de
      tabla, reconstruido fila a fila con los bordes de las celdas.
    """
    height = float(page.rect.height)
    lines = [ln for ln in _page_lines(page) if not _is_footer(ln, height)]
    if not lines:
        return []
    rules = _page_rules(page) if any(_line_kind(ln)[0] == "table" for ln in lines) else None

    blocks: list[TextBlock] = []
    pending_table: list[_Line] = []
    current: list[_Line] = []
    current_kind: tuple[BlockKind, int | None] | None = None

    def indent_of(x0: float) -> float:
        return round(max(0.0, x0 - body_left), 1)

    def flush_table() -> None:
        if pending_table:
            text = _ruled_table_text(pending_table, rules) if rules is not None else None
            blocks.append(
                TextBlock(
                    text=text or _fallback_table_text(pending_table),
                    kind="table",
                    font_size=TABLE_SIZE_PT,
                    page=page_no,
                    indent=indent_of(min(ln.x0 for ln in pending_table)),
                )
            )
            pending_table.clear()

    def flush_current() -> None:
        nonlocal current_kind
        if current and current_kind is not None:
            kind, level = current_kind
            if kind == "code":
                text = "\n".join(ln.text.rstrip() for ln in current)
            else:
                text = join_lines([ln.text for ln in current])
            if kind == "paragraph":
                if text.startswith(FIGURE_PREFIX):
                    kind = "figure"
                elif LIST_MARKER_RE.match(text):
                    kind = "list_item"
            blocks.append(
                TextBlock(
                    text=text,
                    kind=kind,
                    level=level,
                    font_size=current[0].size,
                    page=page_no,
                    indent=indent_of(min(ln.x0 for ln in current)),
                )
            )
        current.clear()
        current_kind = None

    previous: _Line | None = None
    for line in lines:
        kind = _line_kind(line)
        if kind[0] == "table":
            flush_current()
            pending_table.append(line)
            previous = line
            continue
        flush_table()
        starts_new = (
            current_kind is None
            or kind != current_kind
            or (previous is not None and line.block_no != previous.block_no and kind[0] != "code")
            or (kind[0] == "paragraph" and LIST_MARKER_RE.match(line.text) is not None)
        )
        if starts_new:
            flush_current()
            current_kind = kind
        current.append(line)
        previous = line
    flush_current()
    flush_table()
    return blocks


def _document_body_left(pdf: Any) -> float:
    """Margen izquierdo del cuerpo: mínimo ``x0`` de los encabezados (o de todo el texto)."""
    heading_x: list[float] = []
    all_x: list[float] = []
    for index in range(pdf.page_count):
        for line in _page_lines(pdf[index]):
            all_x.append(line.x0)
            if heading_level_for(line.size, line.bold) is not None:
                heading_x.append(line.x0)
    if heading_x:
        return min(heading_x)
    return min(all_x) if all_x else 0.0


def extract_pdf(path: Path, doc_id: str, *, skip_cover: bool = True) -> ExtractedDocument:
    """Extrae bloques por página con ``get_text("dict")`` y detecta encabezados por tamaño.

    Args:
        path: ruta del PDF.
        doc_id: identificador del documento en el manifiesto.
        skip_cover: omite la página 1 si es portada (título de 28 pt).

    Raises:
        ExtractionError: si el archivo no existe o PyMuPDF no puede abrirlo.
    """
    import pymupdf

    if not path.is_file():
        raise ExtractionError(f"No existe el PDF: {path}")
    try:
        pdf = pymupdf.open(path)
    except Exception as exc:  # PyMuPDF lanza varios tipos según el daño del archivo
        raise ExtractionError(f"PyMuPDF no pudo abrir {path.name}: {exc}") from exc
    pages: list[PageContent] = []
    skipped: list[int] = []
    with pdf:
        body_left = _document_body_left(pdf)
        for index in range(pdf.page_count):
            page = pdf[index]
            page_no = index + 1
            if skip_cover and page_no == 1 and is_cover_page(page):
                skipped.append(page_no)
                continue
            pages.append(PageContent(page=page_no, blocks=_page_blocks(page, page_no, body_left)))
        page_count = int(pdf.page_count)
    return ExtractedDocument(
        doc_id=doc_id, source=path.name, pages=pages, page_count=page_count, skipped_pages=skipped
    )


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_HEADING_STYLE_RE = re.compile(r"^(?:heading|t[íi]tulo)\s*(\d)$", re.IGNORECASE)
_MONO_FONTS = ("courier", "consolas", "menlo", "mono")

DOCX_LIST_INDENT_PT = 18.0
"""Sangría sintética por nivel de lista en DOCX: los elementos de lista quedan más
sangrados que el cuerpo (0 pt), igual que en los PDF, para que el chunker sepa dónde
termina la lista."""


def _docx_heading_level(style_name: str) -> int | None:
    """Nivel de encabezado de un estilo de párrafo (``Heading 2`` → 2); máximo 4."""
    match = _HEADING_STYLE_RE.match(style_name.strip())
    if not match:
        return None
    return min(int(match.group(1)), 4)


def _paragraph_segments(paragraph: Any) -> list[tuple[str, int]]:
    """Divide un párrafo en segmentos de texto separados por saltos de página.

    Devuelve ``[(texto, saltos_antes_del_segmento), ...]`` relativos al inicio del párrafo,
    para que el texto que sigue a un ``<w:br w:type="page"/>`` cuente en la página siguiente.
    """
    segments: list[tuple[str, int]] = []
    breaks = 0
    buffer: list[str] = []
    for element in paragraph._p.iter():
        tag = element.tag
        if tag == f"{_W_NS}t":
            buffer.append(element.text or "")
        elif tag == f"{_W_NS}tab":
            buffer.append("\t")
        elif tag == f"{_W_NS}br":
            if element.get(f"{_W_NS}type") == "page":
                segments.append(("".join(buffer), breaks))
                buffer = []
                breaks += 1
            else:
                buffer.append("\n")
        elif tag == f"{_W_NS}lastRenderedPageBreak":
            continue
    segments.append(("".join(buffer), breaks))
    return segments


def _paragraph_is_mono(paragraph: Any) -> bool:
    """``True`` si todas las corridas con texto usan una fuente monoespaciada."""
    runs = [r for r in paragraph.runs if r.text.strip()]
    if not runs:
        return False
    for run in runs:
        name = (run.font.name or paragraph.style.font.name or "").lower()
        if not any(mono in name for mono in _MONO_FONTS):
            return False
    return True


def extract_docx(path: Path, doc_id: str, *, skip_cover: bool = True) -> ExtractedDocument:
    """Extrae párrafos, estilos de título y tablas de un DOCX con páginas por saltos explícitos.

    - ``página = 1 + saltos de página previos`` (convención de los runbooks generados).
    - ``pageBreakBefore`` en el párrafo también cuenta como salto.
    - Si ``skip_cover`` y la página 1 tiene un párrafo de estilo ``Title`` y existen más
      páginas, la página 1 se trata como portada y se omite (igual que en los PDF).

    Raises:
        ExtractionError: si el archivo no existe o no es un DOCX válido.
    """
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    if not path.is_file():
        raise ExtractionError(f"No existe el DOCX: {path}")
    try:
        document = docx.Document(str(path))
    except Exception as exc:  # python-docx lanza errores de zip/XML variados
        raise ExtractionError(f"python-docx no pudo abrir {path.name}: {exc}") from exc

    blocks: list[TextBlock] = []
    title_pages: set[int] = set()
    page = 1
    numbering: dict[str, int] = {}
    previous_was_list = False

    for child in document.element.body.iterchildren():
        if child.tag == f"{_W_NS}tbl":
            table = Table(child, document)
            rows = [[cell.text for cell in row.cells] for row in table.rows]
            text = table_rows_to_text(_dedupe_merged_cells(rows))
            if text:
                blocks.append(TextBlock(text=text, kind="table", page=page))
            previous_was_list = False
            continue
        if child.tag != f"{_W_NS}p":
            continue
        paragraph = Paragraph(child, document)
        ppr = child.find(f"{_W_NS}pPr")
        if ppr is not None and ppr.find(f"{_W_NS}pageBreakBefore") is not None and blocks:
            page += 1
        style = paragraph.style.name if paragraph.style is not None else ""
        level = _docx_heading_level(style)
        segments = _paragraph_segments(paragraph)
        for text, breaks_before in segments:
            segment_page = page + breaks_before
            clean = text.strip()
            if not clean:
                continue
            if style.lower() in ("title", "subtitle", "título", "subtítulo"):
                title_pages.add(segment_page)
            kind: BlockKind = "paragraph"
            indent = 0.0
            if level is not None:
                kind = "heading"
            elif _paragraph_is_mono(paragraph) or "code" in style.lower():
                kind = "code"
                clean = text.rstrip()
            elif clean.startswith(FIGURE_PREFIX):
                kind = "figure"
            elif "list number" in style.lower():
                kind = "list_item"
                numbering[style] = (numbering.get(style, 0) + 1) if previous_was_list else 1
                indent = DOCX_LIST_INDENT_PT * (_style_list_depth(style) + 1)
                if not NUMBERED_MARKER_RE.match(clean):
                    clean = f"{numbering[style]}. {clean}"
            elif "list" in style.lower() and "bullet" in style.lower():
                kind = "list_item"
                indent = DOCX_LIST_INDENT_PT * (_style_list_depth(style) + 1)
                if not BULLET_MARKER_RE.match(clean):
                    clean = f"• {clean}"
            elif LIST_MARKER_RE.match(clean):
                kind = "list_item"
                indent = DOCX_LIST_INDENT_PT
            blocks.append(
                TextBlock(
                    text=clean,
                    kind=kind,
                    level=level,
                    page=segment_page,
                    indent=indent,
                )
            )
        previous_was_list = "list" in style.lower()
        if not previous_was_list:
            numbering.clear()
        page += segments[-1][1]

    last_page = max((b.page for b in blocks), default=1)
    skipped: list[int] = []
    if skip_cover and 1 in title_pages and last_page > 1:
        skipped.append(1)
        blocks = [b for b in blocks if b.page != 1]
    pages: dict[int, list[TextBlock]] = {}
    for block in blocks:
        pages.setdefault(block.page, []).append(block)
    return ExtractedDocument(
        doc_id=doc_id,
        source=path.name,
        pages=[PageContent(page=p, blocks=pages[p]) for p in sorted(pages)],
        page_count=last_page,
        skipped_pages=skipped,
    )


def _style_list_depth(style: str) -> int:
    """Profundidad de lista desde el estilo (``List Number 2`` → 1; ``List Number`` → 0)."""
    match = re.search(r"(\d)\s*$", style)
    return max(0, int(match.group(1)) - 1) if match else 0


def _dedupe_merged_cells(rows: list[list[str]]) -> list[list[str | None]]:
    """python-docx repite el texto de celdas combinadas horizontalmente; se deja una sola."""
    out: list[list[str | None]] = []
    for row in rows:
        cleaned: list[str | None] = []
        for i, cell in enumerate(row):
            cleaned.append(None if i > 0 and cell == row[i - 1] else cell)
        out.append(cleaned)
    return out


# ---------------------------------------------------------------------------
# Despacho
# ---------------------------------------------------------------------------


def extract(path: Path, doc_id: str) -> ExtractedDocument:
    """Despacha a :func:`extract_pdf` o :func:`extract_docx` según la extensión.

    Raises:
        ExtractionError: si la extensión no es ``.pdf`` ni ``.docx``.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path, doc_id)
    if suffix == ".docx":
        return extract_docx(path, doc_id)
    raise ExtractionError(f"Formato no soportado para extracción: {path.name}")
