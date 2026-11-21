"""Renderizado de runbooks: bloques Markdown → HTML (para PDF) y → DOCX.

Convenciones tipográficas COMUNES del proyecto (las usa también la documentación de Odoo
convertida a PDF, de modo que el extractor detecta encabezados por tamaño de fuente):

==========  ========
Elemento    Tamaño
==========  ========
h1          22 pt
h2          17 pt
h3          14 pt
h4          12 pt
cuerpo      10,5 pt
==========  ========

Convención de páginas del DOCX: un DOCX no tiene una paginación fija hasta que se abre en
un procesador de texto. Para que "la página citada" sea determinista, cada sección
principal (encabezado ``##``) termina con un **salto de página explícito** y se valida en
la construcción que ninguna sección excede lo que cabe en una página. Entonces:
página 1 = portada, página 2 = primera sección, y así sucesivamente.
"""

from __future__ import annotations

import html
import io
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_BREAK, WD_LINE_SPACING, WD_PARAGRAPH_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.text.paragraph import Paragraph as DocxParagraph

from docuerp.data.markdown_ast import (
    Block,
    CodeBlock,
    Footnote,
    Heading,
    ListBlock,
    Paragraph,
    Quote,
    Table,
    parse_inline,
    plain_text,
)

DISCLAIMER = "Documento interno sintético — uso académico"
"""Leyenda obligatoria de la portada de todos los documentos del corpus sintético."""

HEADING_SIZES_PT: dict[int, float] = {1: 22.0, 2: 17.0, 3: 14.0, 4: 12.0}
"""Tamaño de fuente de h1…h4 (convención común del proyecto)."""
BODY_SIZE_PT = 10.5
"""Tamaño del cuerpo de texto."""

# Capacidad de una página de DOCX medida en "líneas de cuerpo" estimadas. La página Carta
# con márgenes de 2,2 cm admite ~44 líneas de 10,5 pt; se deja margen de seguridad porque
# la métrica real depende de la fuente instalada en el equipo que abra el documento.
DOCX_PAGE_CAPACITY_LINES = 40.0
_CHARS_PER_LINE = 92

CoverRows = list[tuple[str, str]]


# ---------------------------------------------------------------------------
# HTML (para PDF con Chromium)
# ---------------------------------------------------------------------------

_CSS = f"""
@page {{ size: Letter; margin: 22mm 20mm 24mm 20mm; }}
html {{ -webkit-print-color-adjust: exact; }}
body {{
  font-family: "Helvetica Neue", Helvetica, Arial, "Liberation Sans", sans-serif;
  font-size: {BODY_SIZE_PT}pt; line-height: 1.45; color: #111827; margin: 0;
}}
h1 {{ font-size: {HEADING_SIZES_PT[1]}pt; line-height: 1.2; margin: 0 0 14pt 0; color: #0b1220; }}
h2 {{ font-size: {HEADING_SIZES_PT[2]}pt; line-height: 1.25; margin: 20pt 0 8pt 0;
     color: #0b1220; border-bottom: 1px solid #cbd5e1; padding-bottom: 3pt; }}
h3 {{ font-size: {HEADING_SIZES_PT[3]}pt; line-height: 1.3; margin: 14pt 0 6pt 0; color: #0b1220; }}
h4 {{ font-size: {HEADING_SIZES_PT[4]}pt; line-height: 1.3; margin: 12pt 0 5pt 0; color: #0b1220; }}
h2, h3, h4 {{ break-after: avoid; }}
p {{ margin: 0 0 7pt 0; font-size: {BODY_SIZE_PT}pt; }}
ul, ol {{ margin: 0 0 8pt 0; padding-left: 20pt; }}
li {{ margin: 0 0 3pt 0; font-size: {BODY_SIZE_PT}pt; }}
table {{ border-collapse: collapse; width: 100%; margin: 4pt 0 10pt 0;
        font-size: {BODY_SIZE_PT}pt; }}
th, td {{ border: 1px solid #94a3b8; padding: 3.5pt 6pt; text-align: left; vertical-align: top;
         font-size: {BODY_SIZE_PT}pt; }}
th {{ background: #e2e8f0; font-weight: 700; }}
tr {{ break-inside: avoid; }}
code, pre {{ font-family: Menlo, "Courier New", monospace; font-size: {BODY_SIZE_PT}pt; }}
code {{ background: #f1f5f9; padding: 0 2pt; }}
pre {{ background: #f1f5f9; border: 1px solid #cbd5e1; padding: 6pt 8pt; white-space: pre-wrap;
      break-inside: avoid; margin: 4pt 0 10pt 0; }}
blockquote {{ margin: 6pt 0 10pt 0; padding: 5pt 10pt; border-left: 3pt solid #0f766e;
             background: #f0fdfa; font-size: {BODY_SIZE_PT}pt; break-inside: avoid; }}
blockquote p {{ margin: 0; }}
.footnote {{ margin: 8pt 0 0 0; padding-top: 4pt; border-top: 1px solid #94a3b8;
            font-size: {BODY_SIZE_PT}pt; font-style: italic; }}
sup {{ font-size: 70%; vertical-align: super; line-height: 0; }}
.cover {{ break-after: page; padding-top: 70mm; }}
.cover .code {{ font-size: {BODY_SIZE_PT}pt; font-weight: 700; letter-spacing: 1pt; color: #0f766e;
               margin-bottom: 8pt; }}
.cover table {{ width: 100%; margin-top: 18pt; }}
.cover th {{ width: 32%; }}
.cover .disclaimer {{ margin-top: 26pt; font-size: {BODY_SIZE_PT}pt; font-style: italic;
                     color: #475569; }}
.status-obsoleto {{ color: #b91c1c; font-weight: 700; }}
.status-vigente {{ color: #047857; font-weight: 700; }}
"""


def _runs_to_html(text: str) -> str:
    """Convierte texto con marcas en línea a HTML escapado."""
    out: list[str] = []
    for run in parse_inline(text):
        piece = html.escape(run.text)
        if run.footnote_ref:
            piece = f"<sup>{piece}</sup>"
        if run.code:
            piece = f"<code>{piece}</code>"
        if run.bold:
            piece = f"<strong>{piece}</strong>"
        if run.italic:
            piece = f"<em>{piece}</em>"
        out.append(piece)
    return "".join(out)


def _list_to_html(block: ListBlock) -> str:
    """Renderiza una lista (y sus sublistas) como ``<ul>``/``<ol>``."""
    tag = "ol" if block.ordered else "ul"
    items = []
    for item in block.items:
        children = "".join(_list_to_html(child) for child in item.children)
        items.append(f"<li>{_runs_to_html(item.text)}{children}</li>")
    return f"<{tag}>{''.join(items)}</{tag}>"


def _table_to_html(table: Table) -> str:
    """Renderiza una tabla con encabezado ``<th>``."""
    head = "".join(f"<th>{_runs_to_html(cell)}</th>" for cell in table.header)
    body = "".join(
        "<tr>" + "".join(f"<td>{_runs_to_html(cell)}</td>" for cell in row) + "</tr>"
        for row in table.rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def blocks_to_html(blocks: list[Block]) -> str:
    """Renderiza los bloques del cuerpo a HTML."""
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, Heading):
            parts.append(f"<h{block.level}>{_runs_to_html(block.text)}</h{block.level}>")
        elif isinstance(block, Paragraph):
            parts.append(f"<p>{_runs_to_html(block.text)}</p>")
        elif isinstance(block, ListBlock):
            parts.append(_list_to_html(block))
        elif isinstance(block, Table):
            parts.append(_table_to_html(block))
        elif isinstance(block, CodeBlock):
            parts.append(f"<pre>{html.escape(block.text)}</pre>")
        elif isinstance(block, Quote):
            parts.append(f"<blockquote><p>{_runs_to_html(block.text)}</p></blockquote>")
        elif isinstance(block, Footnote):
            label = html.escape(block.label)
            parts.append(
                f'<div class="footnote"><sup>{label}</sup> {_runs_to_html(block.text)}</div>'
            )
    return "\n".join(parts)


def build_html(
    title: str, code: str, status: str, cover_rows: CoverRows, body_blocks: list[Block]
) -> str:
    """Arma el HTML completo: portada (código, título, versión, estado, acceso, fecha) y cuerpo."""
    rows = []
    for label, value in cover_rows:
        cell = html.escape(value)
        if label == "Estado":
            cell = f'<span class="status-{html.escape(status)}">{cell}</span>'
        rows.append(f"<tr><th>{html.escape(label)}</th><td>{cell}</td></tr>")
    cover = (
        '<section class="cover">'
        f'<div class="code">{html.escape(code)}</div>'
        f"<h1>{html.escape(title)}</h1>"
        f"<table><tbody>{''.join(rows)}</tbody></table>"
        f'<p class="disclaimer">{DISCLAIMER}</p>'
        "</section>"
    )
    return (
        '<!doctype html><html lang="es"><head><meta charset="utf-8">'
        f"<title>{html.escape(code)} — {html.escape(title)}</title><style>{_CSS}</style></head>"
        f"<body>{cover}{blocks_to_html(body_blocks)}</body></html>"
    )


def footer_template(filename: str) -> str:
    """Pie de página de Chromium: ``nombre_archivo · pág N`` (Chromium exige tamaño en px)."""
    return (
        '<div style="width:100%;font-size:9px;font-family:Arial,sans-serif;color:#475569;'
        f'text-align:center;">{html.escape(filename)} · pág <span class="pageNumber"></span></div>'
    )


# ---------------------------------------------------------------------------
# Estimación de página para DOCX
# ---------------------------------------------------------------------------


def _text_lines(text: str, chars_per_line: int = _CHARS_PER_LINE) -> int:
    """Líneas que ocupa ``text`` (sin marcas) con ``chars_per_line`` caracteres por línea."""
    return max(1, math.ceil(len(plain_text(text)) / chars_per_line))


def _list_lines(block: ListBlock, depth: int = 0) -> float:
    """Líneas estimadas de una lista; las sublistas tienen menos ancho útil."""
    width = _CHARS_PER_LINE - 6 * (depth + 1)
    total = 0.0
    for item in block.items:
        total += _text_lines(item.text, width) + 0.25
        total += sum(_list_lines(child, depth + 1) for child in item.children)
    return total


def _table_lines(table: Table) -> float:
    """Líneas estimadas de una tabla: cada fila ocupa lo que su celda más larga."""
    columns = len(table.header)
    total = 0.0
    for row in [table.header, *table.rows]:
        # Se reparte el ancho útil por igual (estimación conservadora para columnas angostas).
        width = max(12, _CHARS_PER_LINE // columns - 3)
        total += max(_text_lines(cell, width) for cell in row) + 0.45
    return total + 0.8


def estimate_lines(block: Block) -> float:
    """Estimación del alto de un bloque en líneas de cuerpo (10,5 pt)."""
    if isinstance(block, Heading):
        return {1: 4.0, 2: 3.2, 3: 2.4, 4: 2.0}[block.level]
    if isinstance(block, Paragraph):
        return _text_lines(block.text) + 0.6
    if isinstance(block, ListBlock):
        return _list_lines(block) + 0.6
    if isinstance(block, Table):
        return _table_lines(block)
    if isinstance(block, CodeBlock):
        return sum(max(1, math.ceil(len(line) / 80)) for line in block.text.split("\n")) + 1.2
    if isinstance(block, Quote):
        return _text_lines(block.text, _CHARS_PER_LINE - 6) + 1.0
    return _text_lines(block.text) + 0.8  # Footnote


def split_sections(blocks: list[Block]) -> list[list[Block]]:
    """Agrupa los bloques en secciones principales; cada ``##`` inicia una sección nueva.

    El h1 (título) y cualquier texto previo al primer ``##`` forman una sección inicial
    solo si existen; los runbooks comienzan sus secciones con ``##`` justo después del título.
    """
    sections: list[list[Block]] = []
    for block in blocks:
        if isinstance(block, Heading) and block.level == 1:
            continue  # el título va en la portada
        if (isinstance(block, Heading) and block.level == 2) or not sections:
            sections.append([])
        sections[-1].append(block)
    return sections


def check_sections_fit(
    sections: list[list[Block]], capacity: float = DOCX_PAGE_CAPACITY_LINES
) -> list[float]:
    """Valida que cada sección quepa en una página y devuelve su estimación de líneas.

    Raises:
        ValueError: si alguna sección excede ``capacity`` (se rompería la regla
            "1 sección principal = 1 página" del DOCX).
    """
    sizes = [sum(estimate_lines(b) for b in section) for section in sections]
    for section, size in zip(sections, sizes, strict=True):
        if size > capacity:
            first = section[0]
            name = first.text if isinstance(first, Heading) else "(sin título)"
            raise ValueError(
                f"La sección «{name}» estima {size:.1f} líneas y no cabe en una página DOCX "
                f"(máximo {capacity:.0f}). Divídala en dos secciones principales."
            )
    return sizes


# ---------------------------------------------------------------------------
# DOCX (python-docx)
# ---------------------------------------------------------------------------

_FONT = "Calibri"
_MONO = "Courier New"
_DARK = RGBColor(0x0B, 0x12, 0x20)


def _set_style_font(
    style, size_pt: float, *, bold: bool = False, color: RGBColor | None = None
) -> None:
    """Fija fuente, tamaño y color de un estilo, eliminando las referencias a fuentes de tema.

    La plantilla de python-docx usa fuentes de tema (``asciiTheme``) en los títulos, que
    en Word tienen prioridad sobre ``w:ascii``; hay que quitarlas para que mande ``Calibri``.
    """
    style.font.name = _FONT
    style.font.size = Pt(size_pt)
    style.font.bold = bold
    if color is not None:
        style.font.color.rgb = color
    r_fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    for attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        key = qn(f"w:{attr}")
        if key in r_fonts.attrib:
            del r_fonts.attrib[key]
    r_fonts.set(qn("w:eastAsia"), _FONT)


def _shading(fill_hex: str):
    """Elemento ``w:shd`` con relleno sólido ``fill_hex``."""
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), fill_hex)
    return shading


def _shade_cell(cell, fill_hex: str) -> None:
    """Sombrea una celda; ``w:shd`` va después de ``w:tcW`` (ya existente), basta con agregarlo."""
    cell._tc.get_or_add_tcPr().append(_shading(fill_hex))


# Hijos de ``w:pPr`` que el esquema OOXML ordena DESPUÉS de ``w:shd``; Word es estricto con
# el orden, por eso el sombreado se inserta antes del primero que exista.
_PPR_AFTER_SHD = tuple(
    qn(f"w:{name}")
    for name in (
        "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap", "overflowPunct", "topLinePunct",
        "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd", "snapToGrid", "spacing", "ind",
        "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc", "textDirection",
        "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr", "sectPr",
        "pPrChange",
    )
)  # fmt: skip


def _shade_paragraph(paragraph: DocxParagraph, fill_hex: str) -> None:
    """Sombrea un párrafo insertando ``w:shd`` en la posición que exige el esquema."""
    p_pr = paragraph._p.get_or_add_pPr()
    shading = _shading(fill_hex)
    for child in p_pr:
        if child.tag in _PPR_AFTER_SHD:
            child.addprevious(shading)
            return
    p_pr.append(shading)


def _add_runs(paragraph: DocxParagraph, text: str, *, force_bold: bool = False) -> None:
    """Agrega al párrafo los fragmentos con su formato (negrita, cursiva, código, nota)."""
    for run in parse_inline(text):
        label = f"[{run.text}]" if run.footnote_ref else run.text
        docx_run = paragraph.add_run(label)
        docx_run.bold = run.bold or force_bold
        docx_run.italic = run.italic
        docx_run.font.size = Pt(BODY_SIZE_PT)
        if run.code:
            docx_run.font.name = _MONO
            docx_run.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:hAnsi"), _MONO)
        if run.footnote_ref:
            docx_run.font.superscript = True


def _add_list(document, block: ListBlock, depth: int = 0) -> None:
    """Agrega una lista como párrafos con sangría francesa.

    La numeración se escribe como texto (``1.``, ``2.``…) porque el estilo ``List Number``
    de python-docx comparte un único contador entre todas las listas del documento y
    continuaría la cuenta de una lista a la siguiente.
    """
    for index, item in enumerate(block.items, start=1):
        paragraph = document.add_paragraph()
        fmt = paragraph.paragraph_format
        fmt.left_indent = Cm(0.9 + 0.7 * depth)
        fmt.first_line_indent = Cm(-0.6)
        fmt.space_after = Pt(2)
        marker = f"{index}.\t" if block.ordered else "•\t"
        fmt.tab_stops.add_tab_stop(Cm(0.9 + 0.7 * depth))
        paragraph.add_run(marker).font.size = Pt(BODY_SIZE_PT)
        _add_runs(paragraph, item.text)
        for child in item.children:
            _add_list(document, child, depth + 1)


def _add_table(document, table: Table, *, header: bool = True) -> None:
    """Agrega una tabla con rejilla, encabezado sombreado y encabezado repetible."""
    columns = len(table.header)
    rows = [table.header, *table.rows]
    doc_table = document.add_table(rows=len(rows), cols=columns)
    doc_table.style = "Table Grid"
    doc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_index, row in enumerate(rows):
        for c_index, cell_text in enumerate(row):
            cell = doc_table.cell(r_index, c_index)
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(1)
            _add_runs(paragraph, cell_text, force_bold=header and r_index == 0)
            if header and r_index == 0:
                _shade_cell(cell, "E2E8F0")
    if header:
        tr_pr = doc_table.rows[0]._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement("w:tblHeader"))
    # Un párrafo mínimo después de la tabla separa el siguiente bloque.
    spacer = document.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)
    spacer.paragraph_format.line_spacing = Pt(4)


def _add_page_break(document) -> None:
    """Inserta un salto de página explícito en un párrafo mínimo (1 pt, sin espacios)."""
    paragraph = document.add_paragraph()
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(0)
    fmt.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    fmt.line_spacing = Pt(1)
    run = paragraph.add_run()
    run.font.size = Pt(1)
    run.add_break(WD_BREAK.PAGE)


def _add_page_number_footer(document, filename: str) -> None:
    """Pie ``nombre_archivo · pág N`` con el campo PAGE de Word.

    El campo se escribe como lo hace Word: ``begin``, instrucción, ``separate``, resultado
    provisional y ``end``, cada pieza en su propio *run*.
    """
    footer_paragraph = document.sections[0].footer.paragraphs[0]
    footer_paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    footer_paragraph.add_run(f"{filename} · pág ").font.size = Pt(9)

    def field_char(kind: str) -> None:
        run = footer_paragraph.add_run()
        run.font.size = Pt(9)
        element = OxmlElement("w:fldChar")
        element.set(qn("w:fldCharType"), kind)
        run._r.append(element)

    field_char("begin")
    instruction_run = footer_paragraph.add_run()
    instruction_run.font.size = Pt(9)
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    instruction_run._r.append(instruction)
    field_char("separate")
    footer_paragraph.add_run("1").font.size = Pt(9)  # resultado provisional; Word lo recalcula
    field_char("end")


def _fixed_zip(raw: bytes, stamp: tuple[int, int, int, int, int, int]) -> bytes:
    """Reescribe el ZIP del DOCX con marcas de tiempo fijas para que sea reproducible."""
    source = zipfile.ZipFile(io.BytesIO(raw))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as target:
        for name in sorted(source.namelist()):
            info = zipfile.ZipInfo(name, date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            target.writestr(info, source.read(name))
    return out.getvalue()


@dataclass(frozen=True)
class DocxPlan:
    """Resultado de :func:`build_docx`: bytes del archivo y páginas de cada sección."""

    data: bytes
    section_pages: list[int]  # página (1-indexed) en que inicia cada sección principal
    total_pages: int


def build_docx(
    *,
    title: str,
    code: str,
    filename: str,
    cover_rows: CoverRows,
    body_blocks: list[Block],
    created: datetime,
) -> DocxPlan:
    """Construye el DOCX con estilos Heading 1–3 y un salto de página tras cada sección.

    Estructura: página 1 = portada; luego una página por cada sección principal (``##``).
    Cada página (portada y secciones) termina con un salto de página explícito, que se
    escribe en un párrafo mínimo justo antes del contenido de la página siguiente; la
    última sección no lleva salto final (así no queda una página en blanco).

    Raises:
        ValueError: si una sección no cabe en una página o se usa un encabezado h4.
    """
    sections = split_sections(body_blocks)
    check_sections_fit(sections)
    for block in body_blocks:
        if isinstance(block, Heading) and block.level > 3:
            raise ValueError("Los DOCX solo admiten encabezados Heading 1 a Heading 3")

    document = Document()
    page = document.sections[0]
    page.page_width, page.page_height = Cm(21.59), Cm(27.94)  # Carta
    page.left_margin = page.right_margin = Cm(2.2)
    page.top_margin = page.bottom_margin = Cm(2.2)

    _set_style_font(document.styles["Normal"], BODY_SIZE_PT)
    document.styles["Normal"].paragraph_format.space_after = Pt(5)
    document.styles["Normal"].paragraph_format.line_spacing = 1.1
    for level in (1, 2, 3):
        style = document.styles[f"Heading {level}"]
        _set_style_font(style, HEADING_SIZES_PT[level], bold=True, color=_DARK)
        style.paragraph_format.space_before = Pt(14 if level > 1 else 0)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.keep_with_next = True

    # Portada (página 1).
    for _ in range(6):
        document.add_paragraph()
    label = document.add_paragraph()
    label_run = label.add_run(code)
    label_run.bold = True
    label_run.font.size = Pt(BODY_SIZE_PT)
    label_run.font.color.rgb = RGBColor(0x0F, 0x76, 0x6E)
    document.add_heading(title, level=1)
    _add_table(document, Table(["Campo", "Valor"], [[k, v] for k, v in cover_rows]))
    disclaimer = document.add_paragraph()
    disclaimer_run = disclaimer.add_run(DISCLAIMER)
    disclaimer_run.italic = True
    disclaimer_run.font.size = Pt(BODY_SIZE_PT)

    # Cuerpo: un salto de página explícito antes de cada sección, incluida la primera.
    section_pages: list[int] = []
    for number, section in enumerate(sections):
        _add_page_break(document)
        section_pages.append(number + 2)
        for block in section:
            if isinstance(block, Heading):
                document.add_heading(plain_text(block.text), level=block.level)
            elif isinstance(block, Paragraph):
                _add_runs(document.add_paragraph(), block.text)
            elif isinstance(block, ListBlock):
                _add_list(document, block)
            elif isinstance(block, Table):
                _add_table(document, block)
            elif isinstance(block, CodeBlock):
                for line in block.text.split("\n"):
                    code_paragraph = document.add_paragraph()
                    code_paragraph.paragraph_format.space_after = Pt(0)
                    _shade_paragraph(code_paragraph, "F1F5F9")
                    run = code_paragraph.add_run(line or " ")
                    run.font.name = _MONO
                    run.font.size = Pt(BODY_SIZE_PT)
            elif isinstance(block, Quote):
                quote = document.add_paragraph()
                quote.paragraph_format.left_indent = Cm(0.5)
                _shade_paragraph(quote, "F0FDFA")
                _add_runs(quote, block.text)
            elif isinstance(block, Footnote):
                note = document.add_paragraph()
                prefix = note.add_run(f"[{block.label}] ")
                prefix.font.size = Pt(BODY_SIZE_PT)
                prefix.italic = True
                _add_runs(note, block.text)
                for run in note.runs:
                    run.italic = True

    _add_page_number_footer(document, filename)

    core = document.core_properties
    core.title = f"{code} — {title}"
    core.author = "Equipo DocuERP"
    core.last_modified_by = "Equipo DocuERP"
    core.subject = DISCLAIMER
    core.created = core.modified = created
    core.revision = 1

    buffer = io.BytesIO()
    document.save(buffer)
    stamp = (created.year, created.month, created.day, 0, 0, 0)
    return DocxPlan(
        data=_fixed_zip(buffer.getvalue(), stamp),
        section_pages=section_pages,
        total_pages=len(sections) + 1,
    )
