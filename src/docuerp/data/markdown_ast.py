"""Analizador mínimo de Markdown para los runbooks sintéticos.

Los runbooks se escriben en un subconjunto pequeño y controlado de Markdown. En lugar de
depender de un convertidor genérico, se analiza el texto a un árbol de bloques propio que
luego se renderiza **dos veces** (HTML para el PDF y python-docx para el DOCX) con
exactamente el mismo contenido. Así ambos formatos nunca divergen.

Subconjunto soportado:

- Encabezados ``#`` a ``####``.
- Párrafos, listas con viñetas (``-``) y numeradas (``1.``) con un nivel de anidamiento.
- Tablas con barras verticales (primera fila = encabezado).
- Bloques de código con triple acento grave.
- Citas con ``>`` (se usan como avisos y notas).
- Notas al pie: referencia ``[^1]`` en el texto y definición ``[^1]: texto`` en su propia línea.
- Énfasis en línea: ``**negrita**``, ``*cursiva*``, ```código```.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import yaml

# ---------------------------------------------------------------------------
# Modelo de bloques
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """Fragmento de texto en línea con su formato."""

    text: str
    bold: bool = False
    italic: bool = False
    code: bool = False
    footnote_ref: bool = False  # ``text`` es la etiqueta de la nota (p. ej. "1")


@dataclass(frozen=True)
class Heading:
    """Encabezado de nivel 1 a 4."""

    level: int
    text: str


@dataclass(frozen=True)
class Paragraph:
    """Párrafo de texto corrido."""

    text: str


@dataclass
class ListItem:
    """Elemento de lista con sublistas opcionales."""

    text: str
    children: list[ListBlock] = field(default_factory=list)


@dataclass
class ListBlock:
    """Lista con viñetas (``ordered=False``) o numerada (``ordered=True``)."""

    ordered: bool
    items: list[ListItem] = field(default_factory=list)


@dataclass(frozen=True)
class Table:
    """Tabla: ``header`` es la primera fila y ``rows`` el resto."""

    header: list[str]
    rows: list[list[str]]


@dataclass(frozen=True)
class CodeBlock:
    """Bloque de código monoespaciado."""

    text: str


@dataclass(frozen=True)
class Quote:
    """Cita o aviso (se renderiza con barra lateral)."""

    text: str


@dataclass(frozen=True)
class Footnote:
    """Definición de nota al pie de una sección."""

    label: str
    text: str


Block = Heading | Paragraph | ListBlock | Table | CodeBlock | Quote | Footnote


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------

_FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def split_front_matter(source: str) -> tuple[dict[str, object], str]:
    """Separa el bloque YAML inicial (entre líneas ``---``) del cuerpo Markdown.

    Raises:
        ValueError: si el documento no comienza con un front matter YAML válido.
    """
    match = _FRONT_MATTER_RE.match(source)
    if match is None:
        raise ValueError("El documento no empieza con un bloque YAML '---' de metadatos")
    data = yaml.safe_load(match.group(1))
    if not isinstance(data, dict):
        raise ValueError("El front matter debe ser un mapeo YAML")
    return data, source[match.end() :]


# ---------------------------------------------------------------------------
# Análisis de bloques
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
_LIST_RE = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")
_TABLE_SEP_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_FOOTNOTE_RE = re.compile(r"^\[\^(\w+)\]:\s*(.+)$")


def _split_row(line: str) -> list[str]:
    """Divide una fila de tabla en celdas, sin las barras de los extremos."""
    cells = line.strip().strip("|").split("|")
    return [cell.strip() for cell in cells]


def _build_list(lines: list[tuple[int, bool, str]]) -> ListBlock:
    """Construye una lista (con un nivel de anidamiento) a partir de líneas ya clasificadas.

    Cada tupla es ``(sangría, ordenada, texto)``. Un elemento más sangrado que el anterior
    se agrega como sublista del último elemento de nivel superior.
    """
    base_indent = min(indent for indent, _, _ in lines)
    root: ListBlock | None = None
    current_item: ListItem | None = None
    sublist: ListBlock | None = None
    for indent, ordered, text in lines:
        if indent <= base_indent:
            if root is None:
                root = ListBlock(ordered=ordered)
            current_item = ListItem(text=text)
            root.items.append(current_item)
            sublist = None
        elif current_item is not None:
            if sublist is None:
                sublist = ListBlock(ordered=ordered)
                current_item.children.append(sublist)
            sublist.items.append(ListItem(text=text))
    assert root is not None  # la primera línea siempre abre la lista
    return root


def parse_blocks(body: str) -> list[Block]:
    """Convierte el cuerpo Markdown (sin front matter) en una lista de bloques."""
    lines = body.split("\n")
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        # Bloque de código cercado.
        if stripped.startswith("```"):
            i += 1
            code_lines: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # cierre del cercado
            blocks.append(CodeBlock("\n".join(code_lines)))
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            blocks.append(Heading(len(heading.group(1)), heading.group(2)))
            i += 1
            continue

        footnote = _FOOTNOTE_RE.match(stripped)
        if footnote:
            blocks.append(Footnote(footnote.group(1), footnote.group(2)))
            i += 1
            continue

        # Tabla: línea con barras seguida de la fila separadora.
        if (
            stripped.startswith("|")
            and i + 1 < len(lines)
            and _TABLE_SEP_RE.match(lines[i + 1].strip())
        ):
            header = _split_row(stripped)
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = _split_row(lines[i])
                if len(row) != len(header):
                    raise ValueError(
                        f"Fila de tabla con {len(row)} celdas (se esperaban {len(header)}): "
                        f"{lines[i]!r}"
                    )
                rows.append(row)
                i += 1
            blocks.append(Table(header, rows))
            continue

        # Cita.
        if stripped.startswith(">"):
            quote_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip().lstrip(">").strip())
                i += 1
            blocks.append(Quote(" ".join(part for part in quote_lines if part)))
            continue

        # Lista.
        if _LIST_RE.match(line):
            raw: list[tuple[int, bool, str]] = []
            while i < len(lines):
                m = _LIST_RE.match(lines[i])
                if m:
                    raw.append((len(m.group(1)), m.group(2)[0].isdigit(), m.group(3).strip()))
                    i += 1
                elif lines[i].strip() and lines[i].startswith(" ") and raw:
                    # Línea de continuación del elemento anterior.
                    indent, ordered, text = raw[-1]
                    raw[-1] = (indent, ordered, f"{text} {lines[i].strip()}")
                    i += 1
                else:
                    break
            blocks.append(_build_list(raw))
            continue

        # Párrafo: líneas consecutivas hasta una línea en blanco o inicio de otro bloque.
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i]
            nxt_stripped = nxt.strip()
            if (
                not nxt_stripped
                or _HEADING_RE.match(nxt)
                or _LIST_RE.match(nxt)
                or nxt_stripped.startswith(("|", ">", "```"))
                or _FOOTNOTE_RE.match(nxt_stripped)
            ):
                break
            para_lines.append(nxt_stripped)
            i += 1
        blocks.append(Paragraph(" ".join(para_lines)))
    return blocks


# ---------------------------------------------------------------------------
# Texto en línea
# ---------------------------------------------------------------------------

_INLINE_RE = re.compile(
    r"(?P<code>`[^`]+`)"
    r"|(?P<bold>\*\*(?!\s).+?(?<!\s)\*\*)"
    r"|(?P<italic>\*(?!\s)[^*]+?(?<!\s)\*)"
    r"|(?P<fn>\[\^\w+\])"
)


def parse_inline(text: str) -> list[Run]:
    """Divide ``text`` en fragmentos con negrita, cursiva, código y referencias a notas."""
    runs: list[Run] = []
    position = 0
    for match in _INLINE_RE.finditer(text):
        if match.start() > position:
            runs.append(Run(text[position : match.start()]))
        token = match.group(0)
        if match.group("code"):
            runs.append(Run(token[1:-1], code=True))
        elif match.group("bold"):
            # La negrita puede contener cursiva o código anidado; se reanaliza el interior.
            for inner in parse_inline(token[2:-2]):
                runs.append(
                    Run(
                        inner.text,
                        bold=True,
                        italic=inner.italic,
                        code=inner.code,
                        footnote_ref=inner.footnote_ref,
                    )
                )
        elif match.group("italic"):
            runs.append(Run(token[1:-1], italic=True))
        else:
            runs.append(Run(token[2:-1], footnote_ref=True))
        position = match.end()
    if position < len(text):
        runs.append(Run(text[position:]))
    return runs


def plain_text(text: str) -> str:
    """Devuelve ``text`` sin marcas de énfasis (útil para medir longitudes)."""
    return "".join(run.text for run in parse_inline(text))
