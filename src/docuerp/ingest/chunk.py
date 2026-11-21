"""Fragmentación: ``StructureChunker`` (respeta estructura) y ``FixedChunker`` (ablación).

Los tamaños se miden en **tokens del tokenizador de bge-m3** (el modelo de embeddings
principal), no en palabras: así sabemos que un chunk cabe en la ventana del modelo
(1024 tokens) y la comparación entre configuraciones es justa.

StructureChunker (configuración principal)
    1. Recorre los bloques en orden y mantiene la ruta de encabezados
       (miga de pan + h1…h4) → ``section_path`` = ``"Inventory > Product type > Goods"``.
    2. Dentro de cada sección agrupa los bloques en **unidades indivisibles**: una tabla
       completa (aunque cruce de página), un bloque de código completo, una lista numerada
       de pasos completa (con sus sub-viñetas, notas y figuras), una viñeta con su
       contenido anidado, o un párrafo. Un párrafo introductorio que termina en ``:`` (o un
       título corto como "See also") se pega a la unidad siguiente.
    3. Empaqueta unidades hasta ``target_tokens`` (450). Al pasar al siguiente chunk de la
       misma sección copia como **solape** (~15 %) las últimas oraciones del chunk anterior.
    4. Si una unidad indivisible excede el objetivo, va sola en su chunk mientras no supere
       ``max_tokens`` (1000, bajo el límite de 1024 de bge-m3); solo por encima de ese tope
       se divide (tablas por filas repitiendo el encabezado, listas por pasos, código por
       líneas, párrafos por oraciones).
    5. Fusiona chunks vecinos de la misma página RST cuando el anterior quedó a medio
       llenar (< 50 % del objetivo) o el actual es diminuto (< ``min_tokens``), sin pasar
       del objetivo; los sub-títulos fusionados quedan como líneas del texto.
    6. Antepone ``section_path`` en la primera línea del texto del chunk: el modelo de
       embeddings y BM25 "ven" el contexto jerárquico aunque el fragmento sea corto.

FixedChunker (línea base de ablación)
    Ventanas deslizantes de ``size`` tokens (256 o 768) con 15 % de solape sobre el texto
    corrido del documento, sin mirar la estructura.
"""

from __future__ import annotations

import bisect
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from docuerp.ingest.extract import NUMBERED_MARKER_RE, ExtractedDocument, TextBlock

log = logging.getLogger(__name__)

DEFAULT_TOKENIZER_ID = "BAAI/bge-m3"
"""Tokenizador por defecto (el de bge-m3); se puede cambiar en ``models.tokenizer``."""

SECTION_SEPARATOR = " > "
"""Separador de niveles en ``section_path``."""

MAX_TOKENS = 1000
"""Tope duro por chunk (bge-m3 admite 1024 tokens; se deja margen para los especiales)."""

UNDERFILL_RATIO = 0.5
"""Un chunk con menos de esta fracción del objetivo se considera "a medio llenar"."""

INDENT_EPSILON = 1.5
"""Tolerancia (pt) al comparar sangrías de bloques."""

_SENTENCE_END = (".", "!", "?", ":", ";", ")", '"', "»", "]")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?:;])\s+(?=[\"'(¿¡\[A-Z0-9•])")


class ChunkDraft(BaseModel):
    """Fragmento sin metadatos de gobernanza (``metadata.py`` lo convierte en ``Chunk``)."""

    model_config = ConfigDict(extra="forbid")

    text: str
    page: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_path: str = ""
    token_count: int = Field(ge=0)


class Chunker(Protocol):
    """Interfaz común de los fragmentadores."""

    name: str

    def chunk(self, doc: ExtractedDocument) -> list[ChunkDraft]:
        """Divide un documento limpio en fragmentos."""
        ...


# ---------------------------------------------------------------------------
# Tokenizadores
# ---------------------------------------------------------------------------


class TextTokenizer(Protocol):
    """Lo mínimo que necesitan los chunkers: contar tokens y obtener sus posiciones."""

    def count(self, text: str) -> int:
        """Número de tokens de ``text`` (sin tokens especiales)."""
        ...

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Posición ``(inicio, fin)`` en caracteres de cada token de ``text``."""
        ...


class HFTokenizer:
    """Tokenizador de Hugging Face (por defecto el de bge-m3). Solo el tokenizador, no el
    modelo: pesa pocos MB y se carga en ~1 s desde la caché local."""

    def __init__(self, model_id: str = DEFAULT_TOKENIZER_ID) -> None:
        from transformers import AutoTokenizer

        try:
            self._tok: Any = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        except OSError:
            log.info("Tokenizador %s no está en caché local; se descarga", model_id)
            self._tok = AutoTokenizer.from_pretrained(model_id)
        # Solo contamos tokens: textos largos no deben disparar el aviso de longitud máxima.
        self._tok.model_max_length = 10**9
        self.model_id = model_id

    def count(self, text: str) -> int:
        """Tokens de ``text`` sin ``<s>``/``</s>``."""
        if not text:
            return 0
        return len(self._tok(text, add_special_tokens=False)["input_ids"])

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Posiciones de cada token en caracteres (``return_offsets_mapping``)."""
        if not text:
            return []
        encoded = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [(int(a), int(b)) for a, b in encoded["offset_mapping"]]


class WhitespaceTokenizer:
    """Tokenizador trivial (palabras separadas por espacios) para pruebas deterministas."""

    _TOKEN_RE = re.compile(r"\S+")

    def count(self, text: str) -> int:
        """Número de palabras."""
        return len(self._TOKEN_RE.findall(text))

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Posición de cada palabra."""
        return [m.span() for m in self._TOKEN_RE.finditer(text)]


@lru_cache(maxsize=2)
def get_tokenizer(model_id: str | None = None) -> HFTokenizer:
    """Tokenizador compartido (singleton por ``model_id``)."""
    if model_id is None:
        from docuerp.config import get_settings

        model_id = get_settings().models.tokenizer or DEFAULT_TOKENIZER_ID
    return HFTokenizer(model_id)


def count_tokens(text: str) -> int:
    """Cuenta tokens con el tokenizador de bge-m3 (cargado de forma perezosa)."""
    return get_tokenizer().count(text)


# ---------------------------------------------------------------------------
# Estructura intermedia
# ---------------------------------------------------------------------------


@dataclass
class _Unit:
    """Unidad indivisible de contenido dentro de una sección."""

    text: str
    tokens: int
    page: int
    page_end: int
    kind: str  # paragraph | table | code | steps | bullet | figure
    atomic: bool
    glue_next: bool = False


@dataclass
class _Section:
    """Sección: ruta de encabezados + bloques en orden."""

    path: tuple[str, ...]
    blocks: list[TextBlock] = field(default_factory=list)
    h1_len: int = 0  # componentes de la ruta hasta el h1 inclusive (miga + título de página)


@dataclass
class _Piece:
    """Chunk en construcción: ruta, cuerpo y rango de páginas."""

    path: tuple[str, ...]
    parts: list[str]
    page: int
    page_end: int
    tokens: int
    h1_key: tuple[str, ...]

    @property
    def body(self) -> str:
        return "\n".join(self.parts)


def format_section_path(path: tuple[str, ...] | list[str]) -> str:
    """``("Inventory", "Product type")`` → ``"Inventory > Product type"``."""
    return SECTION_SEPARATOR.join(p for p in path if p)


def _split_sections(doc: ExtractedDocument) -> list[_Section]:
    """Divide el documento en secciones según los encabezados y la miga de pan.

    La miga de pan (``Inventory › Product management``) precede al h1 de cada página RST
    y aporta los niveles superiores de la ruta; un h1 sin miga empieza una ruta nueva.
    """
    sections: list[_Section] = []
    crumbs: tuple[str, ...] = ()
    pending_crumbs: tuple[str, ...] | None = None
    stack: dict[int, str] = {}
    current = _Section(path=())
    for page in doc.pages:
        for block in page.blocks:
            if block.kind == "breadcrumb":
                pending_crumbs = tuple(
                    part.strip() for part in re.split(r"\s*›\s*", block.text) if part.strip()
                )
                continue
            if block.kind == "heading" and block.level is not None:
                level = max(1, min(block.level, 4))
                if level == 1:
                    crumbs = pending_crumbs or ()
                    stack = {}
                pending_crumbs = None
                stack = {lvl: title for lvl, title in stack.items() if lvl < level}
                stack[level] = block.text.strip()
                path = crumbs + tuple(stack[lvl] for lvl in sorted(stack))
                if current.blocks or current.path:
                    sections.append(current)
                current = _Section(path=path, h1_len=len(crumbs) + (1 if 1 in stack else 0))
                continue
            current.blocks.append(block)
    sections.append(current)
    return [s for s in sections if s.blocks]


def _is_numbered(block: TextBlock) -> bool:
    return block.kind == "list_item" and NUMBERED_MARKER_RE.match(block.text) is not None


def _is_title_like(block: TextBlock) -> bool:
    """Párrafo que introduce lo que sigue: termina en ``:`` o es un título corto."""
    if block.kind != "paragraph":
        return False
    text = block.text.strip()
    if text.endswith(":"):
        return True
    return len(text.split()) <= 3 and not text.endswith((".", "!", "?"))


def _continues_on_next_page(last: TextBlock, nxt: TextBlock) -> bool:
    """``True`` si ``nxt`` es la continuación, en la página siguiente, del párrafo ``last``
    (mismo tipo y sangría, y ``last`` no termina en puntuación de cierre)."""
    return (
        last.kind == "paragraph"
        and nxt.kind == "paragraph"
        and nxt.page == last.page + 1
        and abs(nxt.indent - last.indent) <= INDENT_EPSILON
        and not last.text.rstrip().endswith(_SENTENCE_END)
    )


def _dedupe_table_header(first: str, continuation: str) -> str:
    """Quita de la continuación de una tabla la fila de encabezado repetida por Chromium."""
    head = first.split("\n", 1)[0]
    lines = continuation.split("\n")
    if lines and lines[0] == head:
        lines = lines[1:]
    return "\n".join(lines)


class StructureChunker:
    """Respeta encabezados; no corta tablas, bloques de código ni pasos numerados; antepone
    ``section_path``; ~``target_tokens`` tokens con ``overlap`` de solapamiento."""

    name = "structure"

    def __init__(
        self,
        target_tokens: int = 450,
        overlap: float = 0.15,
        *,
        min_tokens: int = 40,
        max_tokens: int = MAX_TOKENS,
        tokenizer: TextTokenizer | None = None,
    ) -> None:
        if target_tokens < 16:
            raise ValueError("target_tokens debe ser ≥ 16")
        if not 0.0 <= overlap < 0.5:
            raise ValueError("overlap debe estar en [0, 0.5)")
        if max_tokens < target_tokens:
            raise ValueError("max_tokens debe ser ≥ target_tokens")
        self.target_tokens = target_tokens
        self.overlap = overlap
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> TextTokenizer:
        """Tokenizador en uso (bge-m3 por defecto, cargado al primer uso)."""
        if self._tokenizer is None:
            self._tokenizer = get_tokenizer()
        return self._tokenizer

    # -- unidades ---------------------------------------------------------------

    def _make_unit(self, blocks: list[TextBlock], kind: str, atomic: bool) -> _Unit:
        if kind == "table":
            text = blocks[0].text
            for block in blocks[1:]:
                text += "\n" + _dedupe_table_header(blocks[0].text, block.text)
        elif kind == "code":
            text = "\n".join(b.text for b in blocks)
        else:
            text = "\n".join(b.text for b in blocks)
        return _Unit(
            text=text,
            tokens=self.tokenizer.count(text),
            page=min(b.page for b in blocks),
            page_end=max(b.page for b in blocks),
            kind=kind,
            atomic=atomic,
        )

    def _units(self, blocks: list[TextBlock]) -> list[_Unit]:
        """Agrupa los bloques de una sección en unidades indivisibles."""
        units: list[_Unit] = []
        i = 0
        while i < len(blocks):
            block = blocks[i]
            if block.kind in ("table", "code"):
                j = i + 1
                while j < len(blocks) and blocks[j].kind == block.kind:
                    j += 1
                units.append(self._make_unit(blocks[i:j], block.kind, atomic=True))
                i = j
                continue
            if block.kind == "list_item":
                numbered = _is_numbered(block)
                base = block.indent
                j = i + 1
                while j < len(blocks):
                    nxt = blocks[j]
                    if nxt.kind == "heading":
                        break
                    nested = nxt.indent > base + INDENT_EPSILON
                    continuation = nxt.kind != "list_item" and nxt.indent >= base - INDENT_EPSILON
                    sibling_step = (
                        numbered and _is_numbered(nxt) and abs(nxt.indent - base) <= INDENT_EPSILON
                    )
                    if nested or continuation or sibling_step:
                        j += 1
                        continue
                    break
                kind = "steps" if numbered else "bullet"
                units.append(self._make_unit(blocks[i:j], kind, atomic=numbered))
                i = j
                continue
            # Un párrafo cortado por el salto de página continúa en la página siguiente:
            # se reúne en una sola unidad (unido con espacio, como el texto original).
            group = [block]
            while i + len(group) < len(blocks) and _continues_on_next_page(
                group[-1], blocks[i + len(group)]
            ):
                group.append(blocks[i + len(group)])
            unit = self._make_unit(
                group, "figure" if block.kind == "figure" else "paragraph", False
            )
            if len(group) > 1:
                unit.text = " ".join(b.text for b in group)
                unit.tokens = self.tokenizer.count(unit.text)
            unit.glue_next = _is_title_like(group[-1])
            units.append(unit)
            i += len(group)
        return self._glue(units)

    def _glue(self, units: list[_Unit]) -> list[_Unit]:
        """Pega cada unidad introductoria (``…:``/título corto) a la siguiente."""
        glued: list[_Unit] = []
        pending: _Unit | None = None
        for unit in units:
            if pending is not None:
                if pending.tokens + unit.tokens <= self.max_tokens:
                    text = pending.text + "\n" + unit.text
                    unit = _Unit(
                        text=text,
                        tokens=pending.tokens + unit.tokens + 1,
                        page=min(pending.page, unit.page),
                        page_end=max(pending.page_end, unit.page_end),
                        kind=unit.kind,
                        atomic=unit.atomic,
                        glue_next=unit.glue_next,
                    )
                else:
                    glued.append(pending)
                pending = None
            if unit.glue_next:
                pending = unit
                continue
            glued.append(unit)
        if pending is not None:
            glued.append(pending)
        return glued

    # -- división de unidades grandes ----------------------------------------

    def _split_text(self, text: str, budget: int, kind: str) -> list[str]:
        """Divide ``text`` en trozos ≤ ``budget`` tokens por la frontera natural del tipo."""
        if kind == "table":
            rows = text.split("\n")
            header, body = rows[0], rows[1:]
            pieces: list[str] = []
            current: list[str] = [header]
            for row in body:
                candidate = "\n".join([*current, row])
                if len(current) > 1 and self.tokenizer.count(candidate) > budget:
                    pieces.append("\n".join(current))
                    current = [header, row]
                else:
                    current.append(row)
            pieces.append("\n".join(current))
            return pieces
        if kind == "code":
            atoms = text.split("\n")
        elif kind == "steps":
            atoms = re.split(r"\n(?=(?:\d{1,3}|[a-z]|[ivxl]{1,5})[.)]\s)", text)
        else:
            atoms = [s for line in text.split("\n") for s in _SENTENCE_SPLIT_RE.split(line)]
        pieces = []
        current = []
        for atom in atoms:
            candidate = (
                "\n".join([*current, atom])
                if kind in ("code", "steps")
                else " ".join([*current, atom])
            )
            if current and self.tokenizer.count(candidate) > budget:
                pieces.append(("\n" if kind in ("code", "steps") else " ").join(current))
                current = [atom]
            else:
                current.append(atom)
        if current:
            pieces.append(("\n" if kind in ("code", "steps") else " ").join(current))
        # Un átomo más grande que el presupuesto (una oración enorme) se corta por tokens.
        out: list[str] = []
        for piece in pieces:
            out.extend(self._hard_split(piece, budget))
        return out

    def _hard_split(self, text: str, budget: int) -> list[str]:
        """Último recurso: corta por posiciones de token si el texto excede ``budget``."""
        if self.tokenizer.count(text) <= budget:
            return [text]
        offsets = self.tokenizer.offsets(text)
        out: list[str] = []
        for start in range(0, len(offsets), budget):
            window = offsets[start : start + budget]
            out.append(text[window[0][0] : window[-1][1]].strip())
        return [t for t in out if t]

    # -- empaquetado ------------------------------------------------------------

    def _overlap_tail(self, parts: list[str], units_kinds: list[str]) -> list[str]:
        """Últimas oraciones del chunk anterior (≤ ``overlap`` · objetivo) para el solape."""
        budget = int(self.target_tokens * self.overlap)
        if budget <= 0 or not parts:
            return []
        if units_kinds and units_kinds[-1] in ("table", "code", "steps"):
            return []  # nunca se duplica una tabla, código o lista de pasos a medias
        sentences = _SENTENCE_SPLIT_RE.split(parts[-1].split("\n")[-1])
        tail: list[str] = []
        used = 0
        for sentence in reversed(sentences):
            tokens = self.tokenizer.count(sentence)
            if used + tokens > budget:
                break
            tail.insert(0, sentence)
            used += tokens
        return [" ".join(tail)] if tail else []

    def _pack_section(self, section: _Section, h1_key: tuple[str, ...]) -> list[_Piece]:
        prefix_tokens = self.tokenizer.count(format_section_path(section.path) + "\n")
        budget = max(16, self.target_tokens - prefix_tokens)
        hard_budget = max(budget, self.max_tokens - prefix_tokens)
        units = self._units(section.blocks)
        pieces: list[_Piece] = []
        parts: list[str] = []
        kinds: list[str] = []
        tokens = 0
        page = page_end = 0

        def emit() -> None:
            nonlocal parts, kinds, tokens
            if parts:
                pieces.append(
                    _Piece(
                        path=section.path,
                        parts=list(parts),
                        page=page,
                        page_end=page_end,
                        tokens=tokens,
                        h1_key=h1_key,
                    )
                )
            parts, kinds, tokens = [], [], 0

        def add(text: str, n_tokens: int, unit: _Unit) -> None:
            nonlocal tokens, page, page_end
            if not parts:
                page, page_end = unit.page, unit.page_end
            else:
                page, page_end = min(page, unit.page), max(page_end, unit.page_end)
            parts.append(text)
            kinds.append(unit.kind)
            tokens += n_tokens + 1

        for unit in units:
            if tokens + unit.tokens <= budget:
                add(unit.text, unit.tokens, unit)
                continue
            # La unidad no cabe: se cierra el chunk actual y se abre otro con solape.
            tail = self._overlap_tail(parts, kinds) if parts else []
            last_unit_page = page_end
            emit()
            if unit.tokens <= budget or (unit.atomic and unit.tokens <= hard_budget):
                if tail:
                    tail_tokens = self.tokenizer.count(tail[0])
                    if tail_tokens + unit.tokens <= (hard_budget if unit.atomic else budget):
                        tail_unit = _Unit(
                            tail[0], tail_tokens, last_unit_page, last_unit_page, "overlap", False
                        )
                        add(tail[0], tail_tokens, tail_unit)
                add(unit.text, unit.tokens, unit)
                continue
            # Unidad demasiado grande: se divide por su frontera natural.
            for fragment in self._split_text(unit.text, budget, unit.kind):
                fragment_tokens = self.tokenizer.count(fragment)
                if parts and tokens + fragment_tokens > budget:
                    emit()
                add(fragment, fragment_tokens, unit)
        emit()
        return pieces

    def _merge_small(self, pieces: list[_Piece]) -> list[_Piece]:
        """Une chunks vecinos de la misma página RST cuando el anterior está "a medio llenar".

        Una sub-sección corta (p. ej. 80 tokens) produciría un chunk pobre en contexto. Si el
        chunk anterior tiene menos de ``UNDERFILL_RATIO`` · objetivo tokens (o el actual
        menos de ``min_tokens``) y juntos no superan el objetivo, se fusionan: la ruta pasa
        a ser el prefijo común y los sub-títulos quedan como líneas dentro del texto.
        """
        merged: list[_Piece] = []
        for piece in pieces:
            if merged:
                prev = merged[-1]
                same_page = prev.h1_key == piece.h1_key
                fits = prev.tokens + piece.tokens <= self.target_tokens
                underfilled = (
                    prev.tokens < self.target_tokens * UNDERFILL_RATIO
                    or piece.tokens < self.min_tokens
                )
                if same_page and fits and underfilled:
                    merged[-1] = _combine(prev, piece)
                    continue
            merged.append(piece)
        return merged

    def chunk(self, doc: ExtractedDocument) -> list[ChunkDraft]:
        """Divide ``doc`` siguiendo su estructura."""
        pieces: list[_Piece] = []
        for section in _split_sections(doc):
            # Clave de "página RST": la ruta hasta el h1 (miga + título de la página).
            h1_key = section.path[: section.h1_len] if section.h1_len else section.path
            pieces.extend(self._pack_section(section, h1_key))
        drafts: list[ChunkDraft] = []
        for piece in self._merge_small(pieces):
            path = format_section_path(piece.path)
            text = f"{path}\n{piece.body}" if path else piece.body
            n_tokens = self.tokenizer.count(text)
            if n_tokens > self.max_tokens:
                log.warning(
                    "%s: chunk de %d tokens en p.%d excede el tope",
                    doc.doc_id,
                    n_tokens,
                    piece.page,
                )
            drafts.append(
                ChunkDraft(
                    text=text,
                    page=piece.page,
                    page_end=max(piece.page, piece.page_end),
                    section_path=path,
                    token_count=n_tokens,
                )
            )
        return drafts


def _combine(first: _Piece, second: _Piece) -> _Piece:
    """Une dos chunks; la ruta resultante es el prefijo común y se conservan como líneas los
    títulos de los niveles que difieren (para no perder contexto)."""
    common: list[str] = []
    for a, b in zip(first.path, second.path, strict=False):
        if a != b:
            break
        common.append(a)
    prefix = tuple(common)

    def labelled(piece: _Piece) -> list[str]:
        extra = piece.path[len(prefix) :]
        return [format_section_path(extra), *piece.parts] if extra else list(piece.parts)

    return _Piece(
        path=prefix,
        parts=[*labelled(first), *labelled(second)],
        page=min(first.page, second.page),
        page_end=max(first.page_end, second.page_end),
        tokens=first.tokens + second.tokens + 2,
        h1_key=first.h1_key,
    )


# ---------------------------------------------------------------------------
# FixedChunker
# ---------------------------------------------------------------------------


class FixedChunker:
    """Ventanas de tamaño fijo en tokens (``fixed256``/``fixed768``) para ablación."""

    def __init__(
        self, size: int, overlap: float = 0.15, *, tokenizer: TextTokenizer | None = None
    ) -> None:
        if size < 8:
            raise ValueError("size debe ser ≥ 8 tokens")
        if not 0.0 <= overlap < 0.9:
            raise ValueError("overlap debe estar en [0, 0.9)")
        self.size = size
        self.overlap = overlap
        self.name = f"fixed{size}"
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> TextTokenizer:
        """Tokenizador en uso (bge-m3 por defecto)."""
        if self._tokenizer is None:
            self._tokenizer = get_tokenizer()
        return self._tokenizer

    def chunk(self, doc: ExtractedDocument) -> list[ChunkDraft]:
        """Divide ``doc`` en ventanas de ``size`` tokens."""
        # Texto corrido: bloques unidos por saltos de línea, con el mapa carácter → bloque.
        starts: list[int] = []
        meta: list[tuple[int, str]] = []
        pieces: list[str] = []
        cursor = 0
        crumbs: tuple[str, ...] = ()
        pending: tuple[str, ...] | None = None
        stack: dict[int, str] = {}
        for page in doc.pages:
            for block in page.blocks:
                if block.kind == "breadcrumb":
                    pending = tuple(
                        p.strip() for p in re.split(r"\s*›\s*", block.text) if p.strip()
                    )
                    continue
                if block.kind == "heading" and block.level is not None:
                    level = max(1, min(block.level, 4))
                    if level == 1:
                        crumbs, stack = pending or (), {}
                    pending = None
                    stack = {k: v for k, v in stack.items() if k < level}
                    stack[level] = block.text.strip()
                path = format_section_path(crumbs + tuple(stack[k] for k in sorted(stack)))
                starts.append(cursor)
                meta.append((block.page, path))
                pieces.append(block.text)
                cursor += len(block.text) + 1
        text = "\n".join(pieces)
        if not text.strip():
            return []
        offsets = self.tokenizer.offsets(text)
        step = max(1, self.size - int(self.size * self.overlap))
        drafts: list[ChunkDraft] = []
        start = 0
        while start < len(offsets):
            window = offsets[start : start + self.size]
            char_start, char_end = window[0][0], window[-1][1]
            chunk_text = text[char_start:char_end].strip()
            if chunk_text:
                first = meta[max(0, bisect.bisect_right(starts, char_start) - 1)]
                last = meta[max(0, bisect.bisect_right(starts, max(char_start, char_end - 1)) - 1)]
                drafts.append(
                    ChunkDraft(
                        text=chunk_text,
                        page=first[0],
                        page_end=max(first[0], last[0]),
                        section_path=first[1],
                        token_count=self.tokenizer.count(chunk_text),
                    )
                )
            if start + self.size >= len(offsets):
                break
            start += step
        return drafts


def get_chunker(name: str, *, tokenizer: TextTokenizer | None = None) -> Chunker:
    """Fábrica: ``structure`` | ``fixed256`` | ``fixed768`` (cualquier ``fixed<N>``).

    Los parámetros de ``structure`` salen de ``configs/settings.yaml`` (sección ``chunking``).

    Raises:
        ValueError: si el nombre no corresponde a ningún fragmentador.
    """
    from docuerp.config import get_settings

    chunking = get_settings().chunking
    if name == "structure":
        return StructureChunker(
            target_tokens=chunking.target_tokens,
            overlap=chunking.overlap,
            min_tokens=chunking.min_tokens,
            tokenizer=tokenizer,
        )
    match = re.fullmatch(r"fixed(\d+)", name)
    if match:
        return FixedChunker(int(match.group(1)), overlap=chunking.overlap, tokenizer=tokenizer)
    raise ValueError(f"Chunker desconocido: {name!r}. Opciones: structure, fixed256, fixed768")
