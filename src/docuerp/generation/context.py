"""Armado del contexto compacto con presupuesto de tokens.

Meta del proyecto: ≤ 4 000 tokens por consulta. Aquí se acota el prompt **propio** (sistema
+ tarea + pregunta + contexto + resultado de herramienta); el CLI del Agent SDK añade
además ~1 400 tokens fijos que no podemos controlar (ver ``docs/research/agent_sdk.md``).

No se carga el tokenizador de bge-m3 (es pesado y distinto del de Claude): se estima con
``caracteres / 3.5``, una aproximación conservadora para español e inglés técnico.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from docuerp.schemas import ScoredChunk

DEFAULT_TOKEN_BUDGET = 4000
"""Presupuesto total del prompt propio, en tokens estimados."""

CHARS_PER_TOKEN = 3.5
"""Relación caracteres/token usada por :func:`estimate_tokens`."""

MIN_FIRST_CHUNK_TOKENS = 150
"""Aunque el resto del prompt consuma casi todo el presupuesto, el mejor chunk se incluye
(recortado) con al menos este espacio: sin evidencia no hay respuesta posible."""

TRUNCATION_MARK = " …[recortado]"


def estimate_tokens(text: str) -> int:
    """Estima los tokens de ``text`` (``ceil(len / 3.5)``); 0 para texto vacío."""
    return math.ceil(len(text) / CHARS_PER_TOKEN) if text else 0


@dataclass
class ContextSelection:
    """Resultado de recortar los chunks al presupuesto.

    Attributes:
        hits: chunks que entran al prompt, en orden del reranker (el último puede ir
            recortado).
        dropped: ``chunk_id`` descartados por falta de presupuesto.
        truncated: ``chunk_id`` cuyo texto se recortó.
        used_tokens: estimación de tokens que ocupan los chunks incluidos.
    """

    hits: list[ScoredChunk] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    truncated: list[str] = field(default_factory=list)
    used_tokens: int = 0


def _ordered_unique(hits: list[ScoredChunk]) -> list[ScoredChunk]:
    """Ordena por ``rank`` (mejor primero, estable) y quita ``chunk_id`` repetidos."""
    seen: set[str] = set()
    ordered = []
    for hit in sorted(hits, key=lambda h: h.rank):
        if hit.chunk.chunk_id not in seen:
            seen.add(hit.chunk.chunk_id)
            ordered.append(hit)
    return ordered


def _truncate(hit: ScoredChunk, max_tokens: int, overhead_tokens: int) -> ScoredChunk:
    """Copia de ``hit`` con el texto recortado para que el fragmento quepa en ``max_tokens``."""
    allowed_chars = max(
        0, int((max_tokens - overhead_tokens) * CHARS_PER_TOKEN) - len(TRUNCATION_MARK)
    )
    text = hit.chunk.text[:allowed_chars].rstrip() + TRUNCATION_MARK
    return hit.model_copy(update={"chunk": hit.chunk.model_copy(update={"text": text})})


def select_context(
    hits: list[ScoredChunk],
    *,
    base_tokens: int,
    budget: int = DEFAULT_TOKEN_BUDGET,
    render_tokens: Callable[[ScoredChunk], int],
) -> ContextSelection:
    """Elige los chunks que caben en ``budget`` respetando el orden del reranker.

    Se agregan chunks de mejor a peor ranking mientras quepan. El primer chunk que no cabe
    corta la selección (no se saltan para meter uno peor, así el contexto conserva la
    prioridad del reranker); si el mejor chunk ya no cabe, se incluye recortado.

    Args:
        hits: candidatos del recuperador.
        base_tokens: tokens ya ocupados por sistema, tarea, pregunta y herramienta.
        budget: presupuesto total en tokens estimados.
        render_tokens: función que estima los tokens de un chunk ya renderizado como
            ``<fragmento>`` (incluye sus atributos).
    """
    selection = ContextSelection()
    remaining = max(budget - base_tokens, MIN_FIRST_CHUNK_TOKENS)
    ordered = _ordered_unique(hits)
    for index, hit in enumerate(ordered):
        cost = render_tokens(hit)
        if cost <= remaining:
            selection.hits.append(hit)
            selection.used_tokens += cost
            remaining -= cost
            continue
        first_dropped = index
        if index == 0:
            overhead = cost - estimate_tokens(hit.chunk.text)
            cut = _truncate(hit, remaining, overhead)
            selection.hits.append(cut)
            selection.truncated.append(hit.chunk.chunk_id)
            selection.used_tokens += render_tokens(cut)
            first_dropped = 1
        selection.dropped.extend(h.chunk.chunk_id for h in ordered[first_dropped:])
        break
    return selection
