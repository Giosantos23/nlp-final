"""Proveedor ``offline``: respuesta extractiva determinista (sin LLM) con citas de los top
chunks. Se usa en degradación controlada (timeout/caída del LLM) y en CI.

Algoritmo (sin aleatoriedad, mismo prompt ⇒ misma respuesta):

1. Lee del prompt de usuario la pregunta y los ``<fragmento>`` (ver
   :func:`docuerp.generation.prompts.parse_user_prompt`).
2. Parte cada fragmento en oraciones y puntúa cada una por solapamiento léxico con la
   pregunta: suma del IDF de los términos de la pregunta que aparecen en la oración,
   normalizada por la raíz de su longitud, con un pequeño descuento según el ranking del
   fragmento (el reranker ya ordenó por relevancia).
3. Toma las mejores oraciones (máx. 4, máx. 2 por fragmento) y las muestra en el orden
   original con su cita ``[chunk_id]``. Si ninguna comparte al menos los términos mínimos
   con la pregunta, se abstiene con el mensaje fijo.

Los fragmentos con ``advertencia`` (posible instrucción maliciosa marcada en la ingesta) se
ignoran: el modo extractivo copiaría su texto literalmente.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from docuerp.generation.prompts import ParsedFragment, ParsedPrompt, parse_user_prompt
from docuerp.llm.contracts import LLMResult
from docuerp.schemas import ABSTENTION_MESSAGE
from docuerp.text import bm25_tokenize

MAX_SENTENCES = 4
MAX_SENTENCES_PER_CHUNK = 2
MIN_SENTENCE_CHARS = 25
MAX_SENTENCE_CHARS = 400
RANK_DISCOUNT = 0.1
"""Cada puesto de ranking resta este porcentaje al puntaje de sus oraciones."""

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+")

OFFLINE_MODEL = "offline-extractive"


@dataclass(frozen=True)
class _Sentence:
    """Oración candidata con su procedencia y puntaje."""

    text: str
    fragment_index: int
    position: int
    score: float


STEM_LENGTH = 5
"""Los términos se comparan por sus primeros 5 caracteres: un "stemming" mínimo y
determinista que une ``validar``/``valido`` o ``recepción``/``recepciones``."""


def _stems(text: str) -> set[str]:
    """Conjunto de términos normalizados (``bm25_tokenize``) truncados a ``STEM_LENGTH``."""
    return {term[:STEM_LENGTH] if term.isalpha() else term for term in bm25_tokenize(text)}


def _split_sentences(text: str) -> list[str]:
    """Parte un texto en oraciones útiles (líneas y puntuación), descartando las muy cortas."""
    sentences: list[str] = []
    for line in text.splitlines():
        if " > " in line and not line.rstrip().endswith((".", "!", "?", ":", ";")):
            continue  # encabezado "Módulo > Sección": lo antepone el chunker, no es contenido
        for piece in _SENTENCE_SPLIT_RE.split(line.strip()):
            piece = piece.strip(" -•*\t")
            if len(piece) >= MIN_SENTENCE_CHARS:
                sentences.append(piece[:MAX_SENTENCE_CHARS])
    return sentences


def _score_sentences(question: str, fragments: list[ParsedFragment]) -> list[_Sentence]:
    """Puntúa cada oración de los fragmentos por solapamiento léxico (IDF) con la pregunta."""
    query_terms = _stems(question)
    if not query_terms:
        return []
    pool: list[tuple[int, int, str, set[str]]] = []
    for f_index, fragment in enumerate(fragments):
        for position, sentence in enumerate(_split_sentences(fragment.text)):
            pool.append((f_index, position, sentence, _stems(sentence)))
    total = len(pool)
    doc_freq = {term: sum(term in terms for *_, terms in pool) for term in query_terms}
    min_overlap = 2 if len(query_terms) >= 2 else 1
    scored: list[_Sentence] = []
    for f_index, position, sentence, terms in pool:
        matched = query_terms & terms
        if len(matched) < min_overlap:
            continue
        idf_sum = sum(math.log(1 + total / (1 + doc_freq[term])) for term in matched)
        score = idf_sum / math.sqrt(len(terms)) * (1 - RANK_DISCOUNT * f_index)
        scored.append(_Sentence(sentence, f_index, position, score))
    return scored


def _select(scored: list[_Sentence]) -> list[_Sentence]:
    """Elige las mejores oraciones respetando el máximo total y por fragmento; devuelve el
    resultado en orden de aparición (fragmento, posición)."""
    chosen: list[_Sentence] = []
    per_fragment: dict[int, int] = {}
    # Desempate determinista: mayor puntaje, luego fragmento y posición más tempranos.
    for sentence in sorted(scored, key=lambda s: (-s.score, s.fragment_index, s.position)):
        if len(chosen) >= MAX_SENTENCES:
            break
        if per_fragment.get(sentence.fragment_index, 0) >= MAX_SENTENCES_PER_CHUNK:
            continue
        per_fragment[sentence.fragment_index] = per_fragment.get(sentence.fragment_index, 0) + 1
        chosen.append(sentence)
    return sorted(chosen, key=lambda s: (s.fragment_index, s.position))


def _abstention(tool_used: str | None = None, review: bool = False) -> dict[str, Any]:
    """Objeto contractual de abstención (mensaje fijo, sin fuentes)."""
    return {
        "answer": ABSTENTION_MESSAGE,
        "sources": [],
        "abstained": True,
        "tool_used": tool_used,
        "needs_human_review": review,
    }


def _flatten(data: dict[str, Any]) -> list[str]:
    """Convierte el resultado de una herramienta en líneas ``clave: valor`` (solo escalares y
    estructuras pequeñas), omitiendo las claves técnicas ``ok``/``tool``."""
    lines = []
    for key, value in data.items():
        if key in {"ok", "tool", "tool_name", "name"} or value in (None, "", [], {}):
            continue
        rendered = (
            value
            if isinstance(value, str | int | float | bool)
            else json.dumps(value, ensure_ascii=False)
        )
        lines.append(f"- {key}: {rendered}")
    return lines


def _tool_answer(parsed: ParsedPrompt) -> dict[str, Any]:
    """Respuesta determinista a partir del resultado de una herramienta (sin documentos)."""
    data = parsed.tool_data or {}
    if not parsed.tool_ok or data.get("error"):
        return _abstention(parsed.tool_name, review=True)
    review = bool(data.get("needs_human_review")) or parsed.intent == "ticket_create"
    body = "\n".join(_flatten(data))
    prefix = f"Resultado de la herramienta {parsed.tool_name}:"
    if parsed.intent == "ticket_create":
        prefix = (
            "Se preparó un borrador de ticket que aún NO fue creado; requiere tu "
            f"confirmación ({parsed.tool_name}):"
        )
    return {
        "answer": f"{prefix}\n{body}" if body else prefix,
        "sources": [],
        "abstained": False,
        "tool_used": parsed.tool_name,
        "needs_human_review": review,
    }


def build_extractive_answer(parsed: ParsedPrompt) -> dict[str, Any]:
    """Construye el objeto contractual extractivo a partir de un prompt ya leído."""
    if parsed.tool_name is not None:
        return _tool_answer(parsed)
    usable = [f for f in parsed.fragments if not f.flagged]
    chosen = _select(_score_sentences(parsed.question, usable))
    if not chosen:
        return _abstention()
    lines = [
        f"- {sentence.text} [{usable[sentence.fragment_index].chunk_id}]" for sentence in chosen
    ]
    sources: list[dict[str, Any]] = []
    for sentence in chosen:
        fragment = usable[sentence.fragment_index]
        ref = {"source": fragment.source, "page": fragment.page, "chunk_id": fragment.chunk_id}
        if ref not in sources:
            sources.append(ref)
    answer = (
        "Según la documentación recuperada (extractos textuales, sin generación por un "
        "modelo):\n" + "\n".join(lines)
    )
    return {
        "answer": answer,
        "sources": sources,
        "abstained": False,
        "tool_used": None,
        "needs_human_review": False,
    }


class OfflineProvider:
    """Extrae oraciones de los chunks delimitados en ``<contexto>`` del prompt de usuario."""

    name = "offline"

    async def complete(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
        effort: str | None = None,
        timeout_s: float = 45.0,
    ) -> LLMResult:
        """Construye un objeto contractual extractivo a partir del contexto.

        Sin ``json_schema`` y sin un prompt con el formato de DocuERP (p. ej. ``llm ping``)
        devuelve un texto fijo que aclara que no hay LLM. Los argumentos de modelo,
        ``max_tokens``, ``effort`` y ``timeout_s`` se ignoran: no hay llamada externa.
        """
        parsed = parse_user_prompt(user)
        has_structure = bool(parsed.question) or parsed.tool_name is not None
        if not has_structure and json_schema is None:
            text = "Proveedor offline: no hay LLM; solo se responde con extractos del contexto."
            return self._result(text, None)
        payload = build_extractive_answer(parsed)
        return self._result(json.dumps(payload, ensure_ascii=False), payload)

    def _result(self, text: str, parsed: dict[str, Any] | None) -> LLMResult:
        """Empaqueta el resultado: sin tokens, sin costo y con ``calls=0`` (no hubo LLM)."""
        return LLMResult(
            text=text,
            parsed=parsed,
            model=OFFLINE_MODEL,
            provider=self.name,
            calls=0,
        )
