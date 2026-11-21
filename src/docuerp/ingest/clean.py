"""Limpieza de texto y escáner de instrucciones sospechosas (inyección indirecta).

Tres tareas, en este orden:

1. **Normalización** de cada bloque con ``docuerp.text.normalize_text`` (NFC, sin
   caracteres de ancho cero, guiones de corte unidos, espacios colapsados). En los bloques
   de código solo se aplica NFC y se quitan los caracteres invisibles: los espacios
   son significativos.
2. **Encabezados y pies repetidos**: un texto que aparece al inicio o al final de la mayoría
   de las páginas (p. ej. "Confidencial — Altamira" o "pág 3") es ruido de maquetación.
3. **Escáner de instrucciones sospechosas**: un documento del corpus podría contener texto
   dirigido al modelo ("ignora las instrucciones anteriores", "reveal your system
   prompt"). Es la vía de *inyección indirecta* del RAG: el texto llega al prompt como
   contexto recuperado. El escáner **no borra** el texto (el documento es la fuente
   citable), sino que marca el chunk con ``flags=["suspicious_instruction"]`` y lo lista
   en el reporte de ingesta; la recuperación y los guardas deciden qué hacer con él.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from docuerp.ingest.extract import ExtractedDocument, PageContent, TextBlock
from docuerp.text import normalize_text as _shared_normalize
from docuerp.text import strip_accents

SUSPICIOUS_FLAG = "suspicious_instruction"
"""Marca que se agrega a ``Chunk.flags`` cuando el escáner detecta una instrucción."""

REPEATED_MIN_PAGES = 3
"""Mínimo de páginas para intentar detectar encabezados/pies repetidos."""

REPEATED_MIN_RATIO = 0.5
"""Fracción mínima de páginas en las que debe repetirse un texto para considerarlo ruido."""

EDGE_BLOCKS = 2
"""Bloques del inicio y del final de cada página que se examinan como posibles repetidos."""

_ZERO_WIDTH_RE = re.compile("[​‌‍⁠﻿]")
_DIGITS_RE = re.compile(r"\d+")


def normalize_text(text: str) -> str:
    """NFC, espacios colapsados, guiones de corte de línea unidos, caracteres de control.

    Delega en ``docuerp.text.normalize_text`` (fuente única compartida); la rama de ingesta
    puede ampliar la limpieza específica de PDF en las demás funciones de este módulo.
    """
    return _shared_normalize(text)


def _normalize_code(text: str) -> str:
    """Normalización mínima para código: NFC, sin invisibles ni espacios finales de línea."""
    text = _ZERO_WIDTH_RE.sub("", unicodedata.normalize("NFC", text))
    return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")


def _edge_key(text: str) -> str:
    """Clave para comparar encabezados/pies: minúsculas y números reemplazados por ``#``."""
    return _DIGITS_RE.sub("#", " ".join(text.lower().split()))


def remove_repeated_headers(doc: ExtractedDocument) -> ExtractedDocument:
    """Elimina encabezados/pies de página repetidos en la mayoría de páginas.

    Solo se consideran los ``EDGE_BLOCKS`` primeros y últimos bloques de cada página, y
    nunca los encabezados de sección ni las migas de pan (que se repiten legítimamente).
    """
    pages = [p for p in doc.pages if p.blocks]
    if len(pages) < REPEATED_MIN_PAGES:
        return doc
    counts: Counter[str] = Counter()
    for page in pages:
        edge = {_edge_key(b.text) for b in _edge_blocks(page) if _removable(b)}
        counts.update(edge)
    threshold = max(REPEATED_MIN_PAGES, int(len(pages) * REPEATED_MIN_RATIO + 0.999))
    repeated = {key for key, n in counts.items() if n >= threshold and key}
    if not repeated:
        return doc
    new_pages: list[PageContent] = []
    for page in doc.pages:
        edge_ids = {id(b) for b in _edge_blocks(page)}
        kept = [
            b
            for b in page.blocks
            if not (id(b) in edge_ids and _removable(b) and _edge_key(b.text) in repeated)
        ]
        new_pages.append(page.model_copy(update={"blocks": kept}))
    return doc.model_copy(update={"pages": new_pages})


def _edge_blocks(page: PageContent) -> list[TextBlock]:
    """Primeros y últimos ``EDGE_BLOCKS`` bloques de una página (sin duplicar)."""
    blocks = page.blocks
    if len(blocks) <= 2 * EDGE_BLOCKS:
        return list(blocks)
    return [*blocks[:EDGE_BLOCKS], *blocks[-EDGE_BLOCKS:]]


def _removable(block: TextBlock) -> bool:
    """Los encabezados y migas de pan nunca se eliminan como 'repetidos'."""
    return block.kind not in ("heading", "breadcrumb") and len(block.text) <= 200


# ---------------------------------------------------------------------------
# Escáner de instrucciones sospechosas
# ---------------------------------------------------------------------------

_Q = (
    r"(?:all|any|the|your|my|these|those|previous|prior|above|earlier|preceding|system|"
    r"original|other)"
)
_REVEAL_EN = r"(?:reveal|show|print|display|output|repeat|leak|disclose|dump|tell\s+me)"
_REVEAL_ES = r"(?:revela|muestra|imprime|ensena|dime|repite|filtra|divulga|comparte)"

SUSPICIOUS_PATTERNS: dict[str, re.Pattern[str]] = {
    # --- Inglés ---
    "en_ignore_instructions": re.compile(
        rf"\b(?:ignore|disregard|bypass|override|skip)\s+(?:{_Q}\s+)+"
        r"(?:instructions?|prompts?|rules|directions|guidelines|directives|messages?)\b"
    ),
    "en_forget_instructions": re.compile(
        r"\bforget\s+(?:all\s+|everything\s+|your\s+|the\s+|previous\s+|prior\s+|above\s+)*"
        r"(?:instructions?|rules|prompts?|context|(?:that\s+)?you\s+(?:were|have\s+been)\s+told)\b"
        r"|\bforget\s+everything\b"
    ),
    "en_reveal_prompt": re.compile(
        rf"\b{_REVEAL_EN}\s+(?:me\s+)?(?:your|the)\s+"
        r"(?:(?:system|hidden|initial|original|secret|full)\s+)?(?:prompt|system\s+message)\b"
        rf"|\b{_REVEAL_EN}\s+(?:me\s+)?(?:your|the)\s+"
        r"(?:system|hidden|initial|original|secret)\s+instructions\b"
        rf"|\b{_REVEAL_EN}\s+(?:me\s+)?your\s+instructions\b"
    ),
    "en_system_prompt": re.compile(r"\bsystem\s+prompt\b"),
    "en_role_override": re.compile(
        r"\byou\s+are\s+now\s+(?:an?\s+|the\s+|in\s+)?\w+"
        r"|\bfrom\s+now\s+on,?\s+you\s+(?:are|will|must|should)\b"
        r"|\bact\s+as\s+(?:an?\s+|the\s+)?(?:unrestricted|unfiltered|admin(?:istrator)?|root|"
        r"superuser|system|developer|jailbroken|dan|different\s+(?:ai|assistant|model))\b"
        r"|\bpretend\s+(?:to\s+be|you\s+are)\s+(?:an?\s+|the\s+)?(?:admin|administrator|"
        r"root|system|unrestricted|different)\b"
    ),
    "en_jailbreak": re.compile(r"\bjailbr(?:eak|oken)\b|\bdo\s+anything\s+now\b|\bdan\s+mode\b"),
    "en_new_instructions": re.compile(
        r"\bnew\s+instructions?\s*:"
        r"|\binstructions?\s+(?:for|to)\s+(?:the\s+)?(?:ai|assistant|model|llm|chatbot|agent)\b"
        r"|\b(?:dear|attention)\s+(?:ai|assistant|llm|chatbot|language\s+model)\b"
    ),
    "en_disable_safety": re.compile(
        r"\b(?:disable|turn\s+off|bypass|ignore|remove)\s+(?:your|all)\s+(?:safety|security|"
        r"guardrails?|filters?|restrictions|rules|policies)\b"
    ),
    "en_exfiltrate": re.compile(
        r"\bexfiltrat\w*"
        r"|\b(?:send|forward|post|upload|email)\s+(?:all\s+|the\s+|any\s+|every\s+)*"
        r"(?:user\s+)?(?:data|credentials|passwords?|tokens?|api\s+keys?|secrets?|"
        r"conversation|chat\s+history)\s+to\s+(?:https?://|\S+@|this\s+|the\s+following|an?\s+)"
    ),
    # --- Marcado de chat usado para suplantar roles ---
    "chat_markup": re.compile(
        r"<\|?im_start\|?>|<\|?im_end\|?>|<\|system\|>|\[/?inst\]|<<\s*/?sys\s*>>"
        r"|(?:^|\n)\s*#{2,}\s*(?:system|instruction|assistant)\s*:?\s*(?:\n|$)"
        r"|(?:^|\n)\s*(?:system|assistant)\s*:\s*(?:ignore|you\s+are|ignora|eres)\b"
    ),
    # --- Español (texto sin tildes) ---
    "es_ignore_instructions": re.compile(
        r"\b(?:ignora|ignore|ignoren|ignorar|omite|omitir|desatiende|desobedece|saltate|salta|"
        r"pasa\s+por\s+alto)\s+(?:todas\s+|todos\s+|las\s+|los\s+|tus\s+|sus\s+|cualquier\s+|"
        r"estas\s+|esas\s+)*(?:instrucciones|indicaciones|reglas|ordenes|directrices|"
        r"restricciones|politicas)\b"
        r"|\bignora\s+(?:todo\s+)?lo\s+(?:anterior|que\s+te\s+(?:dijeron|indicaron))\b"
    ),
    "es_forget_instructions": re.compile(
        r"\bolvida\s+(?:todo\s+lo\s+anterior|todas?\s+|tus\s+|las\s+|sus\s+)*"
        r"(?:instrucciones|reglas|indicaciones|restricciones|lo\s+que\s+te\s+(?:dijeron|indicaron))\b"
        r"|\bolvida\s+todo\b"
    ),
    "es_reveal_prompt": re.compile(
        rf"\b{_REVEAL_ES}(?:me)?\s+(?:tu|tus|el|las|los)\s+"
        r"(?:prompt|instrucciones\s+(?:del\s+sistema|ocultas|iniciales|originales|secretas)|"
        r"mensaje\s+(?:del|de)\s+sistema|instrucciones\s+de\s+sistema)\b"
    ),
    "es_system_prompt": re.compile(
        r"\b(?:prompt|mensaje|instrucciones)\s+(?:del|de)\s+sistema\b|\bprompt\s+del\s+sistema\b"
    ),
    "es_role_override": re.compile(
        r"\b(?:actua|actuar|actuen|comportate|finge|simula|haz\s+de\s+cuenta\s+que\s+eres)\s+"
        r"(?:como|ser|que\s+eres)?\s*(?:un\s+|una\s+|el\s+|la\s+)?(?:administrador|admin|root|"
        r"superusuario|desarrollador|sistema|asistente\s+sin\s+restricciones|ia\s+sin\s+filtros)\b"
        r"|\ba\s+partir\s+de\s+ahora\s+(?:eres|seras|actuaras|responderas|debes|ignora)\b"
        r"|\bahora\s+eres\s+(?:un|una|el|la)\b"
    ),
    "es_new_instructions": re.compile(
        r"\bnuevas\s+instrucciones\s*:"
        r"|\binstrucciones\s+para\s+(?:la\s+|el\s+)?(?:ia|asistente|modelo|llm|bot|agente)\b"
        r"|\b(?:querida|querido|atencion)\s+(?:ia|asistente|modelo|bot)\b"
    ),
    "es_disable_safety": re.compile(
        r"\b(?:desactiva|deshabilita|apaga|elude|evade|salta|omite)\s+(?:tus\s+|todos\s+los\s+|"
        r"todas\s+las\s+|los\s+|las\s+)?(?:filtros|restricciones|controles\s+de\s+seguridad|"
        r"guardrails|reglas\s+de\s+seguridad|politicas\s+de\s+seguridad)\b"
        r"|\b(?:responde|actua|opera)\s+sin\s+(?:restricciones|filtros|limites)\b"
        r"|\bmodo\s+(?:sin\s+restricciones|dios|desarrollador\s+sin\s+filtros)\b"
    ),
    "es_exfiltrate": re.compile(
        r"\bexfiltra\w*"
        r"|\b(?:envia|enviar|manda|mandar|reenvia|sube|publica|filtra)\s+(?:todos\s+|todas\s+|"
        r"los\s+|las\s+|tus\s+|sus\s+)*(?:datos|credenciales|contrasenas|claves|tokens?|secretos|"
        r"conversacion|historial)\s+(?:del\s+usuario\s+|de\s+los\s+usuarios\s+)?(?:a|al|hacia|por)\s+"
    ),
}
"""Patrones de inyección (ES/EN). Se aplican sobre texto en minúsculas y sin tildes."""


def _scan_view(text: str) -> str:
    """Vista normalizada para el escáner: NFKC, sin invisibles, minúsculas, sin tildes.

    NFKC pliega variantes de compatibilidad (letras de ancho completo, ligaduras), una
    técnica común para esquivar filtros; quitar tildes hace que "actúa" y "actua" coincidan.
    """
    folded = unicodedata.normalize("NFKC", text)
    folded = _ZERO_WIDTH_RE.sub("", folded)
    folded = strip_accents(folded.lower())
    return re.sub(r"[ \t\r\f\v]+", " ", folded)


def scan_suspicious(text: str) -> list[str]:
    """Devuelve los patrones de inyección (ES/EN) encontrados en ``text`` (vacío si ninguno).

    Returns:
        Identificadores de patrón ordenados (p. ej. ``["en_ignore_instructions"]``).
    """
    if not text:
        return []
    view = _scan_view(text)
    return sorted(name for name, pattern in SUSPICIOUS_PATTERNS.items() if pattern.search(view))


def suspicious_excerpts(text: str, *, width: int = 60) -> list[tuple[str, str]]:
    """``[(patrón, fragmento)]`` con el contexto de cada coincidencia (para el reporte)."""
    view = _scan_view(text)
    found: list[tuple[str, str]] = []
    for name, pattern in SUSPICIOUS_PATTERNS.items():
        match = pattern.search(view)
        if match:
            start = max(0, match.start() - width)
            end = min(len(view), match.end() + width)
            found.append((name, " ".join(view[start:end].split())))
    return found


def clean_block(block: TextBlock) -> TextBlock | None:
    """Normaliza un bloque; devuelve ``None`` si queda vacío."""
    text = _normalize_code(block.text) if block.kind == "code" else normalize_text(block.text)
    if not text.strip():
        return None
    if text == block.text:
        return block
    return block.model_copy(update={"text": text})


def clean_document(doc: ExtractedDocument) -> tuple[ExtractedDocument, list[dict[str, str]]]:
    """Aplica toda la limpieza y devuelve el documento y los hallazgos del escáner.

    Returns:
        ``(documento_limpio, hallazgos)``; cada hallazgo es
        ``{"doc_id", "source", "page", "pattern", "excerpt"}``. Si hay hallazgos, el
        documento recibe la marca ``SUSPICIOUS_FLAG`` en ``flags``.
    """
    pages: list[PageContent] = []
    for page in doc.pages:
        blocks = [b for b in (clean_block(block) for block in page.blocks) if b is not None]
        pages.append(page.model_copy(update={"blocks": blocks}))
    cleaned = remove_repeated_headers(doc.model_copy(update={"pages": pages}))

    findings: list[dict[str, str]] = []
    for page in cleaned.pages:
        for name, excerpt in suspicious_excerpts(page.text):
            findings.append(
                {
                    "doc_id": doc.doc_id,
                    "source": doc.source,
                    "page": str(page.page),
                    "pattern": name,
                    "excerpt": excerpt,
                }
            )
    flags = list(cleaned.flags)
    if findings and SUSPICIOUS_FLAG not in flags:
        flags.append(SUSPICIOUS_FLAG)
    return cleaned.model_copy(update={"flags": flags}), findings
