"""Normalización de texto y tokenización léxica compartidas.

Este módulo es la única fuente de verdad para cómo se limpia y tokeniza el texto en
DocuERP. La ingesta (al construir el índice) y la recuperación (al consultar) deben usar
exactamente las mismas funciones; si no, BM25 compararía tokens incompatibles.
"""

from __future__ import annotations

import re
import unicodedata

# Identificadores que deben sobrevivir como un solo token: modelos técnicos de Odoo
# (stock.picking), tickets (TCK-00042), runbooks (RB-N2-003), códigos de error
# (ERR-INV-204), versiones (17.0) y rutas de menú con guion bajo (res_users).
_IDENTIFIER_RE = re.compile(
    r"[a-z0-9]+(?:[._\-/][a-z0-9]+)+",
)
_WORD_RE = re.compile(r"[a-z0-9]+")
_SOFT_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_MULTISPACE_RE = re.compile(r"[ \t ]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")
_ZERO_WIDTH_RE = re.compile(r"[​‌‍⁠﻿]")

# Stopwords mínimas en español e inglés: el corpus mezcla ambos idiomas (consultas en
# español sobre documentación mayormente en inglés). Se mantienen cortas a propósito
# para no borrar términos de dominio.
_STOPWORDS_RAW = """
a al algo algun alguna algunas alguno algunos ante antes como con contra cual cuales
cuando de del desde donde dos e el ella ellas ellos en entre era es esa esas ese eso
esos esta estas este esto estos fue fueron ha han hasta hay la las le les lo los mas
me mi mis muy ni no nos o os otra otras otro otros para pero por porque que quien se
sea ser si sin sobre son su sus tambien te tiene tienen todo todos tu tus un una unas
uno unos y ya yo hacer puedo debo
the of and to in is it that for on with as are be this by or from an at which can
you your if not will has have was were their there these those into then than so
such do does how what when where who why
"""
STOPWORDS: frozenset[str] = frozenset(_STOPWORDS_RAW.split())


def strip_accents(text: str) -> str:
    """Quita tildes y diacríticos (á→a, ñ→n) para que 'recepción' y 'recepcion' coincidan."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    """Normaliza texto extraído de PDF/DOCX sin cambiar su significado.

    - Unicode NFC (forma canónica compuesta).
    - Elimina caracteres de ancho cero (usados también para ocultar inyecciones).
    - Une palabras cortadas por guion al final de línea ("inven-\\ntario" → "inventario").
    - Colapsa espacios y saltos de línea repetidos.
    """
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _SOFT_HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _MULTISPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    return text.strip()


def bm25_tokenize(text: str) -> list[str]:
    """Tokeniza para BM25 conservando identificadores exactos.

    Devuelve primero los identificadores compuestos completos (p. ej. ``stock.picking``,
    ``tck-00042``) y luego las palabras sueltas sin stopwords. Así una consulta por el
    identificador exacto obtiene coincidencia fuerte, y una consulta por sus partes
    ("stock", "picking") también recupera el fragmento.
    """
    lowered = strip_accents(text.lower())
    tokens: list[str] = [m.group(0) for m in _IDENTIFIER_RE.finditer(lowered)]
    tokens.extend(
        word for word in _WORD_RE.findall(lowered) if word not in STOPWORDS and len(word) > 1
    )
    return tokens
