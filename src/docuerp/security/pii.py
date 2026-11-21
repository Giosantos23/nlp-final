"""Enmascaramiento de PII: correos, teléfonos (+502 y genéricos), DPI guatemalteco
(13 dígitos), NIT, tarjetas (Luhn), IBAN (mod-97) y direcciones IP → ``[EMAIL]``,
``[TELEFONO]``, ``[DPI]``, ``[NIT]``, ``[TARJETA]``, ``[IBAN]``, ``[IP]``.

Diseño:

* Cada detector es una expresión regular **más** un validador opcional (Luhn para
  tarjetas, mod-97 para IBAN, departamento/municipio para DPI, rango de octetos para IP).
  El validador reduce falsos positivos: un número de 16 dígitos que no pasa Luhn no es
  una tarjeta y no se toca.
* Los detectores se aplican en un **orden fijo** (de lo más específico a lo más genérico):
  primero correos (contienen dígitos y puntos), luego IBAN, DPI, tarjetas, NIT, IP y al
  final teléfonos. Como los tokens de reemplazo (``[EMAIL]``…) no contienen dígitos, un
  detector posterior nunca vuelve a enmascarar lo que otro ya reemplazó.
* Se protegen identificadores del dominio que *parecen* números (``TCK-00042``,
  ``RB-N2-003``, versiones ``17.0`` o ``17.0.1.0.0``, fechas ``2026-11-17``): los patrones
  usan delimitadores que no los alcanzan.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

PII_TOKENS: dict[str, str] = {
    "email": "[EMAIL]",
    "phone": "[TELEFONO]",
    "dpi": "[DPI]",
    "nit": "[NIT]",
    "card": "[TARJETA]",
    "iban": "[IBAN]",
    "ip": "[IP]",
}
"""Token de reemplazo por tipo de PII."""


def luhn_valid(number: str) -> bool:
    """Verifica el dígito de control Luhn de un número de tarjeta.

    Algoritmo: recorriendo los dígitos de derecha a izquierda, se duplica uno sí y uno no;
    si el doble pasa de 9 se le resta 9; la suma total debe ser múltiplo de 10.
    Se ignoran espacios y guiones; cualquier otro carácter invalida el número.
    """
    digits = re.sub(r"[ -]", "", number)
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 1:  # cada segundo dígito desde la derecha se duplica
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def iban_valid(candidate: str) -> bool:
    """Valida un IBAN con el algoritmo ISO 13616 (mod-97 == 1).

    Se mueven los 4 primeros caracteres al final, cada letra se convierte en número
    (A=10 … Z=35) y el entero resultante módulo 97 debe ser 1.
    """
    compact = re.sub(r"\s", "", candidate).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", compact):
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


def dpi_valid(candidate: str) -> bool:
    """Valida la estructura de un DPI/CUI guatemalteco de 13 dígitos.

    Formato: 8 dígitos de correlativo + 1 verificador + 2 de departamento (01–22) + 2 de
    municipio (≥ 01). Solo se valida departamento y municipio: es suficiente para descartar
    números arbitrarios de 13 dígitos sin depender de un algoritmo de verificación no
    documentado oficialmente.
    """
    digits = re.sub(r"\s", "", candidate)
    if not (digits.isdigit() and len(digits) == 13):
        return False
    department = int(digits[9:11])
    municipality = int(digits[11:13])
    return 1 <= department <= 22 and municipality >= 1


def _always(_: str) -> bool:
    """Validador neutro para patrones que no requieren verificación adicional."""
    return True


@dataclass(frozen=True)
class _Detector:
    """Un tipo de PII: nombre, expresión regular y validador del texto coincidente."""

    kind: str
    pattern: re.Pattern[str]
    validator: Callable[[str], bool] = _always


# Octeto IPv4 0–255 (sin ceros a la izquierda ambiguos más allá de 3 dígitos).
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"

_DETECTORS: tuple[_Detector, ...] = (
    _Detector(
        "email",
        re.compile(
            r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
        ),
    ),
    _Detector(
        "iban",
        # 2 letras de país + 2 dígitos de control + 11..30 alfanuméricos, en grupos de 4
        # opcionalmente separados por espacio (formato impreso habitual).
        re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b"),
        iban_valid,
    ),
    _Detector(
        "dpi",
        # 13 dígitos contiguos o en grupos 4-5-4 separados por espacio (como en el carné).
        re.compile(r"(?<![\d-])\d{4} ?\d{5} ?\d{4}(?![\d-])"),
        dpi_valid,
    ),
    _Detector(
        "card",
        # 13..19 dígitos, opcionalmente separados por espacio o guion; Luhn decide.
        re.compile(r"(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])"),
        luhn_valid,
    ),
    _Detector(
        "nit",
        # NIT guatemalteco: 6..10 dígitos + guion + dígito o K (p. ej. 1234567-8, 123456-K),
        # o el número contiguo cuando va precedido de la palabra NIT. El grupo ``pii`` marca
        # la parte que se reemplaza (la palabra "NIT" se conserva para dar contexto).
        re.compile(
            r"(?i:\bnit\b)[\s:.#]*(?P<pii>\d{6,10}-?[0-9Kk])\b"
            r"|(?<![\w-])(?P<pii2>\d{6,10}-[0-9Kk])(?![\w-])"
        ),
    ),
    _Detector(
        "ip",
        # IPv4 completa; se excluyen versiones de módulos Odoo como 17.0.1.0.0 mediante los
        # lookarounds (no puede haber otro ".dígito" antes ni después).
        re.compile(rf"(?<![\d.]){_OCTET}(?:\.{_OCTET}){{3}}(?!\.?\d)"),
    ),
    _Detector(
        "ip",
        # IPv6 en forma completa (8 grupos); las formas comprimidas se omiten para no
        # confundir horas como 10:30:00.
        re.compile(r"(?<![:\w])(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}(?![:\w])"),
    ),
    _Detector(
        "phone",
        # Guatemala con prefijo internacional: +502 2345-6789, +502 23456789, (502) 2345 6789.
        re.compile(r"(?<![\d+])(?:\+|00)?\(?502\)?[ -]?[2-7]\d{3}[ -]?\d{4}\b(?![ -]\d)"),
    ),
    _Detector(
        "phone",
        # Internacional genérico: +código país y 7..12 dígitos con separadores.
        re.compile(r"\+\d{1,3}[ -]?(?:\(\d{1,4}\)[ -]?)?\d{2,4}(?:[ -]?\d{2,4}){1,3}\b"),
    ),
    _Detector(
        "phone",
        # Local guatemalteco de 8 dígitos con separador: 2345-6789 o 5555 1234.
        # Los lookarounds evitan tomar dos grupos de 4 dentro de un número más largo
        # (p. ej. una tarjeta inválida "4111 1111 1111 1112").
        re.compile(r"(?<![\w-])(?<!\d )[2-7]\d{3}[ -]\d{4}(?![\w-])(?! \d)"),
    ),
)
"""Detectores en el orden en que se aplican (de más específico a más genérico)."""


def _pii_span(match: re.Match[str]) -> tuple[int, int]:
    """Posición del dato personal dentro de la coincidencia.

    Si el patrón define un grupo con nombre ``pii``/``pii2`` (p. ej. el NIT precedido de la
    palabra "NIT"), solo ese grupo se considera dato personal; si no, toda la coincidencia.
    """
    for name in ("pii", "pii2"):
        if name in match.re.groupindex and match.group(name) is not None:
            return match.span(name)
    return match.span()


def _apply(detector: _Detector, text: str) -> tuple[str, list[str]]:
    """Reemplaza las coincidencias válidas de ``detector`` por su token.

    Devuelve el texto resultante y la lista de valores originales reemplazados.
    """
    token = PII_TOKENS[detector.kind]
    replaced_values: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        start, end = _pii_span(match)
        value = match.string[start:end]
        if not detector.validator(value):
            return match.group(0)
        replaced_values.append(value)
        # Se conserva el contexto que no es PII (p. ej. "NIT: ") y se cambia solo el dato.
        offset = match.start()
        whole = match.group(0)
        return whole[: start - offset] + token + whole[end - offset :]

    return detector.pattern.sub(_replace, text), replaced_values


def find_pii(text: str) -> list[tuple[str, str]]:
    """Lista ``(tipo, valor)`` de la PII presente en ``text`` sin modificarlo.

    Usa exactamente la misma lógica y orden que :func:`mask_pii`; sirve para pruebas y
    diagnóstico (nunca debe escribirse su resultado en logs, contiene los datos reales).
    """
    found: list[tuple[str, str]] = []
    working = text
    for detector in _DETECTORS:
        working, values = _apply(detector, working)
        found.extend((detector.kind, value) for value in values)
    return found


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Devuelve ``(texto_enmascarado, tipos_detectados)``.

    ``tipos_detectados`` contiene cada tipo una sola vez, en el orden en que se detectó
    (p. ej. ``["email", "phone"]``). Si no hay PII, devuelve el texto intacto y ``[]``.
    """
    if not text:
        return text, []
    kinds: list[str] = []
    masked = text
    for detector in _DETECTORS:
        masked, values = _apply(detector, masked)
        if values and detector.kind not in kinds:
            kinds.append(detector.kind)
    return masked, kinds


def contains_pii(text: str) -> bool:
    """``True`` si ``text`` contiene al menos un dato personal detectable."""
    return bool(mask_pii(text)[1])


__all__ = [
    "PII_TOKENS",
    "contains_pii",
    "dpi_valid",
    "find_pii",
    "iban_valid",
    "luhn_valid",
    "mask_pii",
]
