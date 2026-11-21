"""Guard de entrada: esquema/longitud, normalización (NFKC, homoglifos, zero-width,
leetspeak, base64/rot13 decodificados para inspección), patrones de inyección ES/EN,
similitud semántica a un banco de ataques y detección de PII.

Arquitectura en capas (cada capa produce *hallazgos* con un peso en [0, 1]):

1. **Normalización anti-evasión.** El texto se lleva a varias *vistas de inspección*:
   la canónica (NFKC, sin caracteres invisibles, homoglifos cirílicos/griegos → latinos,
   minúsculas y sin tildes) y vistas derivadas cuando aplican: leetspeak decodificado
   (``1gn0r4`` → ``ignora``), letras espaciadas unidas (``i g n o r a`` → ``ignora``),
   segmentos base64/hex decodificados, rot13, percent-encoding y texto oculto con
   caracteres *tag* Unicode. Las vistas solo se usan para detectar; nunca se envían al LLM.
2. **Patrones ponderados ES/EN** por categoría (anular instrucciones, revelar prompt,
   cambio de rol, exfiltración, escalamiento, delimitadores falsos, acciones críticas,
   inyección SQL/código). Por categoría se toma el peso máximo.
3. **Similitud semántica** al banco ``security/attack_bank.jsonl`` con
   ``docuerp.embeddings`` (coseno = producto punto de vectores L2-normalizados).
4. **Puntaje combinado** con *noisy-OR*: ``riesgo = 1 − Π(1 − wᵢ)``. Dos señales débiles
   independientes suman más que una sola, pero ninguna señal débil bloquea sola. Se
   bloquea si ``riesgo ≥ injection_threshold`` (configurable, 0.70 por defecto).

Finalmente la PII de la consulta se enmascara (``[EMAIL]``…) en ``sanitized_text``, que es
el texto que continúa por el pipeline.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import json
import logging
import re
import threading
import unicodedata
import urllib.parse
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from docuerp import paths
from docuerp.embeddings import Embedder, get_embedder
from docuerp.schemas import QUERY_MAX_CHARS, ROLE_LEVEL, QueryRequest, ScoredChunk
from docuerp.security.patterns import (
    CATEGORY_LABELS,
    CLAIMED_ROLE_PATTERNS,
    HOMOGLYPHS,
    INJECTION_PATTERNS,
    INVISIBLE_RE,
    SQUASHED_KEYWORDS,
    TAG_CHARS_RE,
    InjectionPattern,
)
from docuerp.security.pii import mask_pii
from docuerp.text import strip_accents

log = logging.getLogger(__name__)

ATTACK_BANK_FILE: Path = paths.SECURITY / "attack_bank.jsonl"
"""Banco de frases de ataque ES/EN para la capa semántica."""


class GuardVerdict(BaseModel):
    """Veredicto del guard de entrada."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool
    sanitized_text: str
    flags: list[str] = Field(default_factory=list)
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str | None = None


# ---------------------------------------------------------------------------
# Capa 1: normalización anti-evasión
# ---------------------------------------------------------------------------

_HOMOGLYPH_TABLE = str.maketrans(HOMOGLYPHS)
_LEET_TABLE = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"}
)
_LEET_TOKEN_RE = re.compile(r"[a-z0-9@$]+")
_SPACED_RE = re.compile(r"(?<!\S)(?:[a-z0-9](?:[ .\-_*·|]{1,3}))+[a-z0-9](?!\S)")
_LETTER_SEP_RE = re.compile(r"[ .\-_*·|]")
_WS_RE = re.compile(r"\s+")
_BASE64_RE = re.compile(r"[A-Za-z0-9+/_-]{16,}={0,2}")
_HEX_RE = re.compile(r"(?:[0-9a-fA-F]{2}[\s:]?){8,}")
_PERCENT_RE = re.compile(r"(?:%[0-9a-fA-F]{2}){3,}")

_COMMON_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "ahora",
        "al",
        "all",
        "and",
        "are",
        "be",
        "como",
        "con",
        "de",
        "del",
        "el",
        "en",
        "es",
        "eso",
        "esta",
        "este",
        "for",
        "in",
        "instrucciones",
        "is",
        "it",
        "la",
        "las",
        "lo",
        "los",
        "mas",
        "me",
        "mi",
        "no",
        "now",
        "of",
        "on",
        "para",
        "por",
        "que",
        "se",
        "sistema",
        "su",
        "system",
        "te",
        "that",
        "the",
        "this",
        "to",
        "todas",
        "todo",
        "todos",
        "tu",
        "tus",
        "un",
        "una",
        "with",
        "y",
        "you",
        "your",
    }
)
"""Palabras funcionales ES/EN para estimar si un texto decodificado es lenguaje natural."""


@dataclass(frozen=True)
class InspectionView:
    """Una vista del texto para inspección: su origen y el texto resultante."""

    origin: str
    """Una de: canonical, leet, despaced, base64, hex, rot13, percent, tags."""
    text: str


def _remove_invisible(text: str) -> tuple[str, int]:
    """Quita caracteres invisibles; devuelve el texto limpio y cuántos se quitaron."""
    cleaned, count = INVISIBLE_RE.subn("", text)
    cleaned, tag_count = TAG_CHARS_RE.subn("", cleaned)
    return cleaned, count + tag_count


def _has_mixed_script(text: str) -> bool:
    """``True`` si alguna palabra mezcla letras latinas con homoglifos cirílicos/griegos."""
    for word in re.findall(r"\w+", text):
        has_latin = any("a" <= ch.lower() <= "z" for ch in word)
        has_glyph = any(ch in HOMOGLYPHS for ch in word)
        if has_latin and has_glyph:
            return True
    return False


def _pre_canonical(text: str) -> str:
    """Canónica sin colapsar espacios (los espacios dobles separan palabras espaciadas)."""
    value = unicodedata.normalize("NFKC", text)
    value, _ = _remove_invisible(value)
    value = value.translate(_HOMOGLYPH_TABLE)
    return strip_accents(value.lower())


def normalize_for_inspection(text: str) -> str:
    """Forma canónica para detección (no se envía al LLM).

    Pasos: NFKC (letras de ancho completo, ligaduras y matemáticas → ASCII), eliminación de
    invisibles, homoglifos → latín, minúsculas, sin tildes y espacios colapsados.
    """
    return _WS_RE.sub(" ", _pre_canonical(text)).strip()


def _deleet(text: str) -> str:
    """Decodifica leetspeak solo en tokens que mezclan letras con dígitos/símbolos.

    Así ``1gn0r4`` → ``ignora`` pero ``17.0`` o ``00042`` (sin letras) quedan intactos.
    """

    def _fix(match: re.Match[str]) -> str:
        token = match.group(0)
        has_alpha = any(ch.isalpha() for ch in token)
        has_leet = any(ch in "0134578@$" for ch in token)
        return token.translate(_LEET_TABLE) if has_alpha and has_leet else token

    return _LEET_TOKEN_RE.sub(_fix, text)


def _despace(text: str) -> str:
    """Une letras sueltas separadas (``i g n o r a  l a s`` → ``ignora las``).

    Dentro de una racha de letras sueltas, un separador simple (espacio, punto, guion…)
    une letras de la misma palabra y dos o más espacios separan palabras. Solo se tocan
    rachas de al menos 4 caracteres para no alterar palabras de una letra ("a", "y").
    """

    def _join(match: re.Match[str]) -> str:
        run = match.group(0)
        if len(_LETTER_SEP_RE.sub("", run)) < 4:
            return run
        words = re.split(r"\s{2,}", run)
        return " ".join(_LETTER_SEP_RE.sub("", word) for word in words)

    return _SPACED_RE.sub(_join, text)


def _plausibility(text: str) -> float:
    """Fracción de palabras funcionales comunes: estima si el texto es lenguaje natural."""
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return 0.0
    return sum(word in _COMMON_WORDS for word in words) / len(words)


def _printable_text(raw: bytes) -> str | None:
    """Decodifica bytes a UTF-8 si casi todo es imprimible y contiene letras; si no, None."""
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if len(decoded) < 4:
        return None
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in decoded) / len(decoded)
    letters = sum(ch.isalpha() for ch in decoded)
    return decoded if printable >= 0.9 and letters >= 3 else None


def _decode_base64_segments(text: str) -> list[str]:
    """Decodifica segmentos base64 (estándar o url-safe) que resulten en texto legible."""
    decoded: list[str] = []
    for match in _BASE64_RE.finditer(text):
        segment = match.group(0)
        if segment.isalpha() and (segment.islower() or segment.isupper()):
            continue  # una palabra larga normal no es base64
        padded = segment + "=" * (-len(segment) % 4)
        for decoder in (base64.b64decode, base64.urlsafe_b64decode):
            try:
                raw = decoder(padded)
            except (binascii.Error, ValueError):
                continue
            readable = _printable_text(raw)
            if readable:
                decoded.append(readable)
                break
    return decoded


def _decode_hex_segments(text: str) -> list[str]:
    """Decodifica secuencias hexadecimales (``69 67 6e 6f …``) a texto legible."""
    decoded: list[str] = []
    for match in _HEX_RE.finditer(text):
        compact = re.sub(r"[\s:]", "", match.group(0))
        if len(compact) % 2:
            compact = compact[:-1]
        try:
            raw = bytes.fromhex(compact)
        except ValueError:
            continue
        readable = _printable_text(raw)
        if readable:
            decoded.append(readable)
    return decoded


def _decode_tag_chars(text: str) -> str | None:
    """Recupera texto ASCII oculto con caracteres *tag* Unicode ("ASCII smuggling")."""
    hidden = "".join(chr(ord(ch) - 0xE0000) for ch in TAG_CHARS_RE.findall(text))
    hidden = "".join(ch for ch in hidden if ch.isprintable())
    return hidden if len(hidden) >= 4 else None


def build_inspection_views(text: str) -> list[InspectionView]:
    """Construye todas las vistas de inspección no redundantes de ``text``.

    La primera es siempre la canónica. Los segmentos codificados se decodifican sobre el
    texto NFKC original (la canónica en minúsculas destruiría el base64).
    """
    nfkc = unicodedata.normalize("NFKC", text)
    canonical = normalize_for_inspection(text)
    views = [InspectionView("canonical", canonical)]
    seen = {canonical}

    def _add(origin: str, candidate: str | None, *, depth: int = 0) -> None:
        if not candidate:
            return
        normalized = normalize_for_inspection(candidate)
        if normalized and normalized not in seen:
            seen.add(normalized)
            views.append(InspectionView(origin, normalized))
            if depth == 0 and origin in {"base64", "hex", "percent", "tags"}:
                # Un nivel extra: ataques que anidan codificaciones (base64 de leetspeak…).
                _add(origin, _deleet(normalized), depth=1)
                for nested in _decode_base64_segments(candidate):
                    _add(origin, nested, depth=1)

    _add("leet", _deleet(canonical))
    pre = _pre_canonical(text)
    despaced = _despace(pre)
    _add("despaced", despaced)
    _add("leet", _deleet(despaced))
    if despaced != pre:
        # Vista "aplastada" (solo alfanuméricos): cubre separadores arbitrarios entre
        # letras (``i.g.n.o.r.a``). Solo se genera si hubo letras espaciadas, porque sobre
        # texto normal uniría palabras y crearía coincidencias espurias.
        views.append(InspectionView("squashed", re.sub(r"[^a-z0-9]", "", pre)))
    cleaned_nfkc, _ = _remove_invisible(nfkc)
    for segment in _decode_base64_segments(cleaned_nfkc):
        _add("base64", segment)
    for segment in _decode_hex_segments(cleaned_nfkc):
        _add("hex", segment)
    if _PERCENT_RE.search(cleaned_nfkc):
        _add("percent", urllib.parse.unquote(cleaned_nfkc))
    hidden = _decode_tag_chars(nfkc)
    if hidden:
        _add("tags", hidden)
    rot = codecs.decode(canonical, "rot_13")
    explicit_rot = any(marker in canonical for marker in ("rot13", "rot-13", "rot 13"))
    # Sin marcador explícito, rot13 solo se considera si el texto tiene suficientes palabras
    # y al rotarlo se vuelve claramente más "natural" (evita vistas espurias).
    enough_words = len(re.findall(r"[a-z]{2,}", canonical)) >= 4
    gain = _plausibility(rot) - _plausibility(canonical)
    if explicit_rot or (enough_words and _plausibility(rot) >= 0.3 and gain >= 0.15):
        _add("rot13", rot)
    return views


# ---------------------------------------------------------------------------
# Capa 3: similitud semántica al banco de ataques
# ---------------------------------------------------------------------------


class AttackPhrase(BaseModel):
    """Una frase del banco de ataques."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str = Field(min_length=3)
    lang: str
    category: str


def load_attack_bank(path: Path | None = None) -> list[AttackPhrase]:
    """Lee y valida ``security/attack_bank.jsonl``.

    Raises:
        FileNotFoundError: si el archivo no existe.
        ValueError: si una línea no es JSON válido o no cumple el esquema.
    """
    source = path or ATTACK_BANK_FILE
    if not source.is_file():
        raise FileNotFoundError(f"No existe el banco de ataques: {source}")
    phrases: list[AttackPhrase] = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            phrases.append(AttackPhrase.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"{source}:{number}: entrada inválida ({exc})") from exc
    return phrases


class SemanticMatcher:
    """Similitud coseno máxima entre un texto y las frases del banco de ataques.

    Las frases se normalizan con :func:`normalize_for_inspection` (igual que la consulta)
    y se embeben una sola vez de forma perezosa; la matriz queda en memoria.
    """

    def __init__(self, embedder: Embedder, bank: list[AttackPhrase]) -> None:
        self.embedder = embedder
        self.bank = bank
        self._matrix: np.ndarray | None = None
        self._lock = threading.Lock()

    def _bank_matrix(self) -> np.ndarray:
        """Matriz (n_frases × dim) de embeddings del banco, calculada una vez."""
        with self._lock:
            if self._matrix is None:
                texts = [normalize_for_inspection(p.text) for p in self.bank]
                self._matrix = np.asarray(self.embedder.embed_documents(texts), dtype=np.float32)
            return self._matrix

    def best_match(
        self, texts: list[str], *, exclude_ids: frozenset[str] = frozenset()
    ) -> tuple[float, AttackPhrase | None]:
        """Devuelve ``(similitud_máxima, frase)`` sobre todas las vistas de ``texts``.

        ``exclude_ids`` permite evaluación *leave-one-out* del propio banco.
        """
        if not texts or not self.bank:
            return 0.0, None
        matrix = self._bank_matrix()
        queries = np.asarray(self.embedder.embed_documents(texts), dtype=np.float32)
        sims = queries @ matrix.T  # coseno: los vectores ya están L2-normalizados
        if exclude_ids:
            mask = np.array([p.id in exclude_ids for p in self.bank])
            sims[:, mask] = -1.0
        flat_index = int(np.argmax(sims))
        _, col = np.unravel_index(flat_index, sims.shape)
        return float(sims.flat[flat_index]), self.bank[int(col)]


# ---------------------------------------------------------------------------
# Capa 4: combinación y decisión
# ---------------------------------------------------------------------------


class InputGuardConfig(BaseModel):
    """Perillas del guard de entrada (por defecto, ``configs/settings.yaml → security``)."""

    model_config = ConfigDict(extra="forbid")

    max_query_chars: int = Field(default=QUERY_MAX_CHARS, ge=3)
    injection_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    semantic_threshold: float = Field(default=0.74, ge=0.0, le=1.0)
    semantic_margin: float = Field(default=0.10, ge=0.0, le=1.0)
    """Ancho de la zona gradual bajo el umbral: aporta riesgo parcial, nunca bloquea sola."""
    semantic_weight: float = Field(default=0.85, ge=0.0, le=1.0)
    semantic_enabled: bool = True
    embedder: str = "bge-m3"
    obfuscation_bonus: float = Field(default=0.30, ge=0.0, le=1.0)
    encoded_payload_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    mask_pii: bool = True

    @classmethod
    def from_settings(cls, embedder: str | None = None) -> InputGuardConfig:
        """Construye la configuración desde ``get_settings()``.

        El embedder por defecto es el de recuperación (ya cargado en el proceso del
        agente), salvo que se indique otro (``fake`` en pruebas).
        """
        from docuerp.config import get_settings

        settings = get_settings()
        sec = settings.security
        return cls(
            max_query_chars=sec.max_query_chars,
            injection_threshold=sec.injection_threshold,
            semantic_threshold=sec.semantic_attack_threshold,
            semantic_enabled=sec.semantic_enabled,
            semantic_margin=sec.semantic_margin,
            embedder=embedder or sec.embedder or settings.retrieval.embedder,
        )


_DECODED_ORIGINS: frozenset[str] = frozenset({"base64", "hex", "rot13", "percent", "tags"})
"""Vistas que provienen de decodificar contenido (no de normalizar)."""

_META_CATEGORIES: frozenset[str] = frozenset(
    {"obfuscation", "anomalous_format", "semantic", "role_claim_mismatch"}
)
"""Categorías que no son un ataque por sí mismas sino señales sobre la forma del texto."""


@dataclass
class Finding:
    """Señal de riesgo detectada por alguna capa."""

    category: str
    name: str
    weight: float
    origin: str = "canonical"


@dataclass
class InputAssessment:
    """Detalle completo de la evaluación (para trazas, red-team y depuración)."""

    verdict: GuardVerdict
    findings: list[Finding] = field(default_factory=list)
    views: list[InspectionView] = field(default_factory=list)
    semantic_score: float = 0.0
    semantic_match_id: str | None = None
    pii_types: list[str] = field(default_factory=list)

    def as_trace_data(self) -> dict[str, Any]:
        """Resumen sanitizado para ``TraceEvent.data`` (sin el texto original)."""
        return {
            "allowed": self.verdict.allowed,
            "risk_score": round(self.verdict.risk_score, 4),
            "flags": self.verdict.flags,
            "categories": sorted({f.category for f in self.findings}),
            "semantic_score": round(self.semantic_score, 4),
            "semantic_match_id": self.semantic_match_id,
            "views": [v.origin for v in self.views],
            "pii_types": self.pii_types,
        }


def combine_risk(weights: list[float]) -> float:
    """Combina pesos con *noisy-OR*: ``1 − Π(1 − w)``; acotado a [0, 1]."""
    remaining = 1.0
    for weight in weights:
        remaining *= 1.0 - min(max(weight, 0.0), 1.0)
    return round(1.0 - remaining, 6)


def sanitize_text(text: str) -> tuple[str, bool]:
    """Texto que continúa por el pipeline: NFKC, sin invisibles y homoglifos de palabras
    con mezcla de alfabetos corregidos. Devuelve ``(texto, hubo_homoglifos)``.

    A diferencia de la vista canónica, conserva mayúsculas, tildes, dígitos e
    identificadores (``TCK-00042``, ``17.0``) para no degradar la recuperación.
    """
    value = unicodedata.normalize("NFKC", text)
    value, _ = _remove_invisible(value)
    mixed = _has_mixed_script(value)

    def _fix_word(match: re.Match[str]) -> str:
        word = match.group(0)
        has_latin = any("a" <= ch.lower() <= "z" for ch in word)
        return word.translate(_HOMOGLYPH_TABLE) if has_latin else word

    if mixed:
        value = re.sub(r"\w+", _fix_word, value)
    value = re.sub(r"[ \t]+", " ", value).strip()
    return value, mixed


class InputGuard:
    """Guard de entrada configurable (patrones + semántica + PII)."""

    def __init__(
        self,
        config: InputGuardConfig | None = None,
        *,
        embedder: Embedder | None = None,
        bank: list[AttackPhrase] | None = None,
    ) -> None:
        self.config = config or InputGuardConfig()
        self._embedder = embedder
        self._bank = bank
        self._matcher: SemanticMatcher | None = None
        self._semantic_failed = False
        self._lock = threading.Lock()

    def _get_matcher(self) -> SemanticMatcher | None:
        """Crea el comparador semántico perezosamente; si falla, degrada a solo patrones."""
        if not self.config.semantic_enabled or self._semantic_failed:
            return None
        with self._lock:
            if self._matcher is None:
                try:
                    embedder = self._embedder or get_embedder(self.config.embedder)  # type: ignore[arg-type]
                    bank = self._bank if self._bank is not None else load_attack_bank()
                    self._matcher = SemanticMatcher(embedder, bank)
                except (OSError, ValueError, RuntimeError, ImportError) as exc:
                    log.warning("Capa semántica no disponible (%s); se usan solo patrones", exc)
                    self._semantic_failed = True
                    return None
            return self._matcher

    # -- capas -------------------------------------------------------------------------

    @staticmethod
    def _pattern_findings(text: str, views: list[InspectionView]) -> list[Finding]:
        """Aplica los patrones a cada vista; registra el origen de cada coincidencia."""
        findings: list[Finding] = []
        cased_view = strip_accents(unicodedata.normalize("NFKC", text).translate(_HOMOGLYPH_TABLE))
        for view in views:
            if view.origin == "squashed":
                findings.extend(
                    Finding(category, f"squashed:{keyword}", 0.8, view.origin)
                    for keyword, category in SQUASHED_KEYWORDS.items()
                    if keyword in view.text
                )
                continue
            for pattern in INJECTION_PATTERNS:
                target = cased_view if pattern.cased else view.text
                if pattern.cased and view.origin != "canonical":
                    continue  # los patrones sensibles a mayúsculas solo aplican al original
                if pattern.regex.search(target):
                    findings.append(
                        Finding(pattern.category, pattern.name, pattern.weight, view.origin)
                    )
        return findings

    @staticmethod
    def _role_mismatch(role: str, canonical: str, findings: list[Finding]) -> Finding | None:
        """Si el texto declara un rol mayor al real (``soy admin`` siendo n1), suma riesgo."""
        if not any(f.category == "escalation" for f in findings):
            return None
        claimed_level = -1
        for regex, claimed in CLAIMED_ROLE_PATTERNS:
            if regex.search(canonical):
                claimed_level = max(claimed_level, ROLE_LEVEL[claimed])
        if claimed_level > ROLE_LEVEL.get(role, 0):
            return Finding("role_claim_mismatch", "claimed_role_above_actual", 0.4)
        return None

    @staticmethod
    def _format_findings(text: str, invisible: int, mixed: bool) -> list[Finding]:
        """Señales de formato: invisibles, homoglifos y repetición masiva (token flooding)."""
        findings: list[Finding] = []
        if invisible >= 3:
            findings.append(Finding("obfuscation", "invisible_chars", 0.2))
        if mixed:
            findings.append(Finding("obfuscation", "homoglyph_mixed_script", 0.25))
        words = re.findall(r"\w+", text.lower())
        if len(words) >= 60:
            most_common = max(words.count(w) for w in set(words))
            if most_common / len(words) > 0.4:
                findings.append(Finding("anomalous_format", "token_flooding", 0.3))
        return findings

    # -- API -----------------------------------------------------------------------------

    def assess(
        self, req: QueryRequest, *, exclude_bank_ids: frozenset[str] = frozenset()
    ) -> InputAssessment:
        """Evalúa ``req`` y devuelve el detalle completo junto con el veredicto."""
        if len(req.query) > self.config.max_query_chars:
            sanitized, _ = sanitize_text(req.query)
            masked, pii_types = mask_pii(sanitized) if self.config.mask_pii else (sanitized, [])
            verdict = GuardVerdict(
                allowed=False,
                sanitized_text=masked[: self.config.max_query_chars],
                flags=["too_long", *(f"pii:{kind}" for kind in pii_types)],
                risk_score=1.0,
                reason=f"Bloqueado: la consulta excede {self.config.max_query_chars} caracteres.",
            )
            return InputAssessment(verdict=verdict, pii_types=pii_types)
        return self._evaluate(req.query, req.role, exclude_bank_ids=exclude_bank_ids)

    def _evaluate(
        self, text: str, role: str, *, exclude_bank_ids: frozenset[str] = frozenset()
    ) -> InputAssessment:
        """Aplica las cuatro capas a ``text`` (sin límite de longitud)."""
        sanitized, mixed = sanitize_text(text)
        _, invisible = _remove_invisible(unicodedata.normalize("NFKC", text))
        masked, pii_types = mask_pii(sanitized) if self.config.mask_pii else (sanitized, [])
        pii_flags = [f"pii:{kind}" for kind in pii_types]

        views = build_inspection_views(text)
        findings = self._pattern_findings(text, views)
        mismatch = self._role_mismatch(role, views[0].text, findings)
        if mismatch:
            findings.append(mismatch)
        findings.extend(self._format_findings(text, invisible, mixed))
        decoded_views = [v for v in views if v.origin in _DECODED_ORIGINS]
        if decoded_views:
            findings.append(
                Finding(
                    "obfuscation",
                    "encoded_payload",
                    self.config.encoded_payload_weight,
                    decoded_views[0].origin,
                )
            )

        semantic_score, match_id = 0.0, None
        matcher = self._get_matcher()
        if matcher is not None:
            try:
                semantic_score, phrase = matcher.best_match(
                    [v.text for v in views], exclude_ids=exclude_bank_ids
                )
                match_id = phrase.id if phrase else None
            except (RuntimeError, ValueError) as exc:
                log.warning("Fallo en la capa semántica (%s); se continúa con patrones", exc)
        semantic_weight = self._semantic_weight(semantic_score)
        if semantic_weight > 0:
            findings.append(Finding("semantic", f"attack_bank:{match_id}", semantic_weight))

        # Bonificación por ofuscación: si una señal de ataque aparece SOLO en una vista
        # derivada, el atacante intentó esconderla, lo que es en sí mismo evidencia.
        canonical_hits = {
            f.category
            for f in findings
            if f.origin == "canonical" and f.category not in _META_CATEGORIES
        }
        hidden_hits = {
            f.category
            for f in findings
            if f.origin != "canonical" and f.category not in _META_CATEGORIES
        } - canonical_hits
        if hidden_hits:
            findings.append(
                Finding("obfuscation", "obfuscated_attack", self.config.obfuscation_bonus)
            )

        per_category: dict[str, float] = {}
        for finding in findings:
            per_category[finding.category] = max(
                per_category.get(finding.category, 0.0), finding.weight
            )
        risk = combine_risk(list(per_category.values()))
        allowed = risk < self.config.injection_threshold

        flags = sorted(
            {f"pattern:{f.name}" for f in findings if f.category not in _META_CATEGORIES}
        )
        flags += sorted({f"signal:{f.name}" for f in findings if f.category in _META_CATEGORIES})
        flags += sorted({f"category:{c}" for c in per_category})
        flags += sorted({f"view:{f.origin}" for f in findings if f.origin != "canonical"})
        if semantic_weight > 0:
            flags.append(f"semantic:{semantic_score:.2f}")
        flags += pii_flags
        verdict = GuardVerdict(
            allowed=allowed,
            sanitized_text=masked,
            flags=flags,
            risk_score=min(max(risk, 0.0), 1.0),
            reason=self._reason(allowed, per_category, risk, pii_types),
        )
        return InputAssessment(
            verdict=verdict,
            findings=findings,
            views=views,
            semantic_score=semantic_score,
            semantic_match_id=match_id,
            pii_types=pii_types,
        )

    def _semantic_weight(self, score: float) -> float:
        """Peso de la similitud según la configuración (ver :func:`semantic_weight`)."""
        return semantic_weight(
            score,
            threshold=self.config.semantic_threshold,
            margin=self.config.semantic_margin,
            full_weight=self.config.semantic_weight,
        )

    def _reason(
        self, allowed: bool, per_category: dict[str, float], risk: float, pii: list[str]
    ) -> str:
        """Razón legible en español de la decisión."""
        if not allowed:
            top = sorted(per_category.items(), key=lambda item: item[1], reverse=True)[:3]
            labels = ", ".join(CATEGORY_LABELS.get(cat, cat) for cat, _ in top)
            limit = self.config.injection_threshold
            return f"Bloqueado por riesgo {risk:.2f} ≥ {limit:.2f}: {labels}."
        parts = [f"Permitido (riesgo {risk:.2f})."]
        if pii:
            parts.append("PII enmascarada: " + ", ".join(pii) + ".")
        return " ".join(parts)

    def inspect_text(self, text: str, *, role: str = "n1") -> InputAssessment:
        """Evalúa un texto arbitrario (p. ej. un chunk recuperado) con las mismas capas.

        Textos largos se recorren en ventanas solapadas de ``max_query_chars`` para que una
        instrucción escondida al final de un chunk extenso no escape; se devuelve la
        evaluación de la ventana más riesgosa.
        """
        windows = text_windows(text, self.config.max_query_chars)
        assessments = [self._evaluate(window, role) for window in windows]
        return max(assessments, key=lambda a: a.verdict.risk_score)


SEMANTIC_PARTIAL_MAX = 0.5
"""Peso máximo de la zona gradual: por sí sola nunca alcanza el umbral de bloqueo 0.70."""


def semantic_weight(score: float, *, threshold: float, margin: float, full_weight: float) -> float:
    """Convierte la similitud al banco de ataques en un peso de riesgo.

    * ``score ≥ threshold`` → ``full_weight`` (0.85 por defecto: bloquea sola).
    * ``threshold − margin < score < threshold`` → peso lineal entre 0 y
      :data:`SEMANTIC_PARTIAL_MAX`; solo bloquea combinada con otra señal.
    * Por debajo → 0.
    """
    if score >= threshold:
        return full_weight
    floor = threshold - margin
    if margin <= 0 or score <= floor:
        return 0.0
    return round(SEMANTIC_PARTIAL_MAX * (score - floor) / margin, 6)


def text_windows(text: str, size: int, overlap: int = 200) -> list[str]:
    """Ventanas de ``size`` caracteres solapadas ``overlap`` (para textos largos)."""
    step = max(size - overlap, 1)
    windows = [text[i : i + size] for i in range(0, max(len(text) - overlap, 1), step)]
    return windows or [text]


@lru_cache(maxsize=4)
def get_input_guard(embedder: str | None = None) -> InputGuard:
    """Guard global (cacheado por embedder) construido desde la configuración."""
    return InputGuard(InputGuardConfig.from_settings(embedder))


def check_input(req: QueryRequest, *, guard: InputGuard | None = None) -> GuardVerdict:
    """Evalúa la consulta y decide si se permite.

    Args:
        req: solicitud ya validada por el contrato ``QueryRequest``.
        guard: instancia a usar (pruebas); por defecto la global de :func:`get_input_guard`.
    """
    return (guard or get_input_guard()).assess(req).verdict


def parse_request(payload: dict[str, Any]) -> tuple[QueryRequest | None, GuardVerdict | None]:
    """Validación estructurada de una solicitud cruda (API/CLI).

    Devuelve ``(solicitud, None)`` si el payload cumple ``QueryRequest``, o
    ``(None, veredicto_bloqueado)`` con la lista de campos inválidos en español.
    """
    try:
        return QueryRequest.model_validate(payload), None
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in err["loc"]) or "payload" for err in exc.errors()})
        raw_query = payload.get("query") if isinstance(payload.get("query"), str) else ""
        masked, _ = mask_pii(str(raw_query)[:QUERY_MAX_CHARS])
        return None, GuardVerdict(
            allowed=False,
            sanitized_text=masked,
            flags=["schema_invalid", *(f"field:{f}" for f in fields)],
            risk_score=1.0,
            reason="Bloqueado: la solicitud no cumple el contrato (" + ", ".join(fields) + ").",
        )


class ContextFinding(BaseModel):
    """Chunk recuperado con instrucciones sospechosas (posible inyección indirecta)."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    risk_score: float
    categories: list[str]
    flagged_at_ingest: bool


def inspect_context(
    hits: list[ScoredChunk], *, guard: InputGuard | None = None, threshold: float | None = None
) -> list[ContextFinding]:
    """Escanea los chunks recuperados en busca de inyección indirecta.

    Un chunk se reporta si la ingesta ya lo marcó (``suspicious_instruction``) o si su
    texto, evaluado con las mismas capas del guard, supera el umbral. El agente debe
    excluirlos del contexto del generador (ver ``permissions.revalidate_hits``).
    """
    active = guard or get_input_guard()
    limit = active.config.injection_threshold if threshold is None else threshold
    findings: list[ContextFinding] = []
    for hit in hits:
        assessment = active.inspect_text(hit.chunk.text)
        flagged = "suspicious_instruction" in hit.chunk.flags
        risk = assessment.verdict.risk_score
        if flagged or risk >= limit:
            findings.append(
                ContextFinding(
                    chunk_id=hit.chunk.chunk_id,
                    risk_score=risk,
                    categories=sorted({f.category for f in assessment.findings}),
                    flagged_at_ingest=flagged,
                )
            )
    return findings


@lru_cache(maxsize=1)
def get_context_guard() -> InputGuard:
    """Guard para escanear chunks recuperados: mismas capas pero **sin** la semántica.

    La calibración (``reports/security/semantic_calibration.md``) muestra que la similitud
    al banco no separa chunks: el runbook envenenado no supera el umbral y los chunks
    legítimos llegan a 0.65. Los patrones sí lo detectan y no marcan ningún otro de los
    1596 chunks del corpus. Además así no se embeben 5 textos largos más por consulta.
    """
    from docuerp.config import get_settings

    sec = get_settings().security
    return InputGuard(
        InputGuardConfig(
            max_query_chars=sec.max_query_chars,
            injection_threshold=sec.injection_threshold,
            semantic_enabled=False,
        )
    )


class SanitizedContext(BaseModel):
    """Contexto del generador después de neutralizar la inyección indirecta."""

    model_config = ConfigDict(extra="forbid")

    hits: list[ScoredChunk]
    sanitized: list[str] = Field(default_factory=list)
    """``chunk_id`` de los fragmentos a los que se les retiraron oraciones inyectadas."""
    quarantined: list[str] = Field(default_factory=list)
    """``chunk_id`` de los fragmentos excluidos completos (nada recuperable o riesgo residual)."""
    removed_segments: int = 0


SANITIZED_FLAG = "instruccion_retirada"
"""Bandera que se agrega a un chunk saneado (el prompt la muestra en ``advertencia``)."""

SANITIZED_NOTE = "[DocuERP retiró de este fragmento una instrucción no confiable.]"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=\S)")


def _segments(text: str) -> list[str]:
    """Oraciones del chunk, línea por línea (conserva las tablas fila a fila)."""
    out: list[str] = []
    for line in text.splitlines():
        out.extend(part for part in _SENTENCE_SPLIT_RE.split(line) if part.strip())
        out.append("\n")
    return out[:-1] if out else out


def sanitize_context(
    hits: list[ScoredChunk], *, guard: InputGuard | None = None
) -> SanitizedContext:
    """Neutraliza la inyección indirecta sin perder el contenido legítimo del fragmento.

    Cada chunk señalado por :func:`inspect_context` se recorre oración por oración con la
    misma capa de patrones: se retiran las oraciones con riesgo ≥ umbral y el resto sigue al
    generador con la bandera :data:`SANITIZED_FLAG` (atributo ``advertencia`` en el prompt).
    Si no se identifica ninguna oración culpable, si no queda texto o si el texto limpio
    sigue sobre el umbral, el chunk sale completo (cuarentena), como antes.
    """
    active = guard or get_context_guard()
    limit = active.config.injection_threshold
    flagged = {f.chunk_id for f in inspect_context(hits, guard=active)}
    kept: list[ScoredChunk] = []
    sanitized: list[str] = []
    quarantined: list[str] = []
    removed_total = 0
    for hit in hits:
        chunk = hit.chunk
        if chunk.chunk_id not in flagged:
            kept.append(hit)
            continue
        parts = _segments(chunk.text)
        clean_parts: list[str] = []
        tainted_line = False  # tras una oración inyectada, el resto de la línea es payload
        for part in parts:
            if part == "\n":
                tainted_line = False
                clean_parts.append(part)
            elif not tainted_line and active.inspect_text(part).verdict.risk_score < limit:
                clean_parts.append(part)
            else:
                tainted_line = True
        removed = sum(1 for p in parts if p != "\n") - sum(1 for p in clean_parts if p != "\n")
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", " ".join(clean_parts)).strip()
        text = re.sub(r"\n{2,}", "\n", text)
        residual = active.inspect_text(text).verdict.risk_score if text else 1.0
        if removed == 0 or not text or residual >= limit:
            quarantined.append(chunk.chunk_id)
            continue
        new_chunk = chunk.model_copy(
            update={
                "text": f"{text}\n{SANITIZED_NOTE}",
                "flags": sorted({*chunk.flags, SANITIZED_FLAG}),
            }
        )
        kept.append(hit.model_copy(update={"chunk": new_chunk}))
        sanitized.append(chunk.chunk_id)
        removed_total += removed
    return SanitizedContext(
        hits=kept, sanitized=sanitized, quarantined=quarantined, removed_segments=removed_total
    )


def quarantine_context(hits: list[ScoredChunk], *, guard: InputGuard | None = None) -> list[str]:
    """``chunk_id`` de los hits que deben salir del contexto del generador.

    Se excluye todo chunk marcado por la ingesta (``suspicious_instruction``) o cuyo texto
    supera el umbral de inyección (ver :func:`inspect_context`). El agente lo llama en
    ``permission_check``, antes de generar: el LLM nunca ve la instrucción envenenada.
    """
    return [f.chunk_id for f in inspect_context(hits, guard=guard or get_context_guard())]


__all__ = [
    "ATTACK_BANK_FILE",
    "CATEGORY_LABELS",
    "HOMOGLYPHS",
    "INJECTION_PATTERNS",
    "SANITIZED_FLAG",
    "SANITIZED_NOTE",
    "AttackPhrase",
    "ContextFinding",
    "Finding",
    "GuardVerdict",
    "InjectionPattern",
    "InputAssessment",
    "InputGuard",
    "InputGuardConfig",
    "InspectionView",
    "SanitizedContext",
    "SemanticMatcher",
    "build_inspection_views",
    "check_input",
    "combine_risk",
    "get_context_guard",
    "get_input_guard",
    "inspect_context",
    "load_attack_bank",
    "normalize_for_inspection",
    "parse_request",
    "quarantine_context",
    "sanitize_context",
    "sanitize_text",
    "semantic_weight",
    "text_windows",
]
