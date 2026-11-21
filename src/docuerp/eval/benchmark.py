"""Banco de pruebas ``eval/benchmark.jsonl`` (42 casos: 20 base + 10 herramientas + 12
seguridad).

Este módulo hace tres cosas:

1. Define el contrato de un caso (:class:`BenchmarkCase`): los siete campos obligatorios del
   enunciado más los extras de la SPEC §5.7.
2. Carga el banco línea a línea con errores que dicen *qué línea* falló
   (:func:`load_benchmark`).
3. Verifica que el banco esté completo y sea coherente: composición mínima por categoría
   (:func:`check_composition`) y reglas de consistencia entre campos
   (:func:`validate_cases`), opcionalmente contra el corpus real
   (:func:`check_against_corpus`).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from docuerp import paths
from docuerp.schemas import USER_ID_PATTERN, ErpVersion, Intent, RiskLevel, UserRole

if TYPE_CHECKING:  # solo para anotaciones: evita un import circular en tiempo de ejecución
    from docuerp.eval.chunks import ChunkStore

Category = Literal[
    # base (20)
    "direct",
    "paraphrase",
    "lexical_id",
    "multipart",
    "ambiguous",
    "unanswerable",
    "temporal",
    # herramientas (10)
    "tool_nominal",
    "tool_wrong_selection",
    "tool_out_of_range",
    "tool_confirmation",
    "tool_unavailable",
    # seguridad (12)
    "sec_injection_direct",
    "sec_injection_indirect",
    "sec_permission",
    "sec_critical_action",
    "sec_hallucination",
]

ExpectedBehavior = Literal[
    "answer", "abstain", "tool_call", "confirm", "deny", "block", "escalate", "clarify"
]

# ---------------------------------------------------------------------------
# Composición mínima del banco (enunciado + anteproyecto)
# ---------------------------------------------------------------------------

BASE_COMPOSITION: dict[str, int] = {
    "direct": 5,
    "paraphrase": 3,
    "lexical_id": 3,
    "multipart": 3,
    "ambiguous": 2,
    "unanswerable": 2,
    "temporal": 2,
}
"""Casos mínimos por categoría base (suman 20)."""

TOOL_CATEGORIES: tuple[str, ...] = (
    "tool_nominal",
    "tool_wrong_selection",
    "tool_out_of_range",
    "tool_confirmation",
    "tool_unavailable",
)
SECURITY_CATEGORIES: tuple[str, ...] = (
    "sec_injection_direct",
    "sec_injection_indirect",
    "sec_permission",
    "sec_critical_action",
    "sec_hallucination",
)
TOOL_TOTAL = 10
"""Casos mínimos de herramientas en total (cada categoría debe tener al menos uno)."""
SECURITY_TOTAL = 12
"""Casos mínimos de seguridad en total (cada categoría debe tener al menos uno)."""
BENCHMARK_TOTAL = sum(BASE_COMPOSITION.values()) + TOOL_TOTAL + SECURITY_TOTAL
"""Tamaño objetivo del banco: 42 casos."""


class EvidenceRef(BaseModel):
    """Evidencia esperada a nivel de página."""

    model_config = ConfigDict(extra="forbid")

    source: str
    page: int = Field(ge=1)


class BenchmarkCase(BaseModel):
    """Caso del banco. Los siete primeros campos son obligatorios por el enunciado.

    ``forbidden_strings`` es un extra opcional: textos que **no** deben aparecer en la
    respuesta (p. ej. contenido de un runbook restringido o la orden inyectada); la
    evaluación de seguridad lo usa para detectar fugas.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Z]{2,4}-\d{2,3}$")
    question: str = Field(min_length=3)
    expected_behavior: ExpectedBehavior
    expected_sources: list[str] = Field(default_factory=list)
    expected_pages: list[int] = Field(default_factory=list)
    answerable: bool
    risk_level: RiskLevel
    category: Category
    split: Literal["dev", "test"] = "test"
    role: UserRole = "n1"
    user_id: str = Field(default="USR-0101", pattern=USER_ID_PATTERN)
    erp_version: ErpVersion = "18.0"
    expected_intent: Intent = "doc_query"
    expected_tool: str | None = None
    expected_evidence: list[EvidenceRef] = Field(default_factory=list)
    forbidden_strings: list[str] = Field(default_factory=list)
    simulated_outages: list[str] = Field(default_factory=list)
    """Herramientas que la evaluación simula caídas solo para este caso (``tool_unavailable``)."""
    notes: str = ""

    @property
    def evidence(self) -> list[EvidenceRef]:
        """Evidencia esperada efectiva.

        Usa ``expected_evidence`` si existe; si no, empareja ``expected_sources`` con
        ``expected_pages`` (solo cuando tienen la misma longitud).
        """
        if self.expected_evidence:
            return self.expected_evidence
        if self.expected_sources and len(self.expected_sources) == len(self.expected_pages):
            return [
                EvidenceRef(source=s, page=p)
                for s, p in zip(self.expected_sources, self.expected_pages, strict=True)
            ]
        return []

    @property
    def is_tool_case(self) -> bool:
        """``True`` si el caso pertenece al grupo de herramientas."""
        return self.category in TOOL_CATEGORIES

    @property
    def is_security_case(self) -> bool:
        """``True`` si el caso pertenece al grupo de seguridad."""
        return self.category in SECURITY_CATEGORIES

    @property
    def has_evidence(self) -> bool:
        """``True`` si el caso tiene evidencia documental esperada (aplica Recall@k)."""
        return bool(self.evidence)


class BenchmarkError(ValueError):
    """El archivo del banco no existe, está mal formado o tiene ids duplicados."""


def load_benchmark(path: Path | None = None, *, split: str | None = None) -> list[BenchmarkCase]:
    """Carga y valida el banco (opcionalmente filtrado por ``split``).

    Args:
        path: archivo JSONL; por defecto ``eval/benchmark.jsonl``.
        split: ``"dev"`` o ``"test"`` para devolver solo ese subconjunto.

    Raises:
        BenchmarkError: archivo inexistente, JSON inválido, caso que no cumple el contrato
            (se indica el número de línea) o ``id`` repetido.
    """
    if split is not None and split not in ("dev", "test"):
        raise BenchmarkError(f"split inválido: {split!r} (use 'dev' o 'test')")
    source = path or paths.BENCHMARK_FILE
    if not source.is_file():
        raise BenchmarkError(f"No existe el banco de pruebas: {source}")

    cases: list[BenchmarkCase] = []
    seen: dict[str, int] = {}
    with source.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                case = BenchmarkCase.model_validate(json.loads(line))
            except json.JSONDecodeError as exc:
                raise BenchmarkError(f"{source.name}:{line_no}: JSON inválido ({exc.msg})") from exc
            except ValidationError as exc:
                detail = "; ".join(
                    f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in exc.errors()
                )
                raise BenchmarkError(f"{source.name}:{line_no}: caso inválido ({detail})") from exc
            if case.id in seen:
                raise BenchmarkError(
                    f"{source.name}:{line_no}: id duplicado {case.id!r} "
                    f"(ya aparece en la línea {seen[case.id]})"
                )
            seen[case.id] = line_no
            cases.append(case)
    if split is not None:
        cases = [c for c in cases if c.split == split]
    return cases


# ---------------------------------------------------------------------------
# Composición mínima
# ---------------------------------------------------------------------------


@dataclass
class CompositionReport:
    """Resultado de comparar el banco con la composición mínima exigida."""

    counts: dict[str, int]
    total: int
    missing: dict[str, int] = field(default_factory=dict)
    """Casos que faltan por categoría o grupo (``"herramientas"``, ``"seguridad"``)."""

    @property
    def ok(self) -> bool:
        """``True`` si no falta ningún caso."""
        return not self.missing

    def lines(self) -> list[str]:
        """Resumen legible (una línea por categoría) para la CLI y el reporte."""
        out = [f"Total de casos: {self.total} (objetivo {BENCHMARK_TOTAL})"]
        targets: dict[str, int | None] = {
            **BASE_COMPOSITION,
            **dict.fromkeys(TOOL_CATEGORIES),
            **dict.fromkeys(SECURITY_CATEGORIES),
        }
        for category, target in targets.items():
            have = self.counts.get(category, 0)
            goal = f"/{target}" if target is not None else "/≥1"
            lacking = self.missing.get(category, 0)
            mark = f"  (faltan {lacking})" if lacking else ""
            out.append(f"  {category:<24}{have}{goal}{mark}")
        for group in ("herramientas", "seguridad"):
            if group in self.missing:
                out.append(f"  Grupo {group}: faltan {self.missing[group]} casos")
        return out


def check_composition(cases: Iterable[BenchmarkCase]) -> CompositionReport:
    """Verifica la composición mínima 5/3/3/3/2/2/2 + 10 herramientas + 12 seguridad.

    Reglas: cada categoría base alcanza su cuota; cada categoría de herramientas y de
    seguridad tiene al menos un caso, y el total de cada grupo alcanza 10 y 12.
    El reporte lista exactamente cuántos casos faltan en cada punto.
    """
    items = list(cases)
    counts: dict[str, int] = dict(Counter(str(c.category) for c in items))
    missing: dict[str, int] = {}

    for category, quota in BASE_COMPOSITION.items():
        lacking = quota - counts.get(category, 0)
        if lacking > 0:
            missing[category] = lacking

    for group, categories, group_total in (
        ("herramientas", TOOL_CATEGORIES, TOOL_TOTAL),
        ("seguridad", SECURITY_CATEGORIES, SECURITY_TOTAL),
    ):
        for category in categories:
            if counts.get(category, 0) < 1:
                missing[category] = 1
        lacking = group_total - sum(counts.get(cat, 0) for cat in categories)
        if lacking > 0:
            missing[group] = lacking
    return CompositionReport(counts=counts, total=len(items), missing=missing)


# ---------------------------------------------------------------------------
# Consistencia entre campos
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationIssue:
    """Problema detectado en un caso. ``error`` impide dar el banco por válido."""

    case_id: str
    severity: Literal["error", "warning"]
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.case_id}: {self.message}"


def validate_cases(cases: Iterable[BenchmarkCase]) -> list[ValidationIssue]:
    """Revisa la coherencia interna de cada caso (sin mirar el corpus)."""
    issues: list[ValidationIssue] = []

    def error(case: BenchmarkCase, message: str) -> None:
        issues.append(ValidationIssue(case.id, "error", message))

    def warn(case: BenchmarkCase, message: str) -> None:
        issues.append(ValidationIssue(case.id, "warning", message))

    for case in cases:
        behavior = case.expected_behavior
        if behavior == "answer" and not case.has_evidence:
            error(case, "espera responder pero no define evidencia (expected_evidence)")
        if behavior == "answer" and not case.answerable:
            error(case, "expected_behavior='answer' contradice answerable=false")
        if behavior == "abstain" and case.answerable:
            error(case, "expected_behavior='abstain' contradice answerable=true")
        if case.category == "unanswerable" and case.answerable:
            error(case, "la categoría 'unanswerable' exige answerable=false")
        if (
            case.expected_sources
            and case.expected_pages
            and not case.expected_evidence
            and len(case.expected_sources) != len(case.expected_pages)
        ):
            warn(case, "expected_sources y expected_pages tienen longitudes distintas")
        if case.expected_evidence:
            evidence_sources = {e.source for e in case.expected_evidence}
            if case.expected_sources and not evidence_sources <= set(case.expected_sources):
                warn(case, "expected_evidence cita fuentes que no están en expected_sources")
            evidence_pages = {e.page for e in case.expected_evidence}
            if case.expected_pages and not evidence_pages <= set(case.expected_pages):
                warn(case, "expected_evidence cita páginas que no están en expected_pages")
        if behavior in ("tool_call", "confirm") and case.expected_tool is None:
            error(case, f"expected_behavior={behavior!r} requiere expected_tool")
        if case.expected_tool is not None and case.expected_intent in (
            "doc_query",
            "out_of_domain",
        ):
            warn(case, "define expected_tool con una intención que no usa herramientas de tickets")
        if (
            case.category in ("tool_confirmation", "sec_critical_action")
            and case.risk_level == "low"
        ):
            warn(case, "una acción crítica debería tener risk_level medium o superior")
        if case.is_security_case and case.risk_level == "low":
            warn(case, "un caso de seguridad con risk_level 'low' es poco verosímil")
    return issues


def check_against_corpus(
    cases: Iterable[BenchmarkCase], store: ChunkStore
) -> list[ValidationIssue]:
    """Comprueba que cada evidencia esperada exista en el corpus ingerido.

    Una evidencia ``(source, page)`` es válida si algún chunk de ``source`` cubre esa
    página. Detecta errores de anotación (PDF o página inexistentes) antes de evaluar.
    """
    issues: list[ValidationIssue] = []
    known_sources = store.sources()
    for case in cases:
        for ev in case.evidence:
            if ev.source not in known_sources:
                issues.append(
                    ValidationIssue(
                        case.id, "error", f"la fuente {ev.source!r} no está en el corpus"
                    )
                )
            elif not store.covers(ev.source, ev.page):
                issues.append(
                    ValidationIssue(
                        case.id,
                        "error",
                        f"ningún chunk de {ev.source!r} cubre la página {ev.page}",
                    )
                )
    return issues


@dataclass
class ValidationReport:
    """Resultado completo de ``docuerp eval validate``."""

    cases: list[BenchmarkCase]
    composition: CompositionReport
    issues: list[ValidationIssue]

    @property
    def errors(self) -> list[ValidationIssue]:
        """Solo las incidencias de severidad ``error``."""
        return [i for i in self.issues if i.severity == "error"]

    @property
    def ok(self) -> bool:
        """El banco es válido si está completo y no tiene errores."""
        return self.composition.ok and not self.errors


def validate_benchmark(
    path: Path | None = None, *, store: ChunkStore | None = None
) -> ValidationReport:
    """Carga el banco y ejecuta todas las verificaciones.

    Args:
        path: archivo JSONL (por defecto ``eval/benchmark.jsonl``).
        store: corpus ingerido opcional para validar las evidencias contra los chunks.

    Raises:
        BenchmarkError: si el archivo no se puede cargar.
    """
    cases = load_benchmark(path)
    issues = validate_cases(cases)
    if store is not None:
        issues.extend(check_against_corpus(cases, store))
    return ValidationReport(cases=cases, composition=check_composition(cases), issues=issues)
