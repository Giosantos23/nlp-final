"""Construye el informe técnico de DocuERP en DOCX y PDF a partir de ``docs/report/*.md``.

Uso (desde la raíz del repositorio)::

    PYTHONPATH=src python scripts/build_report.py            # figuras + DOCX + PDF
    PYTHONPATH=src python scripts/build_report.py --pdf-only
    PYTHONPATH=src python scripts/build_report.py --placeholders   # solo PLACEHOLDERS.md
    PYTHONPATH=src python scripts/build_report.py --check          # verifica marcadores

Fuentes del documento:

* Secciones ``docs/report/NN_nombre.md`` en orden numérico. Cada ``# Título`` es una sección
  numerada (salvo «Bibliografía») y cada ``## Título`` una subsección.
* Tablas con leyenda: un párrafo ``Tabla: {#tab:id} Leyenda.`` justo antes de la tabla.
* Figuras: un párrafo ``::figura id | Leyenda.``. ``id`` es un diagrama de
  ``scripts/report_diagrams.py`` o ``resultados:<nombre>``, que se busca como
  ``<nombre>.png`` en ``reports/figures/`` y luego en ``eval/results/latest/figures/``. Si
  no existe, se deja un recuadro de figura pendiente.
* Referencias cruzadas ``[@tab:id]`` y ``[@fig:id]`` → «Tabla N» / «Figura N».
* Marcadores ``{{nombre}}``: se resuelven desde los resultados (``eval/results/latest/
  summary.json``, ``security/results/``, ``eval/benchmark.jsonl``) o desde
  ``docs/report/values.json`` (textos y cifras manuales). Los no resueltos quedan visibles y
  resaltados. ``--placeholders`` regenera ``docs/report/PLACEHOLDERS.md``.
* Marcadores genéricos ``{{R:clave}}``: toman ``.value`` de ``reports/RESULTS.json`` (lo
  escribe ``scripts/collect_results.py``). Los decimales salen con tres cifras, los IC
  ``[a, b]`` como ``[a, b]`` y las tasas red-team ``{passed, total, rate}`` como
  ``p/t (xx.x %)``. Una subclave de un valor compuesto se pide con puntos
  (``{{R:tau.at_selected.balanced_accuracy}}``) y el formato con ``!``: ``!N`` decimales,
  ``!pct`` porcentaje e ``!int`` entero (``{{R:e2e.final.cost_mean_usd!5}}``). Una clave que
  aún no existe en un grupo conocido queda pendiente; un grupo desconocido es un error.
  Los textos de ``values.json`` pueden contener marcadores, que se resuelven igual.

Salidas en ``docs/report/build/``: ``Informe_DocuERP.docx``, ``Informe_DocuERP.pdf`` y
``build_info.json`` (páginas por parte y marcadores pendientes). El PDF se arma con Chromium
(Playwright) en tres partes (portada, índice y cuerpo) que se unen con PyMuPDF; el índice
toma los números de página del esquema del cuerpo ya paginado.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from html import escape
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import report_diagrams  # noqa: E402

REPORT_DIR = ROOT / "docs" / "report"
BUILD_DIR = REPORT_DIR / "build"
FIGURES_DIR = REPORT_DIR / "figures"
FONTS_DIR = REPORT_DIR / "assets" / "fonts"
VALUES_FILE = REPORT_DIR / "values.json"
PLACEHOLDERS_FILE = REPORT_DIR / "PLACEHOLDERS.md"
EVAL_SUMMARY = ROOT / "eval" / "results" / "latest" / "summary.json"
RESULTS_FILE = ROOT / "reports" / "RESULTS.json"
SECURITY_RESULTS = ROOT / "security" / "results"
BENCHMARK_FILE = ROOT / "eval" / "benchmark.jsonl"
RESULT_FIGURE_DIRS = (
    ROOT / "reports" / "figures",
    ROOT / "eval" / "results" / "latest" / "figures",
)
OUTPUT_NAME = "Informe_DocuERP"
OUT_DIR = BUILD_DIR
"""Carpeta de salida; con ``--preview`` es ``build/_preview`` (no se versiona)."""

TOKENS = json.loads((ROOT / "design" / "tokens.json").read_text(encoding="utf-8"))
ACCENT = TOKENS["color"]["accent"]
INK = "#16191E"

MONTHS = (
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
)

META: dict[str, Any] = {
    "title": "DocuERP",
    "subtitle": (
        "Asistente RAG agéntico con trazabilidad para el soporte funcional de Odoo 17.0 y 18.0"
    ),
    "kind": "Informe técnico · Proyecto final",
    "course": "CC3103 Procesamiento de Lenguaje Natural",
    "university": "Universidad del Valle de Guatemala",
    "teacher": "Luis Esturbán",
    "members": [("Giovanni Santos", "22523"), ("Alexis Mesías", "22562")],
    "company": "Caso de estudio: Altamira Soluciones ERP (empresa ficticia)",
}

UNNUMBERED_SECTIONS = {"Bibliografía"}
BIBLIOGRAPHY = "Bibliografía"
ANNEX_PREFIX = "Anexo"
"""Secciones sin número: la bibliografía y los anexos (van después y no cuentan como cuerpo)."""
NAME_PATTERN = r"R:[\w.:@~/+-]+(?:!\w+)?|[\w-]+"
"""Nombre de marcador: ``R:clave[!formato]`` (RESULTS.json) o un nombre clásico."""
SENTINEL_RE = re.compile(rf"⟦({NAME_PATTERN})⟧")
PLACEHOLDER_RE = re.compile(rf"\{{\{{({NAME_PATTERN})\}}\}}")
RESULT_GROUPS = frozenset(
    {
        "annotation",
        "e2e",
        "judge_agreement",
        "models",
        "paired",
        "retrieval",
        "security",
        "target",
        "tau",
        "v11_vs_baseline",
    }
)
"""Primer segmento válido de una clave ``{{R:…}}`` (grupos que escribe collect_results)."""
REF_RE = re.compile(r"\[@(tab|fig):([\w-]+)\]")
TABLE_CAPTION_RE = re.compile(r"^Tabla:\s*\{#tab:([\w-]+)\}\s*(.+)$")
FIGURE_RE = re.compile(r"^::figura\s+(\S+)\s*\|\s*(.+)$")


# ===========================================================================
# Marcadores: resolución desde los resultados
# ===========================================================================


def _f3(v: float) -> str:
    return f"{v:.3f}"


def _pct(v: float) -> str:
    return f"{v * 100:.1f} %"


def _seconds(v: float) -> str:
    return f"{v / 1000:.1f}"


def _integer(v: float) -> str:
    return f"{round(v):,}".replace(",", " ")


def _usd(v: float) -> str:
    return f"{v:.4f}"


def _ms(v: float) -> str:
    return f"{v:.0f}"


def _delta(v: float) -> str:
    return f"{v:+.3f}"


def _pvalue(v: float) -> str:
    return "< 0.001" if v < 0.001 else f"{v:.3f}"


def _one_decimal(v: float) -> str:
    return f"{v:.1f}"


@dataclass
class Resolution:
    """Resultado de resolver un marcador: valor numérico (si aplica) y texto final."""

    name: str
    description: str
    origin: str
    raw: Any = None
    text: str | None = None


def _get(data: Any, *keys: str) -> Any:
    """Acceso anidado tolerante: ``None`` si falta cualquier clave."""
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


_MISSING = object()


def lookup_result(facts: dict[str, Any], key: str) -> tuple[Any, str | None]:
    """``(valor, origen)`` de ``key`` en RESULTS.json; ``(_MISSING, None)`` si no está.

    Si ``key`` no existe tal cual, se busca el prefijo más largo (separado por puntos) que sí
    exista y se desciende en su valor compuesto con el resto de los segmentos
    (``tau.at_selected.tpr``). Una subclave inexistente en un valor que sí existe es un
    error de redacción y lanza ``KeyError``.
    """
    entry = facts.get(key)
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"], entry.get("source")
    parts = key.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        entry = facts.get(".".join(parts[:cut]))
        if not (isinstance(entry, dict) and "value" in entry):
            continue
        value = entry["value"]
        for part in parts[cut:]:
            if not isinstance(value, dict) or part not in value:
                raise KeyError(f"Subclave inexistente en reports/RESULTS.json: {key}")
            value = value[part]
        return value, entry.get("source")
    return _MISSING, None


def _number(value: float, fmt: str) -> str:
    if fmt == "pct":
        return _pct(value)
    if fmt == "int":
        return _integer(value)
    if fmt.isdigit():
        return f"{value:.{int(fmt)}f}"
    if fmt:
        raise KeyError(f"Formato desconocido en un marcador R: !{fmt}")
    return _f3(value)


def format_result(value: Any, fmt: str = "") -> str | None:
    """Texto de un valor de RESULTS.json; ``None`` si el dato aún no existe.

    Decimales con tres cifras (o el formato pedido), enteros tal cual, IC ``[a, b]``, tasas
    red-team ``p/t (xx.x %)`` y ``NaN`` como «n/d». Un compuesto sin forma conocida lanza
    ``KeyError``: hay que pedir una subclave.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "Sí" if value else "No"
    if isinstance(value, int):
        return _number(float(value), fmt) if fmt else str(value)
    if isinstance(value, float):
        return "n/d" if value != value else _number(value, fmt)
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple):
        if len(value) == 2 and all(isinstance(v, int | float) for v in value):
            return "[" + ", ".join(_number(float(v), fmt) for v in value) + "]"
        if len(value) == 2 and any(v is None for v in value):
            return None
        raise KeyError("Lista sin formato conocido en reports/RESULTS.json")
    if isinstance(value, dict):
        if {"passed", "total", "rate"} <= value.keys():
            if value["rate"] is None:
                return None
            return f"{value['passed']}/{value['total']} ({_pct(float(value['rate']))})"
        if "value" in value:
            return format_result(value["value"], fmt)
        raise KeyError("Valor compuesto en reports/RESULTS.json: pida una subclave")
    raise KeyError(f"Tipo sin formato en reports/RESULTS.json: {type(value).__name__}")


RETRIEVAL_CONFIGS: dict[str, tuple[str, str, str]] = {
    "bm25": ("bm25", "structure", "bge-m3"),
    "dense": ("dense", "structure", "bge-m3"),
    "hybrid": ("hybrid", "structure", "bge-m3"),
    "final": ("hybrid_rerank", "structure", "bge-m3"),
    "mmr": ("hybrid_rerank_mmr", "structure", "bge-m3"),
    "fixed256": ("hybrid_rerank", "fixed256", "bge-m3"),
    "fixed768": ("hybrid_rerank", "fixed768", "bge-m3"),
    "e5": ("hybrid_rerank", "structure", "e5-base"),
}
"""Alias del informe → (modo, segmentación, embedder) en ``summary.json``."""

MODE_PREFERENCE = ("hybrid_rerank", "hybrid_rerank_mmr", "hybrid", "dense", "bm25")

RETRIEVAL_METRICS: dict[str, tuple[str, str]] = {
    "recall1": ("recall@1", "Recall@1"),
    "recall5": ("recall@5", "Recall@5"),
    "mrr": ("mrr", "MRR"),
    "ndcg5": ("ndcg@5", "nDCG@5"),
    "redundancy5": ("redundancy@5", "redundancia en el top-5"),
    "noise5": ("noise@5", "ruido en el top-5"),
}

E2E_METRICS: dict[str, str] = {
    "behavior_correct": "comportamiento esperado",
    "intent_correct": "intención correcta",
    "citation_precision": "precisión de citas",
    "faithfulness": "fidelidad (juez)",
    "coverage": "cobertura fáctica (juez)",
    "correctness": "corrección (juez, 0–1)",
    "abstention_correct": "abstención correcta",
    "abstention_unnecessary": "abstención innecesaria",
    "tool_selection": "selección de herramienta",
    "param_validity": "validez de parámetros",
    "hitl_compliance": "cumplimiento HITL",
    "attack_blocked": "ataques bloqueados (banco)",
    "denial_correct": "denegaciones correctas (banco)",
    "critical_confirmed": "acciones críticas con confirmación (banco)",
    "hallucination_handled": "alucinación forzada manejada",
}
PCT_E2E = {"attack_blocked", "denial_correct", "critical_confirmed"}

OPERATION: dict[str, tuple[tuple[str, ...], Callable[[float], str], str]] = {
    "latency_p50": (("latency_ms", "p50"), _seconds, "latencia de extremo a extremo P50 (s)"),
    "latency_p95": (("latency_ms", "p95"), _seconds, "latencia de extremo a extremo P95 (s)"),
    "api_latency_p50": (("api_latency_ms", "p50"), _seconds, "latencia de la API del LLM P50 (s)"),
    "tokens_mean": (("tokens", "mean"), _integer, "tokens por consulta (media)"),
    "llm_calls": (("llm_calls_mean",), lambda v: f"{v:.2f}", "llamadas al LLM por consulta"),
    "cost_mean": (("cost_usd", "mean"), _usd, "costo medio por consulta (USD, lista)"),
}

LOCAL_STAGE_GROUPS = ("guards", "router", "retrieval", "rerank", "gates", "tools")

OK_TARGETS: dict[str, tuple[str, str, float]] = {
    "recall5": ("recall5_final", "ge", 0.85),
    "mrr": ("mrr_final", "ge", 0.70),
    "redundancy5": ("redundancy5_final", "le", 0.20),
    "faithfulness": ("faithfulness_final", "ge", 0.90),
    "citation_precision": ("citation_precision_final", "ge", 0.95),
    "abstention_correct": ("abstention_correct_final", "ge", 0.90),
    "abstention_unnecessary": ("abstention_unnecessary_final", "le", 0.10),
    "latency_p50": ("latency_p50_final", "le", 4000.0),
    "latency_p95": ("latency_p95_final", "le", 10000.0),
    "tokens_mean": ("tokens_mean_final", "le", 4000.0),
    "tool_selection": ("tool_selection_final", "ge", 0.90),
    "attack_blocked": ("attack_blocked_final", "ge", 0.95),
    "denial_correct": ("denial_correct_final", "ge", 1.0),
    "critical_confirmed": ("critical_confirmed_final", "ge", 1.0),
}

SECURITY_VECTORS = (
    "injection_direct",
    "injection_indirect",
    "permission_evasion",
    "critical_action",
    "forced_hallucination",
    "pii_exfiltration",
    "tool_abuse",
)
SECURITY_GOALS = {
    "sec_attacks_blocked_rate": ("attacks_blocked", "ataques de inyección bloqueados"),
    "sec_denials_rate": ("denials", "evasiones de permisos denegadas"),
    "sec_confirmations_rate": ("confirmations", "acciones críticas con confirmación"),
}
OUTCOME_ES = {
    "blocked": "Bloqueado",
    "denied": "Denegado",
    "abstained": "Abstención",
    "needs_confirmation": "Pide confirmación",
    "masked": "Enmascarado",
    "answered": "Respondió",
    "error": "Error",
}

MANUAL_VALUES: dict[str, str] = {
    "annot_jaccard_pages": "Jaccard medio por pares de las tres pasadas de anotación sobre "
    "la evidencia (fuente, página); reports/benchmark/annotation_agreement.json.",
    "annot_kappa_behavior": "Kappa de Fleiss de las tres pasadas (autor, anotador A y B) "
    "sobre expected_behavior; reports/benchmark/annotation_agreement.json.",
    "annot_n_adjudicated": "Casos con algún desacuerdo entre las tres pasadas, resueltos en "
    "la adjudicación.",
    "judge_human_n": "Afirmaciones revisadas a mano para validar al juez (planilla "
    "eval/annotation/judge_human_review.csv; revisión pendiente).",
    "judge_human_kappa": "Kappa de Cohen entre el juez y la revisión humana (pendiente).",
    "analysis_retrieval": "Párrafo de análisis de la Tabla de recuperación (qué etapa aporta "
    "más y en qué categorías), escrito tras mirar eval/results/latest/report.md; "
    "máximo 60 palabras para no pasar de 15 páginas.",
    "analysis_ablation": "Análisis de la ablación de segmentación y embeddings (máx. 40 "
    "palabras; va en el mismo párrafo que analysis_retrieval).",
    "analysis_e2e": "Análisis de línea base frente a final (máx. 80 palabras).",
    "analysis_models": "Lectura de la ablación del generador (máx. 50 palabras): qué modelo "
    "se elige y por qué, con fidelidad y comportamiento frente a latencia y costo; se escribe "
    "cuando collect_results.py tenga models.sonnet.* y models.opus.*.",
    "failure_analysis": "Autopsia de dos o tres fallas reales del banco (caso, causa, "
    "corrección o decisión), tomada de la sección de fallas de eval/results/latest/report.md; "
    "máximo 130 palabras.",
    "security_findings": "Hallazgos de la corrida red-team de extremo a extremo (fallas, "
    "falsos positivos y mitigaciones agregadas), desde security/results/<run>.md; máximo "
    "120 palabras.",
    "conclusion_hypothesis_verdict": "Verbo del veredicto sobre la hipótesis («confirman», "
    "«confirman en parte» o «no confirman»), según la Tabla de metas.",
}


class Resolver:
    """Resuelve marcadores contra los artefactos de resultados (carga perezosa)."""

    def __init__(self, *, preview: bool = False) -> None:
        self._cache: dict[str, Any] = {}
        self.values: dict[str, Any] = _read_json(VALUES_FILE) or {}
        self.preview = preview

    # --- fuentes ----------------------------------------------------------
    def _load(self, key: str, loader: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = loader()
        return self._cache[key]

    @property
    def eval_summary(self) -> Any:
        return self._load("eval", lambda: _read_json(EVAL_SUMMARY))

    @property
    def results(self) -> dict[str, Any]:
        """``reports/RESULTS.json`` (clave → ``{value, source}``); vacío si no existe."""

        def load() -> dict[str, Any]:
            data = _read_json(RESULTS_FILE)
            return data if isinstance(data, dict) else {}

        return self._load("results", load)

    @property
    def security(self) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
        return self._load("security", _load_security)

    @property
    def benchmark(self) -> list[dict[str, Any]]:
        def load() -> list[dict[str, Any]]:
            if not BENCHMARK_FILE.is_file():
                return []
            lines = BENCHMARK_FILE.read_text(encoding="utf-8").splitlines()
            return [json.loads(line) for line in lines if line.strip()]

        return self._load("bench", load)

    def _retrieval_config(self, alias: str) -> dict[str, Any] | None:
        configs = _get(self.eval_summary, "retrieval") or {}
        mode, chunker, embedder = RETRIEVAL_CONFIGS[alias]
        exact = configs.get(f"{mode}/{chunker}/{embedder}")
        if exact is not None or alias in {"bm25", "dense", "hybrid", "final", "mmr"}:
            return exact
        for candidate in MODE_PREFERENCE:
            found = configs.get(f"{candidate}/{chunker}/{embedder}")
            if found is not None:
                return found
        return None

    # --- resolución ---------------------------------------------------------
    def resolve(self, name: str) -> Resolution:
        """Resuelve ``name``; ``text`` queda en ``None`` si el dato aún no existe."""
        res = self._resolve(name)
        if name in self.values:
            manual = self.values[name]
            res.raw = manual if isinstance(manual, int | float) else res.raw
            res.text = str(manual)
            res.origin += " (valor manual en docs/report/values.json)"
        if res.text is None and self.preview:
            res.text = _preview_value(name)
        return res

    def _resolve_result(self, name: str) -> Resolution:
        """``R:clave[!formato]`` → ``.value`` de ``reports/RESULTS.json``."""
        key, _, fmt = name[2:].partition("!")
        if key.split(".", 1)[0] not in RESULT_GROUPS:
            raise KeyError(f"Marcador desconocido: {{{{{name}}}}} (grupo fuera de RESULTS.json)")
        value, source = lookup_result(self.results, key)
        origin = f"reports/RESULTS.json → {key}"
        if source:
            origin += f" (de `{source}`)"
        res = Resolution(name, f"`{key}` de RESULTS.json", origin)
        if value is _MISSING:
            res.description += " (aún no existe: lo agrega collect_results.py)"
            return res
        res.raw, res.text = value, format_result(value, fmt)
        return res

    def _resolve(self, name: str) -> Resolution:
        if name.startswith("R:"):
            return self._resolve_result(name)
        m = re.fullmatch(r"(recall1|recall5|mrr|ndcg5|redundancy5|noise5)_(\w+?)(_ci)?", name)
        if m and m.group(2) in RETRIEVAL_CONFIGS:
            key, label = RETRIEVAL_METRICS[m.group(1)]
            mode, chunker, embedder = RETRIEVAL_CONFIGS[m.group(2)]
            summary = _get(self._retrieval_config(m.group(2)), "aggregate", key)
            res = Resolution(
                name,
                f"{label} de `{mode}/{chunker}/{embedder}`" + (" (IC 95 %)" if m.group(3) else ""),
                f"eval/results/latest/summary.json → retrieval.<config>.aggregate.{key}",
            )
            return _fill_summary(res, summary, ci=bool(m.group(3)), fmt=_f3)

        m = re.fullmatch(r"ret_p50_(\w+)", name)
        if m and m.group(1) in RETRIEVAL_CONFIGS:
            value = _get(self._retrieval_config(m.group(1)), "latency_ms", "p50")
            res = Resolution(
                name,
                f"latencia P50 (ms) de la recuperación `{'/'.join(RETRIEVAL_CONFIGS[m.group(1)])}`",
                "eval/results/latest/summary.json → retrieval.<config>.latency_ms.p50",
            )
            return _fill_value(res, value, _ms)

        metrics = "|".join(E2E_METRICS)
        m = re.fullmatch(rf"({metrics})_(baseline|final)(_ci)?", name)
        if m:
            metric, run = m.group(1), m.group(2)
            summary = _get(self.eval_summary, "e2e", run, "aggregate", metric)
            res = Resolution(
                name,
                f"{E2E_METRICS[metric]}, corrida {run}" + (" (IC 95 %)" if m.group(3) else ""),
                f"eval/results/latest/summary.json → e2e.{run}.aggregate.{metric}",
            )
            fmt = _pct if metric in PCT_E2E else _f3
            return _fill_summary(res, summary, ci=bool(m.group(3)), fmt=fmt)

        m = re.fullmatch(rf"({metrics})_(delta|delta_ci|p)", name)
        if m:
            metric, part = m.group(1), m.group(2)
            paired = _get(self.eval_summary, "paired", metric) or {}
            res = Resolution(
                name,
                {
                    "delta": f"diferencia final − línea base en {E2E_METRICS[metric]}",
                    "delta_ci": f"IC 95 % pareado de la diferencia en {E2E_METRICS[metric]}",
                    "p": f"valor p (permutación pareada) en {E2E_METRICS[metric]}",
                }[part],
                f"eval/results/latest/summary.json → paired.{metric}",
            )
            if part == "delta":
                return _fill_value(res, paired.get("delta"), _delta)
            if part == "p":
                return _fill_value(res, paired.get("p_value"), _pvalue)
            lo, hi = paired.get("ci_low"), paired.get("ci_high")
            if lo is not None and hi is not None:
                res.raw, res.text = (lo, hi), f"{lo:+.3f} a {hi:+.3f}"
            return res

        m = re.fullmatch(rf"({metrics})_n_(baseline|final|paired)", name)
        if m:
            metric, part = m.group(1), m.group(2)
            if part == "paired":
                path: tuple[str, ...] = ("paired", metric, "n")
                label = f"casos con valor en ambas variantes (pareados) en {E2E_METRICS[metric]}"
            else:
                path = ("e2e", part, "aggregate", metric, "n")
                label = f"casos con valor en la corrida {part} en {E2E_METRICS[metric]}"
            res = Resolution(name, label, f"eval/results/latest/summary.json → {'.'.join(path)}")
            return _fill_value(res, _get(self.eval_summary, *path), _integer)

        ops = "|".join(OPERATION)
        m = re.fullmatch(rf"({ops})_(baseline|final)", name)
        if m:
            path, fmt, label = OPERATION[m.group(1)]
            run = m.group(2)
            value = _get(self.eval_summary, "e2e", run, "operation", *path)
            res = Resolution(
                name,
                f"{label}, corrida {run}",
                f"eval/results/latest/summary.json → e2e.{run}.operation.{'.'.join(path)}",
            )
            return _fill_value(res, value, fmt)

        if name == "latency_local_p50":
            stages = _get(self.eval_summary, "e2e", "final", "operation", "latency_by_stage_ms")
            res = Resolution(
                name,
                "suma de las medianas de las etapas locales (guards, enrutador, recuperación, "
                "rerank, gates y herramientas), en segundos",
                "eval/results/latest/summary.json → e2e.final.operation.latency_by_stage_ms",
            )
            if isinstance(stages, dict):
                values = [_get(stages, g, "p50") for g in LOCAL_STAGE_GROUPS]
                defined = [float(v) for v in values if v is not None]
                if defined:
                    return _fill_value(res, sum(defined), lambda v: f"{v / 1000:.2f}")
            return res

        if name in {"targets_met", "targets_total"}:
            targets = _get(self.eval_summary, "targets")
            res = Resolution(
                name,
                "metas del anteproyecto que cumple el sistema final"
                if name == "targets_met"
                else "metas del anteproyecto evaluadas",
                "eval/results/latest/summary.json → targets[].meets",
            )
            if isinstance(targets, list) and targets:
                count = (
                    sum(1 for t in targets if t.get("meets"))
                    if name == "targets_met"
                    else len(targets)
                )
                res.raw, res.text = count, str(count)
            return res

        m = re.fullmatch(r"ok_(\w+)", name)
        if m and m.group(1) in OK_TARGETS:
            ref, cmp, threshold = OK_TARGETS[m.group(1)]
            base = self.resolve(ref)
            res = Resolution(
                name,
                f"«Sí»/«No» según `{ref}` {'≥' if cmp == 'ge' else '≤'} {threshold:g}",
                f"derivado de {{{{{ref}}}}}",
            )
            raw = base.raw[0] if isinstance(base.raw, tuple) else base.raw
            if isinstance(raw, int | float):
                ok = raw >= threshold - 1e-9 if cmp == "ge" else raw <= threshold + 1e-9
                res.raw, res.text = ok, "Sí" if ok else "No"
            return res

        summary, cases = self.security
        m = re.fullmatch(r"sec_rate_(\w+)", name)
        if m and (m.group(1) in SECURITY_VECTORS or m.group(1) == "overall"):
            key = m.group(1)
            stats = (
                _get(summary, "overall") if key == "overall" else _get(summary, "by_vector", key)
            )
            res = Resolution(
                name,
                f"tasa de defensa exitosa red-team ({key}), con «aprobados/total»",
                "security/results/<última corrida e2e>.summary.json → summary."
                + ("overall" if key == "overall" else f"by_vector.{key}"),
            )
            return _fill_rate(res, stats)

        if name in SECURITY_GOALS:
            goal, label = SECURITY_GOALS[name]
            res = Resolution(
                name,
                f"red-team: {label}",
                f"security/results/<última corrida e2e>.summary.json → summary.goals.{goal}",
            )
            return _fill_rate(res, _get(summary, "goals", goal), with_counts=False)

        simple = {
            "sec_benign_fp": (
                ("benign", "false_positives"),
                _integer,
                "falsos positivos en las consultas benignas",
            ),
            "sec_benign_fpr": (("benign", "fpr"), _pct, "tasa de falsos positivos del guard"),
            "sec_leaks": (("leaks",), _integer, "fugas del canario o del prompt en la suite"),
        }
        if name in simple:
            path, fmt, label = simple[name]
            res = Resolution(
                name,
                label,
                f"security/results/<última corrida e2e>.summary.json → summary.{'.'.join(path)}",
            )
            return _fill_value(res, _get(summary, *path), fmt)

        m = re.fullmatch(r"sec_case_(RT-[A-Z]+-\d+)", name)
        if m:
            record = cases.get(m.group(1))
            res = Resolution(
                name,
                f"resultado observado del caso {m.group(1)} (y si pasa)",
                "security/results/<última corrida e2e>.jsonl → observed, passed",
            )
            if record is not None and record.get("observed") is not None:
                observed = OUTCOME_ES.get(record["observed"], str(record["observed"]))
                verdict = {True: "pasa", False: "falla"}.get(record.get("passed"), "no aplica")
                res.raw, res.text = record.get("passed"), f"{observed} ({verdict})"
            return res

        m = re.fullmatch(r"bench_n_(dev|test)", name)
        if m:
            res = Resolution(
                name,
                f"casos de la partición `{m.group(1)}` del banco",
                "eval/benchmark.jsonl → conteo por split",
            )
            if self.benchmark:
                count = sum(1 for case in self.benchmark if case.get("split") == m.group(1))
                res.raw, res.text = count, str(count)
            return res

        if name in MANUAL_VALUES:
            return Resolution(name, MANUAL_VALUES[name], "docs/report/values.json (manual)")
        raise KeyError(f"Marcador desconocido: {{{{{name}}}}}")


FILLER = (
    "Texto de relleno para medir la maquetación con una extensión parecida a la del análisis "
    "definitivo, que se redactará después de leer los resultados de la corrida. "
)


def _preview_value(name: str) -> str:
    """Valor ficticio de longitud realista (solo para ``--preview``; nunca se versiona)."""
    if name in {"failure_analysis", "security_findings"}:
        return (FILLER * 4).strip()
    if name.startswith("analysis_"):
        return (FILLER * 3).strip()
    if name == "conclusion_hypothesis_verdict":
        return "confirman"
    if name.startswith("ok_"):
        return "Sí"
    if name.startswith("sec_case_"):
        return "Bloqueado (pasa)"
    if name.startswith("sec_rate_"):
        return "100.0 % (17/17)"
    if name.endswith("_ci"):
        return "0.000–0.000"
    return "0.000"


def _fill_value(res: Resolution, value: Any, fmt: Callable[[float], str]) -> Resolution:
    if isinstance(value, int | float):
        res.raw, res.text = value, fmt(float(value))
    return res


def _fill_summary(
    res: Resolution, summary: Any, *, ci: bool, fmt: Callable[[float], str]
) -> Resolution:
    if not isinstance(summary, dict) or summary.get("mean") is None:
        return res
    if not ci:
        return _fill_value(res, summary["mean"], fmt)
    lo, hi = summary.get("ci_low"), summary.get("ci_high")
    if lo is not None and hi is not None:
        res.raw, res.text = (lo, hi), f"{fmt(lo)}–{fmt(hi)}"
    return res


def _fill_rate(res: Resolution, stats: Any, *, with_counts: bool = True) -> Resolution:
    if not isinstance(stats, dict) or stats.get("rate") is None:
        return res
    res.raw = stats["rate"]
    res.text = _pct(stats["rate"])
    if with_counts:
        res.text += f" ({stats.get('passed')}/{stats.get('total')})"
    return res


def _load_security() -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    """Última corrida red-team, prefiriendo el modo de extremo a extremo."""
    if not SECURITY_RESULTS.is_dir():
        return None, {}
    payloads = []
    for path in SECURITY_RESULTS.glob("*.summary.json"):
        data = _read_json(path)
        if isinstance(data, dict):
            payloads.append((path, data))
    if not payloads:
        return None, {}
    e2e = [p for p in payloads if "e2e" in str(p[1].get("mode", ""))]
    path, data = max(e2e or payloads, key=lambda item: item[0].name)
    cases: dict[str, dict[str, Any]] = {}
    jsonl = path.with_name(path.name.replace(".summary.json", ".jsonl"))
    if jsonl.is_file():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                cases[record["id"]] = record
    return data.get("summary"), cases


# ===========================================================================
# Documento: secciones → bloques
# ===========================================================================


@dataclass
class Heading:
    level: int
    number: str
    title: str
    anchor: str


@dataclass
class Block:
    """Bloque del documento ya numerado (lo consumen los renderizadores HTML y DOCX)."""

    kind: str  # heading | para | list | table | figure | pending | code
    heading: Heading | None = None
    inline: Any = None  # tokens inline de markdown-it
    items: list[Any] = field(default_factory=list)
    ordered: bool = False
    header: list[Any] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    number: int = 0
    caption: Any = None
    ref: str = ""
    css: str = ""
    text: str = ""


@dataclass
class Document:
    blocks: list[Block]
    headings: list[Heading]
    placeholders: dict[str, Resolution]
    tables: dict[str, int]
    figures: dict[str, int]


def _markdown() -> Any:
    from markdown_it import MarkdownIt

    return MarkdownIt("commonmark", {"html": False}).enable("table")


def section_files() -> list[Path]:
    return sorted(p for p in REPORT_DIR.glob("[0-9][0-9]_*.md"))


def _figure_key(raw_id: str) -> str:
    return raw_id.split(":", 1)[-1]


def number_captions(texts: Iterable[str]) -> tuple[dict[str, int], dict[str, int]]:
    """Asigna números a tablas y figuras en orden de aparición."""
    tables: dict[str, int] = {}
    figures: dict[str, int] = {}
    for text in texts:
        for line in text.splitlines():
            if m := TABLE_CAPTION_RE.match(line.strip()):
                if m.group(1) in tables:
                    raise ValueError(f"Tabla duplicada: {m.group(1)}")
                tables[m.group(1)] = len(tables) + 1
            elif m := FIGURE_RE.match(line.strip()):
                key = _figure_key(m.group(1))
                if key in figures:
                    raise ValueError(f"Figura duplicada: {key}")
                figures[key] = len(figures) + 1
    return tables, figures


MAX_NESTING = 3
"""Profundidad máxima de marcadores dentro de textos de ``values.json``."""


def substitute_placeholders(
    text: str, resolver: Resolver, placeholders: dict[str, Resolution], *, depth: int = 0
) -> str:
    """Reemplaza cada ``{{…}}`` por su texto; los pendientes quedan como ``⟦nombre⟧``.

    Un texto manual que a su vez contiene marcadores (p. ej. un análisis con
    ``{{R:e2e.final.behavior_correct}}``) se resuelve de forma recursiva; cada marcador
    encontrado se registra en ``placeholders`` para ``PLACEHOLDERS.md``.
    """

    def sub(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in placeholders:
            placeholders[name] = resolver.resolve(name)
        res = placeholders[name]
        if res.text is None:
            return f"⟦{name}⟧"
        if depth < MAX_NESTING and PLACEHOLDER_RE.search(res.text):
            return substitute_placeholders(res.text, resolver, placeholders, depth=depth + 1)
        return res.text

    return PLACEHOLDER_RE.sub(sub, text)


def build_document(resolver: Resolver) -> Document:
    """Lee las secciones, resuelve marcadores y referencias y devuelve los bloques."""
    files = section_files()
    raw_texts = [p.read_text(encoding="utf-8") for p in files]
    tables, figures = number_captions(raw_texts)
    placeholders: dict[str, Resolution] = {}

    def sub_ref(m: re.Match[str]) -> str:
        kind, key = m.group(1), m.group(2)
        table = tables if kind == "tab" else figures
        if key not in table:
            raise ValueError(f"Referencia rota: [@{kind}:{key}]")
        return f"{'Tabla' if kind == 'tab' else 'Figura'} {table[key]}"

    md = _markdown()
    blocks: list[Block] = []
    headings: list[Heading] = []
    h1 = h2 = 0
    for text in raw_texts:
        text = REF_RE.sub(sub_ref, substitute_placeholders(text, resolver, placeholders))
        tokens = md.parse(text)
        pending_caption: tuple[int, Any] | None = None
        in_bibliography = in_backmatter = False
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.type == "heading_open":
                level = int(tok.tag[1])
                title = tokens[i + 1].content.strip()
                if level == 1:
                    numbered = title not in UNNUMBERED_SECTIONS and not title.startswith(
                        ANNEX_PREFIX
                    )
                    in_backmatter = not numbered
                    in_bibliography = title == BIBLIOGRAPHY
                    if numbered:
                        h1 += 1
                        h2 = 0
                    number = str(h1) if numbered else ""
                else:
                    h2 += 1
                    number = f"{h1}.{h2}" if not in_backmatter else ""
                anchor = "sec-" + (number.replace(".", "-") if number else _slug(title))
                head = Heading(level, number, title, anchor)
                headings.append(head)
                blocks.append(Block("heading", heading=head, inline=tokens[i + 1].children))
                i += 3
                continue
            if tok.type == "paragraph_open":
                inline = tokens[i + 1]
                content = inline.content.strip()
                if m := TABLE_CAPTION_RE.match(content):
                    caption = md.parseInline(m.group(2))[0].children
                    pending_caption = (tables[m.group(1)], caption)
                elif m := FIGURE_RE.match(content):
                    key = _figure_key(m.group(1))
                    blocks.append(
                        Block(
                            "figure",
                            number=figures[key],
                            caption=md.parseInline(m.group(2))[0].children,
                            ref=m.group(1),
                        )
                    )
                elif m := SENTINEL_RE.fullmatch(content):
                    res = placeholders[m.group(1)]
                    blocks.append(Block("pending", ref=m.group(1), text=res.description))
                else:
                    css = "bib" if in_bibliography else ""
                    blocks.append(Block("para", inline=inline.children, css=css))
                i += 3
                continue
            if tok.type in {"bullet_list_open", "ordered_list_open"}:
                ordered = tok.type == "ordered_list_open"
                close = "ordered_list_close" if ordered else "bullet_list_close"
                items: list[Any] = []
                depth = 0
                j = i
                while j < len(tokens):
                    t = tokens[j]
                    if t.type == tok.type:
                        depth += 1
                    elif t.type == close:
                        depth -= 1
                        if depth == 0:
                            break
                    elif t.type == "inline" and depth == 1:
                        items.append(t.children)
                    j += 1
                blocks.append(Block("list", items=items, ordered=ordered))
                i = j + 1
                continue
            if tok.type == "table_open":
                header: list[Any] = []
                rows: list[list[Any]] = []
                current: list[Any] = []
                j = i + 1
                in_head = False
                while tokens[j].type != "table_close":
                    t = tokens[j]
                    if t.type == "thead_open":
                        in_head = True
                    elif t.type == "thead_close":
                        in_head = False
                    elif t.type == "tr_open":
                        current = []
                    elif t.type == "tr_close":
                        if in_head:
                            header = current
                        else:
                            rows.append(current)
                    elif t.type == "inline":
                        current.append(t.children)
                    j += 1
                number, caption = pending_caption if pending_caption else (0, None)
                pending_caption = None
                blocks.append(
                    Block("table", header=header, rows=rows, number=number, caption=caption)
                )
                i = j + 1
                continue
            if tok.type in {"fence", "code_block"}:
                blocks.append(Block("code", text=tok.content.rstrip("\n")))
            i += 1
        if pending_caption is not None:
            raise ValueError("Leyenda de tabla sin tabla a continuación")
    return Document(blocks, headings, placeholders, tables, figures)


def _slug(text: str) -> str:
    import unicodedata

    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")


def result_figure(name: str) -> Path | None:
    """PNG de una figura de resultados si ya existe."""
    for folder in RESULT_FIGURE_DIRS:
        candidate = folder / f"{name}.png"
        if candidate.is_file():
            return candidate
    return None


def figure_source(ref: str) -> tuple[str, Path | None]:
    """``("diagram", png)`` o ``("result", png | None)`` para el id de una figura."""
    if ref.startswith("resultados:"):
        return "result", result_figure(ref.split(":", 1)[1])
    if ref not in report_diagrams.DIAGRAMS:
        raise ValueError(f"Figura desconocida: {ref}")
    return "diagram", FIGURES_DIR / f"{ref}.png"


def spanish_date(day: date) -> str:
    return f"{day.day} de {MONTHS[day.month - 1]} de {day.year}"


# ===========================================================================
# HTML (PDF)
# ===========================================================================


def font_faces() -> str:
    faces = []
    for family, stem in (("Geist", "geist"), ("Geist Mono", "geist-mono")):
        for subset, rng in (
            (
                "latin",
                "U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, "
                "U+2000-206F, U+2074, U+20AC, U+2122, U+2190-21FF, U+2212, U+2215, U+2264-2265, "
                "U+FEFF, U+FFFD",
            ),
            (
                "latin-ext",
                "U+0100-02AF, U+0304, U+0308, U+0329, U+1E00-1E9F, U+1EF2-1EFF, "
                "U+2020, U+20A0-20AB, U+20AD-20C0, U+2113, U+2C60-2C7F, U+A720-A7FF",
            ),
        ):
            path = (FONTS_DIR / f"{stem}-{subset}-wght-normal.woff2").as_uri()
            faces.append(
                f"@font-face{{font-family:'{family}';font-style:normal;font-display:block;"
                f"font-weight:100 900;src:url('{path}') format('woff2');unicode-range:{rng};}}"
            )
    return "\n".join(faces)


BASE_CSS = """
:root{--ink:#16191E;--muted:#5B6472;--line:#E2E6EA;--soft:#F4F6F7;--accent:#10B981;
--accent-dark:#0B8A60;--graphite:#0B0D10;}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{font-family:'Geist',Inter,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;
color:var(--ink);font-size:9.2pt;line-height:1.45;font-feature-settings:'ss01','tnum' 0;
-webkit-print-color-adjust:exact;print-color-adjust:exact;text-rendering:geometricPrecision}
code,.mono{font-family:'Geist Mono','JetBrains Mono',ui-monospace,Menlo,monospace}
"""

BODY_CSS = """
@page{size:A4;margin:18mm 17mm 17mm 17mm}
p{margin:0 0 5.4pt;text-align:justify;hyphens:auto;-webkit-hyphens:auto;orphans:3;widows:3}
h1{font-size:17pt;line-height:1.2;font-weight:650;letter-spacing:-0.01em;color:var(--graphite);
margin:17pt 0 8pt;padding-top:10pt;border-top:1px solid var(--line);
break-after:avoid;page-break-after:avoid}
h1 .num{display:inline-block;min-width:1.45em;color:var(--accent);font-weight:700}
h1.first{border-top:none;padding-top:0;margin-top:0}
h1.unnumbered{break-before:page;page-break-before:always;border-top:none;padding-top:0;
margin-top:0}
h2{font-size:11.4pt;line-height:1.3;font-weight:620;color:var(--graphite);margin:13pt 0 5pt;
break-after:avoid;page-break-after:avoid}
h2 .num{color:var(--accent-dark);font-weight:650;margin-right:7pt}
strong{font-weight:620;color:var(--graphite)}
code{font-size:0.86em;background:#EEF2F1;color:#0E5E45;border-radius:3px;padding:0 2px;
white-space:nowrap}
a{color:var(--accent-dark);text-decoration:none}
ul,ol{margin:0 0 7pt;padding-left:15pt}
li{margin:0 0 2.5pt}
.tcap,figcaption{font-size:8.3pt;line-height:1.4;color:var(--muted);text-align:left}
.tcap{margin:10pt 0 4pt;break-after:avoid;page-break-after:avoid}
.tcap b,figcaption b{color:var(--graphite);font-weight:650}
.tcap b::before,figcaption b::before{content:'';display:inline-block;width:6px;height:6px;
border-radius:1.5px;background:var(--accent);margin-right:5px;vertical-align:1px}
table{width:100%;border-collapse:collapse;margin:0 0 9pt;font-size:7.9pt;line-height:1.33;
break-inside:auto;border-top:1.6px solid var(--graphite);border-bottom:1px solid var(--graphite)}
thead{display:table-header-group}
tr{break-inside:avoid;page-break-inside:avoid}
th{text-align:left;font-weight:620;color:var(--graphite);background:var(--soft);
padding:4.2pt 5pt;border-bottom:1px solid #C9D1D6;vertical-align:bottom}
td{padding:3.6pt 5pt;border-bottom:0.6px solid var(--line);vertical-align:top}
tbody tr:last-child td{border-bottom:none}
td code,th code{white-space:normal;word-break:break-word}
td .ph{word-break:break-all;font-size:0.74em}
td:first-child code{white-space:nowrap}
table.idcol td:first-child{white-space:nowrap}
figure{margin:12pt 0 12pt;break-inside:avoid;page-break-inside:avoid}
figure svg{display:block;width:100%;height:auto;border-radius:6px}
#fig-arquitectura svg{width:80%;margin:0 auto}
#fig-grafo_agente svg{width:68%;margin:0 auto}
figure img{display:block;max-width:100%;max-height:66mm;width:auto;height:auto;margin:0 auto;
border-radius:6px}
figcaption{margin-top:6pt}
figure.pending .box{height:62mm;border:1.2px dashed #C9D1D6;border-radius:8px;
background:repeating-linear-gradient(135deg,#FAFBFB 0 10px,#F3F5F6 10px 20px);
display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:8.5pt;
padding:0 16mm}
figure.pending .box span{display:block;text-align:center;line-height:1.6}
figure.pending .box code{white-space:normal}
.ph{font-family:'Geist Mono',ui-monospace,monospace;font-size:0.8em;background:#FFF5D9;
color:#8A5A00;border:1px dashed #E9B949;border-radius:3px;padding:0 2.5px;
-webkit-box-decoration-break:clone;box-decoration-break:clone}
.pending-block{border:1px dashed #E9B949;background:#FFFBEF;border-radius:6px;padding:7pt 9pt;
margin:4pt 0 9pt;font-size:8.6pt;color:#6B4A00}
.pending-block .ph{margin-right:6px}
p.bib{text-align:left;padding-left:1.27cm;text-indent:-1.27cm;margin-bottom:2pt;
font-size:8.2pt;line-height:1.22;word-break:break-word}
p.bib code{white-space:normal}
pre{font-size:8pt;background:var(--soft);padding:7pt;border-radius:5px;white-space:pre-wrap}
"""

TOC_CSS = """
@page{size:A4;margin:24mm 22mm 20mm 22mm}
.toc-title{font-size:24pt;font-weight:680;letter-spacing:-0.02em;color:var(--graphite);
margin:0 0 4pt}
.toc-sub{font-size:8.6pt;color:var(--muted);margin:0 0 16pt;padding-bottom:10pt;
border-bottom:1px solid var(--line)}
.toc-sub b{color:var(--accent);font-weight:650}
.entry{display:flex;align-items:baseline;gap:0;text-decoration:none;color:inherit}
.cols{column-count:2;column-gap:12mm}
.grp{break-inside:avoid;page-break-inside:avoid;padding-bottom:3pt}
.entry .t{flex:0 1 auto;margin-right:6pt}
.entry .dots{flex:1;border-bottom:1px dotted #C3CAD0;transform:translateY(-3px)}
.entry .p{font-variant-numeric:tabular-nums;min-width:16pt;margin-left:6pt;text-align:right;
color:var(--muted)}
.l1{font-size:9.6pt;font-weight:620;color:var(--graphite);margin:6pt 0 2pt}
.l1 .n{color:var(--accent);flex:none;width:20pt;font-weight:700}
.l1 .p{color:var(--graphite)}
.l2{font-size:8.4pt;color:#3A424D;margin:0 0 0 20pt;line-height:1.5}
.l2 .n{color:var(--muted);flex:none;width:24pt}
"""


def _inline_html(md: Any, children: Any) -> str:
    html = md.renderer.renderInline(children or [], md.options, {})
    return SENTINEL_RE.sub(lambda m: _ph_html(m.group(1)), html)


def _ph_html(name: str) -> str:
    """Marcador pendiente visible; ``<wbr>`` tras guiones bajos y puntos para poder partirlo."""
    breakable = escape(name).replace("_", "_<wbr>").replace(".", ".<wbr>")
    return f'<span class="ph">{{{{{breakable}}}}}</span>'


def svg_for(ref: str) -> str:
    return report_diagrams.DIAGRAMS[ref]()


def render_body_html(doc: Document) -> str:
    md = _markdown()
    out: list[str] = []
    first_h1 = True
    for block in doc.blocks:
        if block.kind == "heading":
            h = block.heading
            assert h is not None
            title = _inline_html(md, block.inline)
            if h.level == 1:
                classes = []
                if first_h1:
                    classes.append("first")
                    first_h1 = False
                if not h.number:
                    classes.append("unnumbered")
                cls = f' class="{" ".join(classes)}"' if classes else ""
                num = f'<span class="num">{h.number}</span>' if h.number else ""
                out.append(f'<h1 id="{h.anchor}"{cls}>{num}{title}</h1>')
            else:
                num = f'<span class="num">{h.number}</span>' if h.number else ""
                out.append(f'<h2 id="{h.anchor}">{num}{title}</h2>')
        elif block.kind == "para":
            cls = f' class="{block.css}"' if block.css else ""
            out.append(f"<p{cls}>{_inline_html(md, block.inline)}</p>")
        elif block.kind == "list":
            tag = "ol" if block.ordered else "ul"
            items = "".join(f"<li>{_inline_html(md, it)}</li>" for it in block.items)
            out.append(f"<{tag}>{items}</{tag}>")
        elif block.kind == "table":
            if block.caption is not None:
                out.append(
                    f'<div class="tcap"><b>Tabla {block.number}.</b> '
                    f"{_inline_html(md, block.caption)}</div>"
                )
            head = "".join(f"<th>{_inline_html(md, c)}</th>" for c in block.header)
            body = "".join(
                "<tr>" + "".join(f"<td>{_inline_html(md, c)}</td>" for c in row) + "</tr>"
                for row in block.rows
            )
            # Primera columna de identificadores cortos (p. ej. RT-ID-01): sin cortes de línea.
            short_ids = all(len(_plain(row[0])) <= 10 for row in block.rows if row)
            cls = ' class="idcol"' if short_ids else ""
            out.append(f"<table{cls}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
        elif block.kind == "figure":
            kind, png = figure_source(block.ref)
            caption = (
                f"<figcaption><b>Figura {block.number}.</b> "
                f"{_inline_html(md, block.caption)}</figcaption>"
            )
            if kind == "diagram":
                out.append(f'<figure id="fig-{block.ref}">{svg_for(block.ref)}{caption}</figure>')
            elif png is not None:
                out.append(f'<figure><img src="{png.as_uri()}" alt="">{caption}</figure>')
            else:
                name = block.ref.split(":", 1)[1]
                out.append(
                    '<figure class="pending"><div class="box"><span>Figura pendiente: se '
                    f"insertará automáticamente desde <code>reports/figures/{name}.png</code> o "
                    f"<code>eval/results/latest/figures/{name}.png</code> al ejecutar "
                    f"<code>make eval-report</code>.</span></div>{caption}</figure>"
                )
        elif block.kind == "pending":
            out.append(
                f'<div class="pending-block"><span class="ph">{{{{{block.ref}}}}}</span>'
                f"Texto pendiente. {escape(block.text)}</div>"
            )
        elif block.kind == "code":
            out.append(f"<pre><code>{escape(block.text)}</code></pre>")
    return _page(BODY_CSS, "\n".join(out))


def _page(css: str, body: str, *, title: str = "DocuERP — Informe técnico") -> str:
    return (
        '<!doctype html><html lang="es"><head><meta charset="utf-8">'
        f"<title>{escape(title)}</title><style>{font_faces()}{BASE_CSS}{css}</style></head>"
        f"<body>{body}</body></html>"
    )


def render_toc_html(entries: list[tuple[Heading, int]]) -> str:
    groups: list[list[str]] = []
    for head, page in entries:
        number = f'<span class="n">{head.number}</span>' if head.number else ""
        if not head.number and head.level == 2:
            number = '<span class="n"></span>'
        row = (
            f'<div class="entry l{head.level}">{number}<span class="t">{escape(head.title)}'
            f'</span><span class="dots"></span><span class="p">{page}</span></div>'
        )
        if head.level == 1 or not groups:
            groups.append([])
        groups[-1].append(row)
    body = (
        '<div class="toc-title">Índice</div>'
        f'<div class="toc-sub"><b>DocuERP</b> · {escape(META["kind"])} · '
        f"{escape(META['course'])}</div>"
        '<div class="cols">'
        + "".join(f'<div class="grp">{"".join(g)}</div>' for g in groups)
        + "</div>"
    )
    return _page(TOC_CSS, body)


COVER_CSS = """
@page{size:A4;margin:0}
html,body{width:210mm;height:297mm;overflow:hidden}
.cover{position:relative;width:210mm;height:297mm;color:#E7EAEE;overflow:hidden;
background:radial-gradient(120% 70% at 100% 0%,rgba(16,185,129,.20) 0%,rgba(16,185,129,0) 55%),
radial-gradient(90% 60% at 0% 100%,rgba(96,165,250,.08) 0%,rgba(96,165,250,0) 60%),
linear-gradient(180deg,#0B0D10 0%,#101317 55%,#16181D 100%)}
.grid{position:absolute;inset:0;background-image:
linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),
linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px);background-size:10mm 10mm;
mask-image:linear-gradient(180deg,rgba(0,0,0,.9),rgba(0,0,0,.15));
-webkit-mask-image:linear-gradient(180deg,rgba(0,0,0,.9),rgba(0,0,0,.15))}
.inner{position:absolute;inset:22mm 20mm 20mm 20mm;display:flex;flex-direction:column}
.brand{display:flex;align-items:center;gap:9px;font-size:9pt;letter-spacing:.22em;
text-transform:uppercase;color:#8B94A3;font-weight:600}
.mark{width:22px;height:22px;border-radius:6px;
background:linear-gradient(135deg,#34D399,#10B981 55%,#0B7A57);
box-shadow:0 0 0 1px rgba(52,211,153,.5),0 0 22px rgba(16,185,129,.55)}
.kicker{margin-top:58mm;font-size:9.5pt;letter-spacing:.24em;text-transform:uppercase;
color:#34D399;font-weight:600}
h1{margin:5mm 0 0;font-size:60pt;line-height:.95;letter-spacing:-.035em;font-weight:720;
color:#F3F5F7}
h1 span{color:#10B981}
.sub{margin-top:7mm;max-width:150mm;font-size:16.5pt;line-height:1.32;color:#C6CCD4;
font-weight:420;letter-spacing:-.005em}
.rule{margin-top:11mm;width:30mm;height:2px;background:#10B981;
box-shadow:0 0 14px rgba(16,185,129,.8)}
.pipeline{margin-top:12mm}
.meta{margin-top:auto;display:grid;grid-template-columns:1fr 1fr;gap:6mm 10mm;
padding-top:8mm;border-top:1px solid #232934}
.meta .k{font-size:7.6pt;letter-spacing:.2em;text-transform:uppercase;color:#6F7987;
margin-bottom:1.6mm;font-weight:600}
.meta .v{font-size:10.5pt;color:#E7EAEE;line-height:1.45}
.meta .v small{color:#8B94A3;font-family:'Geist Mono',monospace;font-size:8.8pt;
margin-left:5px}
.foot{margin-top:7mm;font-size:7.8pt;color:#6F7987;display:flex;justify-content:space-between}
"""


def _cover_pipeline_svg() -> str:
    stages = [
        "ingesta",
        "índices",
        "guard",
        "router",
        "híbrido",
        "rerank",
        "gate τ",
        "permisos",
        "LLM",
        "guard",
        "respuesta",
    ]
    width, step = 660, 60
    parts = [f'<svg viewBox="0 0 {width} 46" width="100%" xmlns="http://www.w3.org/2000/svg">']
    parts.append(
        f'<line x1="28" y1="12" x2="{28 + step * (len(stages) - 1)}" y2="12" '
        'stroke="#10B981" stroke-opacity=".45" stroke-width="1.2"/>'
    )
    for i, name in enumerate(stages):
        x = 28 + i * step
        last = i == len(stages) - 1
        fill = "#10B981" if last or i == 0 else "#0B0D10"
        parts.append(
            f'<circle cx="{x}" cy="12" r="{5 if last else 4}" fill="{fill}" stroke="#34D399" '
            'stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="{x}" y="36" fill="#8B94A3" font-size="9.5" text-anchor="middle" '
            f"font-family=\"'Geist Mono',monospace\">{escape(name)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def render_cover_html(day: date) -> str:
    members = "<br>".join(
        f"{escape(name)}<small>{carne}</small>" for name, carne in META["members"]
    )
    body = f"""
<div class="cover"><div class="grid"></div><div class="inner">
<div class="brand"><div class="mark"></div>DocuERP · {escape(META["kind"])}</div>
<div class="kicker">Sistema NLP / RAG agéntico empresarial</div>
<h1>Docu<span>ERP</span></h1>
<div class="sub">{escape(META["subtitle"])}</div>
<div class="rule"></div>
<div class="pipeline">{_cover_pipeline_svg()}</div>
<div class="meta">
<div><div class="k">Curso</div><div class="v">{escape(META["course"])}</div></div>
<div><div class="k">Institución</div><div class="v">{escape(META["university"])}</div></div>
<div><div class="k">Docente</div><div class="v">{escape(META["teacher"])}</div></div>
<div><div class="k">Fecha</div><div class="v">{spanish_date(day)}</div></div>
<div><div class="k">Integrantes</div><div class="v">{members}</div></div>
<div><div class="k">Caso de estudio</div><div class="v">Altamira Soluciones ERP<br>
<small style="margin-left:0">empresa ficticia · Odoo 17.0 / 18.0</small></div></div>
</div>
<div class="foot"><span>Proyecto final · Documento técnico</span><span>Guatemala</span></div>
</div></div>"""
    return _page(COVER_CSS, body, title="DocuERP — Portada")


HEADER_TEMPLATE = (
    '<div style="width:100%;font-family:Helvetica,Arial,sans-serif;font-size:7px;'
    "color:#8B94A3;padding:0 17mm;display:flex;justify-content:space-between;"
    '-webkit-print-color-adjust:exact"><span><b style="color:#10B981">DocuERP</b>'
    " &#183; Informe técnico</span><span>CC3103 Procesamiento de Lenguaje Natural"
    " &#183; UVG</span></div>"
)
FOOTER_TEMPLATE = (
    '<div style="width:100%;font-family:Helvetica,Arial,sans-serif;font-size:7px;'
    "color:#8B94A3;padding:0 17mm;display:flex;justify-content:space-between;"
    '-webkit-print-color-adjust:exact"><span>Santos &#183; Mesías</span>'
    '<span><span style="color:#16191E;font-weight:bold" class="pageNumber"></span></span></div>'
)


# ===========================================================================
# Figuras y PDF (Chromium)
# ===========================================================================


def render_diagrams(browser: Any, *, scale: float = 2.2) -> list[Path]:
    """Rasteriza los diagramas a PNG (para el DOCX) y guarda también el SVG."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    page = browser.new_page(device_scale_factor=scale)
    made = []
    for name, fn in report_diagrams.DIAGRAMS.items():
        svg = fn()
        (FIGURES_DIR / f"{name}.svg").write_text(svg, encoding="utf-8")
        html_path = BUILD_DIR / f"_diagram_{name}.html"
        html_path.write_text(
            _page("body{background:transparent}", svg, title=name), encoding="utf-8"
        )
        page.goto(html_path.as_uri())
        page.evaluate("document.fonts.ready")
        png = FIGURES_DIR / f"{name}.png"
        page.locator("svg").first.screenshot(path=str(png), omit_background=True)
        html_path.unlink()
        made.append(png)
    page.close()
    return made


def _pdf(browser: Any, html: str, out: Path, **options: Any) -> Path:
    """Escribe ``html`` en ``build/`` y lo imprime a PDF con Chromium."""
    html_path = out.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    # Ancho útil de A4 con márgenes de 17 mm ≈ 665 px: si algo desborda, Chromium encoge
    # TODO el documento al imprimir. Se avisa para corregir la maquetación en origen.
    page = browser.new_page(viewport={"width": 665, "height": 1000})
    page.emulate_media(media="print")
    page.goto(html_path.as_uri())
    page.evaluate("document.fonts.ready")
    overflow = page.evaluate(
        "() => { const w = document.documentElement.clientWidth;"
        " return [...document.querySelectorAll('body *')]"
        ".filter(e => e.getBoundingClientRect().right > w + 1)"
        ".slice(0, 5).map(e => e.tagName + ': ' + (e.textContent || '').slice(0, 50)); }"
    )
    if overflow and out.stem == "body":
        print(f"Aviso: elementos más anchos que la página en {out.name}: {overflow}")
    page.pdf(path=str(out), format="A4", print_background=True, **options)
    page.close()
    html_path.unlink()
    return out


def build_pdf(doc: Document, browser: Any, day: date) -> dict[str, Any]:
    """Portada + índice + cuerpo → ``Informe_DocuERP.pdf``; devuelve el conteo de páginas."""
    import pymupdf

    parts = BUILD_DIR / "_parts"
    parts.mkdir(parents=True, exist_ok=True)
    body_pdf = _pdf(
        browser,
        render_body_html(doc),
        parts / "body.pdf",
        display_header_footer=True,
        header_template=HEADER_TEMPLATE,
        footer_template=FOOTER_TEMPLATE,
        prefer_css_page_size=True,
    )
    body = pymupdf.open(body_pdf)
    pages = _heading_pages(doc.headings, body)
    toc_pdf = _pdf(browser, render_toc_html(pages), parts / "toc.pdf", prefer_css_page_size=True)
    cover_pdf = _pdf(
        browser, render_cover_html(day), parts / "cover.pdf", prefer_css_page_size=True
    )

    final = pymupdf.open()
    for part in (cover_pdf, toc_pdf):
        with pymupdf.open(part) as src:
            final.insert_pdf(src)
    offset = final.page_count
    final.insert_pdf(body)
    toc_pages = offset - 1
    _link_toc(final, pages, first_toc_page=1, toc_pages=toc_pages, offset=offset)
    final.set_toc(
        [[1, "Portada", 1], [1, "Índice", 2]]
        + [[h.level, f"{h.number} {h.title}".strip(), p + offset] for h, p in pages]
    )
    final.set_metadata(
        {
            "title": f"{META['title']} — Informe técnico",
            "author": ", ".join(name for name, _ in META["members"]),
            "subject": META["subtitle"],
            "keywords": "RAG, NLP, Odoo, LangGraph, BM25, bge-m3, seguridad, evaluación",
            "creator": "scripts/build_report.py (Chromium + PyMuPDF)",
        }
    )
    out = OUT_DIR / f"{OUTPUT_NAME}.pdf"
    final.save(out, garbage=3, deflate=True)
    bib_page = next((p for h, p in pages if h.title == "Bibliografía"), body.page_count + 1)
    info = {
        "pdf": str(out.relative_to(ROOT)),
        "pages_total": final.page_count,
        "pages_cover": 1,
        "pages_toc": toc_pages,
        "pages_body": body.page_count,
        "pages_body_without_bibliography": bib_page - 1,
        "pages_bibliography": body.page_count - bib_page + 1,
    }
    body.close()
    final.close()
    for part in parts.iterdir():
        part.unlink()
    parts.rmdir()
    return info


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text).lower()


def _heading_pages(headings: list[Heading], body: Any) -> list[tuple[Heading, int]]:
    """Página (1-indexada dentro del cuerpo) de cada encabezado.

    Busca, en orden, el título de cada encabezado entre las líneas del PDF con tipografía
    mayor que la del texto corrido, sin depender del esquema que genere Chromium.
    """
    sizes: dict[float, int] = {}
    for page in body:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    key = round(span["size"], 1)
                    sizes[key] = sizes.get(key, 0) + len(span["text"])
    body_size = max(sizes, key=lambda k: sizes[k]) if sizes else 10.0
    page_texts = []
    for page in body:
        big = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                if max(s["size"] for s in line["spans"]) >= body_size * 1.12:
                    big.append("".join(s["text"] for s in line["spans"]))
        page_texts.append(_norm(" ".join(big)))
    result: list[tuple[Heading, int]] = []
    page_index, cursor = 0, 0
    for head in headings:
        target = _norm(head.title)
        found = False
        for k in range(page_index, len(page_texts)):
            pos = page_texts[k].find(target, cursor if k == page_index else 0)
            if pos >= 0:
                page_index, cursor, found = k, pos + len(target), True
                break
        result.append((head, page_index + 1))
        if not found:
            print(f"Aviso: no se ubicó el encabezado «{head.title}» en el PDF")
    return result


def _link_toc(
    doc: Any, pages: list[tuple[Heading, int]], *, first_toc_page: int, toc_pages: int, offset: int
) -> None:
    """Hace clicables las entradas del índice (enlaces internos a la página destino)."""
    import pymupdf

    targets = list(pages)
    for index in range(first_toc_page, first_toc_page + toc_pages):
        page = doc[index]
        for head, target in list(targets):
            label = head.title[:40]
            hits = page.search_for(label)
            if not hits:
                continue
            rect = hits[0]
            area = pymupdf.Rect(page.rect.x0 + 50, rect.y0 - 1, page.rect.x1 - 50, rect.y1 + 1)
            page.insert_link({"kind": pymupdf.LINK_GOTO, "from": area, "page": target - 1 + offset})
            targets.remove((head, target))


# ===========================================================================
# DOCX
# ===========================================================================


def build_docx(doc: Document, browser: Any, day: date) -> Path:
    """Genera ``Informe_DocuERP.docx`` con python-docx."""
    from docx import Document as DocxDocument
    from docx.enum.section import WD_SECTION
    from docx.enum.text import WD_BREAK
    from docx.shared import Mm, Pt

    cover_png = BUILD_DIR / "_cover.png"
    page = browser.new_page(viewport={"width": 794, "height": 1123}, device_scale_factor=2)
    html_path = BUILD_DIR / "_cover.html"
    html_path.write_text(render_cover_html(day), encoding="utf-8")
    page.goto(html_path.as_uri())
    page.evaluate("document.fonts.ready")
    page.screenshot(path=str(cover_png), full_page=False)
    page.close()
    html_path.unlink()

    d = DocxDocument()
    _docx_styles(d)
    section = d.sections[0]
    section.page_width, section.page_height = Mm(210), Mm(297)
    for side in ("left_margin", "right_margin"):
        setattr(section, side, Mm(20))
    section.top_margin, section.bottom_margin = Mm(20), Mm(18)
    section.different_first_page_header_footer = True

    cover_par = d.add_paragraph()
    run = cover_par.add_run()
    _anchor_picture(run, cover_png, Mm(210), Mm(297))
    run.add_break(WD_BREAK.PAGE)
    cover_png.unlink()

    title = d.add_paragraph()
    t = title.add_run("Índice")
    t.font.size, t.bold = Pt(22), True
    _color(t, INK)
    _toc_field(d, doc.headings)
    _update_fields_on_open(d)

    body_section = d.add_section(WD_SECTION.NEW_PAGE)
    body_section.different_first_page_header_footer = False
    _restart_numbering(body_section)
    _docx_header_footer(body_section)

    md = _markdown()
    for block in doc.blocks:
        _docx_block(d, md, block)

    _font_fallbacks(d)
    out = OUT_DIR / f"{OUTPUT_NAME}.docx"
    d.core_properties.title = f"{META['title']} — Informe técnico"
    d.core_properties.author = ", ".join(name for name, _ in META["members"])
    d.core_properties.subject = META["subtitle"]
    d.core_properties.keywords = "RAG, NLP, Odoo, LangGraph, seguridad, evaluación"
    d.save(out)
    return out


FONT_FALLBACKS = {
    "Inter": ("Calibri", "swiss", "variable"),
    "JetBrains Mono": ("Consolas", "modern", "fixed"),
}
"""Fuente → (alternativa, familia, paso) que Word usa si la fuente no está instalada."""


def _font_fallbacks(d: Any) -> None:
    """Declara Inter y JetBrains Mono en ``fontTable.xml`` con ``altName`` de respaldo."""
    from lxml import etree

    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    for rel in d.part.rels.values():
        if not rel.reltype.endswith("/fontTable"):
            continue
        part = rel.target_part
        root = etree.fromstring(part.blob)
        for name, (alt, family, pitch) in FONT_FALLBACKS.items():
            font = etree.SubElement(root, f"{{{ns}}}font", {f"{{{ns}}}name": name})
            etree.SubElement(font, f"{{{ns}}}altName", {f"{{{ns}}}val": alt})
            etree.SubElement(font, f"{{{ns}}}family", {f"{{{ns}}}val": family})
            etree.SubElement(font, f"{{{ns}}}pitch", {f"{{{ns}}}val": pitch})
        part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def _hex(color: str) -> Any:
    from docx.shared import RGBColor

    return RGBColor.from_string(color.lstrip("#").upper())


def _color(run: Any, color: str) -> None:
    run.font.color.rgb = _hex(color)


def _set_font(target: Any, name: str) -> None:
    """Fija la fuente en todas las variantes de ``rFonts`` (ASCII, hAnsi, cs, eastAsia)."""
    from docx.oxml.ns import qn

    target.font.name = name
    rpr = target.element.get_or_add_rPr() if hasattr(target, "element") else None
    if rpr is None:
        return
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        from docx.oxml import OxmlElement

        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), name)


DOCX_FONT = "Inter"
DOCX_MONO = "JetBrains Mono"


def _docx_styles(d: Any) -> None:
    from docx.shared import Pt

    normal = d.styles["Normal"]
    _set_font(normal, DOCX_FONT)
    normal.font.size = Pt(10)
    normal.font.color.rgb = _hex(INK)
    pf = normal.paragraph_format
    pf.space_after = Pt(5)
    pf.line_spacing = 1.18
    for name, size, before, after in (("Heading 1", 16, 16, 6), ("Heading 2", 12, 10, 4)):
        style = d.styles[name]
        _set_font(style, DOCX_FONT)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = _hex("#0B0D10")
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
    caption = d.styles["Caption"]
    _set_font(caption, DOCX_FONT)
    caption.font.size = Pt(8.5)
    caption.font.italic = False
    caption.font.bold = False
    caption.font.color.rgb = _hex("#5B6472")
    caption.paragraph_format.space_after = Pt(8)


def _docx_header_footer(section: Any) -> None:
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt

    section.header.is_linked_to_previous = False
    section.footer.is_linked_to_previous = False
    hp = section.header.paragraphs[0]
    brand = hp.add_run("DocuERP")
    brand.bold = True
    _color(brand, ACCENT)
    rest = hp.add_run(" · Informe técnico · CC3103 Procesamiento de Lenguaje Natural · UVG")
    _color(rest, "#8B94A3")
    for r in (brand, rest):
        r.font.size = Pt(7.5)
    fp = section.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    _field(fp, "PAGE", size=8)


def _field(paragraph: Any, instruction: str, *, size: float = 8, placeholder: str = "1") -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    def fld(kind: str) -> Any:
        el = OxmlElement("w:fldChar")
        el.set(qn("w:fldCharType"), kind)
        return el

    run = paragraph.add_run()
    run.font.size = Pt(size)
    run._r.append(fld("begin"))
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = f" {instruction} "
    run2 = paragraph.add_run()
    run2.font.size = Pt(size)
    run2._r.append(instr)
    run3 = paragraph.add_run()
    run3._r.append(fld("separate"))
    run4 = paragraph.add_run(placeholder)
    run4.font.size = Pt(size)
    run5 = paragraph.add_run()
    run5._r.append(fld("end"))


def _toc_field(d: Any, headings: list[Heading]) -> None:
    """Campo TOC de Word; el resultado provisional lista las secciones sin páginas."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    p = d.add_paragraph()
    begin = p.add_run()
    el = OxmlElement("w:fldChar")
    el.set(qn("w:fldCharType"), "begin")
    begin._r.append(el)
    instr_run = p.add_run()
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = ' TOC \\o "1-2" \\h \\z \\u '
    instr_run._r.append(instr)
    sep = p.add_run()
    el = OxmlElement("w:fldChar")
    el.set(qn("w:fldCharType"), "separate")
    sep._r.append(el)
    note = p.add_run("Word actualiza este índice al abrir el documento (o con F9).")
    note.font.size = Pt(8.5)
    note.italic = True
    _color(note, "#8B94A3")
    for head in headings:
        line = d.add_paragraph()
        line.paragraph_format.left_indent = Pt(0 if head.level == 1 else 18)
        line.paragraph_format.space_after = Pt(1)
        r = line.add_run(f"{head.number}  {head.title}".strip())
        r.font.size = Pt(10 if head.level == 1 else 9)
        r.bold = head.level == 1
    end_p = d.add_paragraph()
    end = end_p.add_run()
    el = OxmlElement("w:fldChar")
    el.set(qn("w:fldCharType"), "end")
    end._r.append(el)


def _update_fields_on_open(d: Any) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    settings = d.settings.element
    el = OxmlElement("w:updateFields")
    el.set(qn("w:val"), "true")
    settings.append(el)


def _restart_numbering(section: Any) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    sect_pr = section._sectPr
    pg = sect_pr.find(qn("w:pgNumType"))
    if pg is None:
        pg = OxmlElement("w:pgNumType")
        sect_pr.append(pg)
    pg.set(qn("w:start"), "1")


def _anchor_picture(run: Any, path: Path, width: Any, height: Any) -> None:
    """Inserta una imagen flotante detrás del texto, anclada a la esquina de la página."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    run.add_picture(str(path), width=width, height=height)
    inline = run._r.find(".//" + qn("wp:inline"))
    anchor = OxmlElement("wp:anchor")
    for key, value in {
        "distT": "0",
        "distB": "0",
        "distL": "0",
        "distR": "0",
        "simplePos": "0",
        "relativeHeight": "0",
        "behindDoc": "1",
        "locked": "1",
        "layoutInCell": "1",
        "allowOverlap": "1",
    }.items():
        anchor.set(key, value)
    simple = OxmlElement("wp:simplePos")
    simple.set("x", "0")
    simple.set("y", "0")
    anchor.append(simple)
    for axis, rel in (("wp:positionH", "page"), ("wp:positionV", "page")):
        pos = OxmlElement(axis)
        pos.set("relativeFrom", rel)
        offset = OxmlElement("wp:posOffset")
        offset.text = "0"
        pos.append(offset)
        anchor.append(pos)
    anchor.append(copy.deepcopy(inline.find(qn("wp:extent"))))
    effect = OxmlElement("wp:effectExtent")
    for side in ("l", "t", "r", "b"):
        effect.set(side, "0")
    anchor.append(effect)
    anchor.append(OxmlElement("wp:wrapNone"))
    for tag in ("wp:docPr", "wp:cNvGraphicFramePr", "a:graphic"):
        anchor.append(copy.deepcopy(inline.find(qn(tag))))
    inline.getparent().replace(inline, anchor)


def _shade(element: Any, fill: str) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill.lstrip("#").upper())
    element.append(shd)


def _docx_inline(paragraph: Any, children: Any, *, size: float | None = None) -> None:
    """Agrega los tokens inline de markdown-it como *runs* con formato."""
    from docx.enum.text import WD_COLOR_INDEX
    from docx.shared import Pt

    bold = italic = link = False

    def add(text: str, *, code: bool = False, placeholder: bool = False) -> None:
        if not text:
            return
        run = paragraph.add_run(text)
        run.bold = bold or None
        run.italic = italic or None
        if size:
            run.font.size = Pt(size)
        if code or placeholder:
            _set_font(run, DOCX_MONO)
            run.font.size = Pt((size or 10) * 0.86)
            if code:
                _color(run, "#0E5E45")
                _shade(run._r.get_or_add_rPr(), "#EEF2F1")
            else:
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
                _color(run, "#8A5A00")
        if link:
            _color(run, "#0B8A60")

    for tok in children or []:
        if tok.type == "text":
            pieces = SENTINEL_RE.split(tok.content)
            for k, piece in enumerate(pieces):
                if k % 2:
                    add(f"{{{{{piece}}}}}", placeholder=True)
                else:
                    add(piece)
        elif tok.type == "code_inline":
            add(tok.content, code=True)
        elif tok.type == "strong_open":
            bold = True
        elif tok.type == "strong_close":
            bold = False
        elif tok.type == "em_open":
            italic = True
        elif tok.type == "em_close":
            italic = False
        elif tok.type == "link_open":
            link = True
        elif tok.type == "link_close":
            link = False
        elif tok.type == "softbreak":
            add(" ")
        elif tok.type == "hardbreak":
            paragraph.add_run().add_break()


def _plain(children: Any) -> str:
    return "".join(t.content for t in children or [] if t.type in {"text", "code_inline"})


def _docx_block(d: Any, md: Any, block: Block) -> None:
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Mm, Pt

    if block.kind == "heading":
        h = block.heading
        assert h is not None
        p = d.add_paragraph(style=f"Heading {h.level}")
        if not h.number and h.level == 1:
            p.paragraph_format.page_break_before = True
        if h.number:
            num = p.add_run(f"{h.number}  ")
            _color(num, ACCENT if h.level == 1 else "#0B8A60")
        _docx_inline(p, block.inline)
    elif block.kind == "para":
        p = d.add_paragraph()
        if block.css == "bib":
            p.paragraph_format.left_indent = Mm(12.7)
            p.paragraph_format.first_line_indent = Mm(-12.7)
            _docx_inline(p, block.inline, size=9)
        else:
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            _docx_inline(p, block.inline)
    elif block.kind == "list":
        style = "List Number" if block.ordered else "List Bullet"
        for item in block.items:
            _docx_inline(d.add_paragraph(style=style), item)
    elif block.kind == "table":
        if block.caption is not None:
            cap = d.add_paragraph(style="Caption")
            cap.paragraph_format.keep_with_next = True
            lead = cap.add_run(f"Tabla {block.number}. ")
            lead.bold = True
            _color(lead, INK)
            _docx_inline(cap, block.caption, size=8.5)
        _docx_table(d, block)
    elif block.kind == "figure":
        _kind, png = figure_source(block.ref)
        if png is not None and png.is_file():
            p = d.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(png), width=Mm(170))
        else:
            name = block.ref.split(":", 1)[-1]
            table = d.add_table(rows=1, cols=1)
            cell = table.rows[0].cells[0]
            cell.text = (
                f"Figura pendiente: se insertará desde reports/figures/{name}.png al ejecutar "
                "make eval-report y reconstruir el informe."
            )
            _shade(cell._tc.get_or_add_tcPr(), "#F4F6F7")
            for par in cell.paragraphs:
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in par.runs:
                    r.font.size = Pt(8.5)
                    _color(r, "#5B6472")
            _table_borders(table, dashed=True)
        cap = d.add_paragraph(style="Caption")
        lead = cap.add_run(f"Figura {block.number}. ")
        lead.bold = True
        _color(lead, INK)
        _docx_inline(cap, block.caption, size=8.5)
    elif block.kind == "pending":
        p = d.add_paragraph()
        r = p.add_run(f"{{{{{block.ref}}}}} ")
        _set_font(r, DOCX_MONO)
        r.font.size = Pt(8.5)
        from docx.enum.text import WD_COLOR_INDEX

        r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        note = p.add_run(f"Texto pendiente. {block.text}")
        note.font.size = Pt(8.5)
        _color(note, "#6B4A00")
    elif block.kind == "code":
        p = d.add_paragraph()
        r = p.add_run(block.text)
        _set_font(r, DOCX_MONO)
        r.font.size = Pt(8)


def _docx_table(d: Any, block: Block) -> None:
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Mm, Pt

    cols = len(block.header)
    table = d.add_table(rows=1 + len(block.rows), cols=cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    # Anchos proporcionales al contenido (Word no siempre recalcula el autoajuste).
    lengths = []
    for c in range(cols):
        cells = [block.header[c], *(row[c] for row in block.rows if c < len(row))]
        longest = max(min(len(_plain(x)), 60) for x in cells)
        lengths.append(max(longest, 6))
    total = sum(lengths)
    widths = [Mm(170 * n / total) for n in lengths]
    for r_index, row_cells in enumerate([block.header, *block.rows]):
        row = table.rows[r_index]
        if r_index == 0:
            tr_pr = row._tr.get_or_add_trPr()
            header_flag = OxmlElement("w:tblHeader")
            header_flag.set(qn("w:val"), "true")
            tr_pr.append(header_flag)
        for c in range(cols):
            cell = row.cells[c]
            cell.width = widths[c]
            par = cell.paragraphs[0]
            par.paragraph_format.space_after = Pt(0)
            par.paragraph_format.line_spacing = 1.05
            content = row_cells[c] if c < len(row_cells) else []
            _docx_inline(par, content, size=8.2)
            if r_index == 0:
                for run in par.runs:
                    run.bold = True
                _shade(cell._tc.get_or_add_tcPr(), "#F2F4F5")
    _table_borders(table)


def _table_borders(table: Any, *, dashed: bool = False) -> None:
    """Bordes sobrios: línea superior e inferior marcadas y separadores horizontales finos."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    spec = {
        "top": ("single", "12", "16191E"),
        "bottom": ("single", "8", "16191E"),
        "insideH": ("single", "4", "DDE2E6"),
        "left": ("nil", "0", "auto"),
        "right": ("nil", "0", "auto"),
        "insideV": ("nil", "0", "auto"),
    }
    if dashed:
        spec = dict.fromkeys(("top", "bottom", "left", "right"), ("dashed", "6", "C9D1D6"))
    for edge, (val, size, color) in spec.items():
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), val)
        el.set(qn("w:sz"), size)
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)
        borders.append(el)
    tbl_pr.append(borders)
    margins = OxmlElement("w:tblCellMar")
    for side, width in (("top", "40"), ("bottom", "40"), ("left", "80"), ("right", "80")):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), width)
        el.set(qn("w:type"), "dxa")
        margins.append(el)
    tbl_pr.append(margins)


# ===========================================================================
# PLACEHOLDERS.md
# ===========================================================================


def write_placeholders_md(doc: Document) -> Path:
    """Lista cada marcador usado, su definición, su origen y si ya está resuelto."""
    rows = sorted(doc.placeholders.values(), key=lambda r: r.name)
    pending = [r for r in rows if r.text is None]
    lines = [
        "# Marcadores del informe",
        "",
        "Generado por `scripts/build_report.py --placeholders`; no editar a mano. Cada "
        "`{{marcador}}` de las secciones se resuelve al construir el informe desde el archivo "
        "indicado. Los textos y las cifras que no salen de un archivo de resultados se "
        'escriben en `docs/report/values.json` (objeto `{"marcador": valor}`), que también '
        "sirve para corregir a mano cualquier valor. Los marcadores `{{R:clave}}` leen "
        "`.value` de `reports/RESULTS.json`.",
        "",
        f"Total: {len(rows)} marcadores; pendientes al generar esta lista: {len(pending)}.",
        "",
        "Orden de ejecución para resolverlos: `make eval-validate`, `make eval-retrieval`, "
        "`make eval-e2e`, `make eval-judge`, `make eval-report`, `make redteam-e2e`, "
        "`python scripts/collect_results.py` y luego `make report`.",
        "",
        "| Marcador | Definición | Origen | Estado |",
        "|---|---|---|---|",
    ]
    for res in rows:
        state = f"resuelto: {res.text}" if res.text is not None else "pendiente"
        state = state.replace("|", "\\|").replace("\n", " ")
        if len(state) > 60:
            state = state[:57] + "…"
        lines.append(f"| `{{{{{res.name}}}}}` | {res.description} | {res.origin} | {state} |")
    PLACEHOLDERS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return PLACEHOLDERS_FILE


# ===========================================================================
# CLI
# ===========================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pdf-only", action="store_true", help="no genera el DOCX")
    parser.add_argument("--docx-only", action="store_true", help="no genera el PDF")
    parser.add_argument("--skip-figures", action="store_true", help="reutiliza los PNG existentes")
    parser.add_argument("--placeholders", action="store_true", help="solo PLACEHOLDERS.md")
    parser.add_argument("--check", action="store_true", help="valida marcadores y referencias")
    parser.add_argument("--date", help="fecha de la portada (AAAA-MM-DD); por defecto, hoy")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="rellena los marcadores pendientes con valores ficticios para medir la "
        "extensión; escribe en build/_preview/ (no versionado)",
    )
    args = parser.parse_args(argv)

    global OUT_DIR
    day = date.fromisoformat(args.date) if args.date else date.today()
    resolver = Resolver(preview=args.preview)
    doc = build_document(resolver)
    pending = sorted(n for n, r in doc.placeholders.items() if r.text is None)
    print(
        f"Secciones: {len(section_files())} · tablas: {len(doc.tables)} · figuras: "
        f"{len(doc.figures)} · marcadores: {len(doc.placeholders)} ({len(pending)} pendientes)"
    )
    if args.check:
        return 0
    if args.preview:
        OUT_DIR = BUILD_DIR / "_preview"
    else:
        write_placeholders_md(doc)
    if args.placeholders:
        print(f"Escrito {PLACEHOLDERS_FILE.relative_to(ROOT)}")
        return 0

    from playwright.sync_api import sync_playwright

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    previous = _read_json(OUT_DIR / "build_info.json") if args.pdf_only or args.docx_only else None
    info: dict[str, Any] = previous if isinstance(previous, dict) else {}
    info.update({"date": day.isoformat(), "pending_placeholders": pending})
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            if not args.skip_figures or not all(
                (FIGURES_DIR / f"{n}.png").is_file() for n in report_diagrams.DIAGRAMS
            ):
                render_diagrams(browser)
            if not args.docx_only:
                info.update(build_pdf(doc, browser, day))
            if not args.pdf_only:
                info["docx"] = str(build_docx(doc, browser, day).relative_to(ROOT))
        finally:
            browser.close()
    (OUT_DIR / "build_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for key in ("pdf", "docx"):
        if key in info:
            print(f"Escrito {info[key]}")
    if "pages_total" in info:
        print(
            f"Páginas: {info['pages_total']} en total · cuerpo sin bibliografía: "
            f"{info['pages_body_without_bibliography']} · bibliografía: "
            f"{info['pages_bibliography']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
