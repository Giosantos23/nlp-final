"""Evaluación de recuperación por configuración (ablaciones).

Una *configuración* es la tripleta ``modo × chunker × embedder``. Para cada una se corre el
recuperador sobre los casos del banco **que tienen evidencia esperada** y se guarda, por
caso, el ranking completo (con una marca de relevancia por página) y sus métricas
(Recall@k, Precision@k, hit@k, MRR, nDCG@5, redundancia y ruido en el top-5).

Salida en ``eval/results/<run_id>/``: ``config.json``, ``predictions.jsonl`` (una línea por
configuración y caso) y ``metrics.json`` (agregados con IC 95 %, por categoría y por caso).
La corrida es reanudable: si se vuelve a invocar con el mismo ``run_id`` se omiten los pares
``(configuración, caso)`` ya calculados.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from docuerp.config import get_settings
from docuerp.eval import aggregate, metrics, runs
from docuerp.eval.benchmark import BenchmarkCase, load_benchmark
from docuerp.eval.models import RunConfig
from docuerp.eval.stats import DEFAULT_RESAMPLES
from docuerp.retrieval.contracts import RetrievalMode, RetrievalQuery, Retriever

log = logging.getLogger(__name__)

VALID_MODES: tuple[str, ...] = ("bm25", "dense", "hybrid", "hybrid_rerank", "hybrid_rerank_mmr")
EVAL_TOP_K = 10
"""Se recupera hasta k=10 para poder reportar Recall@10."""


@dataclass(frozen=True)
class RetrievalConfigSpec:
    """Una configuración evaluada: modo de recuperación, chunker y embedder."""

    mode: str
    chunker: str
    embedder: str

    @property
    def label(self) -> str:
        """Etiqueta estable ``modo/chunker/embedder`` usada como llave en los resultados."""
        return f"{self.mode}/{self.chunker}/{self.embedder}"


def parse_config_spec(spec: str, *, chunker: str, embedder: str) -> RetrievalConfigSpec:
    """Interpreta ``modo`` o ``modo:chunker:embedder`` (los faltantes usan los por defecto).

    Raises:
        ValueError: modo desconocido o formato con más de tres partes.
    """
    parts = [p.strip() for p in spec.split(":")]
    if not 1 <= len(parts) <= 3 or not parts[0]:
        raise ValueError(f"Configuración inválida: {spec!r} (use modo[:chunker[:embedder]])")
    mode = parts[0]
    if mode not in VALID_MODES:
        raise ValueError(f"Modo desconocido: {mode!r}. Opciones: {', '.join(VALID_MODES)}")
    return RetrievalConfigSpec(
        mode=mode,
        chunker=parts[1] if len(parts) > 1 and parts[1] else chunker,
        embedder=parts[2] if len(parts) > 2 and parts[2] else embedder,
    )


def expand_configs(
    configs: Sequence[str],
    chunkers: Sequence[str] | None = None,
    embedders: Sequence[str] | None = None,
) -> list[RetrievalConfigSpec]:
    """Producto cartesiano ``modos × chunkers × embedders`` sin repetidos.

    Los elementos de ``configs`` que ya traen chunker/embedder (``modo:chunker:embedder``)
    se respetan tal cual; el resto se expande con ``chunkers`` y ``embedders``.
    """
    settings = get_settings()
    chunker_list = list(chunkers or [settings.retrieval.chunker])
    embedder_list = list(embedders or [settings.retrieval.embedder])
    specs: dict[str, RetrievalConfigSpec] = {}
    for raw in configs:
        if ":" in raw:
            spec = parse_config_spec(raw, chunker=chunker_list[0], embedder=embedder_list[0])
            specs.setdefault(spec.label, spec)
            continue
        for chunker in chunker_list:
            for embedder in embedder_list:
                spec = parse_config_spec(raw, chunker=chunker, embedder=embedder)
                specs.setdefault(spec.label, spec)
    return list(specs.values())


class RankedChunk(BaseModel):
    """Un elemento del ranking guardado en la predicción."""

    model_config = ConfigDict(extra="forbid")

    rank: int
    chunk_id: str
    source: str
    page: int
    page_end: int
    score: float
    relevant: bool


class RetrievalCaseRecord(BaseModel):
    """Línea de ``predictions.jsonl`` de una corrida de recuperación."""

    model_config = ConfigDict(extra="forbid")

    config: str
    case_id: str
    ranking: list[RankedChunk] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    evidence_ok: bool | None = None
    top_score: float | None = None
    filtered_out: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


RetrieverFactory = Callable[[RetrievalConfigSpec], Retriever]


def _default_factory(spec: RetrievalConfigSpec) -> Retriever:
    """Construye el recuperador real (carga índices y modelos: puede tardar)."""
    from docuerp.retrieval import build_retriever

    mode: RetrievalMode = spec.mode  # type: ignore[assignment]  # validado en parse_config_spec
    return build_retriever(mode, spec.chunker, spec.embedder)


def evaluate_case(retriever: Retriever, case: BenchmarkCase, label: str) -> RetrievalCaseRecord:
    """Recupera para un caso y calcula sus métricas.

    Un fallo del recuperador no aborta la corrida: se registra en ``error`` y el caso
    cuenta con métricas en cero (el sistema no recuperó nada útil).
    """
    query = RetrievalQuery(
        text=case.question,
        role=case.role,
        erp_version=case.erp_version,
        top_k=EVAL_TOP_K,
        candidates_k=max(get_settings().retrieval.candidates_k, EVAL_TOP_K),
    )
    started = time.perf_counter()
    try:
        result = retriever.retrieve(query)
    except Exception as exc:  # el recuperador es código de terceros dentro de la evaluación
        log.error("Falla de recuperación en %s (%s): %s", case.id, label, exc)
        zero = metrics.retrieval_case_metrics([], case.evidence)
        return RetrievalCaseRecord(
            config=label,
            case_id=case.id,
            metrics=zero,
            timings_ms={"total": (time.perf_counter() - started) * 1000},
            error=f"{type(exc).__name__}: {exc}",
        )
    hits = result.hits[:EVAL_TOP_K]
    chunks = [h.chunk for h in hits]
    relevance = metrics.relevance_vector(chunks, case.evidence)
    timings = dict(result.timings_ms)
    timings.setdefault("total", sum(timings.values()) or (time.perf_counter() - started) * 1000)
    return RetrievalCaseRecord(
        config=label,
        case_id=case.id,
        ranking=[
            RankedChunk(
                rank=i,
                chunk_id=h.chunk.chunk_id,
                source=h.chunk.source,
                page=h.chunk.page,
                page_end=h.chunk.page_end,
                score=h.score,
                relevant=relevance[i - 1],
            )
            for i, h in enumerate(hits, start=1)
        ],
        metrics=metrics.retrieval_case_metrics(chunks, case.evidence),
        evidence_ok=result.evidence_ok,
        top_score=result.top_score,
        filtered_out=result.filtered_out,
        timings_ms=timings,
    )


def finalize_retrieval_run(
    run_dir: Path,
    cases: Sequence[BenchmarkCase],
    *,
    failed_configs: dict[str, str] | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any]:
    """Agrega ``predictions.jsonl`` en ``metrics.json`` y lo devuelve."""
    by_id = {c.id: c for c in cases}
    per_config: dict[str, dict[str, dict[str, float]]] = {}
    latencies: dict[str, list[float]] = {}
    errors: dict[str, int] = {}
    latest: dict[tuple[str, str], RetrievalCaseRecord] = {}
    for record in runs.iter_jsonl(run_dir / runs.PREDICTIONS_FILE, RetrievalCaseRecord):
        latest[(record.config, record.case_id)] = record
    for (label, case_id), record in latest.items():
        if case_id not in by_id:
            continue
        per_config.setdefault(label, {})[case_id] = record.metrics
        latencies.setdefault(label, []).append(record.timings_ms.get("total", 0.0))
        errors[label] = errors.get(label, 0) + (1 if record.error else 0)

    out_configs: dict[str, Any] = {}
    for label, per_case in per_config.items():
        mode, chunker, embedder = label.split("/")
        summary = aggregate.aggregate_retrieval(per_case, by_id, n_resamples=n_resamples)
        lat = latencies[label]
        out_configs[label] = {
            "mode": mode,
            "chunker": chunker,
            "embedder": embedder,
            **summary,
            "errors": errors[label],
            "latency_ms": {
                "mean": sum(lat) / len(lat),
                "p50": metrics.percentile(lat, 50),
                "p95": metrics.percentile(lat, 95),
            },
            "per_case": per_case,
        }
    data = {
        "kind": "retrieval",
        "ks": list(metrics.RETRIEVAL_KS),
        "configs": out_configs,
        "failed_configs": failed_configs or {},
    }
    runs.write_json(run_dir / runs.METRICS_FILE, data)
    return data


def run_retrieval_eval(
    configs: list[str],
    *,
    split: str | None = None,
    chunkers: Sequence[str] | None = None,
    embedders: Sequence[str] | None = None,
    name: str = "retrieval",
    run_id: str | None = None,
    results_dir: Path | None = None,
    cases: Sequence[BenchmarkCase] | None = None,
    retriever_factory: RetrieverFactory | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> Path:
    """Evalúa cada configuración y escribe ``eval/results/<run_id>/``; devuelve el dir.

    Args:
        configs: modos (``bm25``, ``hybrid_rerank``…) o ``modo:chunker:embedder``.
        split: ``dev`` o ``test``; ``None`` usa todo el banco.
        chunkers: chunkers a cruzar con los modos (por defecto el de ``settings``).
        embedders: embedders a cruzar con los modos (por defecto el de ``settings``).
        name: sufijo del ``run_id`` si se genera uno nuevo.
        run_id: id de una corrida existente para **reanudarla** (omite lo ya calculado).
        results_dir: raíz de resultados (por defecto ``eval/results``).
        cases: casos ya cargados (para pruebas); por defecto se lee el banco.
        retriever_factory: constructor de recuperadores (inyectable en pruebas).
        n_resamples: remuestreos bootstrap de los IC.

    Raises:
        ValueError: sin configuraciones o sin casos con evidencia.
        RuntimeError: si ninguna configuración pudo construirse.
    """
    specs = expand_configs(configs, chunkers, embedders)
    if not specs:
        raise ValueError("No se indicó ninguna configuración de recuperación")
    pool = list(cases) if cases is not None else load_benchmark(split=split)
    eval_cases = [c for c in pool if c.has_evidence]
    if not eval_cases:
        raise ValueError("El banco no tiene casos con evidencia esperada para evaluar recuperación")

    settings = get_settings()
    run_dir = (
        runs.run_path(run_id, results_dir)
        if run_id
        else runs.run_path(runs.new_run_id(name), results_dir)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / runs.CONFIG_FILE).is_file():
        runs.write_config(
            run_dir,
            RunConfig(
                kind="retrieval",
                run_id=run_dir.name,
                name=name,
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                split=split,
                seed=settings.seed,
                configs=[s.label for s in specs],
                case_ids=[c.id for c in eval_cases],
                settings=settings.safe_dump()["retrieval"],
            ),
        )

    done = {
        (r.config, r.case_id)
        for r in runs.iter_jsonl(run_dir / runs.PREDICTIONS_FILE, RetrievalCaseRecord)
        if r.error is None
    }
    factory = retriever_factory or _default_factory
    failed: dict[str, str] = {}
    for spec in specs:
        pending = [c for c in eval_cases if (spec.label, c.id) not in done]
        if not pending:
            log.info("Config %s: todo calculado, se omite", spec.label)
            continue
        try:
            retriever = factory(spec)
        except Exception as exc:  # índice ausente, modelo no disponible…
            log.error("No se pudo construir %s: %s", spec.label, exc)
            failed[spec.label] = f"{type(exc).__name__}: {exc}"
            continue
        for case in pending:
            runs.append_jsonl(
                run_dir / runs.PREDICTIONS_FILE, evaluate_case(retriever, case, spec.label)
            )
        log.info("Config %s: %d casos evaluados", spec.label, len(pending))

    data = finalize_retrieval_run(
        run_dir, eval_cases, failed_configs=failed, n_resamples=n_resamples
    )
    if not data["configs"]:
        raise RuntimeError(f"Ninguna configuración pudo evaluarse: {failed}")
    return run_dir
