"""Evaluación extremo a extremo vía ``docuerp.agent.service.answer_query``.

Para cada caso del banco y cada repetición se arma un :class:`~docuerp.schemas.QueryRequest`
(con el rol, usuario y versión del caso), se invoca al sistema y se guarda una
:class:`~docuerp.eval.models.PredictionRecord` con la respuesta, las citas, la intención,
las latencias por etapa y el consumo del LLM.

Dos variantes comparables (mismo contrato de salida):

* ``final``: el servicio completo (agente, herramientas, guardias, reranker, gate τ).
* ``baseline``: BM25 + prompt estructurado, sin agente (:mod:`docuerp.eval.baseline`).

La ejecución es **reanudable**: con el mismo ``run_id`` se omiten las repeticiones ya
guardadas sin error; las que fallaron se reintentan. Las repeticiones se ejecutan de a una
por defecto, para que las latencias medidas no se contaminen entre sí.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docuerp.agent.tools.base import simulate_unavailable
from docuerp.config import get_settings
from docuerp.eval import aggregate
from docuerp.eval import runs as run_io
from docuerp.eval.benchmark import BenchmarkCase, load_benchmark
from docuerp.eval.chunks import ChunkStore
from docuerp.eval.models import PredictionRecord, RunConfig
from docuerp.eval.stats import DEFAULT_RESAMPLES
from docuerp.schemas import QueryRequest, QueryResponse

log = logging.getLogger(__name__)

CONFIGS: tuple[str, ...] = ("baseline", "final")
CASE_TIMEOUT_S = 180.0
"""Tiempo máximo por consulta; si se excede, la repetición se guarda con ``error``."""

AnswerFn = Callable[[QueryRequest], Awaitable[QueryResponse]]


def resolve_answer_fn(config: str) -> AnswerFn:
    """Devuelve la función que responde consultas para la variante ``config``.

    Raises:
        ValueError: si la variante no es ``baseline`` ni ``final``.
    """
    if config == "baseline":
        from docuerp.eval.baseline import answer_query_baseline

        return answer_query_baseline
    if config == "final":
        from docuerp.agent.service import answer_query

        return answer_query
    raise ValueError(f"Variante desconocida: {config!r}. Opciones: {', '.join(CONFIGS)}")


def request_for_case(case: BenchmarkCase, run: int) -> QueryRequest:
    """Construye la consulta del caso. La sesión es única por (caso, repetición), así el
    límite de borradores por sesión de una repetición no afecta a otra."""
    return QueryRequest(
        query=case.question,
        role=case.role,
        user_id=case.user_id,
        erp_version=case.erp_version,
        session_id=f"eval-{case.id}-r{run}",
    )


def prediction_from_response(
    case: BenchmarkCase, run: int, config: str, response: QueryResponse
) -> PredictionRecord:
    """Aplana una ``QueryResponse`` en el registro que se guarda en ``predictions.jsonl``."""
    stage_ms: dict[str, float] = {}
    stage_status: dict[str, str] = {}
    api_ms: float | None = None
    for event in response.events:
        stage_ms[event.stage] = stage_ms.get(event.stage, 0.0) + event.duration_ms
        stage_status[event.stage] = event.status
        value = event.data.get("api_latency_ms")
        if isinstance(value, int | float) and not isinstance(value, bool):
            api_ms = (api_ms or 0.0) + float(value)
    answer = response.response
    return PredictionRecord(
        case_id=case.id,
        run=run,
        config=config,
        question=case.question,
        answer=answer.answer,
        sources=answer.sources,
        abstained=answer.abstained,
        tool_used=answer.tool_used,
        needs_human_review=answer.needs_human_review,
        intent=response.intent,
        blocked=response.blocked,
        block_reason=response.block_reason,
        pending_action=response.pending_action is not None,
        latency_ms=response.latency_ms,
        stage_latency_ms=stage_ms,
        stage_status=stage_status,
        api_latency_ms=api_ms,
        usage=response.usage,
        trace_id=response.trace_id,
    )


async def evaluate_one(
    case: BenchmarkCase, run: int, config: str, answer_fn: AnswerFn
) -> PredictionRecord:
    """Ejecuta una repetición de un caso. Un fallo se registra en ``error`` (no se propaga)."""
    try:
        with simulate_unavailable(case.simulated_outages):
            response = await asyncio.wait_for(
                answer_fn(request_for_case(case, run)), timeout=CASE_TIMEOUT_S
            )
    except Exception as exc:  # el servicio no debe poder abortar la corrida completa
        log.error("Falla en %s (run %d): %s", case.id, run, exc)
        return PredictionRecord(
            case_id=case.id,
            run=run,
            config=config,
            question=case.question,
            error=f"{type(exc).__name__}: {exc}",
        )
    return prediction_from_response(case, run, config, response)


def finalize_e2e_run(
    run_dir: Path,
    cases: Sequence[BenchmarkCase],
    *,
    store: ChunkStore | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any]:
    """Recalcula ``metrics.json`` desde ``predictions.jsonl`` (+ ``judgments.jsonl``).

    Se puede invocar de nuevo después del juez para incorporar fidelidad, corrección y
    respaldo de citas.
    """
    config = run_io.read_config(run_dir)
    store = store if store is not None else ChunkStore.default(get_settings().retrieval.chunker)
    data = aggregate.compute_e2e_metrics(
        cases,
        run_io.read_predictions(run_dir),
        run_io.read_judgments(run_dir),
        spans=store.spans() if store else None,
        n_resamples=n_resamples,
    )
    data.update(
        {
            "run_id": config.run_id,
            "name": config.name,
            "generator_model": config.generator_model,
            "runs": config.runs,
            "split": config.split,
        }
    )
    run_io.write_json(run_dir / run_io.METRICS_FILE, data)
    return data


async def run_e2e_eval(
    config: str = "final",
    *,
    runs: int = 3,
    split: str | None = None,
    run_id: str | None = None,
    name: str | None = None,
    results_dir: Path | None = None,
    cases: Sequence[BenchmarkCase] | None = None,
    answer_fn: AnswerFn | None = None,
    concurrency: int = 1,
    store: ChunkStore | None = None,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> Path:
    """Ejecuta el banco ``runs`` veces y escribe predicciones y métricas.

    Args:
        config: ``baseline`` o ``final``.
        runs: repeticiones por caso (≥ 1).
        split: ``dev`` o ``test``; ``None`` usa todo el banco.
        run_id: id de una corrida existente para reanudarla.
        name: sufijo del ``run_id`` nuevo (por defecto el nombre de la variante).
        results_dir: raíz de resultados (por defecto ``eval/results``).
        cases: casos ya cargados (para pruebas).
        answer_fn: reemplazo de la función que responde (para pruebas).
        concurrency: consultas simultáneas (1 = latencias limpias).
        store: corpus para validar citas por rango de páginas (opcional).
        n_resamples: remuestreos bootstrap de los IC.

    Returns:
        Directorio de la corrida.

    Raises:
        ValueError: parámetros inválidos, banco vacío o ``run_id`` de otra variante.
    """
    if runs < 1 or concurrency < 1:
        raise ValueError("runs y concurrency deben ser ≥ 1")
    fn = answer_fn or resolve_answer_fn(config)
    pool = list(cases) if cases is not None else load_benchmark(split=split)
    if not pool:
        raise ValueError("El banco de pruebas no tiene casos para la partición pedida")

    settings = get_settings()
    run_dir = (
        run_io.run_path(run_id, results_dir)
        if run_id
        else run_io.run_path(run_io.new_run_id(name or config), results_dir)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / run_io.CONFIG_FILE).is_file():
        existing = run_io.read_config(run_dir)
        if existing.name != config:
            raise ValueError(f"{run_dir.name} pertenece a la variante {existing.name!r}")
    else:
        run_io.write_config(
            run_dir,
            RunConfig(
                kind="e2e",
                run_id=run_dir.name,
                name=config,
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                split=split,
                runs=runs,
                seed=settings.seed,
                generator_model=settings.llm.model,
                settings=settings.safe_dump(),
                case_ids=[c.id for c in pool],
            ),
        )

    done = {p.key for p in run_io.read_predictions(run_dir) if p.error is None}
    pending = [(r, c) for r in range(runs) for c in pool if (c.id, r) not in done]
    log.info("e2e %s: %d pendientes de %d", config, len(pending), runs * len(pool))

    semaphore = asyncio.Semaphore(concurrency)

    async def worker(run: int, case: BenchmarkCase) -> None:
        async with semaphore:
            record = await evaluate_one(case, run, config, fn)
        run_io.append_jsonl(run_dir / run_io.PREDICTIONS_FILE, record)

    await asyncio.gather(*(worker(r, c) for r, c in pending))
    finalize_e2e_run(run_dir, pool, store=store, n_resamples=n_resamples)
    return run_dir
