"""CLI de evaluación: ``validate``, ``retrieval``, ``e2e``, ``judge``, ``report`` y ``compare``."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import typer

app = typer.Typer(help="Evaluación: recuperación, extremo a extremo, juez LLM y reportes.")


@contextmanager
def _friendly_errors() -> Iterator[None]:
    """Convierte los errores esperados (datos faltantes, parámetros inválidos) en un
    mensaje claro y código de salida 1, sin traza de Python."""
    try:
        yield
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        typer.secho(f"Error: {exc}", fg="red", err=True)
        raise typer.Exit(code=1) from exc


def _split_csv(value: str) -> list[str]:
    """Separa una lista por comas ignorando vacíos."""
    return [item.strip() for item in value.split(",") if item.strip()]


def _run[T](action: Callable[[], T]) -> T:
    """Ejecuta ``action`` con manejo amable de errores."""
    with _friendly_errors():
        return action()


@app.command("validate")
def validate(
    path: Path | None = typer.Option(None, help="Banco JSONL (por defecto eval/benchmark.jsonl)"),
    chunker: str = typer.Option(
        "structure", help="Chunker cuyo corpus procesado valida las evidencias"
    ),
    corpus: bool = typer.Option(
        True, "--corpus/--no-corpus", help="Verificar las evidencias contra data/processed"
    ),
) -> None:
    """Valida el banco: formato, composición mínima (5/3/3/3/2/2/2 + 10 + 12) y evidencias."""
    from docuerp.eval.benchmark import validate_benchmark
    from docuerp.eval.chunks import ChunkStore

    def action() -> bool:
        store = ChunkStore.default(chunker) if corpus else None
        report = validate_benchmark(path, store=store)
        for line in report.composition.lines():
            typer.echo(line)
        for issue in report.issues:
            typer.secho(str(issue), fg="red" if issue.severity == "error" else "yellow")
        if corpus and store is None:
            typer.secho(
                "Aviso: no hay corpus procesado; no se verificaron las evidencias.", fg="yellow"
            )
        typer.echo(
            f"Errores: {len(report.errors)} · Avisos: {len(report.issues) - len(report.errors)}"
        )
        return report.ok

    ok = _run(action)
    if not ok:
        typer.secho("El banco NO es válido todavía.", fg="red", err=True)
        raise typer.Exit(code=1)
    typer.secho("El banco es válido.", fg="green")


@app.command("retrieval")
def retrieval(
    configs: str = typer.Option(
        "bm25,dense,hybrid,hybrid_rerank,hybrid_rerank_mmr",
        help="Modos (o modo:chunker:embedder) separados por coma",
    ),
    chunkers: str = typer.Option("", help="Chunkers a cruzar con los modos (coma); vacío = config"),
    embedders: str = typer.Option(
        "", help="Embedders a cruzar con los modos (coma); vacío = config"
    ),
    split: str | None = typer.Option(None, help="dev | test"),
    name: str = typer.Option("retrieval", help="Sufijo del run_id"),
    run_id: str | None = typer.Option(None, help="Reanuda una corrida existente"),
) -> None:
    """Evalúa la recuperación por configuración (modos × chunkers × embedders)."""
    from docuerp.eval.retrieval_eval import run_retrieval_eval

    run_dir = _run(
        lambda: run_retrieval_eval(
            _split_csv(configs),
            split=split,
            chunkers=_split_csv(chunkers) or None,
            embedders=_split_csv(embedders) or None,
            name=name,
            run_id=run_id,
        )
    )
    typer.echo(f"Resultados de recuperación en {run_dir}")


@app.command("e2e")
def e2e(
    config: str = typer.Option("final", help="baseline | final"),
    runs: int = typer.Option(3, help="Repeticiones por caso"),
    split: str | None = typer.Option(None, help="dev | test"),
    run_id: str | None = typer.Option(None, help="Reanuda una corrida existente"),
    concurrency: int = typer.Option(1, help="Consultas simultáneas (1 = latencias limpias)"),
) -> None:
    """Evaluación extremo a extremo (reanudable con --run-id)."""
    import asyncio

    from docuerp.eval.e2e import run_e2e_eval

    run_dir = _run(
        lambda: asyncio.run(
            run_e2e_eval(config, runs=runs, split=split, run_id=run_id, concurrency=concurrency)
        )
    )
    typer.echo(f"Resultados e2e en {run_dir}")


@app.command("judge")
def judge(
    run_dir: str = typer.Argument(..., help="Directorio eval/results/<run_id>"),
    model: str | None = typer.Option(
        None, help="Modelo juez (por defecto settings.llm.judge_model)"
    ),
    concurrency: int = typer.Option(4, help="Llamadas simultáneas al LLM"),
) -> None:
    """Ejecuta el LLM-as-judge sobre una corrida e2e (con caché en disco)."""
    import asyncio

    from docuerp.eval.judge import run_judge

    out = _run(lambda: asyncio.run(run_judge(Path(run_dir), model=model, concurrency=concurrency)))
    typer.echo(f"Veredictos en {out}")


@app.command("report")
def report(
    run_dirs: list[Path] = typer.Argument(
        None, help="Corridas a incluir (por defecto la última de cada tipo)"
    ),
) -> None:
    """Genera el reporte comparativo, las figuras y eval/results/latest/summary.json."""
    from docuerp.eval.report import build_report

    path = _run(lambda: build_report(list(run_dirs) if run_dirs else None))
    typer.echo(f"Reporte en {path}")


@app.command("compare")
def compare(
    baseline: Path | None = typer.Argument(
        None, help="Corrida e2e baseline (por defecto la última)"
    ),
    final: Path | None = typer.Argument(None, help="Corrida e2e final (por defecto la última)"),
) -> None:
    """Compara baseline vs final con IC 95 % y prueba pareada (permutación + Wilcoxon)."""
    from docuerp.eval import runs
    from docuerp.eval.report import METRIC_LABELS, compare_runs

    def action() -> dict[str, object]:
        base_dir = baseline or runs.latest_run("e2e", "baseline")
        final_dir = final or runs.latest_run("e2e", "final")
        if base_dir is None or final_dir is None:
            raise ValueError("Se necesitan una corrida e2e baseline y una final")
        return compare_runs(base_dir, final_dir)

    result = _run(action)
    typer.echo(f"Baseline: {result['baseline_run']}  ·  Final: {result['final_run']}")
    typer.echo(f"{'Métrica':<34}{'Base':>8}{'Final':>8}{'Δ':>9}{'p perm.':>10}{'p Wilc.':>10}")
    paired: dict[str, dict[str, float]] = result["paired"]  # type: ignore[assignment]
    for metric, row in paired.items():
        wilcoxon = row["p_wilcoxon"]
        typer.echo(
            f"{METRIC_LABELS[metric]:<34}{row['baseline']:>8.3f}{row['final']:>8.3f}"
            f"{row['delta']:>+9.3f}{row['p_value']:>10.3f}"
            f"{'—' if wilcoxon != wilcoxon else format(wilcoxon, '.3f'):>10}"
        )
