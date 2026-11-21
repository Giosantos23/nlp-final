"""CLI de recuperación: ``docuerp retrieval search`` y ``docuerp retrieval warmup``.

Sirven para inspeccionar a mano qué recupera cada modo y cuánto tarda cada etapa, sin
levantar el agente ni el LLM.
"""

from __future__ import annotations

from typing import get_args

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from docuerp.retrieval.contracts import RetrievalMode, RetrievalQuery, RetrievalResult
from docuerp.retrieval.gate import evidence_score
from docuerp.retrieval.pipeline import build_retriever, warmup
from docuerp.retrieval.store import IndexNotFoundError
from docuerp.schemas import ErpVersion, ScoredChunk, UserRole

app = typer.Typer(help="Recuperación híbrida: búsqueda e inspección de etapas.")


def _console(*, stderr: bool = False) -> Console:
    """Consola rich creada al usarse (no al importar), para respetar el ancho del terminal
    vigente y la salida estándar redirigida (p. ej. en pruebas)."""
    return Console(stderr=stderr)


def _choice(value: str, options: tuple[str, ...], name: str) -> str:
    """Valida que ``value`` esté entre ``options``; si no, error de uso con las opciones."""
    if value not in options:
        raise typer.BadParameter(
            f"{value!r} no es válido para {name}. Opciones: {', '.join(options)}"
        )
    return value


def _hits_table(title: str, hits: list[ScoredChunk]) -> Table:
    """Tabla rich con rank, score, fuente, página, sección y banderas de cada hit."""
    table = Table(title=title, title_justify="left", header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("evid.", justify="right")
    table.add_column("source", overflow="fold")
    table.add_column("page", justify="right")
    table.add_column("section_path", overflow="fold")
    table.add_column("flags", overflow="fold")
    for hit in hits:
        chunk = hit.chunk
        pages = (
            str(chunk.page) if chunk.page_end == chunk.page else f"{chunk.page}-{chunk.page_end}"
        )
        table.add_row(
            str(hit.rank),
            f"{hit.score:.4f}",
            f"{evidence_score(hit):.2f}",
            chunk.source,
            pages,
            chunk.section_path,
            ", ".join(chunk.flags) or "-",
        )
    return table


def _print_result(result: RetrievalResult, *, show_stages: bool) -> None:
    """Imprime los hits finales, el veredicto del gate y las latencias por etapa."""
    if show_stages:
        for name, hits in result.stages.items():
            _console().print(_hits_table(f"Etapa: {name} ({len(hits)})", hits))
    if result.hits:
        _console().print(_hits_table(f"Top {len(result.hits)} final", result.hits))
    else:
        _console().print("[yellow]Sin resultados autorizados para esta consulta.[/yellow]")
    top = "-" if result.top_score is None else f"{result.top_score:.3f}"
    verdict = (
        "[green]suficiente[/green]"
        if result.evidence_ok
        else "[red]insuficiente (abstenerse)[/red]"
    )
    _console().print(f"Evidencia: {verdict}  top_score={top}  filtered_out={result.filtered_out}")
    _console().print(
        "Tiempos (ms): " + "  ".join(f"{name}={ms:.1f}" for name, ms in result.timings_ms.items())
    )


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Consulta en lenguaje natural"),
    role: str = typer.Option("n1", help="portal_user | n1 | n2 | admin"),
    erp_version: str = typer.Option("18.0", "--version", "--erp-version", help="17.0 | 18.0"),
    mode: str = typer.Option(
        "hybrid_rerank", help="bm25 | dense | hybrid | hybrid_rerank | hybrid_rerank_mmr"
    ),
    top_k: int = typer.Option(5, help="Número de resultados"),
    candidates_k: int = typer.Option(20, help="Candidatos por etapa antes del rerank"),
    chunker: str = typer.Option("structure", help="structure | fixed256 | fixed768"),
    embedder: str = typer.Option("bge-m3", help="bge-m3 | e5-base | fake"),
    include_obsolete: bool = typer.Option(
        False, "--include-obsolete", help="Incluir documentos obsoletos"
    ),
    stages: bool = typer.Option(False, "--stages", help="Mostrar también cada etapa intermedia"),
) -> None:
    """Ejecuta una búsqueda y muestra los resultados por etapa."""
    _choice(role, get_args(UserRole), "--role")
    _choice(erp_version, get_args(ErpVersion), "--version")
    _choice(mode, get_args(RetrievalMode), "--mode")
    try:
        request = RetrievalQuery(
            text=query,
            role=role,
            erp_version=erp_version,
            top_k=top_k,
            candidates_k=candidates_k,
            include_obsolete=include_obsolete,
        )
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        retriever = build_retriever(mode, chunker=chunker, embedder=embedder)  # type: ignore[arg-type]
        result = retriever.retrieve(request)
    except IndexNotFoundError as exc:
        _console(stderr=True).print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    _print_result(result, show_stages=stages)


@app.command("warmup")
def warmup_command(
    mode: str = typer.Option("hybrid_rerank", help="Modo cuyos modelos e índices se precargan"),
    chunker: str = typer.Option("structure", help="structure | fixed256 | fixed768"),
    embedder: str = typer.Option("bge-m3", help="bge-m3 | e5-base | fake"),
) -> None:
    """Precarga índices y modelos (embedder, reranker) y reporta el tiempo de cada uno."""
    _choice(mode, get_args(RetrievalMode), "--mode")
    try:
        timings = warmup(mode, chunker=chunker, embedder=embedder)  # type: ignore[arg-type]
    except IndexNotFoundError as exc:
        _console(stderr=True).print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    table = Table(title="Calentamiento", title_justify="left", header_style="bold")
    table.add_column("componente")
    table.add_column("ms", justify="right")
    for name, ms in timings.items():
        table.add_row(name, f"{ms:.0f}")
    _console().print(table)
