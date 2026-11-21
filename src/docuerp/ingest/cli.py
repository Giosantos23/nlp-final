"""CLI de ingesta: ``docuerp ingest run``, ``ingest report`` e ``ingest odoo-pdfs``."""

from __future__ import annotations

import logging

import typer

app = typer.Typer(help="Ingesta del corpus: extracción, limpieza, chunking e indexación.")

_VALID_SOURCES = ("all", "odoo", "internal")


def _setup_logging() -> None:
    """Logging legible en consola (nivel INFO) para las corridas largas."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@app.command("run")
def run(
    chunker: str = typer.Option("structure", help="structure | fixed256 | fixed768"),
    embedder: str = typer.Option("bge-m3", help="bge-m3 | e5-base | fake"),
    sources: str = typer.Option("all", help="all | odoo | internal"),
    skip_embeddings: bool = typer.Option(
        False, "--skip-embeddings", help="Solo produce chunks.{chunker}.jsonl (+ BM25), sin Chroma"
    ),
    bm25: bool = typer.Option(True, "--bm25/--no-bm25", help="Construir también el índice BM25"),
) -> None:
    """Ejecuta el pipeline de ingesta completo."""
    from docuerp.ingest.pipeline import run_ingestion, write_report

    if sources not in _VALID_SOURCES:
        typer.secho(f"--sources debe ser uno de {', '.join(_VALID_SOURCES)}", fg="red", err=True)
        raise typer.Exit(code=2)
    _setup_logging()
    try:
        report = run_ingestion(
            chunker=chunker,
            embedder=embedder,
            sources=sources,  # type: ignore[arg-type]
            skip_embeddings=skip_embeddings,
            build_bm25=bm25,
        )
    except (ValueError, FileNotFoundError) as exc:
        typer.secho(f"Error en la ingesta: {exc}", fg="red", err=True)
        raise typer.Exit(code=1) from exc
    json_path, md_path = write_report(report)
    typer.echo(
        f"{report.documents} documentos, {report.pages} páginas, {report.chunks} chunks "
        f"({report.tokens_mean} tokens de media), {len(report.flagged_chunks)} marcados, "
        f"{report.timings_s.get('total', 0):.1f} s"
    )
    typer.echo(f"Reporte de la corrida: {json_path} · {md_path}")


@app.command("report")
def report() -> None:
    """Regenera el reporte de ingesta a partir de los chunks procesados."""
    from docuerp.ingest.pipeline import build_consolidated_report

    try:
        json_path, md_path = build_consolidated_report()
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Reporte consolidado: {json_path} · {md_path}")


@app.command("odoo-pdfs")
def odoo_pdfs(
    versions: str = typer.Option("17.0,18.0", help="Versiones separadas por coma"),
    modules: str = typer.Option("", help="Módulos separados por coma (vacío = todos)"),
) -> None:
    """Genera los PDF de la documentación de Odoo por módulo y versión."""
    from docuerp.ingest.odoo_pdf import build_odoo_pdf_records

    _setup_logging()
    try:
        records = build_odoo_pdf_records(
            versions=[v.strip() for v in versions.split(",") if v.strip()],
            modules=[m.strip() for m in modules.split(",") if m.strip()] or None,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        typer.secho(f"Error al generar los PDF: {exc}", fg="red", err=True)
        raise typer.Exit(code=1) from exc
    for record in records:
        typer.echo(f"{record.path.name}: {record.rst_files} RST → {record.pages} páginas")
