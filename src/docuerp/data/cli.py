"""CLI del corpus sintético: ``build-runbooks``, ``seed-tickets`` y ``datasets``.

Los nombres ``runbooks`` y ``tickets`` se mantienen como alias ocultos porque el
``Makefile`` raíz (objetivo ``corpus``) los invoca con esos nombres.
"""

from __future__ import annotations

import logging

import typer

app = typer.Typer(help="Corpus sintético: runbooks, BD de tickets y datasets de entrenamiento.")


def _configure_logging() -> None:
    """Muestra los mensajes informativos del módulo en la terminal."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@app.command("build-runbooks")
@app.command("runbooks", hidden=True)
def build_runbooks_cmd() -> None:
    """Genera los PDF/DOCX de los runbooks y el manifiesto data/manifest/internal.yaml."""
    from docuerp.data.runbooks import build_runbooks

    _configure_logging()
    generated = build_runbooks()
    typer.echo(f"{len(generated)} documentos generados:")
    for path in generated:
        typer.echo(f"  {path}")


@app.command("seed-tickets")
@app.command("tickets", hidden=True)
def seed_tickets_cmd(
    n: int = typer.Option(300, "--n", help="Número de tickets sintéticos"),
    seed: int = typer.Option(42, "--seed", help="Semilla del generador"),
) -> None:
    """Crea/recrea data/tickets/tickets.db y escribe data/tickets/README.md."""
    from docuerp.data.tickets import seed_tickets, summarize_db, write_tickets_readme

    db_path = seed_tickets(n_tickets=n, seed=seed)
    readme = write_tickets_readme(db_path.parent / "README.md")
    typer.echo(f"BD de tickets: {db_path}")
    typer.echo(f"README del esquema: {readme}")
    for table, count in summarize_db(db_path).items():
        typer.echo(f"  {table}: {count}")


@app.command("datasets")
def datasets_cmd() -> None:
    """Genera data/router/intents.jsonl y data/classifier/tickets_{train,test}.jsonl."""
    from collections import Counter

    from docuerp.data.datasets import build_classifier_dataset, build_router_dataset, read_jsonl

    router_path = build_router_dataset()
    rows = read_jsonl(router_path)
    typer.echo(f"Router: {router_path} ({len(rows)} ejemplos)")
    for (split, intent), count in sorted(Counter((r["split"], r["intent"]) for r in rows).items()):
        typer.echo(f"  {split:5s} {intent}: {count}")

    classifier_dir = build_classifier_dataset()
    for split in ("train", "test"):
        file = classifier_dir / f"tickets_{split}.jsonl"
        typer.echo(f"Clasificador {split}: {file} ({len(read_jsonl(file))} tickets)")
