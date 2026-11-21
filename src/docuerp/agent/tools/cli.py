"""CLI de herramientas: ``docuerp agent tools list`` y ``train-classifier``."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Herramientas del agente: listado y entrenamiento del clasificador.")
console = Console()


@app.command("list")
def list_tools() -> None:
    """Lista las herramientas con sus roles y límites."""
    from docuerp.agent.tools.base import default_registry

    table = Table(title="Herramientas del agente")
    table.add_column("nombre", no_wrap=True)
    for column in ("roles", "escribe", "confirmación", "por consulta", "por sesión"):
        table.add_column(column)
    table.add_column("timeout (s)", justify="right")
    for spec in default_registry().specs():
        table.add_row(
            spec.name,
            ", ".join(spec.allowed_roles),
            "sí" if spec.writes else "no",
            "sí" if spec.requires_confirmation else "no",
            str(spec.max_calls_per_query),
            str(spec.max_calls_per_session or "—"),
            f"{spec.timeout_s:g}",
        )
    console.print(table)


@app.command("train-classifier")
def train_classifier_cmd(
    kind: str = typer.Option("embeddings", help="tfidf | embeddings | both"),
    embedder: str | None = typer.Option(None, help="bge-m3 | e5-base | fake (por defecto: config)"),
) -> None:
    """Entrena el clasificador de tickets y muestra accuracy/F1 macro en test."""
    from docuerp.agent.tools.classify import CLASSIFIER_KINDS, train_classifier

    kinds = list(CLASSIFIER_KINDS) if kind == "both" else [kind]
    table = Table(title="Clasificador de tickets")
    table.add_column("tipo")
    table.add_column("métrica")
    table.add_column("valor", justify="right")
    for current in kinds:
        try:
            metrics = train_classifier(current, embedder_name=embedder)
        except (FileNotFoundError, ValueError) as exc:
            typer.secho(f"No se pudo entrenar {current}: {exc}", fg="red", err=True)
            raise typer.Exit(code=1) from exc
        for key, value in metrics.items():
            table.add_row(current, key, f"{value:.4f}")
    console.print(table)
