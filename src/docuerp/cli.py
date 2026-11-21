"""Punto de entrada de la CLI: ``uv run docuerp <grupo> <comando>``.

Cada paquete expone ``cli.py`` con ``app = typer.Typer()``. Aquí se montan como grupos de
forma **robusta**: si un subpaquete no importa (p. ej. una rama a medio integrar), su grupo
se reemplaza por uno que explica el error, sin romper el resto de la CLI.
"""

from __future__ import annotations

import importlib
import json

import typer

from docuerp import __version__

app = typer.Typer(
    name="docuerp",
    help="DocuERP: asistente RAG agéntico para soporte funcional de Odoo (CC3103 · UVG).",
    no_args_is_help=True,
    add_completion=False,
)

SUBCOMMANDS: tuple[tuple[str, str], ...] = (
    ("ingest", "docuerp.ingest.cli"),
    ("retrieval", "docuerp.retrieval.cli"),
    ("llm", "docuerp.llm.cli"),
    ("generation", "docuerp.generation.cli"),
    ("agent", "docuerp.agent.cli"),
    ("security", "docuerp.security.cli"),
    ("api", "docuerp.api.cli"),
    ("eval", "docuerp.eval.cli"),
    ("data", "docuerp.data.cli"),
)
"""Pares ``(grupo, módulo)`` registrados en la CLI raíz."""


def _broken_group(name: str, module: str, error: Exception) -> typer.Typer:
    """Crea un grupo sustituto que informa por qué ``module`` no pudo importarse."""
    broken = typer.Typer(help=f"[no disponible] {module} no se pudo importar.")
    message = f"{type(error).__name__}: {error}"

    @broken.callback(invoke_without_command=True)
    def _report() -> None:
        typer.secho(f"El grupo '{name}' no está disponible ({message}).", fg="red", err=True)
        raise typer.Exit(code=1)

    return broken


def _register_subcommands() -> None:
    """Importa y monta cada sub-aplicación."""
    for name, module in SUBCOMMANDS:
        try:
            sub = importlib.import_module(module).app
        except Exception as exc:  # importación robusta: no tumbar la CLI completa
            sub = _broken_group(name, module, exc)
        app.add_typer(sub, name=name)


_register_subcommands()


@app.command("version")
def version() -> None:
    """Muestra la versión instalada de DocuERP."""
    typer.echo(f"docuerp {__version__}")


@app.command("config")
def show_config() -> None:
    """Imprime la configuración efectiva (los secretos aparecen enmascarados)."""
    from docuerp.config import get_settings

    typer.echo(json.dumps(get_settings().safe_dump(), indent=2, ensure_ascii=False))


@app.command(
    "ask",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def ask(ctx: typer.Context) -> None:
    """Atajo de ``docuerp agent ask``."""
    from typer.main import get_command

    from docuerp.agent.cli import app as agent_app

    get_command(agent_app).main(["ask", *ctx.args], standalone_mode=True)


@app.command(
    "serve",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def serve(ctx: typer.Context) -> None:
    """Atajo de ``docuerp api serve``."""
    from typer.main import get_command

    from docuerp.api.cli import app as api_app

    get_command(api_app).main(["serve", *ctx.args], standalone_mode=True)


if __name__ == "__main__":  # pragma: no cover
    app()
