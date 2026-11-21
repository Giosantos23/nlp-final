"""CLI de la API: ``docuerp api serve`` (alias ``docuerp serve``)."""

from __future__ import annotations

import os

import typer

app = typer.Typer(help="API FastAPI de DocuERP.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """API FastAPI de DocuERP.

    El *callback* mantiene ``app`` como grupo aunque tenga un solo comando: sin él, Typer
    lo colapsa a un comando suelto y el atajo ``docuerp serve`` (que invoca ``serve`` por
    nombre) falla con "unexpected extra argument".
    """


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", help="Interfaz de escucha"),
    port: int = typer.Option(8000, help="Puerto"),
    reload: bool = typer.Option(False, help="Recarga automática (desarrollo)"),
    mock: bool = typer.Option(
        False, "--mock", help="Modo mock: respuestas precargadas, sin modelos ni índices"
    ),
) -> None:
    """Levanta el servidor uvicorn (``--mock`` equivale a ``DOCUERP_API_MOCK=1``)."""
    import uvicorn

    if mock:
        # La fábrica create_app lee la variable al construir la app (también con --reload).
        os.environ["DOCUERP_API_MOCK"] = "1"
        typer.echo("Modo MOCK activo: las respuestas son simuladas.")
    uvicorn.run("docuerp.api.app:create_app", host=host, port=port, reload=reload, factory=True)
