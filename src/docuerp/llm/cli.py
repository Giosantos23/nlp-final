"""CLI del LLM: ``docuerp llm ping`` para verificar autenticación y latencia."""

from __future__ import annotations

import asyncio

import typer

from docuerp.llm.contracts import LLMError

app = typer.Typer(help="Proveedores LLM: verificación de conexión, latencia y costo.")

PING_SYSTEM = "Eres un verificador de conexión. Responde en una sola palabra."
PING_USER = "Responde únicamente con la palabra: pong"


@app.command("ping")
def ping(
    provider: str = typer.Option("agent_sdk", help="agent_sdk | anthropic_api | offline | fake"),
    model: str = typer.Option("claude-haiku-5-5", help="ID exacto del modelo"),
    effort: str = typer.Option("low", help="low | medium | high | xhigh | max"),
    timeout_s: float = typer.Option(45.0, "--timeout", help="Tiempo máximo en segundos"),
) -> None:
    """Envía un mensaje mínimo y muestra latencia, tokens y costo (nunca secretos)."""
    from docuerp.llm.factory import get_provider, parse_provider_name

    try:
        llm = get_provider(parse_provider_name(provider))
        result = asyncio.run(
            llm.complete(
                PING_SYSTEM,
                PING_USER,
                model=model,
                max_tokens=256,
                effort=effort,
                timeout_s=timeout_s,
            )
        )
    except LLMError as exc:
        typer.secho(f"FALLÓ ({exc.cause}): {exc}", fg="red", err=True)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.secho(f"Argumento inválido: {exc}", fg="red", err=True)
        raise typer.Exit(code=2) from exc
    typer.secho(f"OK  proveedor={result.provider}  modelo={result.model}", fg="green")
    typer.echo(f"respuesta : {result.text.strip()[:200]}")
    typer.echo(
        f"latencia  : {result.latency_ms:.0f} ms de pared"
        + (f", {result.api_latency_ms:.0f} ms de API" if result.api_latency_ms else "")
    )
    typer.echo(
        f"tokens    : entrada={result.input_tokens} salida={result.output_tokens} "
        f"caché_lectura={result.cache_read_tokens} caché_escritura={result.cache_creation_tokens}"
    )
    typer.echo(f"costo     : US${result.cost_usd:.5f} (precio de lista)")
