"""CLI de generación: inspección de prompts y demostración de respuestas."""

from __future__ import annotations

import asyncio
import json

import typer

from docuerp.generation.demo import SCENARIOS
from docuerp.generation.prompts import (
    CANARY_TOKEN,
    build_system_prompt,
    build_user_prompt,
)
from docuerp.llm.contracts import LLMError

app = typer.Typer(help="Generación fundamentada: inspección de prompts y respuestas.")

CANARY_MASK = "[CANARIO-ENMASCARADO]"


@app.command("show-prompt")
def show_prompt() -> None:
    """Imprime el prompt de sistema (con el canario enmascarado)."""
    typer.echo(build_system_prompt().replace(CANARY_TOKEN, CANARY_MASK))


@app.command("demo")
def demo(
    scenario: str = typer.Option(
        "recepcion", help=f"Escenario de demostración: {', '.join(SCENARIOS)}"
    ),
    provider: str = typer.Option("offline", help="agent_sdk | anthropic_api | offline | fake"),
    model: str = typer.Option("claude-haiku-5-5", help="ID exacto del modelo (si hay LLM)"),
    show_user_prompt: bool = typer.Option(
        False, "--show-prompt", help="Muestra el prompt de usuario"
    ),
) -> None:
    """Genera una respuesta fundamentada con evidencia sintética y muestra consumo y latencia.

    Por defecto usa el proveedor ``offline`` (sin red). Con ``--provider agent_sdk`` llama
    al LLM real usando la suscripción configurada en ``.env``.
    """
    from docuerp.generation.generator import generate_answer
    from docuerp.llm.factory import get_provider, parse_provider_name

    if scenario not in SCENARIOS:
        typer.secho(
            f"Escenario desconocido: {scenario!r}. Opciones: {', '.join(SCENARIOS)}", fg="red"
        )
        raise typer.Exit(code=2)
    case = SCENARIOS[scenario]
    typer.secho(f"Escenario: {scenario} — {case.description}", bold=True)
    typer.echo(f"Pregunta: {case.question}")
    if show_user_prompt:
        typer.echo(
            build_user_prompt(
                case.question, case.hits, intent=case.intent, tool_result=case.tool_result
            )
        )
    try:
        llm = get_provider(parse_provider_name(provider))
        answer, result = asyncio.run(
            generate_answer(
                case.question,
                case.hits,
                intent=case.intent,
                tool_result=case.tool_result,
                provider=llm,
                model=model,
                erp_version="18.0",
            )
        )
    except (LLMError, ValueError) as exc:
        typer.secho(f"Error: {exc}", fg="red", err=True)
        raise typer.Exit(code=1) from exc
    typer.secho("\nRespuesta (contrato §11):", bold=True)
    typer.echo(json.dumps(answer.model_dump(), indent=2, ensure_ascii=False))
    typer.secho("\nConsumo:", bold=True)
    typer.echo(
        f"proveedor={result.provider} modelo={result.model} llamadas={result.calls} "
        f"latencia={result.latency_ms:.0f} ms api={result.api_latency_ms or 0:.0f} ms "
        f"tokens(in/out/cache_r/cache_w)={result.input_tokens}/{result.output_tokens}/"
        f"{result.cache_read_tokens}/{result.cache_creation_tokens} costo=US${result.cost_usd:.5f}"
    )
