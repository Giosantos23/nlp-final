"""CLI del agente: ``docuerp agent ask`` (alias ``docuerp ask``) y ``train-router``.

Ejemplo::

    docuerp agent ask "¿Cómo valido una recepción parcial?" --role n1 --user USR-0101 \\
        --version 18.0

Muestra la respuesta, las citas, la traza por etapa (con latencias) y el JSON contractual
§11. Si la respuesta trae un borrador de ticket, pide confirmación humana en la terminal
antes de registrarlo (HITL); sin un "sí" explícito no se escribe nada.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from docuerp.schemas import QueryRequest, QueryResponse

app = typer.Typer(help="Agente DocuERP: consultas de extremo a extremo con traza.")
console = Console()

_STATUS_STYLE: dict[str, str] = {
    "ok": "green",
    "blocked": "yellow",
    "error": "red",
    "skipped": "bright_black",
    "retry": "magenta",
}


def _summarize(data: dict[str, Any], limit: int = 90) -> str:
    """Resumen de una línea de los datos de un evento (para la tabla de traza)."""
    parts = []
    for key, value in data.items():
        if key == "top" and isinstance(value, list):
            parts.append(f"top={len(value)}")
            continue
        parts.append(f"{key}={value}")
    text = ", ".join(parts)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_response(resp: QueryResponse, *, out: Console | None = None) -> None:
    """Imprime la respuesta, citas, traza y JSON contractual con ``rich``."""
    target = out or console
    answer = resp.response
    if resp.blocked:
        style, title = "yellow", "Bloqueada"
    elif answer.abstained:
        style, title = "bright_black", "Abstención"
    else:
        style, title = "green", "Respuesta"
    if answer.needs_human_review:
        title += " · requiere revisión humana"
    target.print(
        Panel(
            answer.answer,
            title=f"{title} · intención {resp.intent}",
            border_style=style,
            subtitle=f"{resp.latency_ms:.0f} ms · herramienta {answer.tool_used or '—'}",
        )
    )
    if answer.sources:
        sources = Table(title="Citas", show_lines=False)
        sources.add_column("source")
        sources.add_column("page", justify="right")
        sources.add_column("chunk_id")
        for ref in answer.sources:
            sources.add_row(ref.source, str(ref.page), ref.chunk_id)
        target.print(sources)
    trace = Table(title=f"Traza {resp.trace_id}")
    trace.add_column("etapa", no_wrap=True)
    trace.add_column("estado")
    trace.add_column("ms", justify="right")
    trace.add_column("datos", overflow="fold")
    for event in resp.events:
        color = _STATUS_STYLE.get(event.status, "white")
        trace.add_row(
            event.stage,
            f"[{color}]{event.status}[/{color}]",
            f"{event.duration_ms:.1f}",
            _summarize(event.data),
        )
    target.print(trace)
    usage = resp.usage
    target.print(
        f"LLM: {usage.llm_calls} llamada(s), {usage.input_tokens}+{usage.output_tokens} "
        f"tokens, US${usage.cost_usd:.5f}"
    )
    if resp.pending_action is not None:
        draft = resp.pending_action.draft
        target.print(
            Panel(
                f"{draft.summary}\ncategoría={draft.category} prioridad={draft.priority} "
                f"módulo={draft.module} cuenta={draft.account_id}\n"
                f"expira: {resp.pending_action.expires_at}",
                title="Borrador pendiente de confirmación",
                border_style="cyan",
            )
        )
    target.print_json(answer.model_dump_json())


@app.command("ask")
def ask(
    query: str = typer.Argument(..., help="Consulta en español"),
    role: str = typer.Option("n1", help="portal_user | n1 | n2 | admin"),
    user_id: str = typer.Option(
        "USR-0101", "--user", "--user-id", help="Identificador ^USR-\\d{4}$"
    ),
    erp_version: str = typer.Option("18.0", "--version", "--erp-version", help="17.0 | 18.0"),
    session_id: str | None = typer.Option(None, "--session-id", help="Sesión (opcional)"),
    as_json: bool = typer.Option(False, "--json", help="Imprime solo el QueryResponse en JSON"),
    confirm: bool = typer.Option(
        True, "--confirm/--no-confirm", help="Ofrecer confirmar un borrador de ticket (HITL)"
    ),
) -> None:
    """Responde una consulta y muestra la respuesta, citas y traza."""
    from docuerp.agent.service import answer_query, confirm_action

    try:
        req = QueryRequest(
            query=query,
            role=role,  # type: ignore[arg-type]
            user_id=user_id,
            erp_version=erp_version,  # type: ignore[arg-type]
            session_id=session_id,
        )
    except ValidationError as exc:
        fields = ", ".join(sorted({str(err["loc"][0]) for err in exc.errors()}))
        typer.secho(f"Solicitud inválida ({fields}).", fg="red", err=True)
        raise typer.Exit(code=2) from exc
    resp = asyncio.run(answer_query(req))
    if as_json:
        typer.echo(json.dumps(resp.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return
    render_response(resp)
    if confirm and resp.pending_action is not None:
        if typer.confirm("¿Confirmas el registro de este ticket?", default=False):
            token = resp.pending_action.confirmation_token
            render_response(asyncio.run(confirm_action(token, req)))
        else:
            console.print("No se registró el ticket (el borrador expirará solo).")


@app.command("train-router")
def train_router_cmd(
    embedder: str | None = typer.Option(None, help="bge-m3 | e5-base | fake (por defecto: config)"),
) -> None:
    """Entrena el clasificador de intención del router y muestra sus métricas."""
    from docuerp.agent.router import train_router

    try:
        metrics = train_router(embedder_name=embedder)
    except FileNotFoundError as exc:
        typer.secho(f"No hay datos de intención: {exc}", fg="red", err=True)
        raise typer.Exit(code=1) from exc
    table = Table(title="Router de intención")
    table.add_column("métrica")
    table.add_column("valor", justify="right")
    for key, value in metrics.items():
        table.add_row(key, f"{value:.4f}" if not float(value).is_integer() else f"{value:.0f}")
    console.print(table)


def _register_tools() -> None:
    """Monta el sub-grupo ``tools`` (importación robusta)."""
    try:
        from docuerp.agent.tools.cli import app as tools_app
    except Exception:  # pragma: no cover - solo si el subpaquete está roto
        return
    app.add_typer(tools_app, name="tools")


_register_tools()
