"""CLI de seguridad: ``docuerp security redteam`` y ``docuerp security check``."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import typer

from docuerp import paths

app = typer.Typer(help="Seguridad: red-team, guards y auditoría.")


def _display(path: Path) -> str:
    """Ruta relativa a la raíz del repo si está dentro de él (para mensajes legibles)."""
    return str(path.relative_to(paths.ROOT)) if path.is_relative_to(paths.ROOT) else str(path)


@app.command("redteam")
def redteam(
    cases: str = typer.Option("security/redteam_cases.jsonl", help="Banco de casos red-team"),
    guards_only: bool = typer.Option(
        False, "--guards-only", help="Solo la capa de guards (sin el servicio del agente)"
    ),
    embedder: str | None = typer.Option(
        None,
        help="Embedder de la capa semántica (bge-m3 | e5-base | fake); por defecto el de config",
    ),
    name: str | None = typer.Option(None, help="Sufijo del run_id (por defecto guards | e2e)"),
    out_dir: str = typer.Option("security/results", "--out-dir", help="Carpeta de resultados"),
    only: list[str] | None = typer.Option(
        None, "--only", help="Ids de casos a ejecutar (repetible), p. ej. tras corregir una falla"
    ),
) -> None:
    """Ejecuta el banco red-team y escribe el reporte de seguridad."""
    from docuerp.security.input_guard import InputGuard, InputGuardConfig
    from docuerp.security.redteam import (
        bank_leave_one_out,
        load_cases,
        make_run_id,
        run_benign,
        run_e2e_sync,
        run_guards_only,
        summarize,
        write_outputs,
    )

    try:
        bank = load_cases(paths.resolve(cases))
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(f"No se pudo cargar el banco: {exc}", fg="red", err=True)
        raise typer.Exit(code=1) from exc
    if only:
        wanted = set(only)
        unknown = wanted - {c.id for c in bank}
        if unknown:
            typer.secho(f"Casos inexistentes: {', '.join(sorted(unknown))}", fg="red", err=True)
            raise typer.Exit(code=1)
        bank = [c for c in bank if c.id in wanted]

    config = InputGuardConfig.from_settings(embedder)
    guard = InputGuard(config)
    mode = "guards_only" if guards_only else "e2e"
    if guards_only:
        results = run_guards_only(bank, guard)
    else:
        progress = logging.getLogger("docuerp.security.redteam")
        progress.setLevel(logging.INFO)
        if not progress.handlers:
            progress.addHandler(logging.StreamHandler())
        try:
            results = run_e2e_sync(bank)
        except NotImplementedError as exc:
            typer.secho(
                "El servicio del agente aún no está implementado "
                "(docuerp.agent.service.answer_query). Use --guards-only.",
                fg="yellow",
                err=True,
            )
            raise typer.Exit(code=2) from exc

    benign = run_benign(guard)
    summary = summarize(results, benign, bank_loo_recall=bank_leave_one_out(guard))
    run_id = make_run_id(name or ("guards" if guards_only else "e2e"))
    meta = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "embedder": config.embedder,
        "injection_threshold": config.injection_threshold,
        "semantic_threshold": config.semantic_threshold,
        "cases_file": cases,
        "only": sorted(only) if only else None,
    }
    outputs = write_outputs(
        results, summary, run_id=run_id, mode=mode, meta=meta, out_dir=paths.resolve(out_dir)
    )
    overall = summary["overall"]
    typer.echo(f"Run {run_id} ({mode}): {overall['passed']}/{overall['total']} casos pasan.")
    for vector, stats in summary["by_vector"].items():
        typer.echo(f"  {vector:<22} {stats['passed']}/{stats['total']}")
    benign_stats = summary["benign"]
    typer.echo(
        f"  FPR benigno: {benign_stats['false_positives']}/{benign_stats['total']} "
        f"({benign_stats['fpr']})"
    )
    for kind, path in outputs.items():
        typer.echo(f"  {kind}: {_display(path)}")


@app.command("calibrate")
def calibrate_cmd(
    embedder: str = typer.Option("bge-m3", help="Embedder de la capa semántica"),
    chunks: int = typer.Option(
        250, help="Chunks de Odoo muestreados como negativos (0 = sin chunks)"
    ),
    out_dir: str = typer.Option("reports/security", "--out-dir", help="Carpeta del reporte"),
) -> None:
    """Curva de umbral de la capa semántica (TPR/FPR) y elección de umbral y margen."""
    from docuerp.embeddings import get_embedder
    from docuerp.security.calibration import calibrate, write_calibration
    from docuerp.security.input_guard import InputGuardConfig

    config = InputGuardConfig.from_settings("fake")  # solo para leer umbral y margen vigentes
    model = get_embedder(embedder)  # type: ignore[arg-type]
    result = calibrate(
        model,
        include_chunks=chunks > 0,
        chunk_sample=chunks,
        current=(config.semantic_threshold, config.semantic_margin),
    )
    result["embedder"] = embedder
    outputs = write_calibration(result, paths.resolve(out_dir))
    choice = result["choice"]
    typer.echo(
        f"Umbral elegido {choice['threshold']:.2f}, margen {choice['margin']:.2f} "
        f"(AUC {result['auc_semantic_only_queries']:.3f})."
    )
    for kind, path in outputs.items():
        typer.echo(f"  {kind}: {_display(path)}")


@app.command("check")
def check(
    query: str = typer.Argument(..., help="Texto a inspeccionar"),
    role: str = typer.Option("n1", help="portal_user | n1 | n2 | admin"),
    user_id: str = typer.Option("USR-0101", "--user-id", help="Identificador ^USR-\\d{4}$"),
    embedder: str | None = typer.Option(None, help="Embedder de la capa semántica"),
) -> None:
    """Muestra el veredicto del guard de entrada para ``query``.

    Código de salida: 0 permitido, 1 solicitud fuera de contrato, 3 bloqueado.
    """
    from docuerp.security.input_guard import InputGuard, InputGuardConfig, parse_request

    req, rejected = parse_request({"query": query, "role": role, "user_id": user_id})
    if req is None:
        payload = rejected.model_dump() if rejected is not None else {}
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        raise typer.Exit(code=1)
    guard = InputGuard(InputGuardConfig.from_settings(embedder))
    assessment = guard.assess(req)
    payload = {"verdict": assessment.verdict.model_dump(), "detail": assessment.as_trace_data()}
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    if not assessment.verdict.allowed:
        raise typer.Exit(code=3)
