"""Entrada/salida de las corridas de evaluación en ``eval/results/<run_id>/``.

Estructura de un directorio de corrida (SPEC §10)::

    eval/results/20261117-0930_final/
        config.json         # qué se ejecutó y con qué parámetros (RunConfig)
        predictions.jsonl   # una línea por (caso, repetición) o por (config, caso)
        judgments.jsonl     # veredictos del LLM-as-judge (opcional)
        metrics.json        # métricas agregadas y por caso
        report.md, figures/ # solo en corridas de reporte

Los JSONL se escriben **de forma incremental** (una línea por registro, vaciando el búfer)
para que una corrida interrumpida conserve lo ya calculado y pueda reanudarse.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from docuerp import paths
from docuerp.eval.models import Judgment, PredictionRecord, RunConfig

CONFIG_FILE = "config.json"
PREDICTIONS_FILE = "predictions.jsonl"
JUDGMENTS_FILE = "judgments.jsonl"
METRICS_FILE = "metrics.json"


def results_root(results_dir: Path | None = None) -> Path:
    """Directorio raíz de resultados (``eval/results`` por defecto)."""
    return results_dir if results_dir is not None else paths.EVAL_RESULTS


def new_run_id(name: str, *, now: datetime | None = None) -> str:
    """Genera ``YYYYMMDD-HHMM_<nombre>`` (nombre saneado a ``[A-Za-z0-9_-]``)."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M")
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-") or "run"
    return f"{stamp}_{safe}"


def run_path(run_id: str, results_dir: Path | None = None) -> Path:
    """Ruta de la corrida ``run_id`` dentro de la raíz de resultados.

    Raises:
        ValueError: si ``run_id`` intenta salirse de la raíz (``..`` o separadores).
    """
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) or run_id in (".", ".."):
        raise ValueError(f"run_id inválido: {run_id!r}")
    return results_root(results_dir) / run_id


# --- JSON / JSONL ---------------------------------------------------------


def write_json(path: Path, data: Any) -> None:
    """Escribe JSON legible (UTF-8, indentado) creando los directorios necesarios."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, default=str)
        fh.write("\n")


def read_json(path: Path) -> Any:
    """Lee un JSON; ``FileNotFoundError`` con mensaje claro si no existe."""
    if not path.is_file():
        raise FileNotFoundError(f"No existe {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def append_jsonl(path: Path, record: BaseModel) -> None:
    """Agrega un registro al final de un JSONL y lo persiste de inmediato."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")
        fh.flush()


def iter_jsonl[T: BaseModel](path: Path, model: type[T]) -> Iterator[T]:
    """Itera los registros válidos de un JSONL.

    Una línea truncada al final (corrida interrumpida a mitad de escritura) se ignora;
    una línea corrupta en medio del archivo sí es un error.
    """
    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            yield model.model_validate_json(line)
        except ValueError as exc:
            if index == len(lines) - 1:  # última línea truncada: se descarta
                return
            raise ValueError(f"{path.name}:{index + 1}: registro inválido ({exc})") from exc


# --- registros tipados ----------------------------------------------------


def write_config(run_dir: Path, config: RunConfig) -> None:
    """Escribe ``config.json`` de la corrida."""
    write_json(run_dir / CONFIG_FILE, config.model_dump(mode="json"))


def read_config(run_dir: Path) -> RunConfig:
    """Lee ``config.json`` de la corrida."""
    return RunConfig.model_validate(read_json(run_dir / CONFIG_FILE))


def read_predictions(run_dir: Path) -> list[PredictionRecord]:
    """Predicciones de la corrida; si un ``(caso, run)`` aparece varias veces gana la última.

    Así una repetición fallida que se reintenta al reanudar sustituye a la anterior.
    """
    latest: dict[tuple[str, int], PredictionRecord] = {}
    for record in iter_jsonl(run_dir / PREDICTIONS_FILE, PredictionRecord):
        latest[record.key] = record
    return list(latest.values())


def read_judgments(run_dir: Path) -> list[Judgment]:
    """Veredictos del juez de la corrida (el último por ``(caso, run)``)."""
    latest: dict[tuple[str, int], Judgment] = {}
    for record in iter_jsonl(run_dir / JUDGMENTS_FILE, Judgment):
        latest[record.key] = record
    return list(latest.values())


def read_metrics(run_dir: Path) -> dict[str, Any]:
    """Lee ``metrics.json`` de la corrida."""
    data = read_json(run_dir / METRICS_FILE)
    if not isinstance(data, dict):
        raise ValueError(f"{run_dir / METRICS_FILE} debe contener un objeto JSON")
    return data


# --- descubrimiento -------------------------------------------------------


def discover_runs(results_dir: Path | None = None) -> list[tuple[Path, RunConfig]]:
    """Corridas con ``config.json`` válido, de la más antigua a la más reciente."""
    root = results_root(results_dir)
    if not root.is_dir():
        return []
    found: list[tuple[Path, RunConfig]] = []
    for child in sorted(root.iterdir()):
        if not (child / CONFIG_FILE).is_file():
            continue
        try:
            found.append((child, read_config(child)))
        except ValueError:
            continue
    found.sort(key=lambda item: (item[1].created_at, item[0].name))
    return found


def latest_run(kind: str, name: str | None = None, results_dir: Path | None = None) -> Path | None:
    """Directorio de la corrida más reciente de ese ``kind`` (y ``name`` si se indica)."""
    matches = [
        path
        for path, cfg in discover_runs(results_dir)
        if cfg.kind == kind and (name is None or cfg.name == name)
    ]
    return matches[-1] if matches else None
