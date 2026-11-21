"""Fuentes de datos de solo lectura de la API (manifiesto, chunks, evaluación, seguridad).

Hay dos implementaciones del protocolo :class:`DataSource`:

* :class:`FileDataSource` lee los artefactos reales del repositorio (``data/``, ``eval/``,
  ``security/``).
* ``MockDataSource`` (en ``mock.py``) sirve fixtures para desarrollar sin modelos.

Reglas de seguridad que se aplican aquí:

* **RBAC**: un documento o chunk solo se entrega si el rol solicitado figura en
  ``allowed_roles``; si falta ese campo, se deniega (mínimo privilegio).
* **Sin path traversal**: los nombres de archivo se validan con una expresión regular
  estricta y además NUNCA se concatenan a una ruta; se busca una coincidencia exacta de
  nombre entre los archivos de los directorios permitidos.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import yaml

from docuerp import paths as repo_paths
from docuerp.api.errors import ApiError
from docuerp.api.traces import find_in_audit_log
from docuerp.config import Settings
from docuerp.schemas import CHUNK_ID_PATTERN, UserRole

logger = logging.getLogger("docuerp.api")

SOURCE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}\.(pdf|docx)$")
"""Nombre de archivo servible: sin separadores de ruta, sin ``..`` solitario, solo pdf/docx."""

LATEST_DIR = "latest"
"""Carpeta donde ``docuerp eval report`` publica el último reporte (``summary.json``)."""

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
"""Identificador de corrida de evaluación (``YYYYMMDD-HHMM_nombre``)."""

MEDIA_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

FALLBACK_USERS: tuple[dict[str, Any], ...] = (
    {
        "user_id": "USR-0001",
        "display_name": "Usuario de portal (Comercial Los Volcanes)",
        "role": "portal_user",
        "account_id": "ACC-0001",
        "account_name": "Comercial Los Volcanes",
    },
    {
        "user_id": "USR-0101",
        "display_name": "Analista N1",
        "role": "n1",
        "account_id": None,
        "account_name": None,
    },
    {
        "user_id": "USR-0201",
        "display_name": "Analista N2",
        "role": "n2",
        "account_id": None,
        "account_name": None,
    },
    {
        "user_id": "USR-0301",
        "display_name": "Administrador",
        "role": "admin",
        "account_id": None,
        "account_name": None,
    },
)
"""Usuarios demo fijos (SPEC §3) usados si la BD de tickets aún no existe."""


@dataclass(frozen=True)
class DocumentFile:
    """Archivo listo para servir: una ruta en disco o bytes generados en memoria."""

    filename: str
    media_type: str
    path: Path | None = None
    content: bytes | None = None
    rendition_of: str | None = None
    """Si el archivo es la versión PDF de un DOCX, nombre del DOCX original."""


@dataclass(frozen=True)
class ApiPaths:
    """Ubicaciones que lee la API. Se inyectan en pruebas para aislar el sistema de archivos."""

    manifest: Path
    document_dirs: tuple[Path, ...]
    processed: Path
    index: Path
    tickets_db: Path
    eval_results: Path
    security_results: Path
    audit_log: Path
    ingestion_report: Path | None = None
    """``reports/ingestion/ingestion_report.json``: páginas, chunks y tokens por documento."""
    docx_renditions: bool = False
    """Si es ``True``, una fuente ``.docx`` se sirve como su PDF equivalente (``docx_pdf``)."""

    @classmethod
    def from_settings(cls, settings: Settings) -> ApiPaths:
        """Construye las rutas a partir de ``Settings.paths`` y de las convenciones del SPEC."""
        resolve = settings.paths.resolve
        return cls(
            manifest=resolve("manifest"),
            document_dirs=(
                resolve("build") / "pdf",
                repo_paths.resolve("data/runbooks/pdf"),
                repo_paths.resolve("data/runbooks/docx"),
            ),
            processed=resolve("processed"),
            index=resolve("index"),
            tickets_db=resolve("tickets_db"),
            eval_results=resolve("eval_results"),
            security_results=repo_paths.resolve("security/results"),
            audit_log=resolve("audit_log"),
            ingestion_report=resolve("reports") / "ingestion" / "ingestion_report.json",
            docx_renditions=True,
        )


class DataSource(Protocol):
    """Lectura de los datos que la API expone (todas las funciones son de solo lectura)."""

    def documents(self, role: UserRole) -> list[dict[str, Any]]:
        """Entradas del manifiesto visibles para ``role``."""
        ...

    def document_file(self, source: str, role: UserRole, *, original: bool = False) -> DocumentFile:
        """Archivo de ``source`` si ``role`` está autorizado (un DOCX, como PDF equivalente,
        salvo ``original=True``)."""
        ...

    def corpus_stats(self, documents: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Estadísticas del corpus para los documentos visibles (``None`` si no hay datos)."""
        ...

    def chunk(self, chunk_id: str, role: UserRole) -> dict[str, Any]:
        """Chunk completo si ``role`` está autorizado."""
        ...

    def eval_summary(self, run_id: str | None) -> dict[str, Any]:
        """Resumen de una corrida de evaluación (la más reciente si ``run_id`` es ``None``)."""
        ...

    def eval_runs(self) -> dict[str, Any]:
        """Listado de corridas de evaluación disponibles."""
        ...

    def security_summary(self) -> dict[str, Any]:
        """Resumen de la matriz red-team."""
        ...

    def users(self) -> list[dict[str, Any]]:
        """Usuarios demo."""
        ...

    def readiness(self) -> dict[str, bool]:
        """Estado de índices y BD de tickets para ``/api/health``."""
        ...

    def audit_record(self, trace_id: str) -> dict[str, Any] | None:
        """Registro sanitizado de la bitácora de auditoría para ``trace_id`` (si existe)."""
        ...


def role_allowed(entry: dict[str, Any], role: UserRole) -> bool:
    """Indica si ``role`` puede leer ``entry``; sin ``allowed_roles`` se deniega."""
    roles = entry.get("allowed_roles")
    return isinstance(roles, list) and role in roles


def _iso(ts: float) -> str:
    """Convierte un mtime a ISO-8601 UTC."""
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class FileDataSource:
    """Implementación basada en los archivos reales del repositorio."""

    def __init__(self, paths: ApiPaths, *, embedder: str = "bge-m3") -> None:
        self.paths = paths
        self.embedder = embedder
        """Embedder del índice activo (se informa en las estadísticas del corpus)."""
        self._chunk_cache: tuple[tuple[Any, ...], dict[str, dict[str, Any]]] | None = None

    # -- manifiesto y archivos ------------------------------------------------------------

    def _manifest_entries(self) -> list[dict[str, Any]]:
        """Lee todos los ``*.yaml`` del manifiesto (acepta lista o ``{documents: [...]}``)."""
        entries: list[dict[str, Any]] = []
        if not self.paths.manifest.is_dir():
            return entries
        for file in sorted(self.paths.manifest.glob("*.yaml")):
            try:
                data = yaml.safe_load(file.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as exc:
                logger.warning("manifiesto ilegible %s: %s", file.name, type(exc).__name__)
                continue
            if isinstance(data, dict):
                data = data.get("documents") or data.get("sources") or []
            if isinstance(data, list):
                entries.extend(e for e in data if isinstance(e, dict))
        return entries

    def _find_file(self, name: str) -> Path | None:
        """Busca ``name`` por igualdad exacta entre los archivos de los directorios permitidos."""
        for directory in self.paths.document_dirs:
            if not directory.is_dir():
                continue
            for candidate in directory.iterdir():
                if candidate.name == name and candidate.is_file():
                    return candidate
        return None

    def _ingestion_report(self) -> dict[str, Any] | None:
        """Sección ``structure`` del reporte de ingesta (o ``None`` si no existe)."""
        if self.paths.ingestion_report is None:
            return None
        data = _read_json(self.paths.ingestion_report)
        if not isinstance(data, dict):
            return None
        chunkers = data.get("chunkers")
        section = chunkers.get("structure") if isinstance(chunkers, dict) else None
        return section if isinstance(section, dict) else None

    def documents(self, role: UserRole) -> list[dict[str, Any]]:
        """Manifiesto filtrado por rol, con ``source`` (nombre citable) y ``file_available``.

        Si existe el reporte de ingesta, cada entrada suma ``pages``, ``chunks``, ``tokens`` y
        ``flagged_chunks`` del chunker ``structure`` (el que usa el sistema final).
        """
        report = self._ingestion_report() or {}
        per_doc = {
            str(d.get("source")): d
            for d in report.get("per_document", [])
            if isinstance(d, dict) and d.get("source")
        }
        out: list[dict[str, Any]] = []
        for entry in self._manifest_entries():
            if not role_allowed(entry, role):
                continue
            source = Path(str(entry.get("source_file", ""))).name
            available = bool(SOURCE_NAME_RE.fullmatch(source)) and self._find_file(source)
            item = {**entry, "source": source, "file_available": bool(available)}
            stats = per_doc.get(source)
            if stats is not None:
                item.update(
                    {
                        "pages": stats.get("pdf_pages") or stats.get("content_pages"),
                        "chunks": stats.get("chunks"),
                        "tokens": stats.get("tokens"),
                        "flagged_chunks": stats.get("flagged_chunks", 0),
                    }
                )
            if source.lower().endswith(".docx") and self.paths.docx_renditions:
                item["preview_format"] = "pdf"
            out.append(item)
        return out

    def corpus_stats(self, documents: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Estadísticas del corpus visible (para la vista de gobernanza), o ``None``."""
        report = self._ingestion_report()
        if report is None:
            return None
        visible = {d["source"] for d in documents}
        per_doc = [
            d
            for d in report.get("per_document", [])
            if isinstance(d, dict) and d.get("source") in visible
        ]
        by_module: dict[str, int] = {}
        for d in per_doc:
            module = str(d.get("module", "general"))
            by_module[module] = by_module.get(module, 0) + int(d.get("chunks") or 0)
        histogram = report.get("token_histogram")
        flagged = report.get("flagged_chunks")
        return {
            "documents": len(per_doc),
            "pages": sum(int(d.get("pdf_pages") or 0) for d in per_doc),
            "chunks": sum(int(d.get("chunks") or 0) for d in per_doc),
            "tokens": sum(int(d.get("tokens") or 0) for d in per_doc),
            "flagged_chunks": sum(int(d.get("flagged_chunks") or 0) for d in per_doc),
            "duplicates": int(report.get("duplicates_within_doc") or 0),
            "chunker": "structure",
            "embedder": self.embedder,
            "by_module": [{"module": m, "chunks": n} for m, n in sorted(by_module.items())],
            "token_histogram": (
                [{"bucket": k, "chunks": v} for k, v in histogram.items()]
                if isinstance(histogram, dict)
                else []
            ),
            "flagged_ids": flagged if isinstance(flagged, list) else [],
            "generated_at": report.get("run_finished_at"),
        }

    def document_file(self, source: str, role: UserRole, *, original: bool = False) -> DocumentFile:
        """Valida el nombre, comprueba el RBAC contra el manifiesto y localiza el archivo.

        Una fuente ``.docx`` se entrega como su PDF equivalente (misma paginación, ver
        :mod:`docuerp.api.docx_pdf`) para que el visor abra la página citada; con
        ``original=True`` se entrega el DOCX tal cual.
        """
        if not SOURCE_NAME_RE.fullmatch(source):
            raise ApiError(400, "invalid_source", "Nombre de documento no válido.")
        stem = Path(source).stem
        owner = next(
            (
                e
                for e in self._manifest_entries()
                if Path(str(e.get("source_file", ""))).stem == stem
            ),
            None,
        )
        if owner is None:
            raise ApiError(404, "document_not_found", "Documento no encontrado.")
        if not role_allowed(owner, role):
            raise ApiError(403, "forbidden", "Tu rol no tiene acceso a este documento.")
        found = self._find_file(source)
        if found is None:
            raise ApiError(404, "document_not_found", "El archivo del documento no existe.")
        if found.suffix.lower() == ".docx" and self.paths.docx_renditions and not original:
            rendition = self._docx_rendition(found.name)
            if rendition is not None:
                return DocumentFile(
                    filename=f"{found.stem}.pdf",
                    media_type=MEDIA_TYPES[".pdf"],
                    path=rendition,
                    rendition_of=found.name,
                )
        return DocumentFile(
            filename=found.name,
            media_type=MEDIA_TYPES[found.suffix.lower()],
            path=found,
        )

    @staticmethod
    def _docx_rendition(source: str) -> Path | None:
        """PDF equivalente del DOCX ``source`` (``None`` si no se pudo generar)."""
        from docuerp.api.docx_pdf import docx_rendition

        try:
            rendition = docx_rendition(source)
        except RuntimeError as exc:
            logger.warning("sin PDF equivalente para %s: %s", source, exc)
            return None
        return rendition.path if rendition is not None else None

    # -- chunks ---------------------------------------------------------------------------

    def _chunk_index(self) -> dict[str, dict[str, Any]]:
        """Índice ``chunk_id → chunk`` desde ``chunks.*.jsonl``; se recarga si cambian."""
        files = (
            sorted(self.paths.processed.glob("chunks.*.jsonl"))
            if (self.paths.processed.is_dir())
            else []
        )
        signature = tuple((f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in files)
        if self._chunk_cache is not None and self._chunk_cache[0] == signature:
            return self._chunk_cache[1]
        index: dict[str, dict[str, Any]] = {}
        for file in files:
            for record in _read_jsonl(file):
                chunk_id = record.get("chunk_id")
                if isinstance(chunk_id, str):
                    index.setdefault(chunk_id, record)
        self._chunk_cache = (signature, index)
        return index

    def chunk(self, chunk_id: str, role: UserRole) -> dict[str, Any]:
        """Devuelve el chunk si existe y el rol puede leerlo."""
        if len(chunk_id) > 300 or not re.fullmatch(CHUNK_ID_PATTERN, chunk_id):
            raise ApiError(400, "invalid_chunk_id", "Identificador de chunk no válido.")
        record = self._chunk_index().get(chunk_id)
        if record is None:
            raise ApiError(404, "chunk_not_found", "Chunk no encontrado.")
        if not role_allowed(record, role):
            raise ApiError(403, "forbidden", "Tu rol no tiene acceso a este chunk.")
        return record

    # -- evaluación -----------------------------------------------------------------------

    def _run_dirs(self) -> list[Path]:
        """Carpetas de corridas ordenadas de la más reciente a la más antigua.

        Se excluye ``latest/`` (copia publicada del último reporte, no una corrida).
        """
        root = self.paths.eval_results
        if not root.is_dir():
            return []
        dirs = [
            d
            for d in root.iterdir()
            if d.is_dir() and RUN_ID_RE.fullmatch(d.name) and d.name != LATEST_DIR
        ]
        return sorted(dirs, key=lambda d: d.name, reverse=True)

    @staticmethod
    def _load_run(directory: Path) -> dict[str, Any]:
        """Carga ``metrics.json``/``config.json`` de una corrida (tolera archivos ausentes).

        Las corridas de ``docuerp eval`` (``kind`` ``e2e`` o ``retrieval``) se exponen con
        ``format`` y las metas del anteproyecto evaluadas (``targets``), sin el detalle por
        caso para que la respuesta sea liviana.
        """
        metrics = _read_json(directory / "metrics.json")
        config = _read_json(directory / "config.json")
        name = directory.name.split("_", 1)[1] if "_" in directory.name else directory.name
        mtime = directory / "metrics.json"
        stamp = mtime.stat().st_mtime if mtime.is_file() else directory.stat().st_mtime
        base: dict[str, Any] = {
            "run_id": directory.name,
            "name": name,
            "generated_at": _iso(stamp),
            "has_metrics": isinstance(metrics, dict),
            "has_report": (directory / "report.md").is_file(),
            "config": config if isinstance(config, dict) else None,
        }
        if not isinstance(metrics, dict):
            base["metrics"] = None
            return base
        kind = metrics.get("kind")
        if kind in ("e2e", "retrieval"):
            base["format"] = kind
            base["metrics"] = _strip_per_case(metrics)
            targets = _evaluate_targets(metrics)
            if targets is not None:
                base["targets"] = targets
            return base
        # Si metrics.json ya trae su propia estructura de resumen, se expone tal cual.
        base["format"] = "custom"
        base["metrics"] = metrics.get("metrics", metrics)
        for key in (
            "targets",
            "ci95",
            "baseline_vs_final",
            "ablations",
            "by_category",
            "failures",
        ):
            if key in metrics:
                base[key] = metrics[key]
        return base

    def _published_report(self) -> dict[str, Any] | None:
        """``eval/results/latest/summary.json`` de ``docuerp eval report`` (si existe)."""
        latest = self.paths.eval_results / LATEST_DIR
        summary = _read_json(latest / "summary.json")
        if not isinstance(summary, dict):
            return None
        runs = summary.get("runs") if isinstance(summary.get("runs"), dict) else {}
        run_id = runs.get("final") or runs.get("baseline") or runs.get("retrieval") or LATEST_DIR
        return {
            **summary,
            "format": "report",
            "run_id": str(run_id),
            "name": "report",
            "has_metrics": True,
            "has_report": (latest / "report.md").is_file(),
            "source": f"eval/results/{LATEST_DIR}/summary.json",
        }

    def eval_summary(self, run_id: str | None) -> dict[str, Any]:
        """Resumen publicado por ``docuerp eval report`` o, si no hay, la corrida más reciente
        con métricas (o la pedida con ``run_id``)."""
        dirs = self._run_dirs()
        if run_id is not None:
            if not RUN_ID_RE.fullmatch(run_id):
                raise ApiError(400, "invalid_run_id", "Identificador de corrida no válido.")
            chosen = next((d for d in dirs if d.name == run_id), None)
        else:
            report = self._published_report()
            if report is not None:
                return report
            chosen = next((d for d in dirs if (d / "metrics.json").is_file()), None)
        if chosen is None:
            raise ApiError(404, "eval_not_found", "No hay resultados de evaluación disponibles.")
        return self._load_run(chosen)

    def eval_runs(self) -> dict[str, Any]:
        """Todas las corridas (más reciente primero) con sus métricas."""
        return {"runs": [self._load_run(d) for d in self._run_dirs()]}

    # -- seguridad ------------------------------------------------------------------------

    def security_summary(self) -> dict[str, Any]:
        """Lee ``security/results``.

        Orden: ``latest.json`` del runner red-team (``docuerp security redteam``) junto con
        su ``<run_id>.jsonl``; luego un ``summary.json`` propio; por último calcula el resumen
        desde el ``*.jsonl`` más reciente.
        """
        root = self.paths.security_results
        if root.is_dir():
            latest = _read_json(root / "latest.json")
            if isinstance(latest, dict) and isinstance(latest.get("run_id"), str):
                run_id = latest["run_id"]
                rows_file = root / f"{run_id}.jsonl"
                if RUN_ID_RE.fullmatch(run_id) and rows_file.is_file():
                    computed = _security_from_rows(list(_read_jsonl(rows_file)), rows_file)
                    return {
                        **computed,
                        "run_id": run_id,
                        "mode": latest.get("mode"),
                        "generated_at": latest.get("created_at") or computed["generated_at"],
                        "report": latest.get("summary"),
                    }
            summary = root / "summary.json"
            data = _read_json(summary) if summary.is_file() else None
            if isinstance(data, dict):
                return data
            runs = sorted(root.glob("*.jsonl"), key=lambda f: f.name, reverse=True)
            if runs:
                return _security_from_rows(list(_read_jsonl(runs[0])), runs[0])
        raise ApiError(404, "security_not_found", "No hay resultados de seguridad disponibles.")

    # -- usuarios y salud -----------------------------------------------------------------

    def users(self) -> list[dict[str, Any]]:
        """Usuarios demo desde la tabla ``users``; usa la lista fija si no hay BD."""
        db = self.paths.tickets_db
        if not db.is_file():
            return [dict(u) for u in FALLBACK_USERS]
        uri = f"file:{quote(str(db))}?mode=ro"
        try:
            with sqlite3.connect(uri, uri=True) as conn:
                rows = conn.execute(
                    "SELECT u.user_id, u.display_name, u.role, u.account_id, a.name "
                    "FROM users u LEFT JOIN accounts a ON a.account_id = u.account_id "
                    "ORDER BY u.user_id"
                ).fetchall()
        except sqlite3.Error as exc:
            logger.warning("no se pudo leer users: %s", type(exc).__name__)
            return [dict(u) for u in FALLBACK_USERS]
        if not rows:
            return [dict(u) for u in FALLBACK_USERS]
        keys = ("user_id", "display_name", "role", "account_id", "account_name")
        return [dict(zip(keys, row, strict=True)) for row in rows]

    def readiness(self) -> dict[str, bool]:
        """Hay índice si existen chunks procesados y el directorio de índices tiene contenido."""
        chunks = self.paths.processed.is_dir() and any(self.paths.processed.glob("chunks.*.jsonl"))
        index = self.paths.index.is_dir() and any(self.paths.index.iterdir())
        return {
            "index_ready": bool(chunks and index),
            "tickets_db_ready": self.paths.tickets_db.is_file(),
        }

    def audit_record(self, trace_id: str) -> dict[str, Any] | None:
        """Busca la traza en ``logs/audit.jsonl``."""
        return find_in_audit_log(self.paths.audit_log, trace_id)


def _read_json(path: Path) -> Any:
    """Lee un JSON; devuelve ``None`` si no existe o está corrupto."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Itera los objetos de un JSONL ignorando líneas vacías o corruptas."""
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record
    except OSError as exc:
        logger.warning("no se pudo leer %s: %s", path.name, type(exc).__name__)


def _strip_per_case(metrics: dict[str, Any]) -> dict[str, Any]:
    """Copia de ``metrics.json`` sin el detalle por caso (``per_case``)."""
    out = {k: v for k, v in metrics.items() if k != "per_case"}
    configs = out.get("configs")
    if isinstance(configs, dict):
        out["configs"] = {
            label: (
                {k: v for k, v in cfg.items() if k != "per_case"} if isinstance(cfg, dict) else cfg
            )
            for label, cfg in configs.items()
        }
    return out


def _evaluate_targets(metrics: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Metas del anteproyecto evaluadas sobre una corrida (``docuerp.eval.report``)."""
    try:
        from docuerp.eval.report import evaluate_targets

        if metrics.get("kind") == "retrieval":
            return evaluate_targets(metrics.get("configs") or None, None)
        return evaluate_targets(None, metrics)
    except Exception as exc:  # formato inesperado: se muestran las métricas sin metas
        logger.warning("no se pudieron evaluar las metas: %s", type(exc).__name__)
        return None


def _rate(rows: list[dict[str, Any]], outcome: str) -> float | None:
    """Fracción de casos con ``expected_outcome == outcome`` cuyo resultado fue correcto."""
    group = [r for r in rows if r.get("expected_outcome") == outcome]
    return round(sum(1 for r in group if r.get("passed")) / len(group), 4) if group else None


def _security_from_rows(rows: list[dict[str, Any]], source: Path) -> dict[str, Any]:
    """Resume los resultados del runner red-team con la misma forma que ``summary.json``.

    Los casos con ``passed: null`` (no aplican al modo de ejecución) se listan pero no
    cuentan en las tasas.
    """
    applicable = [r for r in rows if r.get("passed") is not None]
    total = len(applicable)
    passed = sum(1 for r in applicable if r.get("passed"))
    vectors: dict[str, list[dict[str, Any]]] = {}
    for row in applicable:
        vectors.setdefault(str(row.get("vector", "desconocido")), []).append(row)
    return {
        "generated_at": _iso(source.stat().st_mtime),
        "source": f"security/results/{source.name}",
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else None,
        "not_applicable": len(rows) - total,
        "metrics": {
            "attacks_blocked_rate": _rate(applicable, "blocked"),
            "denial_rate": _rate(applicable, "denied"),
            "critical_confirmation_rate": _rate(applicable, "needs_confirmation"),
        },
        "by_vector": [
            {
                "vector": vector,
                "total": len(group),
                "passed": sum(1 for r in group if r.get("passed")),
                "pass_rate": round(sum(1 for r in group if r.get("passed")) / len(group), 4),
            }
            for vector, group in vectors.items()
        ],
        "cases": rows,
    }
