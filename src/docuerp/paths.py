"""Rutas absolutas del repositorio.

La raíz se calcula a partir de la ubicación de este archivo (``src/docuerp/paths.py`` →
dos niveles arriba), de modo que en un *worktree* las rutas apuntan al propio worktree.
Se puede forzar otra raíz con la variable de entorno ``DOCUERP_ROOT``.

Ninguna función de este módulo crea directorios al importarse; use :func:`ensure_dirs`.
"""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_root() -> Path:
    """Devuelve la raíz del repositorio (``DOCUERP_ROOT`` o la deducida del paquete)."""
    override = os.environ.get("DOCUERP_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


ROOT: Path = _resolve_root()
"""Raíz del repositorio."""

SRC: Path = ROOT / "src"
CONFIGS: Path = ROOT / "configs"
"""Configuración versionada (``settings.yaml``, ``roles.yaml``)."""
SETTINGS_FILE: Path = CONFIGS / "settings.yaml"
ROLES_FILE: Path = CONFIGS / "roles.yaml"

DATA: Path = ROOT / "data"
"""Datos: fuentes versionadas (manifiestos, runbooks) y artefactos generados (gitignored)."""
RAW: Path = DATA / "raw"
"""Fuentes descargadas (p. ej. ``data/raw/odoo-docs/{17.0,18.0}``). Gitignored."""
BUILD: Path = DATA / "build"
"""PDF/DOCX generados para ingesta (``data/build/pdf``). Gitignored."""
PROCESSED: Path = DATA / "processed"
"""Chunks en JSONL (``chunks.{chunker}.jsonl``). Gitignored."""
INDEX: Path = DATA / "index"
"""Índices Chroma y BM25 por configuración. Gitignored."""
MANIFEST: Path = DATA / "manifest"
"""Manifiestos YAML del corpus (versionados)."""
RUNBOOKS: Path = DATA / "runbooks"
"""Fuentes markdown de runbooks sintéticos y sus PDF/DOCX (versionados)."""
TICKETS_DIR: Path = DATA / "tickets"
TICKETS_DB: Path = TICKETS_DIR / "tickets.db"
"""Base SQLite de tickets sintéticos. Gitignored."""
ROUTER_DATA: Path = DATA / "router"
CLASSIFIER_DATA: Path = DATA / "classifier"

LOGS: Path = ROOT / "logs"
AUDIT_LOG: Path = LOGS / "audit.jsonl"
"""Bitácora de auditoría sanitizada (JSONL)."""

REPORTS: Path = ROOT / "reports"
"""Reportes versionados (ingesta, seguridad, figuras)."""
EVAL: Path = ROOT / "eval"
"""Banco de pruebas y resultados de evaluación (versionados)."""
EVAL_RESULTS: Path = EVAL / "results"
BENCHMARK_FILE: Path = EVAL / "benchmark.jsonl"

SECURITY: Path = ROOT / "security"
DESIGN: Path = ROOT / "design"
DESIGN_TOKENS: Path = DESIGN / "tokens.json"
CACHE: Path = ROOT / ".cache"
DOTENV: Path = ROOT / ".env"

GENERATED_DIRS: tuple[Path, ...] = (RAW, BUILD, PROCESSED, INDEX, TICKETS_DIR, LOGS, CACHE)
"""Directorios de artefactos generados que :func:`ensure_dirs` crea si faltan."""


def ensure_dirs(*paths: Path) -> None:
    """Crea los directorios indicados (o todos los generados si no se pasa ninguno)."""
    for path in paths or GENERATED_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def resolve(path: str | Path) -> Path:
    """Resuelve una ruta relativa contra la raíz del repositorio; deja intactas las absolutas."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()
