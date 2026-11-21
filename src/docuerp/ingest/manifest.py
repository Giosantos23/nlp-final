"""Manifiesto del corpus: un registro de gobernanza por documento fuente.

Lee ``data/manifest/*.yaml`` (``odoo.yaml`` de la rama de ingesta e ``internal.yaml`` de la
rama de corpus sintético). Cada archivo contiene una lista ``documents:`` de entradas.

El manifiesto es la fuente de verdad de procedencia (origen, commit), licencia, propietario,
vigencia, sensibilidad y roles autorizados: el pipeline copia esos campos a cada chunk y la
recuperación los usa para filtrar por rol, versión y vigencia.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from docuerp import paths
from docuerp.schemas import ChunkErpVersion, DocStatus, DocType, ErpModule, Language, UserRole


class SourceEntry(BaseModel):
    """Entrada del manifiesto: metadatos de procedencia, licencia, vigencia y acceso."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="doc_id, p. ej. 'odoo-18.0-inventario'")
    title: str
    source_file: str = Field(description="Nombre citable del archivo (PDF/DOCX)")
    origin: str = Field(description="URL o procedencia (repo, commit, generador sintético)")
    license: str
    owner: str
    format: Literal["pdf", "docx", "md", "rst", "html"]
    version: str = Field(description="Versión del recurso: '18.0@a1b2c3d' o 'v1.1'")
    erp_version: ChunkErpVersion
    status: DocStatus
    date: str = Field(description="Fecha ISO-8601 de publicación o última revisión")
    update_frequency: str
    allowed_roles: list[UserRole] = Field(min_length=1)
    sensitivity: Literal["public", "internal", "restricted"]
    module: ErpModule
    doc_type: DocType
    language: Language
    sha256: str | None = Field(default=None, description="Hash del archivo fuente")
    poisoned: bool = Field(
        default=False,
        description=(
            "Documento con inyección indirecta deliberada (pruebas de seguridad); sigue "
            "siendo vigente e indexable"
        ),
    )


class ManifestError(ValueError):
    """Un archivo de manifiesto es inválido (YAML mal formado, campos o ids repetidos)."""


def _read_entries(path: Path) -> list[dict[str, Any]]:
    """Lee la lista ``documents:`` de un YAML de manifiesto."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ManifestError(f"{path.name}: YAML inválido ({exc})") from exc
    if not isinstance(data, dict) or not isinstance(data.get("documents", []), list):
        raise ManifestError(f"{path.name}: se esperaba un mapeo con la lista 'documents:'")
    entries = data.get("documents") or []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ManifestError(f"{path.name}: la entrada #{i} no es un mapeo")
    return entries


def load_manifest(manifest_dir: Path | None = None) -> list[SourceEntry]:
    """Carga y valida todas las entradas de ``data/manifest/*.yaml``.

    Args:
        manifest_dir: directorio alternativo (por defecto ``paths.MANIFEST``).

    Raises:
        ValueError: si hay ``id`` duplicados entre archivos (``ManifestError``, subclase de
            ``ValueError``), o si una entrada no cumple el esquema.
    """
    directory = manifest_dir or paths.MANIFEST
    if not directory.is_dir():
        return []
    entries: list[SourceEntry] = []
    origin_of: dict[str, str] = {}
    for path in sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")]):
        for i, raw in enumerate(_read_entries(path)):
            try:
                entry = SourceEntry.model_validate(raw)
            except ValidationError as exc:
                raise ManifestError(f"{path.name}: entrada #{i} inválida: {exc}") from exc
            if entry.id in origin_of:
                raise ManifestError(
                    f"id duplicado {entry.id!r} en {path.name} "
                    f"(ya definido en {origin_of[entry.id]})"
                )
            origin_of[entry.id] = path.name
            entries.append(entry)
    return entries


def find_entry(doc_id: str, entries: list[SourceEntry] | None = None) -> SourceEntry:
    """Devuelve la entrada con ``id == doc_id`` (``KeyError`` si no existe)."""
    for entry in entries if entries is not None else load_manifest():
        if entry.id == doc_id:
            return entry
    raise KeyError(f"No hay entrada de manifiesto para {doc_id!r}")
