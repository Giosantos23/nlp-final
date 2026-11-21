"""Acceso de solo lectura a los chunks ingeridos, para la evaluación.

La respuesta del agente solo trae ``chunk_id`` en sus citas. Para (a) saber qué rango de
páginas cubre cada cita, (b) darle al juez el texto citado y (c) validar el banco contra el
corpus, la evaluación necesita consultar los chunks. Se leen de
``data/processed/chunks.{chunker}.jsonl`` (SPEC §8) sin tocar los índices.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

from docuerp import paths
from docuerp.eval.benchmark import EvidenceRef
from docuerp.schemas import Chunk

log = logging.getLogger(__name__)


class ChunkStore:
    """Diccionario ``chunk_id → Chunk`` con consultas por fuente y página."""

    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self._by_id: dict[str, Chunk] = {c.chunk_id: c for c in chunks}

    # --- construcción -----------------------------------------------------

    @classmethod
    def from_jsonl(cls, path: Path) -> ChunkStore:
        """Lee un ``chunks.*.jsonl``; las líneas inválidas se omiten con una advertencia."""
        if not path.is_file():
            raise FileNotFoundError(f"No existe el archivo de chunks: {path}")
        chunks: list[Chunk] = []
        with path.open(encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                if not raw.strip():
                    continue
                try:
                    chunks.append(Chunk.model_validate(json.loads(raw)))
                except ValueError as exc:  # JSON o contrato inválido
                    log.warning("%s:%d: chunk inválido omitido (%s)", path.name, line_no, exc)
        return cls(chunks)

    @classmethod
    def default(cls, chunker: str = "structure") -> ChunkStore | None:
        """Carga ``data/processed/chunks.{chunker}.jsonl`` o devuelve ``None`` si no existe."""
        path = paths.PROCESSED / f"chunks.{chunker}.jsonl"
        if not path.is_file():
            log.info("Sin corpus procesado en %s: se omiten las verificaciones contra chunks", path)
            return None
        return cls.from_jsonl(path)

    # --- consultas --------------------------------------------------------

    def __len__(self) -> int:
        return len(self._by_id)

    def get(self, chunk_id: str) -> Chunk | None:
        """Devuelve el chunk o ``None`` si no se conoce."""
        return self._by_id.get(chunk_id)

    def sources(self) -> set[str]:
        """Conjunto de archivos fuente presentes."""
        return {c.source for c in self._by_id.values()}

    def covers(self, source: str, page: int) -> bool:
        """``True`` si algún chunk de ``source`` incluye la página ``page``."""
        return any(
            c.source == source and c.page <= page <= c.page_end for c in self._by_id.values()
        )

    def for_evidence(self, evidence: Iterable[EvidenceRef]) -> list[Chunk]:
        """Chunks que intersectan alguna evidencia ``(source, page)``, sin repetir."""
        refs = list(evidence)
        return [
            c
            for c in self._by_id.values()
            if any(c.source == e.source and c.page <= e.page <= c.page_end for e in refs)
        ]

    def span(self, chunk_id: str) -> tuple[str, int, int] | None:
        """``(source, page, page_end)`` del chunk, o ``None`` si no se conoce."""
        chunk = self._by_id.get(chunk_id)
        return None if chunk is None else (chunk.source, chunk.page, chunk.page_end)

    def spans(self) -> dict[str, tuple[str, int, int]]:
        """Mapa ``chunk_id → (source, page, page_end)`` de todo el corpus."""
        return {cid: (c.source, c.page, c.page_end) for cid, c in self._by_id.items()}
