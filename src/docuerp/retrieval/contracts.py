"""Contratos del módulo de recuperación (SPEC §5.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from docuerp.schemas import ErpVersion, ScoredChunk, UserRole, normalize_text

RetrievalMode = Literal["bm25", "dense", "hybrid", "hybrid_rerank", "hybrid_rerank_mmr"]
"""Configuraciones de recuperación comparadas en la evaluación (baseline → final)."""

StageName = Literal["bm25", "dense", "fusion", "rerank", "mmr"]
"""Claves válidas de ``RetrievalResult.stages``."""


class RetrievalQuery(BaseModel):
    """Consulta de recuperación con filtros de rol, versión y vigencia."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000)
    role: UserRole
    erp_version: ErpVersion
    top_k: int = Field(default=5, ge=1, le=50)
    candidates_k: int = Field(default=20, ge=1, le=200)
    include_obsolete: bool = False

    @field_validator("text", mode="before")
    @classmethod
    def _normalize(cls, value: object) -> object:
        """Normaliza el texto (NFC + strip)."""
        return normalize_text(value) if isinstance(value, str) else value


class RetrievalResult(BaseModel):
    """Resultado de recuperación con trazabilidad por etapa y gate de evidencia."""

    model_config = ConfigDict(extra="forbid")

    query: RetrievalQuery
    hits: list[ScoredChunk]
    stages: dict[str, list[ScoredChunk]] = Field(default_factory=dict)
    filtered_out: int = Field(default=0, ge=0)
    evidence_ok: bool
    top_score: float | None
    timings_ms: dict[str, float] = Field(default_factory=dict)


@dataclass(frozen=True)
class SearchOutcome:
    """Salida interna de un recuperador individual (BM25 o denso).

    ``rejected_ids`` son los ``chunk_id`` que habrían entrado a la lista de candidatos pero
    fueron descartados por rol, versión o vigencia; el pipeline une los de ambos
    recuperadores para informar ``RetrievalResult.filtered_out``.
    """

    hits: list[ScoredChunk]
    rejected_ids: frozenset[str] = frozenset()


@runtime_checkable
class Retriever(Protocol):
    """Interfaz común de todos los recuperadores."""

    def retrieve(self, q: RetrievalQuery) -> RetrievalResult:
        """Recupera los ``top_k`` chunks autorizados más relevantes para ``q``."""
        ...
