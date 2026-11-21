"""Recuperación híbrida (SPEC §5.2). Propiedad de la rama ``feat/retrieval``."""

from __future__ import annotations

from docuerp.retrieval.contracts import (
    RetrievalMode,
    RetrievalQuery,
    RetrievalResult,
    Retriever,
    StageName,
)
from docuerp.retrieval.pipeline import build_retriever

__all__ = [
    "RetrievalMode",
    "RetrievalQuery",
    "RetrievalResult",
    "Retriever",
    "StageName",
    "build_retriever",
]
