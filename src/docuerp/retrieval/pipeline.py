"""Recuperador híbrido configurable: BM25 + denso → RRF → rerank → MMR → gate τ.

Cada modo activa más etapas sobre la misma base; así la evaluación puede medir la
contribución de cada una (ablación) y el inspector de trazas muestra el resultado de cada
etapa en ``RetrievalResult.stages``:

====================== =================================================================
modo                   etapas
====================== =================================================================
``bm25``               BM25 (baseline léxico)
``dense``              búsqueda vectorial en Chroma
``hybrid``             BM25 + denso → RRF (k=60, pesos 0.5/0.5)
``hybrid_rerank``      hybrid → cross-encoder sobre el top-20 → top-5
``hybrid_rerank_mmr``  hybrid_rerank, pero el top-5 se elige con MMR (λ=0.7)
====================== =================================================================
"""

from __future__ import annotations

import time
from typing import Any, get_args

from docuerp.retrieval.bm25 import BM25Retriever
from docuerp.retrieval.contracts import (
    RetrievalMode,
    RetrievalQuery,
    RetrievalResult,
    Retriever,
)
from docuerp.retrieval.dense import DenseRetriever
from docuerp.retrieval.fusion import rrf_fuse
from docuerp.retrieval.gate import build_result
from docuerp.retrieval.mmr import mmr_select
from docuerp.retrieval.rerank import Reranker
from docuerp.retrieval.store import resolve_settings
from docuerp.schemas import ScoredChunk

RERANK_MODES = ("hybrid_rerank", "hybrid_rerank_mmr")
"""Modos que pasan los candidatos por el cross-encoder."""


def _ms_since(start: float) -> float:
    """Milisegundos transcurridos desde ``start`` (``time.perf_counter()``)."""
    return (time.perf_counter() - start) * 1000


class HybridRetriever:
    """Implementa los modos ``hybrid``, ``hybrid_rerank`` y ``hybrid_rerank_mmr``.

    Args:
        mode: uno de los tres modos híbridos.
        chunker: variante de fragmentación indexada (``structure``, ``fixed256``…).
        embedder: modelo de embeddings del índice denso (``bge-m3``, ``e5-base``…).
        settings: ``docuerp.config.Settings`` (por defecto, la global).
        reranker: reranker a usar en los modos con rerank (por defecto, el cross-encoder
            real, cargado de forma perezosa). Se inyecta un falso en las pruebas.
    """

    def __init__(
        self,
        mode: RetrievalMode = "hybrid_rerank",
        chunker: str = "structure",
        embedder: str = "bge-m3",
        settings: Any = None,
        reranker: Reranker | None = None,
    ) -> None:
        if mode not in ("hybrid", *RERANK_MODES):
            raise ValueError(
                f"HybridRetriever solo admite hybrid, hybrid_rerank y hybrid_rerank_mmr; "
                f"recibió {mode!r}"
            )
        self.mode = mode
        self.chunker = chunker
        self.embedder = embedder
        self.settings = settings
        self._bm25 = BM25Retriever(chunker, settings)
        self._dense = DenseRetriever(chunker, embedder, settings)
        self._reranker = reranker or Reranker(settings=settings)

    def retrieve(self, q: RetrievalQuery) -> RetrievalResult:
        """Ejecuta todas las etapas y registra ``stages`` y ``timings_ms``."""
        cfg = resolve_settings(self.settings).retrieval
        started = time.perf_counter()
        timings: dict[str, float] = {}
        pool = max(q.candidates_k, q.top_k)  # candidatos por etapa (top-20 por defecto)

        t0 = time.perf_counter()
        bm25 = self._bm25.search_with_stats(q, pool)
        timings["bm25"] = _ms_since(t0)

        t0 = time.perf_counter()
        dense = self._dense.search_with_stats(q, pool)
        timings["dense"] = _ms_since(t0)

        t0 = time.perf_counter()
        fused = rrf_fuse(
            {"bm25": bm25.hits, "dense": dense.hits},
            k=cfg.rrf_k,
            weights={"bm25": cfg.weights.bm25, "dense": cfg.weights.dense},
            limit=pool,
        )
        timings["fusion"] = _ms_since(t0)
        stages: dict[str, list[ScoredChunk]] = {
            "bm25": bm25.hits,
            "dense": dense.hits,
            "fusion": fused,
        }

        hits = fused[: q.top_k]
        if self.mode in RERANK_MODES:
            t0 = time.perf_counter()
            reranked = self._reranker.rerank(q.text, fused, top_k=len(fused)) if fused else []
            timings["rerank"] = _ms_since(t0)
            stages["rerank"] = reranked
            hits = reranked[: q.top_k]

            if self.mode == "hybrid_rerank_mmr":
                t0 = time.perf_counter()
                hits = mmr_select(
                    q.text,
                    reranked,
                    top_k=q.top_k,
                    lambda_=cfg.mmr_lambda,
                    embedder=self.embedder,
                    vectors=self._dense.fetch_vectors([c.chunk.chunk_id for c in reranked]),
                )
                timings["mmr"] = _ms_since(t0)
                stages["mmr"] = hits

        timings["total"] = _ms_since(started)
        return build_result(
            q,
            hits,
            stages=stages,
            filtered_out=len(bm25.rejected_ids | dense.rejected_ids),
            timings_ms=timings,
            tau=cfg.tau,
        )


def build_retriever(
    mode: RetrievalMode,
    chunker: str = "structure",
    embedder: str = "bge-m3",
    settings: Any = None,
) -> Retriever:
    """Fábrica de recuperadores (SPEC §5.2). Devuelve un objeto que cumple ``Retriever``.

    Los recuperadores son livianos: los índices (BM25, Chroma) y los modelos (embedder,
    reranker) viven en cachés globales protegidas con locks, de modo que construir muchos
    recuperadores no recarga nada y es seguro usarlos desde varios hilos.

    Raises:
        ValueError: si ``mode`` no es uno de los cinco modos.
    """
    if mode == "bm25":
        return BM25Retriever(chunker, settings)
    if mode == "dense":
        return DenseRetriever(chunker, embedder, settings)
    if mode in get_args(RetrievalMode):
        return HybridRetriever(mode, chunker, embedder, settings)
    raise ValueError(
        f"Modo de recuperación desconocido: {mode!r}. Opciones: {get_args(RetrievalMode)}"
    )


def warmup(
    mode: RetrievalMode = "hybrid_rerank",
    chunker: str = "structure",
    embedder: str = "bge-m3",
    settings: Any = None,
) -> dict[str, float]:
    """Precarga índices y modelos que usa ``mode`` y devuelve el tiempo de cada carga (ms).

    Conviene llamarla al arrancar la API para que la primera consulta real no pague la
    carga de bge-m3 y del reranker.

    Raises:
        IndexNotFoundError: si falta el JSONL o el índice de Chroma (ver ``docuerp ingest``).
    """
    from docuerp.retrieval.models import get_embedder, get_reranker
    from docuerp.retrieval.store import get_collection, load_bm25

    timings: dict[str, float] = {}
    if mode != "dense":
        t0 = time.perf_counter()
        load_bm25(chunker, settings)
        timings["bm25_index"] = _ms_since(t0)
    if mode != "bm25":
        t0 = time.perf_counter()
        get_collection(chunker, embedder, settings)
        timings["chroma"] = _ms_since(t0)
        t0 = time.perf_counter()
        get_embedder(embedder).embed_query("calentamiento")
        timings["embedder"] = _ms_since(t0)
    if mode in RERANK_MODES:
        t0 = time.perf_counter()
        get_reranker(settings).score("calentamiento", ["texto de prueba"])
        timings["reranker"] = _ms_since(t0)
    return timings
