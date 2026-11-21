"""Recuperador denso sobre Chroma (bge-m3 o multilingual-e5-base).

La consulta se convierte en un vector con el **mismo** embedder que se usó al indexar y se
buscan los vecinos más cercanos (HNSW, coseno). Captura paráfrasis y sinónimos que BM25
no ve ("dar de alta una orden de compra" ≈ "crear un purchase order").
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from docuerp.retrieval.contracts import RetrievalQuery, RetrievalResult, SearchOutcome
from docuerp.retrieval.filters import chroma_where, passes_filters
from docuerp.retrieval.gate import build_result, dense_norm
from docuerp.retrieval.models import get_embedder
from docuerp.retrieval.store import chunk_from_metadata, get_collection, resolve_settings
from docuerp.schemas import Chunk, ScoredChunk

WINDOW_FACTOR = 3
"""Se piden ``candidates_k × 3`` vecinos antes de filtrar por rol (SPEC §8)."""


def _distance_space(collection: Any) -> str:
    """Métrica del índice HNSW de la colección (``cosine``, ``ip`` o ``l2``).

    Chroma 1.x la expone en ``collection.configuration``; las colecciones creadas con el
    metadato legado ``hnsw:space`` también se leen. Si no se declara, Chroma usa ``l2``.
    """
    try:
        return str(collection.configuration["hnsw"]["space"])
    except (KeyError, TypeError, AttributeError):
        return str((collection.metadata or {}).get("hnsw:space", "l2"))


def distance_to_cosine(distance: float, space: str) -> float:
    """Convierte la distancia de Chroma en similitud coseno.

    Con vectores L2-normalizados: ``cosine`` e ``ip`` dan ``1 - d`` y ``l2`` (distancia
    euclidiana al cuadrado) da ``1 - d/2``.
    """
    return 1.0 - distance / 2.0 if space == "l2" else 1.0 - distance


class DenseRetriever:
    """Búsqueda HNSW coseno en ``data/index/{chunker}/{embedder}/chroma``."""

    def __init__(
        self, chunker: str = "structure", embedder: str = "bge-m3", settings: Any = None
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.settings = settings

    def _collection(self) -> Any:
        return get_collection(self.chunker, self.embedder, self.settings)

    def _scan(
        self, raw: dict[str, Any], q: RetrievalQuery, k: int, space: str
    ) -> tuple[list[ScoredChunk], set[str]]:
        """Recorre los vecinos devueltos por Chroma en orden de cercanía.

        Aplica el filtro de rol (y versión/vigencia, por si la consulta no los filtró de
        forma nativa) y se detiene al reunir ``k`` autorizados. Devuelve los hits y los ids
        descartados antes de completar ``k``.
        """
        hits: list[ScoredChunk] = []
        rejected: set[str] = set()
        ids = raw["ids"][0]
        for chunk_id, document, metadata, distance in zip(
            ids, raw["documents"][0], raw["metadatas"][0], raw["distances"][0], strict=True
        ):
            chunk: Chunk = chunk_from_metadata(chunk_id, document, metadata)
            if not passes_filters(chunk, q):
                rejected.add(chunk_id)
                continue
            cosine = distance_to_cosine(float(distance), space)
            hits.append(
                ScoredChunk(
                    chunk=chunk,
                    score=cosine,
                    rank=len(hits) + 1,
                    retriever="dense",
                    stage_scores={"dense": cosine, "dense_norm": dense_norm(cosine)},
                )
            )
            if len(hits) == k:
                break
        return hits, rejected

    def search_with_stats(self, q: RetrievalQuery, k: int) -> SearchOutcome:
        """Como :meth:`search` pero también informa los ids descartados por los filtros.

        Estrategia de pre-filtro:

        1. Se piden ``k × 3`` vecinos sin filtro y se descartan en Python los no autorizados.
        2. Si faltan autorizados y quedaban más vecinos, se repite con el filtro nativo de
           Chroma (versión y vigencia, :func:`chroma_where`) y una ventana que se duplica
           hasta reunir ``k`` o agotar la colección. El rol siempre se verifica en Python.
        """
        collection = self._collection()
        total = collection.count()
        if total == 0:
            return SearchOutcome(hits=[])
        space = _distance_space(collection)
        vector = np.asarray(get_embedder(self.embedder).embed_query(q.text), dtype=np.float32)

        window = min(total, k * WINDOW_FACTOR)
        where: dict[str, Any] | None = None
        rejected: set[str] = set()
        while True:
            raw = collection.query(
                query_embeddings=[vector.tolist()],
                n_results=window,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            hits, dropped = self._scan(raw, q, k, space)
            rejected |= dropped
            exhausted = len(raw["ids"][0]) < window or window >= total
            if len(hits) >= k or exhausted:
                return SearchOutcome(hits=hits, rejected_ids=frozenset(rejected))
            where = chroma_where(q)
            window = min(total, window * 2)

    def search(self, q: RetrievalQuery, k: int) -> list[ScoredChunk]:
        """Devuelve hasta ``k`` candidatos autorizados (``retriever="dense"``)."""
        return self.search_with_stats(q, k).hits

    def fetch_vectors(self, chunk_ids: list[str]) -> dict[str, np.ndarray]:
        """Recupera de Chroma los embeddings ya indexados de ``chunk_ids`` (para MMR), lo que
        evita volver a codificar los textos. Los ids inexistentes se omiten."""
        if not chunk_ids:
            return {}
        raw = self._collection().get(ids=chunk_ids, include=["embeddings"])
        embeddings = raw["embeddings"]
        if embeddings is None:
            return {}
        return {
            chunk_id: np.asarray(vector, dtype=np.float32)
            for chunk_id, vector in zip(raw["ids"], embeddings, strict=True)
        }

    def retrieve(self, q: RetrievalQuery) -> RetrievalResult:
        """Recuperación densa completa (top-k + gate de evidencia)."""
        start = time.perf_counter()
        outcome = self.search_with_stats(q, q.top_k)
        elapsed = (time.perf_counter() - start) * 1000
        return build_result(
            q,
            outcome.hits,
            stages={"dense": outcome.hits},
            filtered_out=len(outcome.rejected_ids),
            timings_ms={"dense": elapsed, "total": elapsed},
            tau=resolve_settings(self.settings).retrieval.tau,
        )
