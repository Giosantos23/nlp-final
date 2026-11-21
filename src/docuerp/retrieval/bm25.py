"""Recuperador léxico BM25 (baseline).

BM25 puntúa por coincidencia de términos: es excelente con identificadores exactos
(``stock.picking``, ``RB-N2-003``, ``ERR-INV-204``) y débil con paráfrasis. Por eso es el
baseline y el complemento léxico del recuperador híbrido.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from docuerp.retrieval.contracts import RetrievalQuery, RetrievalResult, SearchOutcome
from docuerp.retrieval.filters import passes_filters
from docuerp.retrieval.gate import build_result
from docuerp.retrieval.store import load_bm25, resolve_settings
from docuerp.schemas import ScoredChunk
from docuerp.text import bm25_tokenize


def _reference_score(bm25: Any, tokens: list[str]) -> float:
    """Score BM25 de referencia para ``tokens``: ``Σ idf(t)``.

    Con BM25 Okapi, un documento de largo promedio que contiene un término exactamente una
    vez obtiene ``idf(t) · (k1+1)/(1+k1) = idf(t)``. Así ``Σ idf(t)`` es el score de un
    fragmento que cubre *todos* los términos de la consulta una vez, y el cociente
    ``score / Σ idf`` mide qué fracción de la masa informativa de la consulta cubre el
    fragmento (se recorta a 1).

    Un término ausente del corpus cuenta con el idf de df=0, ``ln(2N+1)``: una consulta con
    vocabulario desconocido no puede alcanzar un score normalizado alto. Los idf negativos
    (términos presentes en más de la mitad de los documentos) se acotan a 0.
    """
    unseen_idf = math.log(2 * bm25.corpus_size + 1)
    return sum(max(bm25.idf.get(token, unseen_idf), 0.0) for token in tokens)


class BM25Retriever:
    """BM25 sobre ``data/index/{chunker}/bm25.pkl`` con pre-filtro de metadatos.

    El índice se construye en memoria desde ``chunks.{chunker}.jsonl`` y se cachea en disco
    (se invalida si cambia el hash del JSONL); ver :func:`docuerp.retrieval.store.load_bm25`.
    """

    def __init__(self, chunker: str = "structure", settings: Any = None) -> None:
        self.chunker = chunker
        self.settings = settings

    def search_with_stats(self, q: RetrievalQuery, k: int) -> SearchOutcome:
        """Como :meth:`search` pero también informa los ids descartados por los filtros.

        Recorre los chunks con score > 0 de mejor a peor; descarta los que no pasan el
        pre-filtro (rol, versión, vigencia) y se detiene al reunir ``k`` autorizados.
        """
        corpus, index = load_bm25(self.chunker, self.settings)
        tokens = list(dict.fromkeys(bm25_tokenize(q.text)))  # únicos, en orden
        if not tokens:
            return SearchOutcome(hits=[])
        scores = index.bm25.get_scores(tokens)
        reference = _reference_score(index.bm25, tokens)
        hits: list[ScoredChunk] = []
        rejected: set[str] = set()
        for position in np.argsort(-scores, kind="stable"):
            raw = float(scores[position])
            if raw <= 0.0:
                break  # el resto no comparte ningún término con la consulta
            chunk = corpus.chunks[int(position)]
            if not passes_filters(chunk, q):
                rejected.add(chunk.chunk_id)
                continue
            normalized = min(1.0, raw / reference) if reference > 0 else 0.0
            hits.append(
                ScoredChunk(
                    chunk=chunk,
                    score=raw,
                    rank=len(hits) + 1,
                    retriever="bm25",
                    stage_scores={"bm25": raw, "bm25_norm": normalized},
                )
            )
            if len(hits) == k:
                break
        return SearchOutcome(hits=hits, rejected_ids=frozenset(rejected))

    def search(self, q: RetrievalQuery, k: int) -> list[ScoredChunk]:
        """Devuelve hasta ``k`` candidatos autorizados (``retriever="bm25"``)."""
        return self.search_with_stats(q, k).hits

    def retrieve(self, q: RetrievalQuery) -> RetrievalResult:
        """Recuperación BM25 completa (top-k + gate de evidencia)."""
        start = time.perf_counter()
        outcome = self.search_with_stats(q, q.top_k)
        elapsed = (time.perf_counter() - start) * 1000
        return build_result(
            q,
            outcome.hits,
            stages={"bm25": outcome.hits},
            filtered_out=len(outcome.rejected_ids),
            timings_ms={"bm25": elapsed, "total": elapsed},
            tau=resolve_settings(self.settings).retrieval.tau,
        )
