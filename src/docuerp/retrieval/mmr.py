"""Maximal Marginal Relevance (λ=0.7) para reducir redundancia en el top-k.

El top-5 por relevancia suele traer varios fragmentos casi idénticos (p. ej. el solape entre
chunks consecutivos o la misma sección en dos versiones). MMR (Carbonell y Goldstein, 1998)
elige de forma iterativa el candidato que mejor equilibra relevancia con la consulta y
novedad respecto a lo ya elegido:

    MMR(d) = λ · sim(d, consulta) − (1 − λ) · max_{s ∈ elegidos} sim(d, s)

con λ = 0.7 (más peso a la relevancia). Las similitudes son cosenos entre embeddings.
"""

from __future__ import annotations

import numpy as np

from docuerp.retrieval.models import embed_texts
from docuerp.schemas import ScoredChunk


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Normaliza cada fila a norma 1 (los embeddings ya lo están; es una defensa barata
    para que el producto punto sea siempre un coseno)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


def mmr_select(
    query: str,
    candidates: list[ScoredChunk],
    *,
    top_k: int,
    lambda_: float = 0.7,
    embedder: str = "bge-m3",
    vectors: dict[str, np.ndarray] | None = None,
) -> list[ScoredChunk]:
    """Selecciona ``top_k`` chunks maximizando ``λ·rel − (1−λ)·max_sim`` (``retriever="mmr"``).

    Args:
        query: texto de la consulta (se codifica con ``embedder``).
        candidates: candidatos ya ordenados por la etapa previa (se usa su orden para
            desempatar: ante igual valor MMR gana el mejor rankeado).
        top_k: cuántos devolver (menos si hay menos candidatos).
        lambda_: peso de la relevancia en [0, 1]; 1 = solo relevancia, 0 = solo diversidad.
        embedder: modelo de embeddings (``bge-m3``, ``e5-base`` o ``fake``).
        vectors: embeddings ya conocidos por ``chunk_id`` (p. ej. los guardados en Chroma);
            solo se codifican los textos que falten.

    Returns:
        Hasta ``top_k`` resultados en orden de selección. Conservan el ``score`` de la
        etapa previa (así el gate sigue leyendo la relevancia real) y añaden
        ``mmr``, ``mmr_relevance`` y ``mmr_redundancy`` a ``stage_scores``.

    Raises:
        ValueError: si ``top_k < 1`` o ``lambda_`` está fuera de [0, 1].
    """
    if top_k < 1:
        raise ValueError("top_k debe ser ≥ 1")
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ debe estar en [0, 1]")
    if not candidates:
        return []

    known = vectors or {}
    missing = [c for c in candidates if c.chunk.chunk_id not in known]
    computed: dict[str, np.ndarray] = {}
    if missing:
        encoded = embed_texts([c.chunk.text for c in missing], embedder=embedder)
        computed = {c.chunk.chunk_id: encoded[i] for i, c in enumerate(missing)}
    doc_matrix = _unit_rows(
        np.stack(
            [
                np.asarray(known.get(c.chunk.chunk_id, computed.get(c.chunk.chunk_id)))
                for c in candidates
            ]
        ).astype(np.float64)
    )
    query_vec = _unit_rows(
        embed_texts([query], embedder=embedder, is_query=True).astype(np.float64)
    )[0]

    relevance = doc_matrix @ query_vec  # coseno consulta-documento
    similarity = doc_matrix @ doc_matrix.T  # coseno documento-documento

    selected: list[int] = []
    redundancy_of: dict[int, float] = {}
    value_of: dict[int, float] = {}
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < min(top_k, len(candidates)):
        best_index = -1
        best_value = -np.inf
        best_redundancy = 0.0
        for i in remaining:  # recorrido en orden de ranking: el primero gana los empates
            redundancy = max((similarity[i, j] for j in selected), default=0.0)
            value = lambda_ * relevance[i] - (1.0 - lambda_) * redundancy
            if value > best_value:
                best_index, best_value, best_redundancy = i, value, float(redundancy)
        selected.append(best_index)
        remaining.remove(best_index)
        redundancy_of[best_index] = best_redundancy
        value_of[best_index] = float(best_value)

    return [
        ScoredChunk(
            chunk=candidates[i].chunk,
            score=candidates[i].score,
            rank=position,
            retriever="mmr",
            stage_scores={
                **candidates[i].stage_scores,
                "mmr": value_of[i],
                "mmr_relevance": float(relevance[i]),
                "mmr_redundancy": redundancy_of[i],
            },
        )
        for position, i in enumerate(selected, start=1)
    ]
