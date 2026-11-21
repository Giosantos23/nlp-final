"""Fusión Reciprocal Rank Fusion: ``score = Σ w_i / (k + rank_i)`` con k=60.

RRF (Cormack et al., 2009) combina listas ordenadas usando **solo el rango**, no el score
crudo. Es la forma natural de mezclar BM25 (scores sin cota) con coseno (en [-1, 1]) sin
tener que calibrar escalas.
"""

from __future__ import annotations

from docuerp.schemas import ScoredChunk


def rrf_fuse(
    rankings: dict[str, list[ScoredChunk]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
    limit: int | None = None,
) -> list[ScoredChunk]:
    """Fusiona listas ordenadas por ``chunk_id``; conserva los scores de cada etapa en
    ``stage_scores`` y marca ``retriever="hybrid"``.

    Args:
        rankings: ``{nombre_de_lista: hits ordenados de mejor a peor}``. El rango de cada
            hit es su posición en la lista (1-indexed), no su campo ``rank``.
        k: constante de amortiguación de RRF (60 en el artículo original).
        weights: peso por lista (por defecto, reparto uniforme: 0.5/0.5 con dos listas).
        limit: máximo de resultados a devolver (todos si es ``None``).

    Returns:
        Lista fusionada y ordenada por score RRF descendente. Cada ``stage_scores`` incluye
        los de las listas de origen, el ``rrf`` total y ``{nombre}_rank`` por lista.

    Raises:
        ValueError: si ``k < 1``, ``limit < 1``, o los pesos son negativos o nombran listas
            inexistentes.
    """
    if k < 1:
        raise ValueError("k de RRF debe ser ≥ 1")
    if limit is not None and limit < 1:
        raise ValueError("limit debe ser ≥ 1")
    if not rankings:
        return []
    default_weight = 1.0 / len(rankings)
    weights = weights or {}
    unknown = set(weights) - set(rankings)
    if unknown:
        raise ValueError(f"Pesos para listas inexistentes: {sorted(unknown)}")
    if any(w < 0 for w in weights.values()):
        raise ValueError("Los pesos de RRF no pueden ser negativos")

    totals: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    first_seen: dict[str, ScoredChunk] = {}
    merged_stages: dict[str, dict[str, float]] = {}
    for name, hits in rankings.items():
        weight = weights.get(name, default_weight)
        for position, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk.chunk_id
            totals[chunk_id] = totals.get(chunk_id, 0.0) + weight / (k + position)
            best_rank[chunk_id] = min(best_rank.get(chunk_id, position), position)
            first_seen.setdefault(chunk_id, hit)
            stages = merged_stages.setdefault(chunk_id, {})
            stages.update(hit.stage_scores)
            stages.setdefault(name, hit.score)
            stages[f"{name}_rank"] = float(position)

    # Desempate determinista: mejor rango en cualquier lista y luego el chunk_id.
    ordered = sorted(totals, key=lambda cid: (-totals[cid], best_rank[cid], cid))
    if limit is not None:
        ordered = ordered[:limit]
    return [
        ScoredChunk(
            chunk=first_seen[cid].chunk,
            score=totals[cid],
            rank=position,
            retriever="hybrid",
            stage_scores={**merged_stages[cid], "rrf": totals[cid]},
        )
        for position, cid in enumerate(ordered, start=1)
    ]
