"""Reranking con cross-encoder bge-reranker-v2-m3 (top-20 → top-5)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from docuerp.schemas import ScoredChunk

Scorer = Callable[[str, list[str]], list[float]]
"""Función ``(consulta, textos) -> logits`` (un logit por texto). Permite inyectar un falso."""


def sigmoid(x: float) -> float:
    """Normaliza un logit del reranker a [0, 1] para el gate τ.

    Implementación numéricamente estable: nunca evalúa ``exp`` de un número grande y
    positivo, así que no desborda con logits extremos.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


class Reranker:
    """Cross-encoder perezoso; los scores devueltos están en [0, 1] (sigmoide).

    Args:
        scorer: función ``(consulta, textos) -> logits``. Si es ``None`` se usa el
            cross-encoder real (:func:`docuerp.retrieval.models.get_reranker`), que se carga
            recién en la primera llamada a :meth:`rerank`.
        settings: configuración para el modelo real (se ignora si se pasa ``scorer``).
    """

    def __init__(self, scorer: Scorer | None = None, settings: Any = None) -> None:
        self._scorer = scorer
        self._settings = settings

    def _score(self, query: str, texts: list[str]) -> list[float]:
        """Obtiene los logits, cargando el modelo real la primera vez si hace falta."""
        if self._scorer is None:
            from docuerp.retrieval.models import get_reranker

            self._scorer = get_reranker(self._settings).score
        return self._scorer(query, texts)

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        """Reordena ``candidates`` y devuelve ``top_k`` con ``retriever="rerank"``.

        El score de cada resultado es la sigmoide del logit del cross-encoder (en [0, 1]);
        el logit crudo queda en ``stage_scores["rerank_logit"]`` y los scores previos
        (bm25, dense, rrf…) se conservan para el inspector de trazas.

        Raises:
            ValueError: si ``top_k < 1`` o el scorer devuelve una cantidad distinta de
                scores que de candidatos.
        """
        if top_k < 1:
            raise ValueError("top_k debe ser ≥ 1")
        if not candidates:
            return []
        logits = self._score(query, [c.chunk.text for c in candidates])
        if len(logits) != len(candidates):
            raise ValueError(
                f"El reranker devolvió {len(logits)} scores para {len(candidates)} candidatos"
            )
        # sorted es estable: ante empate se conserva el orden previo (el de la fusión).
        order = sorted(range(len(candidates)), key=lambda i: -logits[i])[:top_k]
        reranked: list[ScoredChunk] = []
        for new_rank, i in enumerate(order, start=1):
            prob = sigmoid(logits[i])
            reranked.append(
                ScoredChunk(
                    chunk=candidates[i].chunk,
                    score=prob,
                    rank=new_rank,
                    retriever="rerank",
                    stage_scores={
                        **candidates[i].stage_scores,
                        "rerank": prob,
                        "rerank_logit": logits[i],
                    },
                )
            )
        return reranked
