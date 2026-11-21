"""Gate de evidencia: decide si lo recuperado alcanza para responder o hay que abstenerse.

Regla (SPEC §5.2): ``evidence_ok`` ⇔ hay hits autorizados **y** el mejor score de evidencia
es ≥ τ (``settings.retrieval.tau``, 0.30 por defecto, a calibrar en dev).

El "score de evidencia" siempre vive en [0, 1] para poder compararlo con un único τ, pero
cada familia de recuperadores lo obtiene distinto:

* **Con reranker** (``hybrid_rerank``, ``hybrid_rerank_mmr``): sigmoide del logit del
  cross-encoder (``stage_scores["rerank"]``). Es la señal más confiable.
* **BM25** (``bm25_norm``): ``score / Σ idf(t)`` sobre los términos de la consulta. Con BM25
  Okapi, un fragmento de largo promedio que contiene cada término exactamente una vez
  obtiene justo ``Σ idf(t)``, así que el cociente es la fracción de la "masa informativa" de
  la consulta que cubre el fragmento (recortada a 1). Los términos de la consulta que no
  existen en el corpus cuentan con el idf máximo, de modo que consultas fuera de dominio
  puntúan bajo. Con τ = 0.30 se exige cubrir al menos ~30 % de esa masa.
* **Denso** (``dense_norm``): el coseno de bge-m3 entre textos no relacionados ronda 0.3-0.4
  y entre consulta y pasaje relevante 0.55-0.80. Se reescala linealmente el tramo
  ``[COSINE_FLOOR, 1]`` a ``[0, 1]``; con τ = 0.30 el coseno mínimo exigido es ≈ 0.55.
* **Híbrido sin reranker**: el mayor entre ``bm25_norm`` y ``dense_norm`` (basta con que un
  canal tenga evidencia fuerte; así un identificador exacto pasa por BM25 aunque el
  embedding no lo capte).

Estos criterios son heurísticos (sin reranker no hay una probabilidad real de relevancia) y
se documentan para poder calibrarlos con el banco de evaluación.
"""

from __future__ import annotations

from docuerp.retrieval.contracts import RetrievalQuery, RetrievalResult
from docuerp.schemas import ScoredChunk

COSINE_FLOOR = 0.35
"""Coseno por debajo del cual un par consulta-pasaje se considera ruido (ver docstring)."""


def dense_norm(cosine: float) -> float:
    """Reescala el coseno de ``[COSINE_FLOOR, 1]`` a ``[0, 1]`` (recortando fuera de rango)."""
    return min(1.0, max(0.0, (cosine - COSINE_FLOOR) / (1.0 - COSINE_FLOOR)))


def evidence_score(hit: ScoredChunk) -> float:
    """Score de evidencia de un hit en [0, 1], según las etapas que lo puntuaron.

    Prioriza el reranker; si no lo hay, toma el mayor de ``bm25_norm`` y ``dense_norm``.
    Devuelve 0.0 si el hit no trae ninguna señal normalizada.
    """
    stages = hit.stage_scores
    if "rerank" in stages:
        return stages["rerank"]
    signals = [stages[key] for key in ("bm25_norm", "dense_norm") if key in stages]
    return max(signals) if signals else 0.0


def evaluate_gate(hits: list[ScoredChunk], tau: float) -> tuple[bool, float | None]:
    """Aplica el gate de evidencia.

    Returns:
        ``(evidence_ok, top_score)``. ``top_score`` es el mayor score de evidencia entre los
        ``hits`` (``None`` si no hay ninguno). ``evidence_ok`` exige hits y ``top_score ≥ τ``.
    """
    if not hits:
        return False, None
    top_score = max(evidence_score(hit) for hit in hits)
    return top_score >= tau, top_score


def build_result(
    q: RetrievalQuery,
    hits: list[ScoredChunk],
    *,
    stages: dict[str, list[ScoredChunk]],
    filtered_out: int,
    timings_ms: dict[str, float],
    tau: float,
) -> RetrievalResult:
    """Arma el ``RetrievalResult`` final: aplica el gate de evidencia sobre ``hits``."""
    evidence_ok, top_score = evaluate_gate(hits, tau)
    return RetrievalResult(
        query=q,
        hits=hits,
        stages=stages,
        filtered_out=filtered_out,
        evidence_ok=evidence_ok,
        top_score=top_score,
        timings_ms=timings_ms,
    )
