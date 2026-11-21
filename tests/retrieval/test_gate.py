"""Gate de evidencia: criterio por modo y umbral τ."""

from __future__ import annotations

import pytest

from docuerp.retrieval.gate import COSINE_FLOOR, dense_norm, evaluate_gate, evidence_score


def test_dense_norm_rescales_cosine() -> None:
    assert dense_norm(COSINE_FLOOR) == 0.0
    assert dense_norm(1.0) == 1.0
    assert dense_norm(0.0) == 0.0  # por debajo del piso se recorta
    assert dense_norm(0.675) == pytest.approx(0.5)


def test_evidence_score_prefers_reranker(make_hit) -> None:
    hit = make_hit("a", stage_scores={"rerank": 0.8, "bm25_norm": 0.1, "dense_norm": 0.2})
    assert evidence_score(hit) == 0.8


def test_evidence_score_without_reranker_takes_best_channel(make_hit) -> None:
    assert evidence_score(make_hit("a", stage_scores={"bm25_norm": 0.2, "dense_norm": 0.5})) == 0.5
    assert evidence_score(make_hit("b", stage_scores={"bm25_norm": 0.7})) == 0.7
    assert evidence_score(make_hit("c")) == 0.0


def test_gate_requires_hits(make_hit) -> None:
    assert evaluate_gate([], tau=0.3) == (False, None)


def test_gate_threshold_is_inclusive(make_hit) -> None:
    hit = make_hit("a", stage_scores={"rerank": 0.30})
    assert evaluate_gate([hit], tau=0.30) == (True, 0.30)
    assert evaluate_gate([hit], tau=0.31)[0] is False


def test_gate_uses_best_hit_not_only_the_first(make_hit) -> None:
    hits = [
        make_hit("a", rank=1, stage_scores={"rerank": 0.1}),
        make_hit("b", rank=2, stage_scores={"rerank": 0.9}),
    ]
    ok, top = evaluate_gate(hits, tau=0.3)
    assert ok is True
    assert top == 0.9
