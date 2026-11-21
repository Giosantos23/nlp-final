"""Reranker: sigmoide estable y reordenamiento con un scorer falso."""

from __future__ import annotations

import math

import pytest

from docuerp.retrieval.rerank import Reranker, sigmoid


def test_sigmoid_reference_values() -> None:
    assert sigmoid(0.0) == 0.5
    assert sigmoid(2.0) == pytest.approx(1 / (1 + math.exp(-2)))
    assert sigmoid(-2.0) == pytest.approx(1 - sigmoid(2.0))


def test_sigmoid_is_stable_for_extreme_logits() -> None:
    assert sigmoid(1000.0) == 1.0
    assert sigmoid(-1000.0) == 0.0
    assert 0.0 <= sigmoid(-745.0) <= 1.0


def test_rerank_orders_by_logit_and_normalizes(make_hit) -> None:
    candidates = [
        make_hit("a", rank=1, text="irrelevante"),
        make_hit("b", rank=2, text="muy relevante"),
        make_hit("c", rank=3, text="algo relevante"),
    ]
    logits = {"irrelevante": -3.0, "muy relevante": 4.0, "algo relevante": 0.5}
    reranker = Reranker(scorer=lambda query, texts: [logits[t] for t in texts])

    result = reranker.rerank("consulta", candidates, top_k=2)

    assert [h.chunk.doc_id for h in result] == ["doc-b", "doc-c"]
    assert [h.rank for h in result] == [1, 2]
    assert all(h.retriever == "rerank" for h in result)
    assert result[0].score == pytest.approx(sigmoid(4.0))
    assert result[0].stage_scores["rerank"] == pytest.approx(result[0].score)
    assert result[0].stage_scores["rerank_logit"] == 4.0
    assert all(0.0 <= h.score <= 1.0 for h in result)


def test_rerank_keeps_previous_stage_scores(make_hit) -> None:
    candidate = make_hit("a", stage_scores={"bm25": 3.0, "rrf": 0.02})
    (hit,) = Reranker(scorer=lambda q, texts: [1.0]).rerank("q", [candidate], top_k=5)
    assert hit.stage_scores["bm25"] == 3.0
    assert hit.stage_scores["rrf"] == 0.02


def test_ties_keep_previous_order(make_hit) -> None:
    candidates = [make_hit("a", rank=1), make_hit("b", rank=2), make_hit("c", rank=3)]
    result = Reranker(scorer=lambda q, texts: [1.0, 1.0, 1.0]).rerank("q", candidates, top_k=3)
    assert [h.chunk.doc_id for h in result] == ["doc-a", "doc-b", "doc-c"]


def test_top_k_larger_than_candidates(make_hit) -> None:
    result = Reranker(scorer=lambda q, t: [0.0, 1.0]).rerank(
        "q", [make_hit("a"), make_hit("b")], top_k=10
    )
    assert len(result) == 2


def test_empty_candidates_do_not_call_the_scorer() -> None:
    def explode(query: str, texts: list[str]) -> list[float]:
        raise AssertionError("no debería invocarse")

    assert Reranker(scorer=explode).rerank("q", [], top_k=5) == []


def test_scorer_is_called_once_with_all_texts(make_hit) -> None:
    calls: list[list[str]] = []

    def scorer(query: str, texts: list[str]) -> list[float]:
        calls.append(texts)
        return [0.0] * len(texts)

    Reranker(scorer=scorer).rerank("q", [make_hit("a"), make_hit("b")], top_k=1)
    assert calls == [["texto de a", "texto de b"]]


def test_invalid_top_k_and_mismatched_scores(make_hit) -> None:
    reranker = Reranker(scorer=lambda q, texts: [0.0])
    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank("q", [make_hit("a")], top_k=0)
    with pytest.raises(ValueError, match="scores"):
        reranker.rerank("q", [make_hit("a"), make_hit("b")], top_k=1)
