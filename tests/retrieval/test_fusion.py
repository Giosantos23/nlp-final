"""Fusión RRF: fórmula, pesos, desempates y validaciones."""

from __future__ import annotations

import pytest

from docuerp.retrieval.fusion import rrf_fuse


def test_rrf_scores_match_formula(make_hit) -> None:
    bm25 = [make_hit("a", score=9.0), make_hit("b", score=5.0), make_hit("c", score=1.0)]
    dense = [make_hit("b", score=0.9), make_hit("a", score=0.8), make_hit("d", score=0.7)]
    fused = rrf_fuse({"bm25": bm25, "dense": dense}, k=60)
    by_name = {h.chunk.doc_id: h for h in fused}

    # a: rango 1 en bm25 y 2 en denso; b: 2 y 1 → empatan en score
    assert by_name["doc-a"].score == pytest.approx(0.5 / 61 + 0.5 / 62)
    assert by_name["doc-b"].score == pytest.approx(0.5 / 62 + 0.5 / 61)
    # c solo en bm25 (rango 3); d solo en denso (rango 3)
    assert by_name["doc-c"].score == pytest.approx(0.5 / 63)
    assert by_name["doc-d"].score == pytest.approx(0.5 / 63)
    assert all(h.retriever == "hybrid" for h in fused)


def test_chunk_in_both_lists_beats_chunk_in_one(make_hit) -> None:
    fused = rrf_fuse(
        {"bm25": [make_hit("solo"), make_hit("ambos")], "dense": [make_hit("ambos")]},
    )
    assert [h.chunk.doc_id for h in fused] == ["doc-ambos", "doc-solo"]
    assert [h.rank for h in fused] == [1, 2]


def test_stage_scores_are_preserved_and_annotated(make_hit) -> None:
    bm25 = [make_hit("a", score=7.5, stage_scores={"bm25": 7.5, "bm25_norm": 0.6})]
    dense = [make_hit("a", score=0.8, retriever="dense", stage_scores={"dense": 0.8})]
    (hit,) = rrf_fuse({"bm25": bm25, "dense": dense})
    assert hit.stage_scores["bm25"] == 7.5
    assert hit.stage_scores["bm25_norm"] == 0.6
    assert hit.stage_scores["dense"] == 0.8
    assert hit.stage_scores["bm25_rank"] == 1.0
    assert hit.stage_scores["dense_rank"] == 1.0
    assert hit.stage_scores["rrf"] == pytest.approx(hit.score)


def test_weights_change_the_order(make_hit) -> None:
    bm25 = [make_hit("lexico"), make_hit("semantico")]
    dense = [make_hit("semantico"), make_hit("lexico")]
    favor_bm25 = rrf_fuse({"bm25": bm25, "dense": dense}, weights={"bm25": 0.9, "dense": 0.1})
    favor_dense = rrf_fuse({"bm25": bm25, "dense": dense}, weights={"bm25": 0.1, "dense": 0.9})
    assert favor_bm25[0].chunk.doc_id == "doc-lexico"
    assert favor_dense[0].chunk.doc_id == "doc-semantico"


def test_ties_are_broken_deterministically(make_hit) -> None:
    bm25 = [make_hit("a"), make_hit("b")]
    dense = [make_hit("b"), make_hit("a")]
    first = rrf_fuse({"bm25": bm25, "dense": dense})
    second = rrf_fuse({"dense": dense, "bm25": bm25})
    assert [h.chunk.chunk_id for h in first] == [h.chunk.chunk_id for h in second]
    assert first[0].chunk.doc_id == "doc-a"  # empate total → orden por chunk_id


def test_limit_truncates_and_reranks_positions(make_hit) -> None:
    hits = [make_hit(str(i)) for i in range(10)]
    fused = rrf_fuse({"bm25": hits}, limit=3)
    assert len(fused) == 3
    assert [h.rank for h in fused] == [1, 2, 3]


def test_empty_inputs(make_hit) -> None:
    assert rrf_fuse({}) == []
    assert rrf_fuse({"bm25": [], "dense": []}) == []
    only_dense = rrf_fuse({"bm25": [], "dense": [make_hit("a")]})
    assert [h.chunk.doc_id for h in only_dense] == ["doc-a"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"limit": 0},
        {"weights": {"bm25": -1.0}},
        {"weights": {"inexistente": 1.0}},
    ],
)
def test_invalid_arguments_raise(make_hit, kwargs) -> None:
    with pytest.raises(ValueError):
        rrf_fuse({"bm25": [make_hit("a")]}, **kwargs)
