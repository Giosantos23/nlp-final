"""MMR: reduce redundancia, respeta λ y reutiliza vectores conocidos."""

from __future__ import annotations

import numpy as np
import pytest

from docuerp.embeddings import FakeEmbedder
from docuerp.retrieval import mmr as mmr_module
from docuerp.retrieval.mmr import mmr_select

QUERY = "reglas de reabastecimiento orden de compra automática"


@pytest.fixture
def candidates(make_hit):
    """Tres casi-duplicados muy relevantes y dos fragmentos distintos menos relevantes."""
    texts = [
        "reglas de reabastecimiento generan orden de compra automática cuando baja el stock",
        "reglas de reabastecimiento generan orden de compra automática cuando baja el stock mínimo",
        "reglas de reabastecimiento generan orden de compra automática cuando baja el inventario",
        "orden de compra proveedor productos",
        "reglas de reabastecimiento mínimo máximo almacén",
    ]
    return [
        make_hit(str(i), rank=i + 1, score=1.0 - 0.1 * i, retriever="rerank", text=t)
        for i, t in enumerate(texts)
    ]


def test_mmr_reduces_duplicates_in_top_3(candidates) -> None:
    plain = candidates[:3]  # el top-3 por relevancia son los tres casi-duplicados
    diverse = mmr_select(QUERY, candidates, top_k=3, lambda_=0.7, embedder="fake")

    names = {h.chunk.doc_id for h in diverse}
    assert len(diverse) == 3
    assert names != {h.chunk.doc_id for h in plain}
    assert names & {"doc-3", "doc-4"}, "debe entrar al menos un fragmento distinto"


def test_mmr_average_pairwise_similarity_is_lower_than_plain_top_k(candidates) -> None:
    embedder = FakeEmbedder()

    def mean_similarity(hits) -> float:
        vectors = embedder.embed_documents([h.chunk.text for h in hits])
        sims = vectors @ vectors.T
        n = len(hits)
        return float((sims.sum() - n) / (n * (n - 1)))

    diverse = mmr_select(QUERY, candidates, top_k=3, lambda_=0.7, embedder="fake")
    assert mean_similarity(diverse) < mean_similarity(candidates[:3])


def test_lambda_one_is_pure_relevance(candidates) -> None:
    embedder = FakeEmbedder()
    query_vec = embedder.embed_query(QUERY)
    relevance = [float(embedder.embed_query(c.chunk.text) @ query_vec) for c in candidates]
    expected = [candidates[i].chunk.doc_id for i in np.argsort(-np.array(relevance))[:3]]
    result = mmr_select(QUERY, candidates, top_k=3, lambda_=1.0, embedder="fake")
    assert [h.chunk.doc_id for h in result] == expected


def test_output_contract(candidates) -> None:
    result = mmr_select(QUERY, candidates, top_k=4, embedder="fake")
    assert [h.rank for h in result] == [1, 2, 3, 4]
    assert all(h.retriever == "mmr" for h in result)
    by_id = {c.chunk.chunk_id: c for c in candidates}
    for hit in result:
        # conserva el score de la etapa previa y agrega los de MMR
        assert hit.score == by_id[hit.chunk.chunk_id].score
        assert {"mmr", "mmr_relevance", "mmr_redundancy"} <= set(hit.stage_scores)
    assert result[0].stage_scores["mmr_redundancy"] == 0.0  # el primero no tiene redundancia


def test_top_k_larger_than_candidates_returns_all(candidates) -> None:
    result = mmr_select(QUERY, candidates, top_k=50, embedder="fake")
    assert sorted(h.chunk.chunk_id for h in result) == sorted(c.chunk.chunk_id for c in candidates)


def test_known_vectors_are_not_re_embedded(candidates, monkeypatch) -> None:
    embedder = FakeEmbedder()
    vectors = {c.chunk.chunk_id: embedder.embed_documents([c.chunk.text])[0] for c in candidates}
    embedded: list[int] = []
    real = mmr_module.embed_texts

    def spy(texts, **kwargs):
        embedded.append(len(texts))
        return real(texts, **kwargs)

    monkeypatch.setattr(mmr_module, "embed_texts", spy)
    mmr_select(QUERY, candidates, top_k=3, embedder="fake", vectors=vectors)
    assert embedded == [1], "solo debe codificarse la consulta"


def test_partially_known_vectors_embed_only_missing(candidates, monkeypatch) -> None:
    embedder = FakeEmbedder()
    vectors = {
        c.chunk.chunk_id: embedder.embed_documents([c.chunk.text])[0] for c in candidates[:3]
    }
    sizes: list[int] = []
    real = mmr_module.embed_texts
    monkeypatch.setattr(
        mmr_module,
        "embed_texts",
        lambda texts, **kw: (sizes.append(len(texts)), real(texts, **kw))[1],
    )
    mmr_select(QUERY, candidates, top_k=3, embedder="fake", vectors=vectors)
    assert sorted(sizes) == [1, 2]  # 2 documentos faltantes + 1 consulta


def test_empty_candidates() -> None:
    assert mmr_select("q", [], top_k=3, embedder="fake") == []


@pytest.mark.parametrize(
    "kwargs", [{"top_k": 0}, {"top_k": 3, "lambda_": 1.5}, {"top_k": 3, "lambda_": -0.1}]
)
def test_invalid_arguments(candidates, kwargs) -> None:
    with pytest.raises(ValueError):
        mmr_select(QUERY, candidates, embedder="fake", **kwargs)
