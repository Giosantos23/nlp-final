"""Recuperador BM25: identificadores exactos, filtros, gate y normalización."""

from __future__ import annotations

import pytest

from docuerp.retrieval.bm25 import BM25Retriever
from docuerp.retrieval.filters import passes_filters
from docuerp.text import bm25_tokenize


def ids(hits) -> list[str]:
    return [h.chunk.doc_id for h in hits]


@pytest.mark.parametrize(
    ("query", "expected_doc"),
    [
        ("RB-N2-003", "RB-N2-003@v1.1"),
        ("ERR-INV-204", "RB-N2-003@v1.1"),
        ("RB-ADM-002", "RB-ADM-002@v1.0"),
        ("RB-N1-001", "RB-N1-001@v1.0"),
    ],
)
def test_exact_identifier_wins(toy_env, make_query, query, expected_doc) -> None:
    retriever = BM25Retriever("structure", toy_env)
    result = retriever.retrieve(make_query(query, role="admin"))
    assert result.hits[0].chunk.doc_id == expected_doc


def test_technical_identifier_ranks_model_chunk_first(toy_env, make_query) -> None:
    retriever = BM25Retriever("structure", toy_env)
    hits = retriever.search(make_query("stock.picking transferencias"), 5)
    assert hits[0].chunk.chunk_id == "odoo-18.0-inventario::p0020::c000"


def test_retrieve_contract(toy_env, make_query) -> None:
    result = BM25Retriever("structure", toy_env).retrieve(make_query("validar recepción", top_k=3))
    assert len(result.hits) == 3
    assert [h.rank for h in result.hits] == [1, 2, 3]
    assert all(h.retriever == "bm25" for h in result.hits)
    assert result.stages == {"bm25": result.hits}
    assert set(result.timings_ms) == {"bm25", "total"}
    scores = [h.score for h in result.hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= h.stage_scores["bm25_norm"] <= 1.0 for h in result.hits)


def test_role_filter_hides_restricted_runbooks(toy_env, make_query) -> None:
    retriever = BM25Retriever("structure", toy_env)
    as_portal = retriever.retrieve(
        make_query("RB-N2-003 recepciones duplicadas", role="portal_user")
    )
    as_n2 = retriever.retrieve(make_query("RB-N2-003 recepciones duplicadas", role="n2"))
    assert all("portal_user" in h.chunk.allowed_roles for h in as_portal.hits)
    assert "RB-N2-003@v1.1" not in ids(as_portal.hits)
    assert "RB-N2-003@v1.1" in ids(as_n2.hits)
    assert as_portal.filtered_out > as_n2.filtered_out


def test_version_filter(toy_env, make_query) -> None:
    retriever = BM25Retriever("structure", toy_env)
    hits_17 = retriever.search(make_query("validar recepción", erp_version="17.0"), 10)
    hits_18 = retriever.search(make_query("validar recepción", erp_version="18.0"), 10)
    assert hits_17
    assert hits_18
    assert {h.chunk.erp_version for h in hits_17} <= {"17.0", "any"}
    assert {h.chunk.erp_version for h in hits_18} <= {"18.0", "any"}


def test_obsolete_only_with_include_obsolete(toy_env, make_query) -> None:
    retriever = BM25Retriever("structure", toy_env)
    default = retriever.retrieve(
        make_query("procedimiento antiguo asistente transferencia inmediata")
    )
    with_old = retriever.retrieve(
        make_query("procedimiento antiguo asistente transferencia inmediata", include_obsolete=True)
    )
    assert "odoo-18.0-inventario-legacy" not in ids(default.hits)
    assert with_old.hits[0].chunk.doc_id == "odoo-18.0-inventario-legacy"
    assert default.filtered_out > with_old.filtered_out >= 0


def test_filtered_out_counts_every_rejected_match(toy_env, make_query, corpus_chunks) -> None:
    """portal_user no ve ningún runbook: todo chunk que contenga los términos se descarta."""
    text = "RB-ADM-002 RB-N2-003"
    q = make_query(text, role="portal_user")
    result = BM25Retriever("structure", toy_env).retrieve(q)
    terms = set(bm25_tokenize(text))
    expected = {
        c.chunk_id
        for c in corpus_chunks
        if terms & set(bm25_tokenize(c.text)) and not passes_filters(c, q)
    }
    assert expected, "el escenario debe tener coincidencias restringidas"
    assert result.filtered_out == len(expected)


def test_gate_passes_with_strong_match_and_fails_out_of_vocabulary(toy_env, make_query) -> None:
    retriever = BM25Retriever("structure", toy_env)
    strong = retriever.retrieve(make_query("RB-N2-003", role="n2"))
    assert strong.evidence_ok is True
    assert strong.top_score is not None
    assert strong.top_score >= 0.30

    weak = retriever.retrieve(make_query("paella valenciana con mariscos y azafrán"))
    assert weak.hits == []
    assert weak.evidence_ok is False
    assert weak.top_score is None


def test_partial_vocabulary_overlap_scores_low(toy_env, make_query) -> None:
    """Una consulta con un solo término conocido y muchos desconocidos no debe pasar τ."""
    result = BM25Retriever("structure", toy_env).retrieve(
        make_query("recepción astronomía cuántica gastronomía submarina")
    )
    assert result.top_score is not None
    assert result.top_score < 0.30
    assert result.evidence_ok is False


def test_query_of_only_stopwords_returns_nothing(toy_env, make_query) -> None:
    result = BM25Retriever("structure", toy_env).retrieve(make_query("de la el y"))
    assert result.hits == []
    assert result.evidence_ok is False


def test_search_respects_k(toy_env, make_query) -> None:
    hits = BM25Retriever("structure", toy_env).search(make_query("orden compra"), 2)
    assert len(hits) == 2
