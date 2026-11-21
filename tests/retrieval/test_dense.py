"""Recuperador denso sobre Chroma con ``FakeEmbedder``."""

from __future__ import annotations

import numpy as np
import pytest

from docuerp.embeddings import FakeEmbedder
from docuerp.retrieval import dense as dense_module
from docuerp.retrieval.dense import DenseRetriever, distance_to_cosine
from docuerp.retrieval.store import IndexNotFoundError


def test_distance_to_cosine_per_space() -> None:
    assert distance_to_cosine(0.2, "cosine") == pytest.approx(0.8)
    assert distance_to_cosine(0.2, "ip") == pytest.approx(0.8)
    assert distance_to_cosine(0.4, "l2") == pytest.approx(0.8)  # ‖a-b‖² = 2(1-cos)


def test_identical_text_is_the_top_hit_with_cosine_one(toy_env, make_query, corpus_chunks) -> None:
    target = corpus_chunks[0]  # recepción
    result = DenseRetriever("structure", "fake", toy_env).retrieve(make_query(target.text))
    top = result.hits[0]
    assert top.chunk.chunk_id == target.chunk_id
    assert top.score == pytest.approx(1.0, abs=1e-3)
    assert top.stage_scores["dense_norm"] == pytest.approx(1.0, abs=1e-3)
    assert result.evidence_ok is True


def test_chunk_is_rebuilt_from_metadata_without_extra_keys(
    toy_env, make_query, corpus_chunks
) -> None:
    target = corpus_chunks[0]
    result = DenseRetriever("structure", "fake", toy_env).retrieve(make_query(target.text))
    assert result.hits[0].chunk == target


def test_flags_roundtrip(toy_env, make_query, corpus_chunks) -> None:
    flagged = next(c for c in corpus_chunks if c.flags)
    result = DenseRetriever("structure", "fake", toy_env).retrieve(
        make_query(flagged.text, role="n1")
    )
    assert result.hits[0].chunk.flags == ["suspicious_instruction"]


def test_retrieve_contract(toy_env, make_query) -> None:
    result = DenseRetriever("structure", "fake", toy_env).retrieve(
        make_query("validar recepción orden", top_k=4)
    )
    assert len(result.hits) == 4
    assert [h.rank for h in result.hits] == [1, 2, 3, 4]
    assert all(h.retriever == "dense" for h in result.hits)
    assert result.stages == {"dense": result.hits}
    assert set(result.timings_ms) == {"dense", "total"}
    cosines = [h.score for h in result.hits]
    assert cosines == sorted(cosines, reverse=True)


@pytest.mark.parametrize("role", ["portal_user", "n1", "n2", "admin"])
def test_only_authorized_chunks_for_each_role(toy_env, make_query, role) -> None:
    hits = DenseRetriever("structure", "fake", toy_env).search(
        make_query("runbook procedimiento restringido administradores base de datos", role=role), 10
    )
    assert hits
    assert all(role in h.chunk.allowed_roles for h in hits)
    assert all(h.chunk.status == "vigente" for h in hits)
    assert all(h.chunk.erp_version in ("18.0", "any") for h in hits)


def test_window_widens_when_top_neighbors_are_unauthorized(toy_env, make_query) -> None:
    """Con k=1 la ventana inicial es de 3 vecinos; si los 3 son de rol superior, se amplía."""
    retriever = DenseRetriever("structure", "fake", toy_env)
    q = make_query(
        "RB-ADM-002 restaurar copia seguridad productiva base datos Odoo", role="portal_user"
    )
    hits = retriever.search(q, 1)
    assert len(hits) == 1
    assert "portal_user" in hits[0].chunk.allowed_roles
    outcome = retriever.search_with_stats(q, 1)
    assert outcome.rejected_ids, "debe registrar los descartados por rol"


def test_filtered_out_matches_rejected_ids(toy_env, make_query) -> None:
    text = "RB-ADM-002 restaurar copia seguridad productiva"
    retriever = DenseRetriever("structure", "fake", toy_env)
    as_portal = make_query(text, role="portal_user")
    result = retriever.retrieve(as_portal)
    assert result.filtered_out == len(retriever.search_with_stats(as_portal, 5).rejected_ids)
    assert result.filtered_out > retriever.retrieve(make_query(text, role="admin")).filtered_out


def test_obsolete_documents_need_the_flag(toy_env, make_query, corpus_chunks) -> None:
    legacy = next(c for c in corpus_chunks if c.status == "obsoleto" and c.erp_version == "18.0")
    retriever = DenseRetriever("structure", "fake", toy_env)
    default = retriever.search(make_query(legacy.text), 10)
    included = retriever.search(make_query(legacy.text, include_obsolete=True), 10)
    assert legacy.chunk_id not in [h.chunk.chunk_id for h in default]
    assert included[0].chunk.chunk_id == legacy.chunk_id


def test_version_filter_prefers_matching_version(toy_env, make_query, corpus_chunks) -> None:
    only17 = next(c for c in corpus_chunks if c.doc_id == "odoo-17.0-inventario" and c.page == 9)
    hits = DenseRetriever("structure", "fake", toy_env).search(
        make_query(only17.text, erp_version="17.0"), 5
    )
    assert hits[0].chunk.chunk_id == only17.chunk_id
    assert all(h.chunk.erp_version in ("17.0", "any") for h in hits)


def test_k_larger_than_authorized_returns_everything_available(toy_env, make_query) -> None:
    hits = DenseRetriever("structure", "fake", toy_env).search(
        make_query("recepción", role="portal_user"), 200
    )
    assert 0 < len(hits) < 41
    assert len({h.chunk.chunk_id for h in hits}) == len(hits)


def test_fetch_vectors_returns_stored_embeddings(toy_env, corpus_chunks) -> None:
    retriever = DenseRetriever("structure", "fake", toy_env)
    target = corpus_chunks[3]
    vectors = retriever.fetch_vectors([target.chunk_id, "no-existe"])
    assert set(vectors) == {target.chunk_id}
    expected = FakeEmbedder().embed_documents([target.text])[0]
    assert np.allclose(vectors[target.chunk_id], expected, atol=1e-5)
    assert retriever.fetch_vectors([]) == {}


def test_missing_index_raises_clear_error(toy_env, make_query) -> None:
    with pytest.raises(IndexNotFoundError, match="--embedder e5-base"):
        DenseRetriever("structure", "e5-base", toy_env).retrieve(make_query("consulta"))


def test_l2_collections_are_converted_to_cosine(toy_env, make_query, monkeypatch) -> None:
    monkeypatch.setattr(dense_module, "_distance_space", lambda collection: "l2")
    retriever = DenseRetriever("structure", "fake", toy_env)
    # Los datos reales son coseno: con la conversión l2 el score deja de ser 1.0 para el
    # texto idéntico (distancia 0 → 1.0 en ambas) pero cambia para los demás.
    q = make_query("validar recepción orden")
    l2_scores = [h.score for h in retriever.search(q, 5)]
    monkeypatch.undo()
    cos_scores = [h.score for h in retriever.search(q, 5)]
    assert l2_scores != cos_scores
    assert all(a > b for a, b in zip(l2_scores, cos_scores, strict=True))
