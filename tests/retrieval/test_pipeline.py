"""Pipeline completo (BM25 + denso → RRF → rerank → MMR → gate) con modelos falsos."""

from __future__ import annotations

import pytest

from docuerp.config import RetrievalWeights
from docuerp.retrieval import (
    RetrievalQuery,
    Retriever,
    build_retriever,
)
from docuerp.retrieval.bm25 import BM25Retriever
from docuerp.retrieval.dense import DenseRetriever
from docuerp.retrieval.pipeline import HybridRetriever, warmup
from docuerp.retrieval.rerank import Reranker
from docuerp.retrieval.store import IndexNotFoundError

ALL_MODES = ["bm25", "dense", "hybrid", "hybrid_rerank", "hybrid_rerank_mmr"]
ROLES = ["portal_user", "n1", "n2", "admin"]


@pytest.fixture
def retriever_for(toy_env, fake_scorer):
    """Fábrica de recuperadores sobre el corpus de juguete con reranker falso."""

    def _build(mode: str, settings=None):
        cfg = settings or toy_env
        if mode in ("bm25", "dense"):
            return build_retriever(mode, "structure", "fake", cfg)  # type: ignore[arg-type]
        return HybridRetriever(
            mode,  # type: ignore[arg-type]
            "structure",
            "fake",
            cfg,
            reranker=Reranker(scorer=fake_scorer),
        )

    return _build


def test_build_retriever_returns_the_right_class(toy_env) -> None:
    assert isinstance(build_retriever("bm25", "structure", "fake", toy_env), BM25Retriever)
    assert isinstance(build_retriever("dense", "structure", "fake", toy_env), DenseRetriever)
    for mode in ("hybrid", "hybrid_rerank", "hybrid_rerank_mmr"):
        built = build_retriever(mode, "structure", "fake", toy_env)  # type: ignore[arg-type]
        assert isinstance(built, HybridRetriever)
        assert built.mode == mode
    assert all(
        isinstance(build_retriever(m, "structure", "fake", toy_env), Retriever)  # type: ignore[arg-type]
        for m in ALL_MODES
    )


def test_build_retriever_rejects_unknown_mode(toy_env) -> None:
    with pytest.raises(ValueError, match="desconocido"):
        build_retriever("magia", "structure", "fake", toy_env)  # type: ignore[arg-type]


def test_hybrid_retriever_rejects_non_hybrid_modes(toy_env) -> None:
    with pytest.raises(ValueError, match="solo admite"):
        HybridRetriever("bm25", "structure", "fake", toy_env)  # type: ignore[arg-type]


def test_hybrid_stages_and_rrf(retriever_for, make_query) -> None:
    result = retriever_for("hybrid").retrieve(make_query("validar recepción orden de recepción"))
    assert set(result.stages) == {"bm25", "dense", "fusion"}
    assert [h.rank for h in result.hits] == [1, 2, 3, 4, 5]
    assert all(h.retriever == "hybrid" for h in result.hits)
    # k=60 y pesos 0.5/0.5 según la fórmula
    top = result.hits[0]
    expected = 0.0
    for name in ("bm25", "dense"):
        if f"{name}_rank" in top.stage_scores:
            expected += 0.5 / (60 + top.stage_scores[f"{name}_rank"])
    assert top.stage_scores["rrf"] == pytest.approx(expected)
    assert result.hits == result.stages["fusion"][:5]
    assert set(result.timings_ms) == {"bm25", "dense", "fusion", "total"}


def with_retrieval(settings, **changes):
    """Copia de ``settings`` con otros valores en la sección ``retrieval``."""
    return settings.model_copy(update={"retrieval": settings.retrieval.model_copy(update=changes)})


def test_rrf_uses_configured_weights(toy_env, retriever_for, make_query) -> None:
    q = make_query("validar recepción orden de recepción", top_k=3)
    only_bm25 = with_retrieval(toy_env, weights=RetrievalWeights(bm25=1.0, dense=0.0))
    result = retriever_for("hybrid", only_bm25).retrieve(q)
    bm25_order = [h.chunk.chunk_id for h in result.stages["bm25"][:3]]
    assert [h.chunk.chunk_id for h in result.hits] == bm25_order
    assert result.hits[0].stage_scores["rrf"] == pytest.approx(1.0 / 61)


def test_rrf_k_comes_from_settings(toy_env, retriever_for, make_query) -> None:
    q = make_query("validar recepción", top_k=1)
    custom = with_retrieval(toy_env, rrf_k=10)
    top = retriever_for("hybrid", custom).retrieve(q).hits[0]
    expected = sum(
        0.5 / (10 + top.stage_scores[f"{n}_rank"])
        for n in ("bm25", "dense")
        if f"{n}_rank" in top.stage_scores
    )
    assert top.stage_scores["rrf"] == pytest.approx(expected)


def test_rerank_stage_reorders_and_normalizes(retriever_for, make_query) -> None:
    q = make_query("RB-N2-003 recepciones duplicadas stock.picking", role="n2")
    result = retriever_for("hybrid_rerank").retrieve(q)
    assert set(result.stages) == {"bm25", "dense", "fusion", "rerank"}
    assert len(result.stages["fusion"]) <= q.candidates_k
    assert len(result.hits) == 5
    assert all(h.retriever == "rerank" for h in result.hits)
    scores = [h.score for h in result.hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert result.hits[0].chunk.doc_id == "RB-N2-003@v1.1"
    assert result.hits == result.stages["rerank"][:5]
    assert result.top_score == pytest.approx(max(scores))
    assert "rerank" in result.timings_ms


def test_mmr_mode_selects_from_reranked_candidates(retriever_for, make_query) -> None:
    q = make_query("reglas de reabastecimiento orden de compra automática stock mínimo", top_k=3)
    plain = retriever_for("hybrid_rerank").retrieve(q)
    diverse = retriever_for("hybrid_rerank_mmr").retrieve(q)

    assert set(diverse.stages) == {"bm25", "dense", "fusion", "rerank", "mmr"}
    assert all(h.retriever == "mmr" for h in diverse.hits)
    assert {h.chunk.chunk_id for h in diverse.hits} <= {
        h.chunk.chunk_id for h in diverse.stages["rerank"]
    }
    # el top-3 puro son los 3 casi-duplicados de reabastecimiento; MMR debe variar la mezcla
    assert {h.chunk.chunk_id for h in diverse.hits} != {h.chunk.chunk_id for h in plain.hits}
    assert diverse.top_score == pytest.approx(max(h.stage_scores["rerank"] for h in diverse.hits))


def test_mmr_lowers_redundancy_against_plain_rerank(retriever_for, make_query) -> None:
    from docuerp.embeddings import FakeEmbedder

    q = make_query("reglas de reabastecimiento orden de compra automática stock mínimo", top_k=3)
    embedder = FakeEmbedder()

    def mean_similarity(hits) -> float:
        vectors = embedder.embed_documents([h.chunk.text for h in hits])
        sims = vectors @ vectors.T
        n = len(hits)
        return float((sims.sum() - n) / (n * (n - 1)))

    plain = retriever_for("hybrid_rerank").retrieve(q).hits
    diverse = retriever_for("hybrid_rerank_mmr").retrieve(q).hits
    assert mean_similarity(diverse) < mean_similarity(plain)


@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("version", ["17.0", "18.0"])
def test_every_hit_in_every_stage_is_authorized(
    retriever_for, make_query, mode, role, version
) -> None:
    q = make_query(
        "procedimiento recepción runbook base de datos stock.picking",
        role=role,
        erp_version=version,
    )
    result = retriever_for(mode).retrieve(q)
    for hits in [result.hits, *result.stages.values()]:
        for hit in hits:
            assert role in hit.chunk.allowed_roles
            assert hit.chunk.erp_version in (version, "any")
            assert hit.chunk.status == "vigente"


def test_filtered_out_is_the_union_across_retrievers(retriever_for, make_query) -> None:
    q = make_query("RB-ADM-002 restaurar copia de seguridad base de datos", role="portal_user")
    bm25 = retriever_for("bm25").retrieve(q)
    dense = retriever_for("dense").retrieve(q)
    hybrid = retriever_for("hybrid").retrieve(q)
    assert hybrid.filtered_out >= max(bm25.filtered_out, dense.filtered_out)
    assert hybrid.filtered_out > 0
    admin = retriever_for("hybrid").retrieve(q.model_copy(update={"role": "admin"}))
    assert hybrid.filtered_out > admin.filtered_out


def test_include_obsolete_reaches_the_pipeline(retriever_for, make_query) -> None:
    text = "procedimiento antiguo asistente transferencia inmediata validar recepciones"
    default = retriever_for("hybrid").retrieve(make_query(text))
    included = retriever_for("hybrid").retrieve(make_query(text, include_obsolete=True))
    legacy = "odoo-18.0-inventario-legacy::p0003::c000"
    assert legacy not in [h.chunk.chunk_id for h in default.hits]
    assert legacy in [h.chunk.chunk_id for h in included.hits]


def test_exact_identifier_wins_in_bm25_and_survives_fusion(retriever_for, make_query) -> None:
    q = make_query("ERR-INV-204", role="n2", top_k=3)
    assert retriever_for("bm25").retrieve(q).hits[0].chunk.doc_id == "RB-N2-003@v1.1"
    fused = retriever_for("hybrid").retrieve(q)
    assert fused.hits[0].chunk.doc_id == "RB-N2-003@v1.1"
    assert fused.hits[0].stage_scores["bm25_rank"] == 1.0


def test_gate_with_reranker_passes_relevant_and_blocks_unrelated(retriever_for, make_query) -> None:
    relevant = retriever_for("hybrid_rerank").retrieve(
        make_query("validar recepción orden de recepción botón Validar")
    )
    assert relevant.evidence_ok is True
    assert relevant.top_score is not None
    assert relevant.top_score >= 0.30

    unrelated = retriever_for("hybrid_rerank").retrieve(
        make_query("paella valenciana con mariscos y azafrán")
    )
    assert unrelated.evidence_ok is False


def test_gate_tau_is_configurable(toy_env, retriever_for, make_query) -> None:
    q = make_query("validar recepción orden de recepción botón Validar")
    strict = retriever_for("hybrid_rerank", with_retrieval(toy_env, tau=0.999)).retrieve(q)
    lenient = retriever_for("hybrid_rerank", with_retrieval(toy_env, tau=0.0)).retrieve(
        make_query("paella valenciana con mariscos y azafrán")
    )
    assert strict.evidence_ok is False
    assert strict.top_score is not None
    assert strict.top_score < 0.999
    assert lenient.evidence_ok is True  # con τ=0 basta con que existan hits autorizados


def test_gate_is_false_without_authorized_hits(retriever_for, make_query) -> None:
    # portal_user + consulta solo cubierta por runbooks restringidos y vocabulario ajeno
    result = retriever_for("bm25").retrieve(
        make_query("RB-ADM-001 rotación claves", role="portal_user")
    )
    assert result.hits == []
    assert result.evidence_ok is False
    assert result.top_score is None
    assert result.filtered_out >= 1


def test_candidates_k_limits_each_stage(retriever_for, make_query) -> None:
    q = make_query("orden de compra proveedor", top_k=2, candidates_k=4)
    result = retriever_for("hybrid_rerank").retrieve(q)
    assert len(result.stages["bm25"]) <= 4
    assert len(result.stages["dense"]) <= 4
    assert len(result.stages["fusion"]) <= 4
    assert len(result.hits) == 2


def test_top_k_larger_than_candidates_k_still_returns_top_k(retriever_for, make_query) -> None:
    q = make_query("orden de compra proveedor", top_k=6, candidates_k=3)
    assert len(retriever_for("hybrid").retrieve(q).hits) == 6


def test_missing_dense_index_fails_clearly(toy_env, make_query) -> None:
    retriever = HybridRetriever("hybrid", "structure", "e5-base", toy_env)
    with pytest.raises(IndexNotFoundError):
        retriever.retrieve(make_query("consulta"))


def test_result_is_a_valid_contract_roundtrip(retriever_for, make_query) -> None:
    result = retriever_for("hybrid_rerank_mmr").retrieve(make_query("validar recepción"))
    from docuerp.retrieval import RetrievalResult

    assert RetrievalResult.model_validate_json(result.model_dump_json()) == result


def test_warmup_reports_each_component(toy_env) -> None:
    timings = warmup("hybrid", "structure", "fake", toy_env)
    assert set(timings) == {"bm25_index", "chroma", "embedder"}
    assert all(ms >= 0 for ms in timings.values())
    assert set(warmup("bm25", "structure", "fake", toy_env)) == {"bm25_index"}
    assert set(warmup("dense", "structure", "fake", toy_env)) == {"chroma", "embedder"}


def test_warmup_loads_the_reranker_only_when_needed(toy_env, monkeypatch) -> None:
    loaded: list[object] = []

    class FakeModel:
        def score(self, query: str, texts: list[str]) -> list[float]:
            loaded.append(texts)
            return [0.0] * len(texts)

    monkeypatch.setattr("docuerp.retrieval.models.get_reranker", lambda settings=None: FakeModel())
    assert "reranker" not in warmup("hybrid", "structure", "fake", toy_env)
    assert loaded == []
    assert "reranker" in warmup("hybrid_rerank", "structure", "fake", toy_env)
    assert len(loaded) == 1


def test_query_contract_is_unchanged() -> None:
    q = RetrievalQuery(text="  hola  ", role="n1", erp_version="18.0")
    assert (q.text, q.top_k, q.candidates_k, q.include_obsolete) == ("hola", 5, 20, False)
    assert RetrievalWeights().bm25 == 0.5


def test_concurrent_retrievals_are_consistent(retriever_for, make_query) -> None:
    from concurrent.futures import ThreadPoolExecutor

    retriever = retriever_for("hybrid_rerank_mmr")
    q = make_query("validar recepción orden de recepción")
    expected = [h.chunk.chunk_id for h in retriever.retrieve(q).hits]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: retriever.retrieve(q), range(16)))
    assert all([h.chunk.chunk_id for h in r.hits] == expected for r in results)
