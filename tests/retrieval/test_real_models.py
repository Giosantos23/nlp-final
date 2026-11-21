"""Prueba con modelos reales (bge-m3 + bge-reranker-v2-m3) sobre el corpus de juguete.

Marcada ``slow``: carga ~2 GB de modelos desde la caché de Hugging Face. Ejecutar con
``pytest tests/retrieval/test_real_models.py -m slow -s`` para ver las latencias medidas.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import pytest

from docuerp.retrieval import RetrievalQuery, build_retriever, store
from docuerp.retrieval.models import get_reranker, resolve_device

pytestmark = pytest.mark.slow

QUERIES = [
    # (consulta, rol, doc_id esperado en el top-3)
    ("¿Cómo valido una recepción en Inventario?", "n1", "odoo-18.0-inventario"),
    ("Qué hago con el error ERR-INV-204", "n2", "RB-N2-003@v1.1"),
    (
        "reglas para que se generen compras automáticas cuando falta stock",
        "n1",
        "odoo-18.0-inventario",
    ),
    ("procedimiento para restaurar la copia de seguridad productiva", "admin", "RB-ADM-002@v1.0"),
]


def test_real_models_end_to_end_latency(
    tmp_path: Path, corpus_chunks, corpus_writer, settings_factory
) -> None:
    store.clear_caches()
    corpus_writer(corpus_chunks, tmp_path / "processed", tmp_path / "index", embedder="bge-m3")
    settings = settings_factory(tmp_path, embedder="bge-m3")
    device = resolve_device(settings.retrieval.device)

    retriever = build_retriever("hybrid_rerank_mmr", "structure", "bge-m3", settings)
    t0 = time.perf_counter()
    first = retriever.retrieve(
        RetrievalQuery(text=QUERIES[0][0], role=QUERIES[0][1], erp_version="18.0")
    )
    cold_ms = (time.perf_counter() - t0) * 1000

    warm_totals: list[float] = []
    per_stage: dict[str, list[float]] = {}
    for text, role, expected_doc in QUERIES:
        q = RetrievalQuery(text=text, role=role, erp_version="18.0")
        result = retriever.retrieve(q)
        warm_totals.append(result.timings_ms["total"])
        for stage, ms in result.timings_ms.items():
            per_stage.setdefault(stage, []).append(ms)
        top3 = [h.chunk.doc_id for h in result.hits[:3]]
        assert expected_doc in top3, f"{text!r} → {top3}"
        assert result.evidence_ok, f"{text!r} no pasó el gate (top={result.top_score})"
        assert all(0.0 <= h.score <= 1.0 for h in result.hits)

    unrelated = retriever.retrieve(
        RetrievalQuery(
            text="receta de paella valenciana con mariscos", role="n1", erp_version="18.0"
        )
    )
    assert unrelated.evidence_ok is False, f"top_score={unrelated.top_score}"

    print(f"\n[latencia] dispositivo={device} reranker_fp16={get_reranker(settings).fp16}")
    print(f"[latencia] primera consulta (carga de modelos incluida): {cold_ms:.0f} ms")
    p50, worst = statistics.median(warm_totals), max(warm_totals)
    print(f"[latencia] consultas en caliente: P50={p50:.0f} ms, max={worst:.0f} ms")
    for stage, values in per_stage.items():
        print(f"[latencia]   {stage:>7}: media={statistics.mean(values):.0f} ms")
    print(f"[gate] top relevante={first.top_score:.3f}, fuera de dominio={unrelated.top_score:.3f}")
    store.clear_caches()
