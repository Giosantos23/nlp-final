"""Pruebas de las utilidades compartidas de texto y embeddings (sin modelos pesados)."""

from __future__ import annotations

import numpy as np

from docuerp.embeddings import get_embedder
from docuerp.text import bm25_tokenize, normalize_text, strip_accents


def test_bm25_tokenize_preserves_identifiers_and_parts() -> None:
    tokens = bm25_tokenize("Revisa stock.picking del ticket TCK-00042 según RB-N2-003")
    assert "stock.picking" in tokens
    assert "tck-00042" in tokens
    assert "rb-n2-003" in tokens
    assert "stock" in tokens
    assert "picking" in tokens
    assert "del" not in tokens  # stopword


def test_accents_and_normalization() -> None:
    assert strip_accents("Recepción añadida") == "Recepcion anadida"
    assert normalize_text("inven-\ntario   listo​") == "inventario listo"


def test_fake_embedder_is_normalized_and_deterministic() -> None:
    emb = get_embedder("fake")
    a = emb.embed_query("ajuste de inventario")
    b = emb.embed_documents(["ajuste de inventario"])[0]
    assert np.allclose(a, b)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5
