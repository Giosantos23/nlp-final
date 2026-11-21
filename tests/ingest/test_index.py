"""Pruebas de persistencia: JSONL, BM25 y Chroma (con ``FakeEmbedder``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from docuerp import paths
from docuerp.embeddings import get_embedder
from docuerp.ingest import index
from docuerp.ingest.chunk import ChunkDraft
from docuerp.ingest.manifest import SourceEntry
from docuerp.ingest.metadata import build_chunks
from docuerp.schemas import Chunk


@pytest.fixture
def chunks(entry: SourceEntry) -> list[Chunk]:
    texts = [
        "Inventory > Receipts\nValidate the receipt WH/IN/00014 in two steps.",
        "Inventory > Scrap\nScrap damaged products from stock.picking forms.",
        "Purchase > RfQ\nCreate a request for quotation for vendor Azure Interior.",
    ]
    drafts = [
        ChunkDraft(text=t, page=i + 2, page_end=i + 2, section_path="x", token_count=10)
        for i, t in enumerate(texts)
    ]
    return build_chunks(drafts, entry, flags_by_index={2: ["suspicious_instruction"]})


@pytest.fixture
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(paths, "INDEX", tmp_path / "index")
    monkeypatch.setattr(paths, "PROCESSED", tmp_path / "processed")
    return tmp_path


def test_paths_follow_spec(isolated_paths: Path) -> None:
    assert (
        index.chroma_dir("structure", "bge-m3") == isolated_paths / "index/structure/bge-m3/chroma"
    )
    assert index.bm25_path("fixed256") == isolated_paths / "index/fixed256/bm25.pkl"
    assert index.chunks_path("structure") == isolated_paths / "processed/chunks.structure.jsonl"


def test_flatten_roundtrip() -> None:
    assert index.flatten_list(["n1", "n2"]) == "|n1|n2|"
    assert index.flatten_list([]) == ""
    assert index.unflatten_list("|n1|n2|") == ["n1", "n2"]
    assert index.unflatten_list("") == []


def test_metadata_roundtrip_returns_identical_chunk(chunks: list[Chunk]) -> None:
    for chunk in chunks:
        meta = index.chunk_to_metadata(chunk)
        assert all(isinstance(v, str | int | float | bool) for v in meta.values())
        assert meta["allowed_roles"] == "|portal_user|n1|n2|admin|"
        assert "text" not in meta
        assert index.chunk_from_record(chunk.text, meta) == chunk


def test_jsonl_roundtrip_and_errors(isolated_paths: Path, chunks: list[Chunk]) -> None:
    path = index.write_chunks(chunks, "structure")
    assert path.read_text(encoding="utf-8").count("\n") == 3
    assert index.load_chunks("structure") == chunks
    with pytest.raises(FileNotFoundError, match="ingest run"):
        index.load_chunks("fixed768")
    path.write_text('{"chunk_id": "broken"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":1:"):
        index.load_chunks("structure")


def test_bm25_build_and_load(isolated_paths: Path, chunks: list[Chunk]) -> None:
    from docuerp.text import bm25_tokenize

    index.build_bm25_index(chunks, "structure")
    bm25, ids = index.load_bm25_index("structure")
    assert ids == [c.chunk_id for c in chunks]
    scores = bm25.get_scores(bm25_tokenize("stock.picking scrap"))
    assert ids[int(scores.argmax())] == chunks[1].chunk_id
    with pytest.raises(ValueError):
        index.build_bm25_index([], "structure")
    with pytest.raises(FileNotFoundError):
        index.load_bm25_index("fixed256")


def test_chroma_build_query_and_reconstruct(isolated_paths: Path, chunks: list[Chunk]) -> None:
    directory = index.build_chroma_index(chunks, "structure", "fake")
    assert directory == isolated_paths / "index/structure/fake/chroma"
    collection = index.open_chroma_collection("structure", "fake")
    assert collection.count() == 3
    assert collection.configuration_json["hnsw"]["space"] == "cosine"
    query = get_embedder("fake").embed_query(chunks[2].text)
    result = collection.query(query_embeddings=[query.tolist()], n_results=1)
    assert result["ids"][0][0] == chunks[2].chunk_id
    rebuilt = index.chunk_from_record(result["documents"][0][0], result["metadatas"][0][0])
    assert rebuilt == chunks[2]
    assert rebuilt.flags == ["suspicious_instruction"]
    # Filtro escalar en Chroma + filtro por rol en Python sobre la lista aplanada.
    filtered = collection.get(where={"erp_version": "18.0"}, include=["metadatas"])
    assert len(filtered["ids"]) == 3
    assert all("|n1|" in m["allowed_roles"] for m in filtered["metadatas"])
    assert collection.get(where={"erp_version": "17.0"})["ids"] == []
    # Reconstruir el índice lo reemplaza (no duplica).
    index.build_chroma_index(chunks[:1], "structure", "fake")
    assert index.open_chroma_collection("structure", "fake").count() == 1


def test_chroma_errors(isolated_paths: Path, chunks: list[Chunk]) -> None:
    with pytest.raises(ValueError, match="No hay chunks"):
        index.build_chroma_index([], "structure", "fake")
    with pytest.raises(ValueError, match="repetidos"):
        index.build_chroma_index([chunks[0], chunks[0]], "structure", "fake")
    with pytest.raises(FileNotFoundError):
        index.open_chroma_collection("fixed768", "fake")
