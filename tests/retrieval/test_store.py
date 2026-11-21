"""Carga del corpus, caché BM25 y reconstrucción de ``Chunk`` desde Chroma."""

from __future__ import annotations

import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from docuerp.retrieval import store
from docuerp.retrieval.store import IndexNotFoundError, chunk_from_metadata


def test_load_corpus_reads_all_chunks_and_caches(toy_env, corpus_chunks) -> None:
    path = store.chunks_path("structure", toy_env)
    first = store.load_corpus(path)
    assert len(first.chunks) == len(corpus_chunks) == 41
    assert first.by_id[corpus_chunks[0].chunk_id] == corpus_chunks[0]
    assert store.load_corpus(path) is first  # segunda llamada: caché en memoria


def test_load_corpus_reloads_when_file_changes(
    toy_env, corpus_writer, corpus_chunks, tmp_path
) -> None:
    path = store.chunks_path("structure", toy_env)
    before = store.load_corpus(path)
    path.write_text(
        path.read_text(encoding="utf-8")
        + corpus_chunks[0]
        .model_dump_json()
        .replace(corpus_chunks[0].chunk_id, "odoo-18.0-inventario::p0099::c000")
        + "\n",
        encoding="utf-8",
    )
    after = store.load_corpus(path)
    assert len(after.chunks) == len(before.chunks) + 1
    assert after.sha256 != before.sha256


def test_load_corpus_missing_file_explains_how_to_generate(tmp_path) -> None:
    with pytest.raises(IndexNotFoundError, match="docuerp ingest run"):
        store.load_corpus(tmp_path / "no-existe.jsonl")


def test_load_corpus_rejects_invalid_and_duplicate_chunks(toy_env, corpus_chunks) -> None:
    path = store.chunks_path("structure", toy_env)
    path.write_text('{"chunk_id": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="línea 1"):
        store.load_corpus(path)
    line = corpus_chunks[0].model_dump_json()
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicados"):
        store.load_corpus(path)


def test_bm25_cache_is_written_and_reused(toy_env, monkeypatch) -> None:
    cache = store.bm25_cache_path("structure", toy_env)
    assert not cache.exists()
    _, first = store.load_bm25("structure", toy_env)
    assert cache.is_file()

    # Proceso "nuevo": sin caché en memoria, debe leer el pkl y no reconstruir.
    store.clear_caches()
    monkeypatch.setattr(store, "_build_bm25", lambda corpus: pytest.fail("no debe reconstruir"))
    _, second = store.load_bm25("structure", toy_env)
    assert second.chunk_ids == first.chunk_ids
    assert second.corpus_sha256 == first.corpus_sha256


def test_bm25_cache_is_invalidated_when_jsonl_changes(
    toy_env, corpus_chunks, make_chunk, tmp_path
) -> None:
    _, before = store.load_bm25("structure", toy_env)
    path = store.chunks_path("structure", toy_env)
    extra = make_chunk("odoo-18.0-nuevo", 1, 0, "documento nuevo sobre telepatía logística")
    path.write_text(
        path.read_text(encoding="utf-8") + extra.model_dump_json() + "\n", encoding="utf-8"
    )

    store.clear_caches()  # simula reinicio: la caché en disco existe pero quedó obsoleta
    corpus, after = store.load_bm25("structure", toy_env)
    assert after.corpus_sha256 != before.corpus_sha256
    assert extra.chunk_id in after.chunk_ids
    assert len(corpus.chunks) == len(corpus_chunks) + 1
    payload = pickle.loads(store.bm25_cache_path("structure", toy_env).read_bytes())
    assert payload["corpus_sha256"] == after.corpus_sha256  # el pkl se reescribió


@pytest.mark.parametrize(
    "content",
    [b"esto no es un pickle", pickle.dumps({"formato": "ajeno"}), pickle.dumps([1, 2, 3])],
)
def test_corrupt_or_foreign_cache_is_rebuilt(toy_env, content) -> None:
    cache = store.bm25_cache_path("structure", toy_env)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(content)
    _, index = store.load_bm25("structure", toy_env)
    assert len(index.chunk_ids) == 41
    assert pickle.loads(cache.read_bytes())["version"] == store.BM25_CACHE_VERSION


def test_unwritable_cache_dir_is_not_fatal(toy_env, monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise OSError("solo lectura")

    monkeypatch.setattr(store.os, "replace", fail)
    _, index = store.load_bm25("structure", toy_env)
    assert len(index.chunk_ids) == 41
    assert not list(store.bm25_cache_path("structure", toy_env).parent.glob("*.tmp"))


def test_concurrent_loads_build_the_index_once(toy_env, monkeypatch) -> None:
    builds: list[int] = []
    real = store._build_bm25

    def counting(corpus):
        builds.append(1)
        return real(corpus)

    monkeypatch.setattr(store, "_build_bm25", counting)
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        return store.load_bm25("structure", toy_env)[1]

    with ThreadPoolExecutor(8) as pool:
        indexes = list(pool.map(lambda _: worker(), range(8)))
    assert len(builds) == 1
    assert all(i is indexes[0] for i in indexes)


def test_empty_corpus_cannot_build_index(toy_env) -> None:
    store.chunks_path("structure", toy_env).write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="vacío"):
        store.load_bm25("structure", toy_env)


def test_chunk_from_metadata_parses_pipes_and_drops_extra_keys(corpus_chunks) -> None:
    chunk = corpus_chunks[0]
    meta = chunk.model_dump(exclude={"text"})
    meta["allowed_roles"] = "|portal_user|n1|n2|admin|"
    meta["flags"] = ""
    meta["clave_extra"] = "x"
    rebuilt = chunk_from_metadata(chunk.chunk_id, chunk.text, meta)
    assert rebuilt == chunk
    assert rebuilt.allowed_roles == ["portal_user", "n1", "n2", "admin"]
    assert rebuilt.flags == []


def test_chunk_from_metadata_keeps_flags(corpus_chunks) -> None:
    chunk = corpus_chunks[0]
    meta = chunk.model_dump(exclude={"text"})
    meta["allowed_roles"] = "|admin|"
    meta["flags"] = "|suspicious_instruction|otra|"
    rebuilt = chunk_from_metadata(chunk.chunk_id, chunk.text, meta)
    assert rebuilt.allowed_roles == ["admin"]
    assert rebuilt.flags == ["suspicious_instruction", "otra"]


def test_chunk_from_metadata_invalid_raises_with_chunk_id(corpus_chunks) -> None:
    chunk = corpus_chunks[0]
    meta = chunk.model_dump(exclude={"text"})
    meta["allowed_roles"] = ""  # un chunk sin roles no es válido
    with pytest.raises(ValueError, match=chunk.chunk_id):
        chunk_from_metadata(chunk.chunk_id, chunk.text, meta)


def test_get_collection_missing_dir_explains_how_to_generate(toy_env) -> None:
    with pytest.raises(IndexNotFoundError, match="docuerp ingest run"):
        store.get_collection("structure", "e5-base", toy_env)
