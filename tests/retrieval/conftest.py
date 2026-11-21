"""Fixtures de recuperación: corpus de juguete (~40 chunks) con JSONL y Chroma temporales.

El corpus (``toy_corpus.jsonl``) es sintético pero cubre los casos que importan para la
lógica de recuperación: documentos públicos y restringidos por rol, versiones 17.0/18.0/any,
fragmentos obsoletos, casi-duplicados (para MMR), identificadores exactos
(``stock.picking``, ``RB-N2-003``, ``ERR-INV-204``) y un chunk con bandera sospechosa.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chromadb
import pytest

from docuerp.config import PathsSettings, RetrievalSettings, Settings
from docuerp.embeddings import get_embedder
from docuerp.retrieval import store
from docuerp.retrieval.contracts import RetrievalQuery
from docuerp.schemas import Chunk, ScoredChunk, UserRole, make_chunk_id
from docuerp.text import bm25_tokenize

ALL_ROLES: list[UserRole] = ["portal_user", "n1", "n2", "admin"]
N1_UP: list[UserRole] = ["n1", "n2", "admin"]
N2_UP: list[UserRole] = ["n2", "admin"]
ADMIN_ONLY: list[UserRole] = ["admin"]

CHUNKER = "structure"
EMBEDDER = "fake"


def build_chunk(
    doc_id: str,
    page: int,
    idx: int,
    text: str,
    *,
    section: str = "General",
    erp: str = "18.0",
    status: str = "vigente",
    roles: list[UserRole] | None = None,
    module: str = "inventario",
    doc_type: str = "odoo_doc",
    page_end: int | None = None,
    flags: list[str] | None = None,
) -> Chunk:
    """Construye un ``Chunk`` válido con valores por defecto razonables."""
    return Chunk(
        chunk_id=make_chunk_id(doc_id, page, idx),
        doc_id=doc_id,
        text=text,
        source=f"{doc_id.split('@')[0]}.pdf",
        page=page,
        page_end=page_end or page,
        version="18.0@abc1234",
        erp_version=erp,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        section_path=section,
        module=module,  # type: ignore[arg-type]
        doc_type=doc_type,  # type: ignore[arg-type]
        language="es",
        allowed_roles=roles or ALL_ROLES,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        ingested_at="2026-11-17T08:00:00Z",
        token_count=len(text.split()),
        flags=flags or [],
    )


def toy_corpus() -> list[Chunk]:
    """Carga el corpus de juguete (41 chunks) desde ``toy_corpus.jsonl``."""
    path = Path(__file__).with_name("toy_corpus.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Chunk.model_validate_json(line) for line in lines if line.strip()]


def chroma_metadata(chunk: Chunk) -> dict[str, Any]:
    """Metadatos planos como los guarda la ingesta: listas como ``"|a|b|"`` y una clave extra."""
    meta = chunk.model_dump(exclude={"text"})
    for name in ("allowed_roles", "flags"):
        meta[name] = "|" + "|".join(meta[name]) + "|" if meta[name] else ""
    meta["clave_extra_de_ingesta"] = "se-descarta"
    return meta


def write_corpus(
    chunks: list[Chunk], processed: Path, index: Path, embedder: str = EMBEDDER
) -> None:
    """Escribe el JSONL y una colección Chroma (coseno) en ``index``, codificada con ``embedder``
    (``fake`` por defecto)."""
    processed.mkdir(parents=True, exist_ok=True)
    with (processed / f"chunks.{CHUNKER}.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(chunk.model_dump_json() + "\n")
    chroma_dir = index / CHUNKER / embedder / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.create_collection("chunks", configuration={"hnsw": {"space": "cosine"}})
    vectors = get_embedder(embedder).embed_documents([c.text for c in chunks])
    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=[v.tolist() for v in vectors],
        documents=[c.text for c in chunks],
        metadatas=[chroma_metadata(c) for c in chunks],
    )


def make_settings(root: Path, **retrieval: Any) -> Settings:
    """``Settings`` que apunta a ``root`` (processed/ e index/) con embedder ``fake``."""
    return Settings(
        paths=PathsSettings(processed=str(root / "processed"), index=str(root / "index")),
        retrieval=RetrievalSettings(**{"embedder": EMBEDDER, **retrieval}),
    )


@pytest.fixture
def corpus_chunks() -> list[Chunk]:
    """Los chunks del corpus de juguete."""
    return toy_corpus()


@pytest.fixture
def toy_env(tmp_path: Path, corpus_chunks: list[Chunk]) -> Settings:
    """Corpus de juguete escrito en disco (JSONL + Chroma); devuelve los ``Settings``.

    Se vacían las cachés globales antes y después para aislar cada prueba.
    """
    store.clear_caches()
    write_corpus(corpus_chunks, tmp_path / "processed", tmp_path / "index")
    yield make_settings(tmp_path)  # type: ignore[misc]
    store.clear_caches()


@pytest.fixture
def make_query() -> Callable[..., RetrievalQuery]:
    """Fábrica de ``RetrievalQuery`` con valores por defecto (rol n2, versión 18.0)."""

    def _make(
        text: str, role: UserRole = "n2", erp_version: str = "18.0", **kwargs: Any
    ) -> RetrievalQuery:
        return RetrievalQuery(text=text, role=role, erp_version=erp_version, **kwargs)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def make_hit() -> Callable[..., ScoredChunk]:
    """Fábrica de ``ScoredChunk`` mínimos a partir de un ``chunk_id`` corto."""

    def _make(
        name: str,
        rank: int = 1,
        score: float = 1.0,
        retriever: str = "bm25",
        text: str | None = None,
        stage_scores: dict[str, float] | None = None,
    ) -> ScoredChunk:
        chunk = build_chunk(f"doc-{name}", 1, 0, text or f"texto de {name}")
        return ScoredChunk(
            chunk=chunk,
            score=score,
            rank=rank,
            retriever=retriever,  # type: ignore[arg-type]
            stage_scores=stage_scores or {},
        )

    return _make


def overlap_scorer(query: str, texts: list[str]) -> list[float]:
    """Reranker falso: logit = 8 · (fracción de términos de la consulta presentes) − 4.

    Determinista y sin modelos: el texto que contiene todos los términos obtiene +4 (≈0.98
    tras la sigmoide) y uno sin ninguno −4 (≈0.02).
    """
    terms = set(bm25_tokenize(query))
    out = []
    for text in texts:
        present = terms & set(bm25_tokenize(text))
        out.append(8.0 * (len(present) / len(terms) if terms else 0.0) - 4.0)
    return out


@pytest.fixture
def fake_scorer() -> Callable[[str, list[str]], list[float]]:
    """Scorer falso (``overlap_scorer``) para inyectar en ``Reranker``."""
    return overlap_scorer


@pytest.fixture
def make_chunk() -> Callable[..., Chunk]:
    """Fábrica de ``Chunk`` (``build_chunk``) para pruebas unitarias."""
    return build_chunk


@pytest.fixture
def corpus_writer() -> Callable[..., None]:
    """Escritor de corpus (``write_corpus``) para pruebas que modifican el JSONL."""
    return write_corpus


@pytest.fixture
def settings_factory() -> Callable[..., Settings]:
    """Fábrica de ``Settings`` (``make_settings``) apuntando a un directorio temporal."""
    return make_settings
