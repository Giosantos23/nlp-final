"""Acceso a los artefactos de ingesta: corpus JSONL, índice BM25 y colección Chroma.

Todo se carga **de forma perezosa** y se cachea en memoria (singletons protegidos con
``threading.Lock``), porque la API atiende consultas concurrentes y reconstruir un índice
por consulta sería inaceptable.

Rutas (SPEC §5.1/§8), relativas a ``settings.paths``:

* ``processed/chunks.{chunker}.jsonl``: un ``Chunk`` por línea (lo escribe la ingesta).
* ``index/{chunker}/bm25.pkl``: caché del índice BM25 (lo escribe este módulo).
* ``index/{chunker}/{embedder}/chroma``: colección ``chunks`` de Chroma (la escribe la ingesta).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docuerp.schemas import Chunk
from docuerp.text import bm25_tokenize

log = logging.getLogger(__name__)

COLLECTION_NAME = "chunks"
"""Nombre de la colección de Chroma que crea la ingesta."""

BM25_CACHE_VERSION = 1
"""Versión del formato de ``bm25.pkl``; cualquier otro formato se reconstruye."""

_LIST_FIELDS = ("allowed_roles", "flags")
"""Campos de ``Chunk`` que Chroma guarda como texto ``"|a|b|"`` (no admite listas)."""


class IndexNotFoundError(FileNotFoundError):
    """Falta un artefacto de ingesta (JSONL o Chroma). El mensaje indica cómo generarlo."""


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------


def resolve_settings(settings: Any) -> Any:
    """Devuelve ``settings`` o la configuración global si es ``None``."""
    if settings is not None:
        return settings
    from docuerp.config import get_settings

    return get_settings()


def chunks_path(chunker: str, settings: Any = None) -> Path:
    """Ruta de ``chunks.{chunker}.jsonl``."""
    cfg = resolve_settings(settings)
    return cfg.paths.resolve("processed") / f"chunks.{chunker}.jsonl"


def bm25_cache_path(chunker: str, settings: Any = None) -> Path:
    """Ruta de la caché del índice BM25 (``index/{chunker}/bm25.pkl``)."""
    cfg = resolve_settings(settings)
    return cfg.paths.resolve("index") / chunker / "bm25.pkl"


def chroma_path(chunker: str, embedder: str, settings: Any = None) -> Path:
    """Ruta del directorio persistente de Chroma (``index/{chunker}/{embedder}/chroma``)."""
    cfg = resolve_settings(settings)
    return cfg.paths.resolve("index") / chunker / embedder / "chroma"


# ---------------------------------------------------------------------------
# Corpus JSONL
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Corpus:
    """Chunks del JSONL en orden de archivo, con índice por ``chunk_id`` y hash del archivo."""

    chunks: tuple[Chunk, ...]
    by_id: dict[str, Chunk]
    sha256: str


_CORPUS_LOCK = threading.Lock()
_CORPORA: dict[Path, tuple[tuple[int, int], Corpus]] = {}


def _stat_signature(path: Path) -> tuple[int, int]:
    """``(mtime_ns, tamaño)``: detecta rápido que el archivo cambió sin releerlo."""
    st = path.stat()
    return st.st_mtime_ns, st.st_size


def _parse_corpus(path: Path) -> Corpus:
    """Lee y valida el JSONL completo; calcula el sha256 de sus bytes."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    chunks: list[Chunk] = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            chunks.append(Chunk.model_validate(json.loads(line)))
        except ValueError as exc:  # JSONDecodeError y ValidationError heredan de ValueError
            raise ValueError(f"{path.name} línea {number}: chunk inválido ({exc})") from exc
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(by_id) != len(chunks):
        raise ValueError(f"{path.name}: hay chunk_id duplicados")
    return Corpus(chunks=tuple(chunks), by_id=by_id, sha256=digest)


def load_corpus(path: Path) -> Corpus:
    """Carga (con caché en memoria) el corpus JSONL. Se recarga si el archivo cambió.

    Raises:
        IndexNotFoundError: si el archivo no existe.
        ValueError: si alguna línea no es un ``Chunk`` válido o hay ids duplicados.
    """
    if not path.is_file():
        raise IndexNotFoundError(
            f"No existe {path}. Ejecuta primero: docuerp ingest run --chunker <chunker>"
        )
    signature = _stat_signature(path)
    with _CORPUS_LOCK:
        cached = _CORPORA.get(path)
        if cached is not None and cached[0] == signature:
            return cached[1]
        corpus = _parse_corpus(path)
        _CORPORA[path] = (signature, corpus)
        return corpus


# ---------------------------------------------------------------------------
# Índice BM25
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BM25Index:
    """Índice BM25Okapi y el ``chunk_id`` de cada documento (misma posición que el corpus)."""

    bm25: Any
    chunk_ids: tuple[str, ...]
    corpus_sha256: str


_BM25_LOCK = threading.Lock()
_BM25_INDEXES: dict[Path, tuple[tuple[int, int], BM25Index]] = {}


def _build_bm25(corpus: Corpus) -> BM25Index:
    """Tokeniza cada chunk con ``bm25_tokenize`` y construye ``BM25Okapi`` en memoria."""
    from rank_bm25 import BM25Okapi

    if not corpus.chunks:
        raise ValueError("El corpus está vacío: no se puede construir el índice BM25")
    tokenized = [bm25_tokenize(chunk.text) for chunk in corpus.chunks]
    return BM25Index(
        bm25=BM25Okapi(tokenized),
        chunk_ids=tuple(chunk.chunk_id for chunk in corpus.chunks),
        corpus_sha256=corpus.sha256,
    )


def _read_bm25_cache(cache: Path, corpus: Corpus) -> BM25Index | None:
    """Lee ``bm25.pkl``; devuelve ``None`` si falta, está corrupto, es de otro formato o su
    hash no coincide con el del JSONL actual (caché obsoleta)."""
    if not cache.is_file():
        return None
    try:
        with cache.open("rb") as fh:
            payload = pickle.load(fh)  # archivo local generado por este mismo módulo
        if (
            payload["version"] != BM25_CACHE_VERSION
            or payload["corpus_sha256"] != corpus.sha256
            or tuple(payload["chunk_ids"]) != tuple(c.chunk_id for c in corpus.chunks)
        ):
            log.info("Caché BM25 obsoleta (%s): se reconstruye", cache)
            return None
        return BM25Index(
            bm25=payload["bm25"],
            chunk_ids=tuple(payload["chunk_ids"]),
            corpus_sha256=payload["corpus_sha256"],
        )
    except Exception as exc:  # caché corrupta o de otro formato: se reconstruye sin fallar
        log.warning("No se pudo leer la caché BM25 %s (%s); se reconstruye", cache, exc)
        return None


def _write_bm25_cache(cache: Path, index: BM25Index) -> None:
    """Escribe ``bm25.pkl`` de forma atómica (archivo temporal + ``os.replace``).

    Un fallo de escritura (disco de solo lectura, permisos) no es fatal: el índice ya está en
    memoria y solo se pierde la caché en disco.
    """
    payload = {
        "version": BM25_CACHE_VERSION,
        "corpus_sha256": index.corpus_sha256,
        "chunk_ids": list(index.chunk_ids),
        "bm25": index.bm25,
    }
    tmp = cache.with_name(f"{cache.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache)
    except OSError as exc:
        log.warning("No se pudo escribir la caché BM25 %s: %s", cache, exc)
        tmp.unlink(missing_ok=True)


def load_bm25(chunker: str, settings: Any = None) -> tuple[Corpus, BM25Index]:
    """Devuelve ``(corpus, índice BM25)`` del chunker, usando (en orden) la caché en memoria,
    la caché en disco ``bm25.pkl`` (si su hash coincide con el JSONL) o una reconstrucción.

    Raises:
        IndexNotFoundError: si no existe ``chunks.{chunker}.jsonl``.
    """
    source = chunks_path(chunker, settings)
    corpus = load_corpus(source)
    signature = _stat_signature(source)
    with _BM25_LOCK:
        cached = _BM25_INDEXES.get(source)
        if (
            cached is not None
            and cached[0] == signature
            and cached[1].corpus_sha256 == corpus.sha256
        ):
            return corpus, cached[1]
        cache = bm25_cache_path(chunker, settings)
        index = _read_bm25_cache(cache, corpus)
        if index is None:
            log.info("Construyendo índice BM25 de %s (%d chunks)", chunker, len(corpus.chunks))
            index = _build_bm25(corpus)
            _write_bm25_cache(cache, index)
        _BM25_INDEXES[source] = (signature, index)
        return corpus, index


# ---------------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------------

_CHROMA_LOCK = threading.Lock()
_CHROMA_COLLECTIONS: dict[Path, Any] = {}


def get_collection(chunker: str, embedder: str, settings: Any = None) -> Any:
    """Devuelve (cacheada) la colección ``chunks`` de Chroma para ``(chunker, embedder)``.

    Raises:
        IndexNotFoundError: si no existe el directorio de Chroma o la colección.
    """
    path = chroma_path(chunker, embedder, settings)
    with _CHROMA_LOCK:
        if path in _CHROMA_COLLECTIONS:
            return _CHROMA_COLLECTIONS[path]
        if not path.is_dir():
            raise IndexNotFoundError(
                f"No existe el índice denso {path}. "
                f"Ejecuta primero: docuerp ingest run --chunker {chunker} --embedder {embedder}"
            )
        import chromadb

        client = chromadb.PersistentClient(path=str(path))
        try:
            collection = client.get_collection(COLLECTION_NAME)
        except Exception as exc:  # chromadb lanza NotFoundError/ValueError según la versión
            raise IndexNotFoundError(
                f"{path} no contiene la colección '{COLLECTION_NAME}': {exc}"
            ) from exc
        _CHROMA_COLLECTIONS[path] = collection
        return collection


def _split_pipe_list(value: Any) -> list[str]:
    """Convierte ``"|n1|n2|"`` en ``["n1", "n2"]`` (acepta también una lista ya hecha)."""
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if not value:
        return []
    return [part for part in str(value).split("|") if part]


def chunk_from_metadata(chunk_id: str, document: str, metadata: dict[str, Any]) -> Chunk:
    """Reconstruye un ``Chunk`` desde un registro de Chroma.

    ``document`` es el texto; ``metadata`` son los campos planos del chunk, con las listas
    (``allowed_roles``, ``flags``) como ``"|a|b|"``. Se **descartan las claves que no son
    campos de ``Chunk``** (p. ej. metadatos auxiliares que añada la ingesta).

    Raises:
        ValueError: si los metadatos no producen un ``Chunk`` válido.
    """
    fields = {key: metadata[key] for key in Chunk.model_fields if key in metadata}
    for name in _LIST_FIELDS:
        if name in fields:
            fields[name] = _split_pipe_list(fields[name])
    fields["chunk_id"] = chunk_id
    fields["text"] = document
    try:
        return Chunk.model_validate(fields)
    except ValueError as exc:
        raise ValueError(f"Metadatos de Chroma inválidos para {chunk_id}: {exc}") from exc


def clear_caches() -> None:
    """Vacía las cachés en memoria (corpus, BM25 y Chroma). Útil en pruebas y tras reingestar."""
    with _CORPUS_LOCK:
        _CORPORA.clear()
    with _BM25_LOCK:
        _BM25_INDEXES.clear()
    with _CHROMA_LOCK:
        _CHROMA_COLLECTIONS.clear()
