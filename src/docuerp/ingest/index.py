"""Indexación: Chroma persistente (HNSW coseno) + BM25 (``rank_bm25``).

Rutas: ``data/index/{chunker}/{embedder}/chroma`` y ``data/index/{chunker}/bm25.pkl``;
chunks en ``data/processed/chunks.{chunker}.jsonl``. El mismo ``chunk_id`` enlaza ambos.

Por contrato los metadatos son *planos* (str, int, float, bool). Las listas del ``Chunk``
(``allowed_roles``, ``flags``) se guardan como cadenas delimitadas ``"|n1|n2|admin|"``
(SPEC §8), que se reconstruyen sin ambigüedad y permiten comprobar un rol con
``"|n1|" in metadata["allowed_roles"]`` (el delimitador inicial y final evita que ``n1``
coincida dentro de otro valor). Ojo: en Chroma 1.x el operador ``$contains`` de ``where``
no busca subcadenas en metadatos de texto, así que el filtro por rol se hace en Python
sobre los candidatos; los campos escalares (``erp_version``, ``status``, ``module``) sí se
pueden filtrar con ``where``. :func:`chunk_from_record` devuelve un ``Chunk`` validado.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

from docuerp import paths
from docuerp.schemas import Chunk

# Tokenización BM25 compartida con recuperación (fuente única en ``docuerp.text``).
from docuerp.text import bm25_tokenize

log = logging.getLogger(__name__)

COLLECTION_NAME = "chunks"
"""Nombre de la colección Chroma (uno por directorio ``(chunker, embedder)``)."""

LIST_FIELDS: tuple[str, ...] = ("allowed_roles", "flags")
"""Campos lista del ``Chunk`` que se aplanan como ``"|a|b|"`` en los metadatos."""

LIST_DELIMITER = "|"

BM25_FORMAT_VERSION = 1
"""Versión del formato del pickle BM25 (se valida al cargar)."""


def chroma_dir(chunker: str, embedder: str) -> Path:
    """Directorio del índice Chroma para ``(chunker, embedder)``."""
    return paths.INDEX / chunker / embedder / "chroma"


def bm25_path(chunker: str) -> Path:
    """Ruta del pickle BM25 para ``chunker``."""
    return paths.INDEX / chunker / "bm25.pkl"


def chunks_path(chunker: str) -> Path:
    """Ruta del JSONL de chunks procesados para ``chunker``."""
    return paths.PROCESSED / f"chunks.{chunker}.jsonl"


# ---------------------------------------------------------------------------
# JSONL de chunks
# ---------------------------------------------------------------------------


def write_chunks(chunks: list[Chunk], chunker: str, *, path: Path | None = None) -> Path:
    """Escribe los chunks en JSONL (una línea por chunk), de forma atómica.

    Se escribe primero un archivo temporal y luego se renombra: si el proceso se
    interrumpe, el JSONL anterior queda intacto.
    """
    target = path or chunks_path(chunker)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(chunk.model_dump_json() + "\n")
    tmp.replace(target)
    return target


def load_chunks(chunker: str, *, path: Path | None = None) -> list[Chunk]:
    """Lee los chunks procesados de ``chunker``.

    Raises:
        FileNotFoundError: si aún no se ejecutó la ingesta para ese chunker.
        ValueError: si una línea no es un ``Chunk`` válido (indica el número de línea).
    """
    source = path or chunks_path(chunker)
    if not source.is_file():
        raise FileNotFoundError(
            f"No existe {source}. Ejecute `docuerp ingest run --chunker {chunker}` primero."
        )
    chunks: list[Chunk] = []
    with source.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                chunks.append(Chunk.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{source.name}:{lineno}: chunk inválido ({exc})") from exc
    return chunks


# ---------------------------------------------------------------------------
# Metadatos planos ⇄ Chunk
# ---------------------------------------------------------------------------


def flatten_list(values: list[str]) -> str:
    """``["n1", "n2"]`` → ``"|n1|n2|"``; lista vacía → ``""``."""
    if not values:
        return ""
    return LIST_DELIMITER + LIST_DELIMITER.join(values) + LIST_DELIMITER


def unflatten_list(value: str) -> list[str]:
    """``"|n1|n2|"`` → ``["n1", "n2"]`` (inverso de :func:`flatten_list`)."""
    return [part for part in value.split(LIST_DELIMITER) if part]


def chunk_to_metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
    """Metadatos planos del chunk para Chroma (todo menos ``text``, que va en ``documents``)."""
    data = chunk.model_dump(exclude={"text"})
    flat: dict[str, str | int | float | bool] = {}
    for key, value in data.items():
        if key in LIST_FIELDS:
            flat[key] = flatten_list(list(value))
        elif isinstance(value, str | int | float | bool):
            flat[key] = value
        else:
            flat[key] = json.dumps(value, ensure_ascii=False)
    return flat


def chunk_from_record(document: str, metadata: dict[str, Any]) -> Chunk:
    """Reconstruye y valida un ``Chunk`` desde el texto y los metadatos de Chroma."""
    data: dict[str, Any] = dict(metadata)
    for key in LIST_FIELDS:
        raw = data.get(key, "")
        data[key] = unflatten_list(raw) if isinstance(raw, str) else list(raw or [])
    data["text"] = document
    return Chunk.model_validate(data)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


def build_bm25_index(chunks: list[Chunk], chunker: str, *, path: Path | None = None) -> Path:
    """Construye y persiste el índice BM25 (``BM25Okapi`` sobre ``bm25_tokenize``).

    El pickle guarda ``{"format", "chunk_ids", "bm25", "n_chunks"}``; ``chunk_ids[i]`` es el
    chunk de la fila ``i`` de la matriz de BM25.

    Raises:
        ValueError: si no hay chunks.
    """
    from rank_bm25 import BM25Okapi

    if not chunks:
        raise ValueError("No hay chunks para indexar en BM25")
    corpus = [bm25_tokenize(chunk.text) or ["_vacio_"] for chunk in chunks]
    payload = {
        "format": BM25_FORMAT_VERSION,
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
        "bm25": BM25Okapi(corpus),
        "n_chunks": len(chunks),
    }
    target = path or bm25_path(chunker)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as fh:
        pickle.dump(payload, fh)
    return target


def load_bm25_index(chunker: str, *, path: Path | None = None) -> tuple[Any, list[str]]:
    """Carga ``(BM25Okapi, chunk_ids)`` alineados por posición.

    Seguridad: ``pickle`` ejecuta código al cargar; solo se leen pickles generados
    localmente por este mismo pipeline en ``data/index`` (nunca archivos externos).

    Raises:
        FileNotFoundError: si el índice no existe.
        ValueError: si el formato no es el esperado.
    """
    source = path or bm25_path(chunker)
    if not source.is_file():
        raise FileNotFoundError(f"No existe el índice BM25 {source}")
    with source.open("rb") as fh:
        payload = pickle.load(fh)  # archivo generado localmente por este pipeline
    # Formato de ingesta ({"format": 1, ...}) o el de recuperación ({"version", ...}).
    known = isinstance(payload, dict) and "bm25" in payload and "chunk_ids" in payload
    if known and (payload.get("format") == BM25_FORMAT_VERSION or "corpus_sha256" in payload):
        return payload["bm25"], list(payload["chunk_ids"])
    raise ValueError(f"Formato de índice BM25 desconocido en {source}")


# ---------------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------------


def _client(directory: Path) -> Any:
    """Cliente Chroma persistente sin telemetría."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    directory.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(directory), settings=ChromaSettings(anonymized_telemetry=False)
    )


def build_chroma_index(
    chunks: list[Chunk],
    chunker: str,
    embedder: str,
    *,
    directory: Path | None = None,
    batch_size: int = 16,
) -> Path:
    """Calcula embeddings y persiste la colección Chroma (metadatos = campos del chunk).

    La colección se **recrea** en cada corrida (índice reproducible a partir del JSONL) y
    usa distancia coseno; los vectores ya vienen L2-normalizados desde ``docuerp.embeddings``.

    Raises:
        ValueError: si no hay chunks o hay ``chunk_id`` repetidos.
    """
    from docuerp.embeddings import get_embedder

    if not chunks:
        raise ValueError("No hay chunks para indexar en Chroma")
    ids = [chunk.chunk_id for chunk in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Hay chunk_id repetidos: el índice quedaría inconsistente")
    target = directory or chroma_dir(chunker, embedder)
    client = _client(target)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:  # la colección aún no existe (NotFoundError según la versión)
        log.debug("La colección %s no existía en %s", COLLECTION_NAME, target)
    collection = client.create_collection(
        COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"chunker": chunker, "embedder": embedder},
        embedding_function=None,
    )
    model = get_embedder(embedder)  # ValueError si el nombre no es un embedder conocido
    step = max(1, min(int(client.get_max_batch_size()), 512))
    for start in range(0, len(chunks), step):
        batch = chunks[start : start + step]
        vectors = model.embed_documents([c.text for c in batch], batch_size=batch_size)
        collection.add(
            ids=[c.chunk_id for c in batch],
            embeddings=vectors.tolist(),
            documents=[c.text for c in batch],
            metadatas=[chunk_to_metadata(c) for c in batch],
        )
    log.info("Chroma %s/%s: %d chunks indexados", chunker, embedder, collection.count())
    return target


def open_chroma_collection(chunker: str, embedder: str, *, directory: Path | None = None) -> Any:
    """Abre la colección Chroma existente (solo lectura lógica).

    Raises:
        FileNotFoundError: si el índice no se ha construido.
    """
    target = directory or chroma_dir(chunker, embedder)
    if not target.is_dir():
        raise FileNotFoundError(
            f"No existe el índice Chroma {target}. Ejecute `docuerp ingest run "
            f"--chunker {chunker} --embedder {embedder}`."
        )
    return _client(target).get_collection(COLLECTION_NAME)
