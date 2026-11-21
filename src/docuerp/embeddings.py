"""Modelos de embeddings compartidos por ingesta, recuperación y router.

Representación vectorial del curso: cada fragmento y cada consulta se convierten en un
vector denso L2-normalizado, de modo que la similitud coseno es un producto punto.

- ``bge-m3`` (BAAI/bge-m3, 1024 dim, multilingüe): modelo principal. Elegido porque las
  consultas son en español y gran parte del corpus está en inglés.
- ``e5-base`` (intfloat/multilingual-e5-base, 768 dim): modelo de ablación. Requiere los
  prefijos "query: " y "passage: " según su ficha técnica.
- ``fake``: vectores deterministas derivados de un hash, para pruebas unitarias rápidas
  sin descargar modelos.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Protocol

import numpy as np

log = logging.getLogger(__name__)

EmbedderName = Literal["bge-m3", "e5-base", "fake"]


@dataclass(frozen=True)
class EmbedderSpec:
    """Ficha de un modelo de embeddings: id en Hugging Face, dimensión y prefijos."""

    hf_id: str
    dim: int
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_length: int = 1024


SPECS: dict[str, EmbedderSpec] = {
    "bge-m3": EmbedderSpec(hf_id="BAAI/bge-m3", dim=1024, max_seq_length=1024),
    "e5-base": EmbedderSpec(
        hf_id="intfloat/multilingual-e5-base",
        dim=768,
        query_prefix="query: ",
        passage_prefix="passage: ",
        max_seq_length=512,
    ),
    "fake": EmbedderSpec(hf_id="fake", dim=64),
}


class Embedder(Protocol):
    """Interfaz mínima que usan el resto de módulos."""

    name: str
    dim: int

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def pick_device() -> str:
    """Usa la GPU de Apple (MPS) si existe; si no, CPU."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class SentenceTransformerEmbedder:
    """Envoltura de sentence-transformers con prefijos y normalización L2."""

    def __init__(self, name: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.spec = SPECS[name]
        self.dim = self.spec.dim
        self.device = device or pick_device()
        log.info("Cargando embeddings %s en %s", self.spec.hf_id, self.device)
        try:  # primero la caché local: evita descargas (p. ej. safetensors de refs/pr)
            self._model = SentenceTransformer(
                self.spec.hf_id, device=self.device, local_files_only=True
            )
        except OSError:
            log.info("%s no está en caché local; se descarga", self.spec.hf_id)
            self._model = SentenceTransformer(self.spec.hf_id, device=self.device)
        self._model.max_seq_length = self.spec.max_seq_length
        # sentence-transformers no es seguro para llamadas concurrentes sobre MPS.
        self._lock = threading.Lock()

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        with self._lock:
            vectors = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 256,
            )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        prefixed = [self.spec.passage_prefix + t for t in texts]
        return self._encode(prefixed, batch_size)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.spec.query_prefix + text], batch_size=1)[0]


class FakeEmbedder:
    """Embeddings deterministas basados en hash de palabras (bolsa de palabras hasheada).

    No captura semántica real, pero textos que comparten palabras quedan cerca, lo que
    basta para probar la lógica de fusión, filtros y MMR sin modelos pesados.
    """

    def __init__(self, dim: int = 64) -> None:
        self.name = "fake"
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in text.lower().split():
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            vec[digest[0] % self.dim] += 1.0 if digest[1] % 2 == 0 else -1.0
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return vec / norm

    def embed_documents(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


@lru_cache(maxsize=4)
def get_embedder(name: EmbedderName = "bge-m3", device: str | None = None) -> Embedder:
    """Devuelve (y cachea) el embedder pedido. Cargar bge-m3 toma varios segundos."""
    if name == "fake":
        return FakeEmbedder(SPECS["fake"].dim)
    if name not in SPECS:
        raise ValueError(f"Embedder desconocido: {name!r}. Opciones: {sorted(SPECS)}")
    return SentenceTransformerEmbedder(name, device=device)
