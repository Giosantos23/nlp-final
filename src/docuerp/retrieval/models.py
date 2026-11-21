"""Carga perezosa y caché (singleton) de embedders y reranker; selección de dispositivo.

Los embeddings viven en ``docuerp.embeddings`` (compartido con ingesta y router); aquí solo
se reexportan con la interfaz que usa la recuperación. El reranker lo implementa la rama
de recuperación.

El reranker es un *cross-encoder*: a diferencia del embedder (que codifica consulta y
documento por separado), recibe el par ``(consulta, fragmento)`` completo y emite un único
logit de relevancia. Es más preciso pero más lento, por eso solo se aplica a los ~20
candidatos que sobreviven a la fusión.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any

import numpy as np

from docuerp.embeddings import Embedder, pick_device
from docuerp.embeddings import get_embedder as _get_embedder

log = logging.getLogger(__name__)

RERANKER_MAX_LENGTH = 512
"""Tokens máximos por par (consulta + fragmento) que ve el cross-encoder."""


def resolve_device(preference: str = "auto") -> str:
    """Devuelve ``mps`` si está disponible (Apple Silicon), luego ``cuda``, luego ``cpu``."""
    return pick_device() if preference == "auto" else preference


def get_embedder(name: str = "bge-m3") -> Embedder:
    """Devuelve el modelo de embeddings ``name`` (bge-m3 | e5-base | fake), cacheado."""
    return _get_embedder(name)


def embed_texts(
    texts: list[str], *, embedder: str = "bge-m3", is_query: bool = False
) -> np.ndarray:
    """Embeddings normalizados L2 (``np.ndarray`` de forma ``(n, d)``). Aplica los prefijos
    ``query:``/``passage:`` cuando el modelo es e5."""
    model = get_embedder(embedder)
    if is_query:
        return np.stack([model.embed_query(t) for t in texts])
    return model.embed_documents(texts)


class CrossEncoderScorer:
    """Envoltura del cross-encoder ``BAAI/bge-reranker-v2-m3`` (sentence-transformers).

    ``score`` devuelve **logits** crudos (no probabilidades); la normalización a [0, 1]
    con sigmoide la hace :mod:`docuerp.retrieval.rerank`.
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        *,
        use_fp16: bool = False,
        max_length: int = RERANKER_MAX_LENGTH,
        batch_size: int = 16,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        # sentence-transformers no es seguro para llamadas concurrentes (sobre todo en MPS).
        self._lock = threading.Lock()
        half = use_fp16 and device != "cpu"
        self.fp16 = half
        self._model = self._load(half)
        if half and not self._fp16_is_stable():
            log.warning("fp16 inestable en %s; se recarga el reranker en fp32", device)
            self.fp16 = False
            self._model = self._load(False)

    def _load(self, half: bool) -> Any:
        """Carga el modelo (en fp16 si ``half``) y lo deja en modo de inferencia."""
        import torch
        from sentence_transformers import CrossEncoder

        log.info("Cargando reranker %s en %s (fp16=%s)", self.model_id, self.device, half)
        model_kwargs = {"torch_dtype": torch.float16} if half else {}
        kwargs: dict[str, Any] = {
            "device": self.device,
            "max_length": self.max_length,
            "model_kwargs": model_kwargs,
            "activation_fn": torch.nn.Identity(),  # logits crudos; la sigmoide es nuestra
        }
        try:  # primero la caché local (sin red)
            return CrossEncoder(self.model_id, local_files_only=True, **kwargs)
        except OSError:
            log.info("%s no está en caché local; se descarga", self.model_id)
            return CrossEncoder(self.model_id, **kwargs)

    def _fp16_is_stable(self) -> bool:
        """Prueba de humo: en fp16 los logits deben ser finitos (MPS a veces da NaN)."""
        probe = self.score("consulta de prueba", ["fragmento de prueba", "otro fragmento"])
        return all(math.isfinite(v) for v in probe)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Logit de relevancia de cada ``texts[i]`` frente a ``query`` (mayor = más relevante)."""
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        with self._lock:
            logits = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return [float(v) for v in np.asarray(logits, dtype=np.float32).reshape(-1)]


_RERANKER_LOCK = threading.Lock()
_RERANKERS: dict[tuple[str, str, bool], CrossEncoderScorer] = {}


def get_reranker(settings: Any = None) -> CrossEncoderScorer:
    """Devuelve el cross-encoder ``BAAI/bge-reranker-v2-m3`` cacheado (singleton por
    configuración). La primera llamada tarda varios segundos; las siguientes son inmediatas.

    Args:
        settings: ``docuerp.config.Settings`` (por defecto, la configuración global).
            Determina el modelo (``models.reranker``), el dispositivo y fp16.
    """
    if settings is None:
        from docuerp.config import get_settings

        settings = get_settings()
    device = resolve_device(settings.retrieval.device)
    key = (settings.models.reranker, device, bool(settings.models.use_fp16))
    with _RERANKER_LOCK:
        if key not in _RERANKERS:
            _RERANKERS[key] = CrossEncoderScorer(
                key[0],
                device,
                use_fp16=key[2],
                batch_size=settings.retrieval.batch_size,
            )
        return _RERANKERS[key]
