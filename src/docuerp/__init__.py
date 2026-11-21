"""DocuERP: asistente RAG agéntico para soporte funcional N1/N2 de Odoo.

Responde en español con citas verificables (source, page, chunk_id), filtra por rol y
versión del ERP, se abstiene sin evidencia y usa herramientas acotadas para tickets con
confirmación humana. Ver ``docs/SPEC.md``.
"""

from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"

__all__ = ["__version__"]

_REQUIRED_MODELS = ("BAAI--bge-m3", "BAAI--bge-reranker-v2-m3")


def _prefer_offline_models() -> None:
    """Si los modelos ya están en la caché de Hugging Face, trabaja sin red.

    Evita consultas al Hub en cada carga y que ``transformers`` descargue en segundo plano
    una conversión a safetensors de ~2 GB. ``HF_HUB_OFFLINE=0`` en el entorno lo desactiva.
    """
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = Path(os.environ.get("HF_HUB_CACHE", home / "hub"))
    if all((hub / f"models--{name}").is_dir() for name in _REQUIRED_MODELS):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


_prefer_offline_models()
