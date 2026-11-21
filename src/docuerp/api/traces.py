"""Almacenamiento de trazas: caché LRU en memoria y lectura de la bitácora de auditoría."""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docuerp.schemas import QueryResponse

logger = logging.getLogger("docuerp.api")


@dataclass(frozen=True)
class StoredTrace:
    """Respuesta guardada junto con el usuario que la originó."""

    user_id: str
    response: QueryResponse


class TraceStore:
    """Caché LRU de las últimas ``capacity`` respuestas, indexadas por ``trace_id``.

    Al leer o escribir una traza se marca como la más reciente; al exceder la capacidad se
    descarta la menos usada. Es seguro entre hilos.
    """

    def __init__(self, capacity: int = 500) -> None:
        if capacity < 1:
            raise ValueError("capacity debe ser >= 1")
        self.capacity = capacity
        self._items: OrderedDict[str, StoredTrace] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, response: QueryResponse, user_id: str) -> None:
        """Guarda (o reemplaza) la respuesta bajo su ``trace_id``."""
        with self._lock:
            self._items[response.trace_id] = StoredTrace(user_id=user_id, response=response)
            self._items.move_to_end(response.trace_id)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

    def get(self, trace_id: str) -> StoredTrace | None:
        """Devuelve la traza guardada o ``None``; la marca como recién usada."""
        with self._lock:
            item = self._items.get(trace_id)
            if item is not None:
                self._items.move_to_end(trace_id)
            return item

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def find_in_audit_log(path: Path, trace_id: str) -> dict[str, Any] | None:
    """Busca ``trace_id`` en ``audit.jsonl`` y devuelve el último registro que coincide.

    El archivo ya está sanitizado por ``security.audit`` (sin PII). Se filtra primero por
    subcadena para no parsear cada línea; las líneas corruptas se ignoran.
    """
    if not path.is_file():
        return None
    found: dict[str, Any] | None = None
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if trace_id not in line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and record.get("trace_id") == trace_id:
                    found = record
    except OSError as exc:
        logger.warning("no se pudo leer la bitácora de auditoría: %s", type(exc).__name__)
        return None
    return found
