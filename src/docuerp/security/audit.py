"""Auditoría sanitizada en ``logs/audit.jsonl`` (ids, intención, filtros, candidatos/scores,
herramienta, latencias; consulta enmascarada). Nunca PII ni secretos.

Cada consulta produce **una línea JSON** con:

* ``trace_id``, ``timestamp`` (UTC ISO-8601), ``role``, ``user_hash`` (HMAC-SHA256 del
  ``user_id`` truncado: permite correlacionar sin exponer la identidad), ``session_hash``.
* ``intent``, ``filters`` (rol, versión y los que reporte la recuperación),
  ``candidates`` (``chunk_id`` + ``score`` por etapa, máx. 20 por etapa),
  ``tool_used``, ``guards`` (decisión de cada guard), ``latencies_ms`` por etapa,
  ``usage`` (tokens, costo).
* ``query_masked`` y ``answer_masked``: con PII enmascarada y recortadas.

Sanitización (defensa en profundidad, se aplica a TODO el registro, no solo a la consulta):
claves con nombres de secreto (``token``, ``secret``, ``password``, ``api_key``…) se
reemplazan por ``[REDACTED]``; todo string pasa por ``mask_pii`` y se elimina el canario.

Rotación simple por tamaño: si el archivo supera ``max_bytes`` se renombra a
``audit.jsonl.1`` (y ``.1`` → ``.2``…) conservando ``backups`` copias.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from docuerp import paths
from docuerp.schemas import CANARY_TOKEN, QueryRequest, QueryResponse
from docuerp.security.pii import mask_pii

log = logging.getLogger(__name__)

REDACTED = "[REDACTED]"
SECRET_KEY_MARKERS: tuple[str, ...] = (
    "token",
    "secret",
    "password",
    "passwd",
    "contrasena",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "hmac",
    "credential",
    "private_key",
)
"""Fragmentos de nombre de clave cuyo valor nunca se registra."""

SAFE_KEYS: frozenset[str] = frozenset(
    {"input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens", "token_count"}
)
"""Claves que contienen "token" pero son métricas numéricas, no secretos."""

MAX_STRING_CHARS = 2000
MAX_CANDIDATES_PER_STAGE = 20
RETRIEVAL_STAGES: tuple[str, ...] = (
    "retrieval.bm25",
    "retrieval.dense",
    "retrieval.fusion",
    "rerank",
    "mmr",
    "evidence_gate",
)
GUARD_STAGES: tuple[str, ...] = ("input_guard", "permission_check", "output_guard")
_DEV_PEPPER = b"docuerp-audit-dev-pepper"
"""Clave de respaldo para el hash de usuario cuando no hay ``DOCUERP_HMAC_SECRET``."""


def _secret_key(name: str, value: Any = None) -> bool:
    """``True`` si el nombre de la clave sugiere un secreto (y no es una métrica).

    Un conteo numérico cuyo nombre termina en ``tokens`` (p. ej. ``total_tokens``) es una
    métrica de uso del LLM, no un secreto.
    """
    lowered = name.lower()
    if lowered in SAFE_KEYS:
        return False
    is_count = isinstance(value, int | float) and not isinstance(value, bool)
    if is_count and lowered.endswith("tokens"):
        return False
    return any(marker in lowered for marker in SECRET_KEY_MARKERS)


def sanitize(value: Any) -> Any:
    """Sanitiza recursivamente: redacta secretos, enmascara PII, quita el canario y recorta."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            str(k): (REDACTED if _secret_key(str(k), v) and v is not None else sanitize(v))
            for k, v in value.items()
        }
    if isinstance(value, list | tuple | set):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        masked, _ = mask_pii(value.replace(CANARY_TOKEN, "[CANARY]"))
        if len(masked) > MAX_STRING_CHARS:
            return masked[:MAX_STRING_CHARS] + "…[recortado]"
        return masked
    if value is None or isinstance(value, bool | int | float):
        return value
    return sanitize(str(value))


def _pepper() -> bytes:
    """Clave del HMAC de identidades: ``DOCUERP_HMAC_SECRET`` o una de desarrollo."""
    secret = os.environ.get("DOCUERP_HMAC_SECRET")
    return secret.encode("utf-8") if secret else _DEV_PEPPER


def hash_identifier(value: str | None) -> str | None:
    """HMAC-SHA256 truncado a 16 hex del identificador (``None`` si no hay valor)."""
    if not value:
        return None
    return hmac.new(_pepper(), value.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def _candidates(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Extrae ``chunk_id`` + ``score`` de los eventos de recuperación.

    Convención con el agente: ``TraceEvent.data["candidates"]`` (o ``"hits"``) es una lista
    de dicts con al menos ``chunk_id`` y ``score``.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        stage = str(event.get("stage", ""))
        data = event.get("data") or {}
        items = data.get("candidates") or data.get("hits")
        if stage not in RETRIEVAL_STAGES or not isinstance(items, list):
            continue
        out[stage] = [
            {"chunk_id": item.get("chunk_id"), "score": item.get("score")}
            for item in items[:MAX_CANDIDATES_PER_STAGE]
            if isinstance(item, dict)
        ]
    return out


def build_record(
    trace: QueryResponse | dict[str, Any],
    *,
    request: QueryRequest | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construye el registro de auditoría (aún sin sanitizar) a partir de la traza."""
    data = trace.model_dump(mode="json") if isinstance(trace, BaseModel) else dict(trace)
    events: list[dict[str, Any]] = [e for e in data.get("events", []) if isinstance(e, dict)]
    response = data.get("response") or {}
    req = (
        request.model_dump(mode="json") if request is not None else dict(data.get("request") or {})
    )

    filters: dict[str, Any] = {"role": req.get("role"), "erp_version": req.get("erp_version")}
    guards: dict[str, Any] = {}
    latencies: dict[str, float] = {}
    for event in events:
        stage = str(event.get("stage", ""))
        latencies[stage] = round(float(event.get("duration_ms", 0.0)), 3)
        event_data = event.get("data") or {}
        if isinstance(event_data.get("filters"), dict):
            filters.update(event_data["filters"])
        if stage in GUARD_STAGES or stage.startswith("tool."):
            guards[stage] = {
                "status": event.get("status"),
                **{
                    k: event_data[k]
                    for k in ("allowed", "risk_score", "flags", "errors", "decision", "reason")
                    if k in event_data
                },
            }
    latencies["total"] = round(float(data.get("latency_ms", 0.0)), 3)

    record: dict[str, Any] = {
        "trace_id": data.get("trace_id"),
        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "role": req.get("role"),
        "user_hash": hash_identifier(req.get("user_id")),
        "session_hash": hash_identifier(req.get("session_id")),
        "intent": data.get("intent"),
        "filters": filters,
        "candidates": _candidates(events),
        "tool_used": response.get("tool_used"),
        "guards": guards,
        "blocked": data.get("blocked", False),
        "block_reason": data.get("block_reason"),
        "abstained": response.get("abstained"),
        "needs_human_review": response.get("needs_human_review"),
        "sources": [s.get("chunk_id") for s in response.get("sources", []) if isinstance(s, dict)],
        "pending_action": bool(data.get("pending_action")),
        "latencies_ms": latencies,
        "usage": data.get("usage") or {},
        "query_masked": req.get("query"),
        "answer_masked": response.get("answer"),
    }
    if extra:
        record["extra"] = extra
    return record


class AuditLogger:
    """Escritor JSONL de auditoría (seguro entre hilos)."""

    def __init__(
        self, path: Path | None = None, *, max_bytes: int = 5_000_000, backups: int = 3
    ) -> None:
        if max_bytes <= 0 or backups < 0:
            raise ValueError("max_bytes debe ser > 0 y backups ≥ 0")
        self.path = path or paths.AUDIT_LOG
        self.max_bytes = max_bytes
        self.backups = backups
        self._lock = threading.Lock()

    def _rotate_if_needed(self, incoming: int) -> None:
        """Rota el archivo si al agregar ``incoming`` bytes superaría ``max_bytes``."""
        if not self.path.exists() or self.path.stat().st_size + incoming <= self.max_bytes:
            return
        if self.backups == 0:
            self.path.unlink()
            return
        oldest = self.path.with_name(f"{self.path.name}.{self.backups}")
        if oldest.exists():
            oldest.unlink()
        for index in range(self.backups - 1, 0, -1):
            source = self.path.with_name(f"{self.path.name}.{index}")
            if source.exists():
                source.rename(self.path.with_name(f"{self.path.name}.{index + 1}"))
        self.path.rename(self.path.with_name(f"{self.path.name}.1"))

    def log(
        self,
        trace: Any,
        *,
        request: QueryRequest | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Serializa y sanitiza una traza (``QueryResponse`` o dict) y la agrega al log.

        Returns:
            El registro sanitizado tal como se escribió.

        Raises:
            TypeError: si ``trace`` no es un ``QueryResponse`` ni un dict.
            OSError: si no se puede escribir el archivo (no se silencia: perder auditoría
                debe notarse).
        """
        if not isinstance(trace, QueryResponse | dict):
            raise TypeError("trace debe ser QueryResponse o dict")
        record = sanitize(build_record(trace, request=request, extra=extra))
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        encoded = line.encode("utf-8")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed(len(encoded))
            with self.path.open("ab") as fh:
                fh.write(encoded)
        return record

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Itera los registros del archivo actual (omite líneas corruptas con aviso)."""
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    log.warning("Línea %d corrupta en %s; se omite", number, self.path)

    def tail(self, limit: int = 50) -> list[dict[str, Any]]:
        """Últimos ``limit`` registros (más reciente al final)."""
        records = list(self.iter_records())
        return records[-limit:] if limit > 0 else []

    def find(self, trace_id: str) -> dict[str, Any] | None:
        """Busca el registro de ``trace_id`` en el archivo actual."""
        for record in self.iter_records():
            if record.get("trace_id") == trace_id:
                return record
        return None


__all__ = ["REDACTED", "AuditLogger", "build_record", "hash_identifier", "sanitize"]
