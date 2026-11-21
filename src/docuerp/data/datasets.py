"""Datasets de entrenamiento: intenciones del router y tickets etiquetados del clasificador.

Formatos (SPEC §10):

- ``data/router/intents.jsonl``: ``{"text", "intent", "split", "hard"}``.
- ``data/classifier/tickets_{train,test}.jsonl``: ``{"text", "category", "module",
  "priority", "ticket_id"}``.

Ambos se dividen 80/20 de forma **estratificada** (la proporción de cada clase se conserva en
train y en test) y de forma determinista con la semilla indicada.
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections import defaultdict
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any

from docuerp import paths
from docuerp.data.router_examples import EXAMPLES_BY_INTENT

TEST_FRACTION = 0.20
"""Fracción de cada estrato que va a ``test``."""


def stratified_split[T](
    items: Sequence[T],
    strata: Sequence[Hashable],
    *,
    seed: int,
    test_fraction: float = TEST_FRACTION,
) -> tuple[list[T], list[T]]:
    """Divide ``items`` en (train, test) conservando la proporción de cada estrato.

    Dentro de cada estrato se baraja con una semilla fija y se envía a ``test`` el
    ``round(len * test_fraction)`` de los elementos (al menos uno si el estrato tiene dos o
    más elementos). El orden de los estratos es el de su primera aparición, de modo que el
    resultado es reproducible.

    Raises:
        ValueError: si ``items`` y ``strata`` no tienen la misma longitud.
    """
    if len(items) != len(strata):
        raise ValueError("items y strata deben tener la misma longitud")
    groups: dict[Hashable, list[T]] = defaultdict(list)
    for item, stratum in zip(items, strata, strict=True):
        groups[stratum].append(item)
    rng = random.Random(seed)
    train: list[T] = []
    test: list[T] = []
    for stratum in groups:
        members = groups[stratum]
        rng.shuffle(members)
        n_test = round(len(members) * test_fraction)
        if len(members) >= 2:
            n_test = max(1, n_test)
        test.extend(members[:n_test])
        train.extend(members[n_test:])
    return train, test


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Escribe una fila JSON por línea, en UTF-8 y sin escapar tildes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Lee un archivo JSONL a una lista de diccionarios."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def router_rows(seed: int = 42) -> list[dict[str, Any]]:
    """Construye las filas del dataset del router con su split (train/test).

    El estrato es ``(intención, difícil)``: así tanto las clases como los negativos difíciles
    quedan repartidos 80/20.

    Raises:
        ValueError: si hay textos duplicados (contaminarían train y test).
    """
    flat: list[tuple[str, str, bool]] = [
        (text, intent, hard)
        for intent, examples in EXAMPLES_BY_INTENT.items()
        for text, hard in examples
    ]
    texts = [text.strip().lower() for text, _, _ in flat]
    if len(set(texts)) != len(texts):
        duplicated = sorted({t for t in texts if texts.count(t) > 1})
        raise ValueError(f"Ejemplos de router duplicados: {duplicated}")
    train, test = stratified_split(flat, [(intent, hard) for _, intent, hard in flat], seed=seed)
    rows: list[dict[str, Any]] = []
    for split, members in (("train", train), ("test", test)):
        for text, intent, hard in members:
            rows.append({"text": text, "intent": intent, "split": split, "hard": hard})
    return rows


def build_router_dataset(out_path: Path | None = None, *, seed: int = 42) -> Path:
    """Genera ``data/router/intents.jsonl`` (texto, intención, split, difícil).

    Returns:
        Ruta del archivo escrito.
    """
    target = out_path or (paths.ROUTER_DATA / "intents.jsonl")
    _write_jsonl(target, router_rows(seed))
    return target


def classifier_rows(db_path: Path | None = None) -> list[dict[str, Any]]:
    """Exporta los tickets de la BD como ejemplos del clasificador.

    El texto es ``resumen + salto de línea + descripción``: es lo que el clasificador verá
    en producción (``classify_ticket(text)``).

    Raises:
        FileNotFoundError: si la base de tickets no existe (ejecute ``docuerp data seed-tickets``).
    """
    source = db_path or paths.TICKETS_DB
    if not source.is_file():
        raise FileNotFoundError(
            f"No existe {source}. Cree la base con: uv run docuerp data seed-tickets"
        )
    connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        cursor = connection.execute(
            "SELECT ticket_id, summary, description, category, module, priority "
            "FROM tickets ORDER BY ticket_id"
        )
        return [
            {
                "text": f"{summary}\n{description}".strip(),
                "category": category,
                "module": module,
                "priority": priority,
                "ticket_id": ticket_id,
            }
            for ticket_id, summary, description, category, module, priority in cursor
        ]
    finally:
        connection.close()


def build_classifier_dataset(
    out_path: Path | None = None, *, seed: int = 42, db_path: Path | None = None
) -> Path:
    """Genera ``data/classifier/tickets_train.jsonl`` y ``tickets_test.jsonl`` desde la BD.

    Args:
        out_path: ruta base opcional (``.../tickets.jsonl``); se derivan ``<base>_train.jsonl``
            y ``<base>_test.jsonl``. Por defecto ``data/classifier/tickets.jsonl``.
        seed: semilla del split estratificado por categoría.
        db_path: base de tickets (por defecto ``data/tickets/tickets.db``).

    Returns:
        El directorio que contiene los dos archivos.
    """
    base = out_path or (paths.CLASSIFIER_DATA / "tickets.jsonl")
    rows = classifier_rows(db_path)
    train, test = stratified_split(rows, [row["category"] for row in rows], seed=seed)
    # Se ordena por ticket_id para que el archivo sea estable y fácil de revisar con diff.
    for split, members in (("train", train), ("test", test)):
        members.sort(key=lambda row: row["ticket_id"])
        _write_jsonl(base.with_name(f"{base.stem}_{split}{base.suffix}"), members)
    return base.parent
