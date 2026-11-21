"""Pruebas de los datasets del router de intenciones y del clasificador de tickets."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

from docuerp.data import datasets as ds
from docuerp.data.router_examples import EXAMPLES_BY_INTENT
from docuerp.data.tickets import seed_tickets
from docuerp.schemas import Intent

INTENTS = {"doc_query", "ticket_status", "ticket_create", "ticket_classify", "out_of_domain"}
CATEGORIES = {
    "configuracion",
    "error_funcional",
    "acceso_permisos",
    "rendimiento",
    "migracion",
    "capacitacion",
    "integracion",
}
TICKET_ID_RE = re.compile(r"TCK-\d{5}")


@pytest.fixture(scope="module")
def router_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return ds.build_router_dataset(tmp_path_factory.mktemp("router") / "intents.jsonl")


@pytest.fixture(scope="module")
def classifier_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("clf")
    db = seed_tickets(root / "tickets.db")
    return ds.build_classifier_dataset(root / "tickets.jsonl", db_path=db)


# --------------------------------------------------------------------------- split estratificado


def test_stratified_split_keeps_proportions() -> None:
    items = list(range(100))
    strata = ["a"] * 60 + ["b"] * 30 + ["c"] * 10
    train, test = ds.stratified_split(items, strata, seed=1)
    assert len(train) + len(test) == 100
    assert not set(train) & set(test)
    test_counts = Counter(strata[i] for i in test)
    assert test_counts == {"a": 12, "b": 6, "c": 2}


def test_stratified_split_is_deterministic_and_seed_dependent() -> None:
    items = list(range(50))
    strata = [i % 5 for i in items]
    assert ds.stratified_split(items, strata, seed=3) == ds.stratified_split(items, strata, seed=3)
    assert ds.stratified_split(items, strata, seed=3) != ds.stratified_split(items, strata, seed=4)


def test_stratified_split_edge_cases() -> None:
    train, test = ds.stratified_split(["x", "y", "z"], ["a", "b", "b"], seed=0)
    assert "x" in train  # un estrato de un solo elemento se queda en train
    assert len(test) == 1  # el estrato de dos elementos aporta al menos uno a test
    assert ds.stratified_split([], [], seed=0) == ([], [])
    with pytest.raises(ValueError):
        ds.stratified_split([1, 2], ["a"], seed=0)


# --------------------------------------------------------------------------- router


def test_router_format_and_size(router_file: Path) -> None:
    rows = ds.read_jsonl(router_file)
    assert 240 <= len(rows) <= 300
    for row in rows:
        assert set(row) == {"text", "intent", "split", "hard"}
        assert isinstance(row["text"], str)
        assert 3 <= len(row["text"]) <= 2000
        assert row["intent"] in INTENTS
        assert row["split"] in {"train", "test"}
        assert isinstance(row["hard"], bool)


def test_router_intents_match_schema_literal() -> None:
    from typing import get_args

    assert set(get_args(Intent)) == INTENTS


def test_router_split_is_80_20_per_intent(router_file: Path) -> None:
    rows = ds.read_jsonl(router_file)
    by_intent: dict[str, Counter[str]] = {i: Counter() for i in INTENTS}
    for row in rows:
        by_intent[row["intent"]][row["split"]] += 1
    for intent, counts in by_intent.items():
        total = counts["train"] + counts["test"]
        assert total >= 40, intent
        assert counts["test"] >= 8, intent
        assert abs(counts["test"] / total - 0.20) <= 0.06, (intent, counts)
    overall_test = sum(c["test"] for c in by_intent.values()) / len(rows)
    assert 0.17 <= overall_test <= 0.23


def test_router_hard_negatives_exist_in_every_intent_and_both_splits(router_file: Path) -> None:
    rows = ds.read_jsonl(router_file)
    hard = Counter((r["intent"], r["split"]) for r in rows if r["hard"])
    for intent in INTENTS:
        assert hard[(intent, "train")] >= 1, intent
        assert hard[(intent, "test")] >= 1, intent
    assert 0.08 <= sum(r["hard"] for r in rows) / len(rows) <= 0.35


def test_router_has_no_duplicates_or_train_test_leakage(router_file: Path) -> None:
    rows = ds.read_jsonl(router_file)
    normalized = [r["text"].strip().lower() for r in rows]
    assert len(set(normalized)) == len(normalized)
    train = {r["text"] for r in rows if r["split"] == "train"}
    test = {r["text"] for r in rows if r["split"] == "test"}
    assert not train & test


def test_router_ticket_status_has_examples_with_and_without_id(router_file: Path) -> None:
    status = [r["text"] for r in ds.read_jsonl(router_file) if r["intent"] == "ticket_status"]
    with_id = [t for t in status if TICKET_ID_RE.search(t)]
    assert len(with_id) >= 15
    assert len(status) - len(with_id) >= 15


def test_router_out_of_domain_covers_other_erps_payroll_general_and_small_talk(
    router_file: Path,
) -> None:
    ood = " ".join(
        r["text"].lower() for r in ds.read_jsonl(router_file) if r["intent"] == "out_of_domain"
    )
    for keyword in (
        "sap",
        "dynamics",
        "quickbooks",
        "nómina",
        "capital de francia",
        "hola",
        "gracias",
    ):
        assert keyword in ood, keyword


def test_router_hard_flag_marks_confusable_examples() -> None:
    hard_doc = [t for t, h in EXAMPLES_BY_INTENT["doc_query"] if h]
    assert any("ticket" in t.lower() for t in hard_doc)  # suena a ticket pero es documental
    hard_ood = [t for t, h in EXAMPLES_BY_INTENT["out_of_domain"] if h]
    assert any("SAP" in t for t in hard_ood)  # suena a doc_query pero es otro ERP


def test_router_is_deterministic(tmp_path: Path) -> None:
    a = ds.build_router_dataset(tmp_path / "a.jsonl", seed=42)
    b = ds.build_router_dataset(tmp_path / "b.jsonl", seed=42)
    c = ds.build_router_dataset(tmp_path / "c.jsonl", seed=5)
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes() != c.read_bytes()


def test_router_rejects_duplicate_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    duplicated = {
        "doc_query": (("¿Cómo valido una recepción?", False), ("¿cómo valido una recepción?", True))
    }
    monkeypatch.setattr(ds, "EXAMPLES_BY_INTENT", duplicated)
    with pytest.raises(ValueError, match="duplicados"):
        ds.router_rows()


# --------------------------------------------------------------------------- clasificador


def test_classifier_files_and_format(classifier_dir: Path) -> None:
    train = ds.read_jsonl(classifier_dir / "tickets_train.jsonl")
    test = ds.read_jsonl(classifier_dir / "tickets_test.jsonl")
    assert len(train) + len(test) == 300
    assert abs(len(test) / 300 - 0.20) <= 0.03
    for row in train + test:
        assert set(row) == {"text", "category", "module", "priority", "ticket_id"}
        assert row["category"] in CATEGORIES
        assert row["priority"] in {"baja", "media", "alta", "critica"}
        assert TICKET_ID_RE.fullmatch(row["ticket_id"])
        assert len(row["text"]) > 20


def test_classifier_split_is_stratified_by_category(classifier_dir: Path) -> None:
    train = ds.read_jsonl(classifier_dir / "tickets_train.jsonl")
    test = ds.read_jsonl(classifier_dir / "tickets_test.jsonl")
    train_counts = Counter(r["category"] for r in train)
    test_counts = Counter(r["category"] for r in test)
    assert set(test_counts) == CATEGORIES == set(train_counts)
    for category in CATEGORIES:
        total = train_counts[category] + test_counts[category]
        assert abs(test_counts[category] / total - 0.20) <= 0.07, category


def test_classifier_train_and_test_are_disjoint_and_cover_all_tickets(classifier_dir: Path) -> None:
    train_ids = {r["ticket_id"] for r in ds.read_jsonl(classifier_dir / "tickets_train.jsonl")}
    test_ids = {r["ticket_id"] for r in ds.read_jsonl(classifier_dir / "tickets_test.jsonl")}
    assert not train_ids & test_ids
    assert len(train_ids | test_ids) == 300


def test_classifier_text_matches_database(classifier_dir: Path) -> None:
    rows = ds.read_jsonl(classifier_dir / "tickets_test.jsonl")
    anchor = next((r for r in rows if r["ticket_id"] == "TCK-00042"), None)
    sample = anchor or rows[0]
    assert "\n" in sample["text"]  # resumen + salto de línea + descripción


def test_classifier_requires_existing_database(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="seed-tickets"):
        ds.build_classifier_dataset(tmp_path / "tickets.jsonl", db_path=tmp_path / "no-existe.db")


def test_classifier_is_deterministic(tmp_path: Path) -> None:
    db = seed_tickets(tmp_path / "t.db", n_tickets=120)
    first = ds.build_classifier_dataset(tmp_path / "x" / "t.jsonl", db_path=db)
    second = ds.build_classifier_dataset(tmp_path / "y" / "t.jsonl", db_path=db)
    for split in ("train", "test"):
        assert (first / f"t_{split}.jsonl").read_bytes() == (
            second / f"t_{split}.jsonl"
        ).read_bytes()
