"""``classify_ticket``: TF-IDF+LR y embeddings+LR, métricas, persistencia y umbral."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docuerp.agent.tools.classify import (
    ClassifierNotReadyError,
    ClassifyTicketInput,
    TicketClassifier,
    TicketExample,
    classify_ticket,
    load_examples,
    train_classifier,
)


def test_tfidf_predicts_expected_category(tfidf_classifier: TicketClassifier) -> None:
    pred = tfidf_classifier.predict("el usuario no tiene permisos de acceso")
    assert pred["category"] == "acceso_permisos"
    assert pred["module"] == "accesos"
    assert 0.0 <= pred["category_confidence"] <= 1.0
    assert len(pred["category_alternatives"]) == 2
    assert pred["kind"] == "tfidf"


def test_train_and_save_reports_test_metrics(
    tfidf_classifier: TicketClassifier, tmp_path: Path
) -> None:
    metrics = tfidf_classifier.train_and_save()
    assert metrics["n_train"] == 12
    assert metrics["n_test"] == 3
    for target in ("category", "priority", "module"):
        assert 0.0 <= metrics[f"{target}_accuracy"] <= 1.0
        assert 0.0 <= metrics[f"{target}_f1_macro"] <= 1.0
    assert metrics["category_accuracy"] == pytest.approx(1.0)
    assert tfidf_classifier.model_path.is_file()
    saved = json.loads((tmp_path / "models" / "tfidf.metrics.json").read_text())
    assert saved == metrics


def test_persisted_model_is_loaded_without_data(
    tfidf_classifier: TicketClassifier, tmp_path: Path
) -> None:
    tfidf_classifier.train_and_save()
    fresh = TicketClassifier("tfidf", model_dir=tmp_path / "models", data_dir=tmp_path / "vacio")
    pred = fresh.predict("rendimiento lento del sistema")
    assert pred["category"] == "rendimiento"
    assert fresh.metrics["n_train"] == 12


def test_embeddings_kind_with_fake_embedder(classifier_data: Path, tmp_path: Path) -> None:
    clf = TicketClassifier(
        "embeddings", embedder_name="fake", model_dir=tmp_path / "emb", data_dir=classifier_data
    )
    metrics = clf.train_and_save()
    assert metrics["n_test"] == 3
    loaded = TicketClassifier("embeddings", model_dir=tmp_path / "emb").load()
    assert loaded.embedder_name == "fake"
    assert loaded.predict("rendimiento lento")["kind"] == "embeddings"


def test_low_confidence_needs_review(classifier_data: Path, tmp_path: Path) -> None:
    strict = TicketClassifier(
        "tfidf", model_dir=tmp_path / "m", data_dir=classifier_data, min_confidence=0.99
    )
    assert strict.predict("texto ambiguo sin relacion")["needs_human_review"] is True


async def test_tool_marks_review_and_ok(tfidf_classifier: TicketClassifier) -> None:
    result = await classify_ticket(
        ClassifyTicketInput(text="error al validar la recepcion"), classifier=tfidf_classifier
    )
    assert result.ok
    assert result.data["category"] == "error_funcional"
    assert result.needs_human_review == result.data["needs_human_review"]


async def test_tool_degrades_without_model_or_data(tmp_path: Path) -> None:
    clf = TicketClassifier("tfidf", model_dir=tmp_path / "m", data_dir=tmp_path / "nada")
    with pytest.raises(ClassifierNotReadyError):
        clf.ensure_ready()
    result = await classify_ticket(ClassifyTicketInput(text="lo que sea"), classifier=clf)
    assert not result.ok
    assert result.data["unavailable"] is True


def test_input_contract_limits() -> None:
    with pytest.raises(ValueError):
        ClassifyTicketInput(text="x" * 2001)
    with pytest.raises(ValueError):
        ClassifyTicketInput(text="ab")


def test_load_examples_reports_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"text": "ok", "category": "migracion", "module": "general", "priority": "baja"}\n'
        '{"text": "x", "category": "inventada", "module": "general", "priority": "baja"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"bad.jsonl:2"):
        load_examples(path)


def test_fit_validations(tmp_path: Path) -> None:
    clf = TicketClassifier("tfidf", model_dir=tmp_path)
    with pytest.raises(ValueError, match="al menos 2 ejemplos"):
        clf.fit([])
    same = [
        TicketExample(text=f"texto {i}", category="migracion", module="general", priority="baja")
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="2 clases"):
        clf.fit(same)
    with pytest.raises(ClassifierNotReadyError):
        clf.evaluate(same)
    with pytest.raises(ValueError):
        TicketClassifier("bayes")  # type: ignore[arg-type]


def test_load_rejects_other_kind(tfidf_classifier: TicketClassifier, tmp_path: Path) -> None:
    path = tfidf_classifier.train_and_save() and tfidf_classifier.model_path
    with pytest.raises(ValueError, match="no 'embeddings'"):
        TicketClassifier("embeddings", model_dir=tmp_path).load(path)


def test_train_classifier_function(classifier_data: Path, tmp_path: Path) -> None:
    metrics = train_classifier(
        "tfidf", data_dir=classifier_data, model_dir=tmp_path / "out", embedder_name="fake"
    )
    assert (tmp_path / "out" / "tfidf.joblib").is_file()
    assert metrics["category_f1_macro"] >= 0.5
    with pytest.raises(ValueError):
        train_classifier("otro")


def test_hybrid_picks_representation_per_target(classifier_data: Path, tmp_path: Path) -> None:
    from docuerp.agent.tools.classify import HybridTicketClassifier

    tfidf = TicketClassifier("tfidf", model_dir=tmp_path, data_dir=classifier_data)
    emb = TicketClassifier(
        "embeddings", embedder_name="fake", model_dir=tmp_path, data_dir=classifier_data
    )
    hybrid = HybridTicketClassifier(tfidf, emb, min_confidence=0.0)
    hybrid.ensure_ready()
    # Se fuerza la elección con métricas de CV conocidas: prioridad → TF-IDF.
    tfidf.metrics.update({"priority_cv_accuracy_train": 0.9, "category_cv_accuracy_train": 0.5})
    emb.metrics.update({"priority_cv_accuracy_train": 0.6, "category_cv_accuracy_train": 0.8})
    assert hybrid.choice() == {
        "category": "embeddings",
        "priority": "tfidf",
        "module": "embeddings",
    }
    out = hybrid.predict("No podemos facturar: error al validar la factura, urgente")
    assert out["kind"] == "hybrid"
    assert out["per_target"]["priority"] == "tfidf"
    assert {"category", "priority", "module", "needs_human_review"} <= set(out)
