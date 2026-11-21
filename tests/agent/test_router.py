"""Router híbrido: reglas, clasificador sobre embeddings, fuera de dominio y persistencia."""

from __future__ import annotations

from pathlib import Path

import pytest

from docuerp.agent.router import IntentRouter, RouteDecision, rule_intent, train_router


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("¿Cuál es el estado del TCK-00042?", "ticket_status"),
        ("estado de tck-00042", "ticket_status"),
        ("¿En qué estado está mi ticket?", "ticket_status"),
        ("Crea un ticket: no puedo validar la recepción", "ticket_create"),
        ("Ábreme un caso porque falla la factura", "ticket_create"),
        ("Registra una incidencia para ACC-0001", "ticket_create"),
        ("Levanta un nuevo ticket por favor", "ticket_create"),
        ("Necesito un nuevo ticket urgente", "ticket_create"),
        ("Crea un ticket parecido al TCK-00042", "ticket_create"),  # creación gana
        ("Clasifica este problema: el sistema está lento", "ticket_classify"),
        ("Categoriza el TCK-00010", "ticket_classify"),
    ],
)
def test_rules(text: str, intent: str) -> None:
    rule = rule_intent(text)
    assert rule is not None
    assert rule[0] == intent


@pytest.mark.parametrize(
    "text",
    [
        "¿Cómo creo un ticket en el módulo Helpdesk de Odoo?",
        "Pasos para registrar una incidencia en Odoo 18",
        "¿Cómo valido una recepción parcial?",
    ],
)
def test_howto_questions_are_not_commands(text: str) -> None:
    assert rule_intent(text) is None


@pytest.fixture
def router(intents_path: Path, tmp_path: Path) -> IntentRouter:
    return IntentRouter(
        embedder_name="fake",
        data_path=intents_path,
        model_path=tmp_path / "router" / "intent_lr.joblib",
        ood_threshold=0.35,
        min_confidence=0.2,
    )


def test_rule_decision(router: IntentRouter) -> None:
    decision = router.decide("estado del TCK-00042")
    assert decision == RouteDecision(
        intent="ticket_status",
        confidence=0.95,
        rationale="regla: identificador TCK-xxxxx",
        method="rules",
    )
    assert router.route("estado del TCK-00042")[0] == "ticket_status"


def test_classifier_trains_lazily_and_persists(router: IntentRouter) -> None:
    assert not router.model_path.exists()
    decision = router.decide("como valido una recepcion parcial en inventario")
    assert decision.method == "classifier"
    assert decision.intent == "doc_query"
    assert router.model_path.is_file()


def test_ood_by_similarity(router: IntentRouter) -> None:
    decision = router.decide("zzz qqq www")
    assert decision.intent == "out_of_domain"
    assert decision.method == "ood_similarity"
    assert "umbral" in decision.rationale


def test_low_confidence_goes_to_gated_doc_branch(intents_path: Path, tmp_path: Path) -> None:
    strict = IntentRouter(
        embedder_name="fake",
        data_path=intents_path,
        model_path=tmp_path / "m.joblib",
        ood_threshold=0.0,
        min_confidence=0.99,
    )
    decision = strict.decide("como valido una recepcion parcial en inventario")
    # Sin intención clara nunca se dispara una acción: rama documental con gate de evidencia.
    assert decision.intent == "doc_query"
    assert decision.method == "low_confidence"
    assert "gate" in decision.rationale


def test_procedure_question_is_not_a_status_lookup() -> None:
    from docuerp.agent.router import rule_intent

    assert rule_intent("¿Cuál es el procedimiento para consultar el estado de un ticket?") is None
    assert rule_intent("¿Cuál es el estado del ticket TCK-00042?")[0] == "ticket_status"
    assert rule_intent("quiero saber el estado de mi ticket")[0] == "ticket_status"


def test_fallback_without_model_or_data(tmp_path: Path) -> None:
    lonely = IntentRouter(
        embedder_name="fake", data_path=tmp_path / "no.jsonl", model_path=tmp_path / "no.joblib"
    )
    decision = lonely.decide("algo cualquiera sobre odoo")
    assert decision.intent == "doc_query"
    assert decision.method == "fallback"


def test_corrupt_data_falls_back(tmp_path: Path) -> None:
    bad = tmp_path / "intents.jsonl"
    bad.write_text("{no es json}\n", encoding="utf-8")
    broken = IntentRouter(embedder_name="fake", data_path=bad, model_path=tmp_path / "x.joblib")
    assert broken.decide("como configuro inventario").method == "fallback"


def test_train_router_metrics_and_reload(intents_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "persist" / "intent_lr.joblib"
    metrics = train_router(out, data_path=intents_path, embedder_name="fake")
    assert metrics["n_train"] == 15
    assert metrics["n_test"] == 2
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert 0.0 <= metrics["hard_accuracy"] <= 1.0  # hay un caso ``hard`` en test
    reloaded = IntentRouter(model_path=out, data_path=tmp_path / "no.jsonl").load()
    assert reloaded.embedder_name == "fake"
    assert reloaded.decide("como valido una recepcion parcial en inventario").method in {
        "classifier",
        "low_confidence",
    }


def test_train_router_requires_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        train_router(tmp_path / "x.joblib", data_path=tmp_path / "missing.jsonl")


def test_fit_needs_two_intents(tmp_path: Path) -> None:
    from docuerp.agent.router import IntentExample

    router = IntentRouter(embedder_name="fake", model_path=tmp_path / "m.joblib")
    with pytest.raises(ValueError, match="2 intenciones"):
        router.fit([IntentExample(text="hola", intent="doc_query")])
    with pytest.raises(RuntimeError):
        router.save()


class _StubModel:
    """Clasificador falso que siempre predice ``ticket_create`` con alta confianza."""

    classes_ = ("doc_query", "ticket_create")

    def predict_proba(self, _x: object) -> list[list[float]]:
        return [[0.1, 0.9]]


@pytest.mark.parametrize(
    ("text", "intent", "method"),
    [
        ("No me deja validar, ¿qué hago?", "doc_query", "action_guard"),
        (
            "nos llegaron cajas golpeadas, ¿cómo las saco del inventario?",
            "doc_query",
            "action_guard",
        ),
        ("el cliente no puede facturar, ábrele algo urgente", "ticket_create", "classifier"),
        ("quiero reportar que el portal no carga", "ticket_create", "classifier"),
    ],
)
def test_classifier_needs_explicit_cue_to_create(
    router: IntentRouter, text: str, intent: str, method: str
) -> None:
    import numpy as np

    router._model = _StubModel()
    router._references = np.ones((1, 4), dtype=float)
    router._ensure_model = lambda: True  # type: ignore[method-assign]
    router._embed = lambda texts: np.ones((len(texts), 4), dtype=float)  # type: ignore[method-assign]
    decision = router.decide(text)
    assert (decision.intent, decision.method) == (intent, method)
