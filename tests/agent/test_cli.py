"""CLI ``docuerp agent``: ``ask`` (rich + JSON + HITL), ``train-router`` y ``tools``."""

from __future__ import annotations

import json
import types
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from docuerp.agent import service as service_mod
from docuerp.agent.cli import app
from docuerp.agent.service import AgentService

runner = CliRunner()


@pytest.fixture(autouse=True)
def installed(make_service: Callable[..., AgentService]) -> Iterator[AgentService]:
    svc = make_service()
    service_mod.set_agent_service(svc)
    yield svc
    service_mod.set_agent_service(None)


def test_ask_pretty_output() -> None:
    result = runner.invoke(
        app,
        [
            "ask",
            "¿Cómo valido una recepción parcial?",
            "--role",
            "n1",
            "--user",
            "USR-0101",
            "--version",
            "18.0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Respuesta" in result.output
    assert "Traza" in result.output
    assert "evidence_gate" in result.output
    assert '"abstained": false' in result.output  # JSON contractual §11
    assert "odoo-18.0-inventario.pdf" in result.output


def test_ask_json_output() -> None:
    result = runner.invoke(app, ["ask", "estado del TCK-00001", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["intent"] == "ticket_status"
    assert payload["response"]["tool_used"] == "get_ticket_status"


def test_ask_rejects_invalid_identity() -> None:
    result = runner.invoke(app, ["ask", "hola mundo", "--user", "juan"])
    assert result.exit_code == 2
    assert "user_id" in result.output


def test_ask_create_and_confirm(tickets_db: Path, kit: types.SimpleNamespace) -> None:
    query = "Crea un ticket: no puedo validar la recepción en inventario, cuenta ACC-0001"
    declined = runner.invoke(app, ["ask", query, "--session-id", "cli-sesion-1"], input="n\n")
    assert declined.exit_code == 0, declined.output
    assert "Borrador pendiente" in declined.output
    assert "No se registró" in declined.output
    assert kit.count_tickets(tickets_db) == 3
    accepted = runner.invoke(app, ["ask", query, "--session-id", "cli-sesion-1"], input="y\n")
    assert accepted.exit_code == 0, accepted.output
    assert "TCK-00004 registrado" in accepted.output
    assert kit.count_tickets(tickets_db) == 4


def test_ask_no_confirm_flag(tickets_db: Path, kit: types.SimpleNamespace) -> None:
    query = "Crea un ticket: no puedo validar la recepción en inventario, cuenta ACC-0001"
    result = runner.invoke(app, ["ask", query, "--no-confirm"])
    assert result.exit_code == 0
    assert "¿Confirmas" not in result.output
    assert kit.count_tickets(tickets_db) == 3


def test_train_router_command(
    intents_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from docuerp.agent import router

    monkeypatch.setattr(
        router, "_default_paths", lambda: (intents_path, tmp_path / "m" / "intent_lr.joblib")
    )
    result = runner.invoke(app, ["train-router", "--embedder", "fake"])
    assert result.exit_code == 0, result.output
    assert "accuracy" in result.output
    assert (tmp_path / "m" / "intent_lr.joblib").is_file()


def test_train_router_without_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from docuerp.agent import router

    monkeypatch.setattr(router, "_default_paths", lambda: (tmp_path / "no.jsonl", tmp_path / "m"))
    result = runner.invoke(app, ["train-router", "--embedder", "fake"])
    assert result.exit_code == 1


def test_tools_list() -> None:
    result = runner.invoke(app, ["tools", "list"])
    assert result.exit_code == 0, result.output
    for name in ("search_documents", "get_ticket_status", "classify_ticket", "create_ticket"):
        assert name in result.output


def test_tools_train_classifier(
    classifier_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from docuerp.agent.tools import classify

    monkeypatch.setattr(classify, "_default_dirs", lambda: (classifier_data, tmp_path / "models"))
    result = runner.invoke(
        app, ["tools", "train-classifier", "--kind", "both", "--embedder", "fake"]
    )
    assert result.exit_code == 0, result.output
    assert "category_f1_macro" in result.output
    assert (tmp_path / "models" / "tfidf.joblib").is_file()
    assert (tmp_path / "models" / "embeddings.joblib").is_file()
    bad = runner.invoke(app, ["tools", "train-classifier", "--kind", "bayes"])
    assert bad.exit_code == 1
