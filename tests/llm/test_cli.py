"""CLI: ``docuerp llm ping``, ``docuerp generation demo`` y ``show-prompt``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from docuerp.cli import app
from docuerp.generation.demo import SCENARIOS
from docuerp.llm.factory import clear_provider_cache
from docuerp.schemas import ABSTENTION_MESSAGE, CANARY_TOKEN, Answer

runner = CliRunner()


def test_llm_ping_offline_succeeds_without_network() -> None:
    result = runner.invoke(app, ["llm", "ping", "--provider", "offline"])
    assert result.exit_code == 0, result.output
    assert "proveedor=offline" in result.output
    assert "costo" in result.output


def test_llm_ping_reports_typed_failure_with_exit_code_1() -> None:
    # El proveedor fake no tiene respuestas programadas: debe fallar de forma controlada.
    result = runner.invoke(app, ["llm", "ping", "--provider", "fake"])
    assert result.exit_code == 1
    assert "FALLÓ (unknown)" in result.output


def test_llm_ping_agent_sdk_without_token_reports_auth_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Un .env inexistente y sin variable de entorno ⇒ error de autenticación, sin llamar a nada.
    monkeypatch.setattr("docuerp.paths.DOTENV", tmp_path / "no-existe.env")
    clear_provider_cache()  # el proveedor cacheado pudo capturar la ruta real de .env
    result = runner.invoke(app, ["llm", "ping", "--provider", "agent_sdk"])
    assert result.exit_code == 1
    assert "FALLÓ (auth)" in result.output
    assert "CLAUDE_CODE_OAUTH_TOKEN" in result.output


def test_llm_ping_rejects_unknown_provider_and_effort() -> None:
    assert runner.invoke(app, ["llm", "ping", "--provider", "gpt"]).exit_code == 2
    bad_effort = runner.invoke(app, ["llm", "ping", "--provider", "anthropic_api", "--effort", "x"])
    assert bad_effort.exit_code in {1, 2}  # sin clave falla antes por autenticación


def test_show_prompt_masks_the_canary() -> None:
    result = runner.invoke(app, ["generation", "show-prompt"])
    assert result.exit_code == 0
    assert CANARY_TOKEN not in result.output
    assert "[CANARIO-ENMASCARADO]" in result.output
    assert "<contexto>" in result.output


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_demo_offline_produces_a_valid_contract_answer(scenario: str) -> None:
    result = runner.invoke(app, ["generation", "demo", "--scenario", scenario])
    assert result.exit_code == 0, result.output
    start = result.output.index("{")
    end = result.output.rindex("}") + 1
    answer = Answer.model_validate(json.loads(result.output[start:end]))
    assert "Consumo:" in result.output
    if scenario == "sin-evidencia":
        assert answer.abstained is True
        assert answer.answer == ABSTENTION_MESSAGE
    if scenario == "recepcion":
        assert answer.sources[0].page == 12
    if scenario == "ticket":
        assert answer.tool_used == "get_ticket_status"


def test_demo_unknown_scenario_and_provider_fail_cleanly() -> None:
    assert runner.invoke(app, ["generation", "demo", "--scenario", "no-existe"]).exit_code == 2
    assert runner.invoke(app, ["generation", "demo", "--provider", "gpt"]).exit_code == 1


def test_demo_can_show_the_user_prompt() -> None:
    result = runner.invoke(app, ["generation", "demo", "--show-prompt"])
    assert result.exit_code == 0
    assert "<fragmento" in result.output
