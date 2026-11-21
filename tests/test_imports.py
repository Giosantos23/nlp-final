"""Todos los módulos del paquete importan sin efectos secundarios ni dependencias rotas."""

from __future__ import annotations

import importlib
import pkgutil

import pytest
import typer

import docuerp


def _all_modules() -> list[str]:
    return sorted(m.name for m in pkgutil.walk_packages(docuerp.__path__, prefix="docuerp."))


MODULES = _all_modules()

CLI_PACKAGES = [
    "ingest",
    "retrieval",
    "llm",
    "generation",
    "agent",
    "agent.tools",
    "security",
    "api",
    "eval",
    "data",
]


def test_discovers_modules() -> None:
    assert len(MODULES) > 40


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)


@pytest.mark.parametrize("package", CLI_PACKAGES)
def test_cli_apps(package: str) -> None:
    mod = importlib.import_module(f"docuerp.{package}.cli")
    assert isinstance(mod.app, typer.Typer)


def test_root_cli_registers_all_groups() -> None:
    from typer.testing import CliRunner

    from docuerp.cli import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for group in [
        "ingest",
        "retrieval",
        "eval",
        "security",
        "data",
        "agent",
        "api",
        "ask",
        "serve",
    ]:
        assert group in result.output
    assert "no disponible" not in result.output


def test_settings_load() -> None:
    from docuerp.config import get_settings

    s = get_settings()
    assert s.retrieval.candidates_k == 20
    assert s.retrieval.top_k == 5
    assert s.retrieval.rrf_k == 60
    assert s.retrieval.tau == pytest.approx(0.0)  # calibrado a nivel sistema (tau_calibration.md)
    assert s.chunking.target_tokens == 450
    assert s.models.embedders["bge-m3"] == "BAAI/bge-m3"
    assert s.agent.max_steps == 4
    assert s.agent.max_tool_calls == 3
    assert s.llm.price_for("claude-haiku-5-5") is not None


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from docuerp.config import Settings

    monkeypatch.setenv("DOCUERP_RETRIEVAL__TOP_K", "8")
    monkeypatch.setenv("DOCUERP_LLM_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("DOCUERP_HMAC_SECRET", "s3cr3t-de-prueba")
    s = Settings()
    assert s.retrieval.top_k == 8
    assert s.retrieval.candidates_k == 20  # el resto sigue viniendo del YAML
    assert s.llm.model == "claude-sonnet-5-5"
    assert s.security.hmac_secret is not None
    assert "s3cr3t" not in str(s.safe_dump())


def test_paths_are_absolute() -> None:
    from docuerp import paths

    for name in [
        "DATA",
        "RAW",
        "BUILD",
        "PROCESSED",
        "INDEX",
        "TICKETS_DB",
        "LOGS",
        "REPORTS",
        "EVAL",
        "CONFIGS",
    ]:
        assert getattr(paths, name).is_absolute()
    assert (paths.CONFIGS / "settings.yaml").is_file()


def test_design_tokens() -> None:
    import json

    from docuerp import paths

    tokens = json.loads(paths.DESIGN_TOKENS.read_text(encoding="utf-8"))
    assert tokens["color"]["accent"] == "#10B981"
    assert tokens["color"]["bg"] == "#0A0C10"
