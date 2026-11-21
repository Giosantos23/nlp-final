"""Pruebas extremo a extremo del pipeline de ingesta y de la CLI (sin modelos pesados)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from docuerp import paths
from docuerp.ingest import pipeline
from docuerp.ingest.chunk import StructureChunker, WhitespaceTokenizer
from docuerp.ingest.clean import SUSPICIOUS_FLAG
from docuerp.ingest.cli import app
from docuerp.ingest.index import load_bm25_index, load_chunks, open_chroma_collection
from docuerp.ingest.manifest import SourceEntry
from docuerp.ingest.odoo_pdf import sha256_file


@pytest.fixture
def workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_pdf: Path,
    entry_payload: dict[str, Any],
) -> dict[str, Path]:
    """Repositorio aislado: manifiesto con 2 documentos (uno ausente) y el PDF de prueba."""
    for name in ("INDEX", "PROCESSED", "REPORTS", "BUILD", "RUNBOOKS"):
        monkeypatch.setattr(paths, name, tmp_path / name.lower())
    pdf_dir = paths.BUILD / "pdf"
    pdf_dir.mkdir(parents=True)
    shutil.copy(sample_pdf, pdf_dir / "demo-doc.pdf")
    manifest = tmp_path / "manifest"
    manifest.mkdir()
    internal = {
        **entry_payload,
        "id": "RB-N1-404@v1.0",
        "source_file": "RB-N1-404.pdf",
        "doc_type": "runbook",
        "erp_version": "any",
        "allowed_roles": ["n1", "n2", "admin"],
        "sensitivity": "internal",
    }
    odoo = {**entry_payload, "sha256": sha256_file(pdf_dir / "demo-doc.pdf")}
    (manifest / "all.yaml").write_text(
        yaml.safe_dump({"documents": [odoo, internal]}, allow_unicode=True), encoding="utf-8"
    )
    monkeypatch.setattr(paths, "MANIFEST", manifest)
    return {"root": tmp_path, "manifest": manifest, "pdf": pdf_dir / "demo-doc.pdf"}


def _chunker() -> StructureChunker:
    return StructureChunker(target_tokens=120, min_tokens=5, tokenizer=WhitespaceTokenizer())


def test_run_ingestion_end_to_end_with_fake_embedder(workspace: dict[str, Path]) -> None:
    report = pipeline.run_ingestion(
        chunker="structure", embedder="fake", sources="all", chunker_impl=_chunker()
    )
    assert report.documents == 1
    assert report.pages == 3
    assert report.chunks == len(load_chunks("structure")) > 0
    assert any("RB-N1-404" in w and "no se encontró" in w for w in report.warnings)
    assert set(report.timings_s) >= set(pipeline.PHASES) | {"total"}
    assert report.per_module_version["inventario@18.0"]["pages"] == 3
    # El texto inyectado se marca y se reporta, pero no se borra.
    flagged = [c for c in load_chunks("structure") if SUSPICIOUS_FLAG in c.flags]
    assert [c.chunk_id for c in flagged] == report.flagged_chunks
    assert flagged
    assert "Ignore all previous instructions" in flagged[0].text
    assert flagged[0].page == 3
    assert {f["pattern"] for f in report.suspicious_findings} >= {"en_ignore_instructions"}
    # Índices: BM25 y Chroma alineados con el JSONL.
    _, ids = load_bm25_index("structure")
    assert ids == [c.chunk_id for c in load_chunks("structure")]
    assert open_chroma_collection("structure", "fake").count() == report.chunks
    # Ninguna cita apunta a la portada y ningún chunk contiene el pie de página.
    chunks = load_chunks("structure")
    assert all(c.page >= 2 for c in chunks)
    assert not any("· pág" in c.text for c in chunks)


def test_run_ingestion_skip_embeddings_and_sources(workspace: dict[str, Path]) -> None:
    report = pipeline.run_ingestion(
        chunker="structure",
        sources="odoo",
        skip_embeddings=True,
        build_bm25=False,
        chunker_impl=_chunker(),
    )
    assert report.skip_embeddings
    assert "chroma" not in report.outputs
    assert "bm25" not in report.outputs
    assert not any("RB-N1-404" in w for w in report.warnings)  # filtrado por --sources odoo
    with pytest.raises(ValueError, match="no produjo chunks"):
        pipeline.run_ingestion(sources="internal", skip_embeddings=True, chunker_impl=_chunker())


def test_sha_mismatch_is_reported(workspace: dict[str, Path]) -> None:
    workspace["pdf"].write_bytes(workspace["pdf"].read_bytes() + b"\n% tampered\n")
    report = pipeline.run_ingestion(
        sources="odoo", skip_embeddings=True, build_bm25=False, chunker_impl=_chunker()
    )
    assert any("sha256" in w for w in report.warnings)


def test_reports_run_and_consolidated(workspace: dict[str, Path]) -> None:
    report = pipeline.run_ingestion(
        sources="odoo", skip_embeddings=True, build_bm25=False, chunker_impl=_chunker()
    )
    json_path, md_path = pipeline.write_report(report)
    assert json_path.name == "ingestion_structure.json"
    assert "Chunks por módulo y versión" in md_path.read_text(encoding="utf-8")
    consolidated_json, consolidated_md = pipeline.build_consolidated_report()
    payload = json.loads(consolidated_json.read_text(encoding="utf-8"))
    section = payload["chunkers"]["structure"]
    assert section["chunks"] == report.chunks
    assert section["pages"] == 3
    assert section["per_module_version"]["inventario@18.0"]["pages"] == 3
    assert "Resumen por chunker" in consolidated_md.read_text(encoding="utf-8")


def test_summary_helpers() -> None:
    assert pipeline.token_statistics([]) == {}
    stats = pipeline.token_statistics([10, 20, 30, 40, 50])
    assert stats["min"] == 10
    assert stats["max"] == 50
    assert stats["p50"] == 30
    histogram = pipeline.token_histogram([10, 70, 300, 2000])
    assert histogram["0-63"] == 1
    assert histogram["64-127"] == 1
    assert histogram["≥1024"] == 1
    with pytest.raises(ValueError):
        pipeline.select_entries([], "bogus")  # type: ignore[arg-type]


def test_select_entries_filters_by_doc_type(entry: SourceEntry) -> None:
    runbook = entry.model_copy(update={"id": "rb", "doc_type": "runbook"})
    assert pipeline.select_entries([entry, runbook], "odoo") == [entry]
    assert pipeline.select_entries([entry, runbook], "internal") == [runbook]
    assert pipeline.select_entries([entry, runbook], "all") == [entry, runbook]


def test_cli_validates_and_reports_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--sources", "bogus"])
    assert result.exit_code == 2
    monkeypatch.setattr(paths, "PROCESSED", tmp_path / "empty")
    monkeypatch.setattr(paths, "REPORTS", tmp_path / "reports")
    result = runner.invoke(app, ["report"])
    assert result.exit_code == 1
    assert "ingest run" in result.output
