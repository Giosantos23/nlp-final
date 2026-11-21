"""Pruebas de metadatos de chunk y del manifiesto del corpus."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from docuerp.ingest.chunk import ChunkDraft
from docuerp.ingest.manifest import ManifestError, SourceEntry, find_entry, load_manifest
from docuerp.ingest.metadata import build_chunks, content_hash, utc_now_iso


def _draft(text: str, page: int, page_end: int | None = None) -> ChunkDraft:
    return ChunkDraft(
        text=text, page=page, page_end=page_end or page, section_path="A > B", token_count=3
    )


def test_content_hash_ignores_whitespace_and_unicode_form() -> None:
    composed = "recepción  lista"
    decomposed = "recepción lista"
    assert content_hash(composed) == content_hash(decomposed)
    assert re.fullmatch(r"[0-9a-f]{64}", content_hash("x"))
    assert content_hash("a") != content_hash("b")


def test_utc_now_iso_format() -> None:
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", utc_now_iso())


def test_build_chunks_ids_are_per_start_page(entry: SourceEntry) -> None:
    drafts = [_draft("uno", 2), _draft("dos", 2, 3), _draft("tres", 3), _draft("cuatro", 2)]
    chunks = build_chunks(
        drafts,
        entry,
        flags_by_index={1: ["suspicious_instruction"]},
        ingested_at="2026-01-01T00:00:00Z",
    )
    assert [c.chunk_id for c in chunks] == [
        "demo-doc::p0002::c000",
        "demo-doc::p0002::c001",
        "demo-doc::p0003::c000",
        "demo-doc::p0002::c002",
    ]
    second = chunks[1]
    assert (second.page, second.page_end) == (2, 3)
    assert second.flags == ["suspicious_instruction"]
    assert chunks[0].flags == []
    assert second.source == "demo-doc.pdf"
    assert second.version == "18.0@abc1234"
    assert second.allowed_roles == ["portal_user", "n1", "n2", "admin"]
    assert {c.ingested_at for c in chunks} == {"2026-01-01T00:00:00Z"}
    # Determinismo: mismos borradores → mismos ids y hashes.
    again = build_chunks(drafts, entry, ingested_at="2026-01-01T00:00:00Z")
    assert [c.chunk_id for c in again] == [c.chunk_id for c in chunks]
    assert [c.content_hash for c in again] == [c.content_hash for c in chunks]


def _write_manifest(directory: Path, name: str, entries: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        yaml.safe_dump({"documents": entries}, allow_unicode=True), encoding="utf-8"
    )


def test_load_manifest_reads_all_files(tmp_path: Path, entry_payload: dict[str, Any]) -> None:
    other = {**entry_payload, "id": "RB-N1-001@v1.0", "doc_type": "runbook", "erp_version": "any"}
    _write_manifest(tmp_path, "odoo.yaml", [entry_payload])
    _write_manifest(tmp_path, "internal.yaml", [other])
    entries = load_manifest(tmp_path)
    assert sorted(e.id for e in entries) == ["RB-N1-001@v1.0", "demo-doc"]
    assert find_entry("demo-doc", entries).source_file == "demo-doc.pdf"
    with pytest.raises(KeyError):
        find_entry("nope", entries)


def test_load_manifest_rejects_duplicates(tmp_path: Path, entry_payload: dict[str, Any]) -> None:
    _write_manifest(tmp_path, "a.yaml", [entry_payload])
    _write_manifest(tmp_path, "b.yaml", [entry_payload])
    with pytest.raises(ValueError, match="duplicado"):
        load_manifest(tmp_path)


def test_load_manifest_rejects_invalid_entries(
    tmp_path: Path, entry_payload: dict[str, Any]
) -> None:
    _write_manifest(tmp_path, "a.yaml", [{**entry_payload, "allowed_roles": ["root"]}])
    with pytest.raises(ManifestError, match="inválida"):
        load_manifest(tmp_path)
    (tmp_path / "a.yaml").write_text("documents: {not: a list}", encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(tmp_path)
    (tmp_path / "a.yaml").write_text("documents: [\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="YAML"):
        load_manifest(tmp_path)


def test_load_manifest_missing_dir_is_empty(tmp_path: Path) -> None:
    assert load_manifest(tmp_path / "nope") == []
