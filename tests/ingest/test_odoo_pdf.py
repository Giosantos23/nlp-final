"""Pruebas de la conversión RST → HTML → PDF de la documentación de Odoo."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from docuerp import paths
from docuerp.ingest import odoo_pdf
from docuerp.ingest.manifest import SourceEntry
from docuerp.ingest.odoo_pdf import (
    HEADING_SIZES_PT,
    PDF_BASE_CSS,
    BuiltPdf,
    OdooSnapshot,
    build_catalog,
    collect_module_pages,
    manifest_entry,
    parse_toctree_entries,
    prepare_rst_source,
    read_head_commit,
    rst_title,
    rst_to_html,
    write_odoo_manifest,
)


def test_heading_contract_matches_project_typography() -> None:
    assert HEADING_SIZES_PT == {1: 22.0, 2: 17.0, 3: 14.0, 4: 12.0}
    assert "font-size: 22.0pt" in PDF_BASE_CSS
    assert "font-size: 10.5pt" in PDF_BASE_CSS


def test_rst_title_handles_overline_and_underline() -> None:
    assert rst_title("=====\nTitle\n=====\n\ntext") == "Title"
    assert rst_title(":orphan:\n\nPlain\n-----\n") == "Plain"
    assert rst_title("no title here\n") is None


def test_parse_toctree_entries_ignores_options_and_reads_explicit_titles() -> None:
    text = (
        ".. toctree::\n   :titlesonly:\n   :hidden:\n\n   a/b\n   Custom <c/d>\n\n"
        "Paragraph after.\n"
    )
    assert parse_toctree_entries(text) == ["a/b", "c/d"]


def test_prepare_rst_source_strips_fields_and_injects_substitutions(rst_content: Path) -> None:
    catalog = build_catalog(rst_content)
    source = prepare_rst_source(":nosearch:\n\nUses |SO| here.\n", catalog)
    assert ":nosearch:" not in source
    assert ".. default-role:: literal" in source
    assert ".. |SO| replace:: :abbr:`SO (Sales Order)`" in source


def test_rst_to_html_renders_sphinx_constructs(rst_content: Path) -> None:
    html = rst_to_html(rst_content / "inventory" / "type.rst", base_dir=rst_content)
    assert "<h1>Product type</h1>" in html
    assert '<strong class="menuselection">Inventory app ‣ Products ‣ Products</strong>' in html
    assert '<strong class="guilabel">New</strong>' in html
    assert "SO (Sales Order)" in html  # sustitución + abbr con expansión
    assert "[cog]" in html  # ícono por nombre
    assert "the routes guide" in html  # :doc: con título explícito
    assert '<em class="xref">Tracked goods</em>' in html  # :ref: → título de la sección
    assert '<strong class="admonition-title">Note:</strong>' in html
    assert "[Figura: Designate a product as a good or service.]" in html
    assert "<p>1. Open the product form.</p>" in html  # marcadores de lista como texto
    assert "• nested bullet inside the step" in html
    assert "Tab: Filters" in html
    assert "<strong>Reordering rules</strong> (Recommended)" in html
    assert html.count("<table") == 2  # list-table y tabla grid conservadas
    assert "&lt;record id=" in html  # code-block
    assert "toctree" not in html
    assert "missing_child" not in html
    assert "<h3>Subsection</h3>" in html


def test_rst_to_html_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        rst_to_html(tmp_path / "nope.rst")


def test_unknown_directive_keeps_content(tmp_path: Path) -> None:
    content = tmp_path / "content"
    content.mkdir()
    page = content / "page.rst"
    page.write_text("Title\n=====\n\n.. brandnew:: Head\n\n   Inner text kept.\n", encoding="utf-8")
    html = rst_to_html(page, base_dir=content)
    assert "Inner text kept." in html


def _content_tree(root: Path) -> Path:
    content = root / "17.0" / "content"
    base = content / "applications" / "inventory_and_mrp"
    (base / "purchase" / "manage").mkdir(parents=True)
    (base / "purchase.rst").write_text(
        "========\nPurchase\n========\n\n.. toctree::\n\n   purchase/manage\n   purchase/ghost\n",
        encoding="utf-8",
    )
    (base / "purchase" / "manage.rst").write_text(
        "Manage\n======\n\n.. toctree::\n\n   manage/rfq\n", encoding="utf-8"
    )
    (base / "purchase" / "manage" / "rfq.rst").write_text("RfQ\n===\n\nText.\n", encoding="utf-8")
    (base / "purchase" / "orphan.rst").write_text("Orphan\n======\n\nText.\n", encoding="utf-8")
    return content


def test_collect_module_pages_follows_toctree_and_tolerates_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _content_tree(tmp_path / "odoo-docs")
    monkeypatch.setattr(paths, "RAW", tmp_path)
    collection = collect_module_pages("17.0", "compras")
    names = [p.docname.rsplit("/", 1)[-1] for p in collection.pages]
    assert names == ["purchase", "manage", "rfq", "orphan"]
    rfq = collection.pages[2]
    assert rfq.breadcrumb == ("Purchase", "Manage")
    assert any("ghost" in m for m in collection.missing)
    assert collection.orphans == ["applications/inventory_and_mrp/purchase/orphan"]


def test_collect_module_pages_rejects_unknown_module() -> None:
    with pytest.raises(ValueError, match="Módulo desconocido"):
        collect_module_pages("18.0", "rrhh")


def test_read_head_commit_from_packed_refs(tmp_path: Path) -> None:
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/18.0\n", encoding="utf-8")
    (git / "packed-refs").write_text(
        "# pack-refs\n" + "a" * 40 + " refs/remotes/origin/18.0\n", encoding="utf-8"
    )
    assert read_head_commit(tmp_path) == "a" * 40
    assert read_head_commit(tmp_path / "missing") is None


def test_manifest_entry_is_valid_source_entry(tmp_path: Path) -> None:
    pdf = tmp_path / "odoo-18.0-compras.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    record = BuiltPdf(
        version="18.0",
        module="compras",
        path=pdf,
        pages=3,
        rst_files=2,
        missing=[],
        orphans=[],
        sha256="b" * 64,
        snapshot=OdooSnapshot("18.0", "160a12ca9f533e0dfe0027c856ab18ff335fe55c", "2026-11-16"),
    )
    entry = SourceEntry.model_validate(manifest_entry(record))
    assert entry.id == "odoo-18.0-compras"
    assert entry.version == "18.0@160a12c"
    assert entry.allowed_roles == ["portal_user", "n1", "n2", "admin"]
    assert entry.sensitivity == "public"
    manifest = write_odoo_manifest([record], tmp_path / "odoo.yaml")
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert [d["id"] for d in data["documents"]] == ["odoo-18.0-compras"]


def test_versioned_odoo_manifest_has_twelve_documents() -> None:
    data = yaml.safe_load((paths.MANIFEST / "odoo.yaml").read_text(encoding="utf-8"))
    entries = [SourceEntry.model_validate(d) for d in data["documents"]]
    assert len(entries) == 12
    assert {e.erp_version for e in entries} == {"17.0", "18.0"}
    assert {e.module for e in entries} == set(odoo_pdf.MODULE_ROOTS)
    assert all(e.sha256 and len(e.sha256) == 64 for e in entries)
    assert all(e.license.startswith("CC BY-SA 4.0") for e in entries)


def test_html_to_pdf_writes_footer_and_cover(sample_pdf: Path) -> None:
    import pymupdf

    with pymupdf.open(sample_pdf) as pdf:
        assert pdf.page_count == 3  # portada + 2 páginas RST (cada una en página nueva)
        assert "demo-doc.pdf · pág 2" in pdf[1].get_text()
        assert "Injected page" in pdf[2].get_text()
