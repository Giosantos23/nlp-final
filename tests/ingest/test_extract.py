"""Pruebas de extracción estructurada (PDF real de Chromium y DOCX de python-docx)."""

from __future__ import annotations

from pathlib import Path

import pytest

from docuerp.ingest.extract import (
    ExtractionError,
    extract,
    extract_docx,
    extract_pdf,
    heading_level_for,
    join_lines,
    table_rows_to_text,
)


def test_heading_level_by_size_requires_bold() -> None:
    assert heading_level_for(22.0, True) == 1
    assert heading_level_for(16.99, True) == 2
    assert heading_level_for(14.0, True) == 3
    assert heading_level_for(12.0, True) == 4
    assert heading_level_for(12.0, False) is None
    assert heading_level_for(10.5, True) is None


def test_join_lines_keeps_compound_hyphen() -> None:
    assert join_lines(["auto-", "populate the form"]) == "auto-populate the form"
    assert join_lines(["one", "two"]) == "one two"


def test_table_rows_to_text_drops_empty_columns_and_rows() -> None:
    rows: list[list[str | None]] = [["A", "", "B"], ["1", None, "2"], ["", "", ""]]
    assert table_rows_to_text(rows) == "A | B\n1 | 2"


def test_extract_pdf_structure(sample_pdf: Path) -> None:
    doc = extract_pdf(sample_pdf, "demo-doc")
    assert doc.page_count == 3
    assert doc.skipped_pages == [1]  # portada omitida
    assert [p.page for p in doc.pages] == [2, 3]
    blocks = doc.pages[0].blocks
    kinds = [(b.kind, b.level) for b in blocks]
    assert kinds[0] == ("breadcrumb", None)
    assert blocks[0].text == "Inventory › Product management"
    assert ("heading", 1) in kinds
    assert ("heading", 2) in kinds
    assert ("heading", 3) in kinds
    texts = [b.text for b in blocks]
    assert all("pág" not in t for t in texts)  # pie eliminado
    steps = [b for b in blocks if b.kind == "list_item" and b.text[0].isdigit()]
    assert [s.text[:2] for s in steps] == ["1.", "2.", "3."]
    nested = next(b for b in blocks if b.text.startswith("• nested"))
    assert nested.indent > steps[0].indent  # la viñeta anidada está más sangrada
    table = next(b for b in blocks if b.kind == "table")
    assert table.text.splitlines()[:3] == [
        "Operation | Tracked",
        "Show on-hand quantity | Yes",
        "Use reordering rules | Yes",
    ]
    code = next(b for b in blocks if b.kind == "code")
    assert code.text.splitlines()[1].startswith("    <field")  # sangría conservada
    figure = next(b for b in blocks if b.kind == "figure")
    assert figure.text == "[Figura: Designate a product as a good or service.]"


def test_extract_dispatch_and_errors(tmp_path: Path, sample_pdf: Path) -> None:
    assert extract(sample_pdf, "x").source == "demo-doc.pdf"
    bad = tmp_path / "file.txt"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ExtractionError, match="no soportado"):
        extract(bad, "x")
    with pytest.raises(ExtractionError, match="No existe"):
        extract_pdf(tmp_path / "missing.pdf", "x")
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"not a pdf at all")
    with pytest.raises(ExtractionError):
        extract_pdf(corrupt, "x")


def _make_docx(path: Path) -> Path:
    import docx
    from docx.enum.text import WD_BREAK

    document = docx.Document()
    document.add_paragraph("Runbook de prueba", style="Title")
    document.add_paragraph("Portada con metadatos")
    document.paragraphs[-1].add_run().add_break(WD_BREAK.PAGE)
    document.add_heading("Procedimiento", level=1)
    document.add_paragraph("Introducción al procedimiento:")
    document.add_paragraph("Abrir el módulo", style="List Number")
    document.add_paragraph("Validar el ajuste", style="List Number")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Campo", "Valor"
    table.cell(1, 0).text, table.cell(1, 1).text = "Ubicación", "WH/Stock"
    last = document.add_paragraph("Fin de la página dos.")
    last.add_run().add_break(WD_BREAK.PAGE)
    document.add_heading("Escalamiento", level=2)
    document.add_paragraph("Contactar a N2.", style="List Bullet")
    document.save(str(path))
    return path


def test_extract_docx_pages_by_explicit_breaks(tmp_path: Path) -> None:
    doc = extract_docx(_make_docx(tmp_path / "RB-N1-001.docx"), "RB-N1-001@v1.0")
    assert doc.skipped_pages == [1]  # portada (estilo Title) omitida
    assert [p.page for p in doc.pages] == [2, 3]
    page2 = doc.pages[0].blocks
    assert (page2[0].kind, page2[0].level, page2[0].text) == ("heading", 1, "Procedimiento")
    assert [b.text for b in page2 if b.kind == "list_item"] == [
        "1. Abrir el módulo",
        "2. Validar el ajuste",
    ]
    assert next(b for b in page2 if b.kind == "table").text == "Campo | Valor\nUbicación | WH/Stock"
    page3 = doc.pages[1].blocks
    assert (page3[0].kind, page3[0].level) == ("heading", 2)
    assert page3[1].text == "• Contactar a N2."


def test_extract_docx_invalid_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.docx"
    bad.write_bytes(b"PK not really")
    with pytest.raises(ExtractionError):
        extract_docx(bad, "x")
