"""Pruebas de fragmentación estructural y de tamaño fijo."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from docuerp.ingest.chunk import (
    FixedChunker,
    HFTokenizer,
    StructureChunker,
    WhitespaceTokenizer,
    format_section_path,
    get_chunker,
)
from docuerp.ingest.extract import ExtractedDocument, PageContent, TextBlock


def _b(
    text: str, page: int = 1, kind: str = "paragraph", level: int | None = None, indent: float = 0.0
) -> TextBlock:
    return TextBlock(text=text, page=page, kind=kind, level=level, indent=indent)  # type: ignore[arg-type]


def _doc(blocks: list[TextBlock]) -> ExtractedDocument:
    pages: dict[int, list[TextBlock]] = {}
    for block in blocks:
        pages.setdefault(block.page, []).append(block)
    return ExtractedDocument(
        doc_id="d", source="d.pdf", pages=[PageContent(page=p, blocks=b) for p, b in pages.items()]
    )


def _words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}." if i % 12 == 11 else f"{prefix}{i}" for i in range(n))


def test_section_path_comes_from_breadcrumb_and_headings(ws_tokenizer: WhitespaceTokenizer) -> None:
    doc = _doc(
        [
            _b("Inventory › Product management", kind="breadcrumb"),
            _b("Product type", kind="heading", level=1),
            _b(_words(60, "a")),
            _b("Goods", kind="heading", level=2),
            _b(_words(60, "b")),
        ]
    )
    chunker = StructureChunker(target_tokens=80, min_tokens=5, tokenizer=ws_tokenizer)
    chunks = chunker.chunk(doc)
    assert [c.section_path for c in chunks] == [
        "Inventory > Product management > Product type",
        "Inventory > Product management > Product type > Goods",
    ]
    assert chunks[1].text.startswith("Inventory > Product management > Product type > Goods\n")
    assert all(c.token_count == ws_tokenizer.count(c.text) for c in chunks)


def test_h1_without_breadcrumb_resets_path(ws_tokenizer: WhitespaceTokenizer) -> None:
    doc = _doc(
        [
            _b("Root", kind="breadcrumb"),
            _b("First", kind="heading", level=1),
            _b(_words(50)),
            _b("Second", kind="heading", level=1, page=2),
            _b(_words(50), page=2),
        ]
    )
    chunks = StructureChunker(target_tokens=60, min_tokens=5, tokenizer=ws_tokenizer).chunk(doc)
    assert [c.section_path for c in chunks] == ["Root > First", "Second"]


def test_table_is_never_cut_and_spans_pages(ws_tokenizer: WhitespaceTokenizer) -> None:
    header = "Col A | Col B"
    rows_p1 = "\n".join(f"r{i} a b c d | x y z" for i in range(10))
    rows_p2 = "\n".join(f"s{i} a b c d | x y z" for i in range(10))
    doc = _doc(
        [
            _b("Page", kind="heading", level=1),
            _b(_words(30)),
            _b(f"{header}\n{rows_p1}", kind="table"),
            _b(f"{header}\n{rows_p2}", page=2, kind="table"),
            _b(_words(30, "z"), page=2),
        ]
    )
    chunker = StructureChunker(target_tokens=100, min_tokens=5, tokenizer=ws_tokenizer)
    chunks = chunker.chunk(doc)
    with_table = [c for c in chunks if "r0 a b" in c.text]
    assert len(with_table) == 1
    table_chunk = with_table[0]
    assert "r9 a b" in table_chunk.text
    assert "s9 a b" in table_chunk.text
    assert table_chunk.text.count(header) == 1  # encabezado repetido por página eliminado
    assert (table_chunk.page, table_chunk.page_end) == (1, 2)


def test_numbered_steps_stay_together(ws_tokenizer: WhitespaceTokenizer) -> None:
    steps = [
        _b(f"{i}. " + _words(14, f"s{i}_"), kind="list_item", indent=13.6) for i in range(1, 6)
    ]
    nested = _b("• nested detail " + _words(5), kind="list_item", indent=27.0)
    doc = _doc(
        [
            _b("Page", kind="heading", level=1),
            _b(_words(40)),
            _b("Follow these steps:"),
            *steps[:2],
            nested,
            *steps[2:],
            _b("After the list " + _words(30, "t")),
        ]
    )
    chunks = StructureChunker(target_tokens=70, min_tokens=5, tokenizer=ws_tokenizer).chunk(doc)
    holder = [c for c in chunks if "1. s1_0" in c.text]
    assert len(holder) == 1
    text = holder[0].text
    assert "Follow these steps:" in text  # el párrafo introductorio va con la lista
    for i in range(1, 6):
        assert f"{i}. s{i}_0" in text
    assert "• nested detail" in text
    assert "After the list" not in text  # el párrafo posterior no es parte de los pasos


def test_paragraph_cut_by_page_break_is_rejoined(ws_tokenizer: WhitespaceTokenizer) -> None:
    doc = _doc(
        [
            _b("Page", kind="heading", level=1),
            _b("Click Validate to", page=4),
            _b("receive the products into stock.", page=5),
            _b("Next paragraph starts here.", page=5),
        ]
    )
    chunks = StructureChunker(target_tokens=100, min_tokens=1, tokenizer=ws_tokenizer).chunk(doc)
    assert "Click Validate to receive the products into stock.\n" in chunks[0].text
    assert (chunks[0].page, chunks[0].page_end) == (4, 5)


def test_bullet_unit_ends_at_body_paragraph(ws_tokenizer: WhitespaceTokenizer) -> None:
    doc = _doc(
        [
            _b("Page", kind="heading", level=1),
            _b("• first bullet", kind="list_item", indent=18.0),
            _b("• second bullet", kind="list_item", indent=18.0),
            _b(_words(40, "body")),
        ]
    )
    chunker = StructureChunker(target_tokens=30, min_tokens=1, tokenizer=ws_tokenizer)
    chunks = chunker.chunk(doc)
    bullets = next(c for c in chunks if "• first bullet" in c.text)
    assert "body0" not in bullets.text


def test_code_block_is_not_cut(ws_tokenizer: WhitespaceTokenizer) -> None:
    code = "\n".join(f"line {i} = value {i}" for i in range(20))
    doc = _doc([_b("Page", kind="heading", level=1), _b(_words(50)), _b(code, kind="code")])
    chunks = StructureChunker(target_tokens=60, min_tokens=5, tokenizer=ws_tokenizer).chunk(doc)
    assert sum("line 0 = value 0" in c.text and "line 19 = value 19" in c.text for c in chunks) == 1


def test_overlap_repeats_tail_sentences(ws_tokenizer: WhitespaceTokenizer) -> None:
    paragraphs = [
        _b(" ".join(f"Sentence {k} of paragraph {i} goes here." for k in range(6)))
        for i in range(4)
    ]
    doc = _doc([_b("Page", kind="heading", level=1), *paragraphs])
    chunker = StructureChunker(
        target_tokens=100, overlap=0.15, min_tokens=5, tokenizer=ws_tokenizer
    )
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 2
    for first, second in itertools.pairwise(chunks):
        tail = first.text.split("\n")[-1].split(". ")[-1]
        assert tail in second.text  # la última oración del chunk previo abre el siguiente
    assert all(c.token_count <= 100 for c in chunks)


def test_oversized_table_is_split_by_rows_with_header(ws_tokenizer: WhitespaceTokenizer) -> None:
    rows = "\n".join(f"row{i} " + " ".join(["x"] * 8) for i in range(40))
    doc = _doc([_b("Page", kind="heading", level=1), _b(f"H1 | H2\n{rows}", kind="table")])
    chunker = StructureChunker(
        target_tokens=60, max_tokens=120, min_tokens=5, tokenizer=ws_tokenizer
    )
    chunks = chunker.chunk(doc)
    assert len(chunks) > 2
    assert all("H1 | H2" in c.text for c in chunks)
    assert all(c.token_count <= 120 for c in chunks)
    body = "\n".join(c.text for c in chunks)
    assert all(f"row{i} " in body for i in range(40))


def test_small_sections_are_merged_within_page(ws_tokenizer: WhitespaceTokenizer) -> None:
    doc = _doc(
        [
            _b("Page", kind="heading", level=1),
            _b("Tiny", kind="heading", level=2),
            _b("just five words here."),
            _b("Other", kind="heading", level=2),
            _b(_words(20)),
        ]
    )
    chunks = StructureChunker(target_tokens=100, min_tokens=10, tokenizer=ws_tokenizer).chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].section_path == "Page"
    assert "Tiny\njust five words here." in chunks[0].text
    assert "\nOther\n" in chunks[0].text


def test_fixed_chunker_windows_and_pages(ws_tokenizer: WhitespaceTokenizer) -> None:
    doc = _doc(
        [
            _b("Page", kind="heading", level=1),
            _b(_words(150, "a")),
            _b(_words(150, "b"), page=2),
        ]
    )
    chunker = FixedChunker(100, overlap=0.15, tokenizer=ws_tokenizer)
    chunks = chunker.chunk(doc)
    assert chunker.name == "fixed100"
    assert all(c.token_count <= 100 for c in chunks)
    assert chunks[0].page == 1
    assert chunks[-1].page_end == 2
    assert any(c.page == 1 and c.page_end == 2 for c in chunks)  # ventana que cruza página
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-15:] == second[:15]  # 15 % de solape
    assert all(c.section_path == "Page" for c in chunks)


def test_empty_document_produces_no_chunks(ws_tokenizer: WhitespaceTokenizer) -> None:
    empty = ExtractedDocument(doc_id="d", source="d.pdf", pages=[])
    assert StructureChunker(tokenizer=ws_tokenizer).chunk(empty) == []
    assert FixedChunker(256, tokenizer=ws_tokenizer).chunk(empty) == []


def test_constructor_validation() -> None:
    with pytest.raises(ValueError):
        StructureChunker(target_tokens=8)
    with pytest.raises(ValueError):
        StructureChunker(overlap=0.6)
    with pytest.raises(ValueError):
        FixedChunker(4)


def test_get_chunker_factory(ws_tokenizer: WhitespaceTokenizer) -> None:
    structure = get_chunker("structure", tokenizer=ws_tokenizer)
    assert isinstance(structure, StructureChunker)
    assert structure.target_tokens == 450
    assert structure.overlap == pytest.approx(0.15)
    assert get_chunker("fixed256", tokenizer=ws_tokenizer).name == "fixed256"
    assert get_chunker("fixed768", tokenizer=ws_tokenizer).name == "fixed768"
    with pytest.raises(ValueError, match="desconocido"):
        get_chunker("semantic")


def test_format_section_path_skips_empty() -> None:
    assert format_section_path(("A", "", "B")) == "A > B"


def _bge_m3_cached() -> bool:
    cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-m3"
    return cache.is_dir()


@pytest.mark.skipif(not _bge_m3_cached(), reason="tokenizador de bge-m3 no está en caché")
def test_real_bge_m3_tokenizer_counts_and_offsets() -> None:
    tokenizer = HFTokenizer("BAAI/bge-m3")
    text = "Ajuste de inventario en WH/Stock"
    assert 4 <= tokenizer.count(text) <= 20
    offsets = tokenizer.offsets(text)
    assert len(offsets) == tokenizer.count(text)
    assert offsets[-1][1] == len(text)
