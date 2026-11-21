"""Pre-filtro de rol, versión y vigencia."""

from __future__ import annotations

import pytest

from docuerp.retrieval.filters import chroma_where, passes_filters, split_by_filters


def test_public_chunk_visible_for_every_role(make_chunk, make_query) -> None:
    chunk = make_chunk("d", 1, 0, "texto")
    for role in ("portal_user", "n1", "n2", "admin"):
        assert passes_filters(chunk, make_query("consulta", role=role))


@pytest.mark.parametrize(
    ("allowed", "visible", "hidden"),
    [
        (["n1", "n2", "admin"], ["n1", "n2", "admin"], ["portal_user"]),
        (["n2", "admin"], ["n2", "admin"], ["portal_user", "n1"]),
        (["admin"], ["admin"], ["portal_user", "n1", "n2"]),
    ],
)
def test_role_must_be_in_allowed_roles(make_chunk, make_query, allowed, visible, hidden) -> None:
    chunk = make_chunk("d", 1, 0, "texto", roles=allowed)
    assert all(passes_filters(chunk, make_query("consulta", role=r)) for r in visible)
    assert not any(passes_filters(chunk, make_query("consulta", role=r)) for r in hidden)


def test_version_filter_accepts_exact_and_any(make_chunk, make_query) -> None:
    q18 = make_query("consulta", erp_version="18.0")
    assert passes_filters(make_chunk("d", 1, 0, "t", erp="18.0"), q18)
    assert passes_filters(make_chunk("d", 1, 1, "t", erp="any"), q18)
    assert not passes_filters(make_chunk("d", 1, 2, "t", erp="17.0"), q18)


def test_obsolete_excluded_unless_requested(make_chunk, make_query) -> None:
    obsolete = make_chunk("d", 1, 0, "t", status="obsoleto")
    assert not passes_filters(obsolete, make_query("consulta"))
    assert passes_filters(obsolete, make_query("consulta", include_obsolete=True))


def test_include_obsolete_does_not_bypass_role_or_version(make_chunk, make_query) -> None:
    hidden_role = make_chunk("d", 1, 0, "t", status="obsoleto", roles=["admin"])
    other_version = make_chunk("d", 1, 1, "t", status="obsoleto", erp="17.0")
    q = make_query("consulta", role="n1", include_obsolete=True)
    assert not passes_filters(hidden_role, q)
    assert not passes_filters(other_version, q)


def test_split_by_filters_keeps_order(make_chunk, make_query) -> None:
    chunks = [
        make_chunk("d", 1, 0, "a"),
        make_chunk("d", 1, 1, "b", roles=["admin"]),
        make_chunk("d", 1, 2, "c"),
        make_chunk("d", 1, 3, "d", status="obsoleto"),
    ]
    kept, dropped = split_by_filters(chunks, make_query("consulta", role="n1"))
    assert [c.text for c in kept] == ["a", "c"]
    assert [c.text for c in dropped] == ["b", "d"]


def test_chroma_where_current_documents(make_query) -> None:
    where = chroma_where(make_query("consulta", erp_version="17.0"))
    assert where == {
        "$and": [
            {"erp_version": {"$in": ["17.0", "any"]}},
            {"status": {"$eq": "vigente"}},
        ]
    }


def test_chroma_where_with_obsolete_only_filters_version(make_query) -> None:
    where = chroma_where(make_query("consulta", include_obsolete=True))
    assert where == {"erp_version": {"$in": ["18.0", "any"]}}
