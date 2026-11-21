"""RBAC: jerarquía portal_user ⊂ n1 ⊂ n2 ⊂ admin, filtrado, revalidación y herramientas."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from docuerp.schemas import ScoredChunk
from docuerp.security import permissions
from docuerp.security.permissions import (
    ConfirmationClaim,
    authorize_tool_call,
    can_read,
    can_use_tool,
    filter_authorized,
    load_policy,
    revalidate_hits,
    role_at_least,
    verify_confirmation,
)

ROLES = ["portal_user", "n1", "n2", "admin"]
DOC_CLASSES = {
    "odoo-18.0-inventario": ["portal_user", "n1", "n2", "admin"],
    "RB-N1-002@v1.0": ["n1", "n2", "admin"],
    "POL-SLA-001@v1.0": ["n1", "n2", "admin"],
    "RB-N2-003@v1.1": ["n2", "admin"],
    "RB-ADM-006@v1.0": ["admin"],
}

# --- jerarquía ----------------------------------------------------------------------


@pytest.mark.parametrize(("lower", "higher"), [(ROLES[i], ROLES[i + 1]) for i in range(3)])
def test_hierarchy_is_strictly_increasing(lower: str, higher: str) -> None:
    assert role_at_least(higher, lower)  # type: ignore[arg-type]
    assert not role_at_least(lower, higher)  # type: ignore[arg-type]


def test_unknown_roles_never_qualify() -> None:
    assert not role_at_least("root", "portal_user")  # type: ignore[arg-type]
    assert not role_at_least("admin", "superadmin")  # type: ignore[arg-type]


def test_document_access_is_monotonic_in_the_hierarchy(
    make_hit: Callable[..., ScoredChunk],
) -> None:
    """Lo que lee un rol lo leen todos los superiores (portal_user ⊂ n1 ⊂ n2 ⊂ admin)."""
    readable: dict[str, set[str]] = {role: set() for role in ROLES}
    for doc_id, allowed in DOC_CLASSES.items():
        chunk = make_hit(doc_id, allowed_roles=allowed).chunk
        for role in ROLES:
            if can_read(role, chunk):  # type: ignore[arg-type]
                readable[role].add(doc_id)
    for lower, higher in itertools.pairwise(ROLES):
        assert readable[lower] < readable[higher]
    assert readable["portal_user"] == {"odoo-18.0-inventario"}
    assert readable["admin"] == set(DOC_CLASSES)


def test_can_read_rejects_unknown_role(make_hit: Callable[..., ScoredChunk]) -> None:
    assert not can_read("superuser", make_hit().chunk)  # type: ignore[arg-type]


# --- filtrado y revalidación ---------------------------------------------------------


def _mixed_hits(make_hit: Callable[..., ScoredChunk]) -> list[ScoredChunk]:
    return [
        make_hit(doc_id, allowed_roles=allowed, rank=rank)
        for rank, (doc_id, allowed) in enumerate(DOC_CLASSES.items(), start=1)
    ]


@pytest.mark.parametrize(("role", "kept"), [("portal_user", 1), ("n1", 3), ("n2", 4), ("admin", 5)])
def test_filter_authorized_counts(
    make_hit: Callable[..., ScoredChunk], role: str, kept: int
) -> None:
    hits = _mixed_hits(make_hit)
    authorized, dropped = filter_authorized(role, hits)  # type: ignore[arg-type]
    assert len(authorized) == kept
    assert dropped == len(hits) - kept
    # Se conservan rank y score originales (la traza refleja la recuperación real).
    assert [h.rank for h in authorized] == sorted(h.rank for h in authorized)
    assert all(h in hits for h in authorized)


def test_revalidate_hits_reasons_and_precedence(make_hit: Callable[..., ScoredChunk]) -> None:
    hits = [
        make_hit("odoo-18.0-inventario", idx=0),  # se conserva
        make_hit("RB-ADM-006@v1.0", allowed_roles=["admin"], idx=1),  # rol
        make_hit("odoo-17.0-inventario", erp_version="17.0", idx=2),  # versión
        make_hit("RB-N1-001@v1.0", allowed_roles=["n1", "n2", "admin"], erp_version="any", idx=3),
        make_hit("RB-N1-002@v0.9", allowed_roles=["n1", "n2", "admin"], status="obsoleto", idx=4),
        make_hit(
            "RB-N1-008@v1.0",
            allowed_roles=["n1", "n2", "admin"],
            flags=["suspicious_instruction"],
            idx=5,
        ),
        # Rol y versión a la vez: gana el rol (precedencia documentada).
        make_hit("RB-ADM-007@v1.0", allowed_roles=["admin"], erp_version="17.0", idx=6),
    ]
    result = revalidate_hits("n1", "18.0", hits)
    assert [h.chunk.doc_id for h in result.kept] == ["odoo-18.0-inventario", "RB-N1-001@v1.0"]
    assert result.dropped == {"role": 2, "version": 1, "obsolete": 1, "suspicious": 1}
    assert result.dropped_total == 5
    assert len(result.dropped_ids) == 5


def test_revalidate_keeps_obsolete_and_suspicious_when_asked(
    make_hit: Callable[..., ScoredChunk],
) -> None:
    hits = [
        make_hit("RB-N2-003@v1.0", allowed_roles=["n2", "admin"], status="obsoleto"),
        make_hit("RB-N2-004@v1.0", allowed_roles=["n2", "admin"], flags=["suspicious_instruction"]),
    ]
    result = revalidate_hits("n2", "18.0", hits, include_obsolete=True, drop_suspicious=False)
    assert len(result.kept) == 2
    assert result.dropped == {}


def test_revalidation_catches_index_leak(make_hit: Callable[..., ScoredChunk]) -> None:
    """Si el índice dejara pasar un runbook ADM a un portal_user, la revalidación lo corta."""
    leaked = make_hit("RB-ADM-006@v1.0", allowed_roles=["admin"])
    result = revalidate_hits("portal_user", "18.0", [leaked])
    assert result.kept == []
    assert result.dropped == {"role": 1}


# --- herramientas --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("role", "tool", "allowed"),
    [
        ("portal_user", "search_documents", True),
        ("portal_user", "get_ticket_status", True),
        ("portal_user", "classify_ticket", False),
        ("portal_user", "create_ticket", False),
        ("n1", "classify_ticket", True),
        ("n1", "create_ticket", True),
        ("admin", "delete_ticket", False),  # no declarada: prohibida para todos
        ("hacker", "search_documents", False),
    ],
)
def test_can_use_tool(role: str, tool: str, allowed: bool) -> None:
    assert can_use_tool(role, tool) is allowed  # type: ignore[arg-type]


def test_read_tools_are_allowed_with_valid_args() -> None:
    decision = authorize_tool_call("n1", "get_ticket_status", {"ticket_id": "TCK-00042"})
    assert decision.decision == "allow"
    decision = authorize_tool_call("n1", "search_documents", {"query": "recepciones", "top_k": 5})
    assert decision.decision == "allow"


@pytest.mark.parametrize(
    ("tool", "args", "violation"),
    [
        ("get_ticket_status", {"ticket_id": "TCK-00001' OR 1=1--"}, "ticket_id_invalido"),
        ("get_ticket_status", {"ticket_id": "tck-00042"}, "ticket_id_invalido"),
        ("search_documents", {"query": "recepciones", "top_k": 50}, "top_k_fuera_de_rango"),
        ("search_documents", {"query": "recepciones", "top_k": 0}, "top_k_fuera_de_rango"),
        ("search_documents", {"query": "recepciones", "top_k": True}, "top_k_no_entero"),
        ("search_documents", {"query": "ok", "top_k": 5}, "query_invalida"),
        ("classify_ticket", {"text": "x" * 2001}, "texto_fuera_de_rango"),
        ("classify_ticket", {"text": "   "}, "texto_fuera_de_rango"),
    ],
)
def test_invalid_args_are_denied(tool: str, args: dict[str, object], violation: str) -> None:
    decision = authorize_tool_call("admin", tool, args)
    assert decision.decision == "deny"
    assert violation in decision.violations


def test_unknown_and_internal_tools_are_denied() -> None:
    assert authorize_tool_call("admin", "delete_ticket").violations == ["herramienta_desconocida"]
    internal = authorize_tool_call("portal_user", "create_ticket_escalated")
    assert internal.decision == "deny"


def test_call_limit_per_query() -> None:
    decision = authorize_tool_call(
        "n1", "get_ticket_status", {"ticket_id": "TCK-00042"}, calls_in_query=3
    )
    assert decision.violations == ["limite_llamadas"]


def test_write_tool_needs_confirmation_or_escalates() -> None:
    assert authorize_tool_call("n1", "create_ticket").decision == "needs_confirmation"
    assert authorize_tool_call("portal_user", "create_ticket").decision == "escalate"
    limited = authorize_tool_call("n2", "create_ticket", tickets_created_in_session=3)
    assert limited.violations == ["limite_sesion"]


def test_role_without_permission_is_denied() -> None:
    decision = authorize_tool_call("portal_user", "classify_ticket", {"text": "lento"})
    assert decision.decision == "deny"
    assert decision.violations == ["rol_no_autorizado"]


# --- confirmación HITL ---------------------------------------------------------------


def _claim(**overrides: object) -> ConfirmationClaim:
    data: dict[str, object] = {
        "owner_user_id": "USR-0101",
        "session_id": "sess-12345678",
        "expires_at": datetime.now(UTC) + timedelta(minutes=5),
    }
    data.update(overrides)
    return ConfirmationClaim.model_validate(data)


def test_confirmation_ok() -> None:
    check = verify_confirmation(_claim(), user_id="USR-0101", session_id="sess-12345678")
    assert check.ok
    assert check.violations == []


def test_confirmation_accumulates_all_violations() -> None:
    claim = _claim(used=True, expires_at=datetime.now(UTC) - timedelta(seconds=1))
    check = verify_confirmation(claim, user_id="USR-0201", session_id="otra-sesion-99")
    assert not check.ok
    assert check.violations == [
        "token_ajeno",
        "sesion_distinta",
        "token_reutilizado",
        "token_expirado",
    ]


def test_confirmation_with_naive_expiry_is_treated_as_utc() -> None:
    naive = (datetime.now(UTC) + timedelta(minutes=1)).replace(tzinfo=None)
    check = verify_confirmation(
        _claim(expires_at=naive), user_id="USR-0101", session_id="sess-12345678"
    )
    assert check.ok


# --- política -----------------------------------------------------------------------


def test_policy_loads_from_roles_yaml() -> None:
    policy = permissions.get_policy()
    assert policy.hierarchy == ROLES
    assert policy.tools["create_ticket_escalated"] == ["portal_user"]


def test_missing_policy_falls_back_to_spec(tmp_path: Path) -> None:
    policy = load_policy(tmp_path / "no-existe.yaml")
    assert policy.tools["classify_ticket"] == ["n1", "n2", "admin"]


def test_corrupt_policy_fails_closed(tmp_path: Path) -> None:
    broken = tmp_path / "roles.yaml"
    broken.write_text("hierarchy: [portal_user, n1]\ntools: {search_documents: [god]}\n")
    with pytest.raises(ValueError, match="Política RBAC inválida"):
        load_policy(broken)
