"""Pruebas de los contratos compartidos (SPEC §4)."""

from __future__ import annotations

import unicodedata
from typing import Any

import pytest
from pydantic import ValidationError

from docuerp.schemas import (
    ABSTENTION_MESSAGE,
    CANARY_TOKEN,
    Answer,
    Chunk,
    PendingAction,
    QueryRequest,
    QueryResponse,
    ScoredChunk,
    SourceRef,
    TicketDraft,
    TraceEvent,
    Usage,
    is_ticket_id,
    make_chunk_id,
)


def _req(**overrides: Any) -> QueryRequest:
    base: dict[str, Any] = {
        "query": "¿Cómo valido una recepción?",
        "role": "n1",
        "user_id": "USR-0101",
    }
    base.update(overrides)
    return QueryRequest(**base)


# --------------------------------------------------------------------------- QueryRequest


class TestQueryRequest:
    def test_defaults(self) -> None:
        req = _req()
        assert req.erp_version == "18.0"
        assert req.session_id is None
        assert req.confirmation_token is None

    def test_query_is_stripped_and_nfc(self) -> None:
        decomposed = unicodedata.normalize("NFD", "  ¿Cómo configuro la facturación?  ")
        req = _req(query=decomposed)
        assert req.query == "¿Cómo configuro la facturación?"
        assert unicodedata.is_normalized("NFC", req.query)

    @pytest.mark.parametrize("query", ["", "  ", "ab", "  ab  "])
    def test_query_too_short(self, query: str) -> None:
        with pytest.raises(ValidationError):
            _req(query=query)

    def test_query_length_bounds(self) -> None:
        assert len(_req(query="x" * 2000).query) == 2000
        with pytest.raises(ValidationError):
            _req(query="x" * 2001)

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            _req(admin=True)

    @pytest.mark.parametrize("user_id", ["USR-0001", "USR-0101", "USR-9999"])
    def test_valid_user_ids(self, user_id: str) -> None:
        assert _req(user_id=user_id).user_id == user_id

    @pytest.mark.parametrize(
        "user_id", ["USR-001", "USR-00001", "usr-0001", "USR_0001", "USR-0001 ", "x", ""]
    )
    def test_invalid_user_ids(self, user_id: str) -> None:
        with pytest.raises(ValidationError):
            _req(user_id=user_id)

    @pytest.mark.parametrize("role", ["guest", "N1", "root", ""])
    def test_invalid_role(self, role: str) -> None:
        with pytest.raises(ValidationError):
            _req(role=role)

    @pytest.mark.parametrize("version", ["16.0", "18", "17.0 ", "any"])
    def test_invalid_erp_version(self, version: str) -> None:
        with pytest.raises(ValidationError):
            _req(erp_version=version)

    def test_session_id_pattern(self) -> None:
        assert _req(session_id="sess_abc-123").session_id == "sess_abc-123"
        for bad in ["short", "has space here", "a" * 65, "ñandú-123"]:
            with pytest.raises(ValidationError):
                _req(session_id=bad)


# --------------------------------------------------------------------------- Chunk


class TestChunk:
    def test_valid(self, chunk_payload: dict[str, Any]) -> None:
        chunk = Chunk(**chunk_payload)
        assert chunk.flags == []
        assert chunk.chunk_id == "odoo-18.0-inventario::p0012::c000"

    def test_make_chunk_id(self) -> None:
        assert make_chunk_id("RB-N2-003@v1.1", 3, 7) == "RB-N2-003@v1.1::p0003::c007"
        with pytest.raises(ValueError):
            make_chunk_id("doc", 0, 0)
        with pytest.raises(ValueError):
            make_chunk_id("doc", 1, 1000)

    def test_page_end_before_page(self, chunk_payload: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            Chunk(**{**chunk_payload, "page_end": 11})

    def test_chunk_id_must_match_doc(self, chunk_payload: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            Chunk(**{**chunk_payload, "chunk_id": "otro-doc::p0012::c000"})
        with pytest.raises(ValidationError):
            Chunk(**{**chunk_payload, "chunk_id": "odoo-18.0-inventario-12"})

    def test_hash_and_roles(self, chunk_payload: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            Chunk(**{**chunk_payload, "content_hash": "xyz"})
        with pytest.raises(ValidationError):
            Chunk(**{**chunk_payload, "allowed_roles": []})
        with pytest.raises(ValidationError):
            Chunk(**{**chunk_payload, "allowed_roles": ["superuser"]})

    def test_erp_version_any(self, chunk_payload: dict[str, Any]) -> None:
        assert Chunk(**{**chunk_payload, "erp_version": "any"}).erp_version == "any"

    def test_extra_forbidden(self, chunk_payload: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            Chunk(**{**chunk_payload, "embedding": [0.1]})

    def test_scored_chunk(self, chunk_payload: dict[str, Any]) -> None:
        sc = ScoredChunk(chunk=Chunk(**chunk_payload), score=0.8, rank=1, retriever="rerank")
        assert sc.stage_scores == {}
        with pytest.raises(ValidationError):
            ScoredChunk(chunk=Chunk(**chunk_payload), score=0.8, rank=0, retriever="rerank")
        with pytest.raises(ValidationError):
            ScoredChunk(chunk=Chunk(**chunk_payload), score=0.8, rank=1, retriever="splade")


# --------------------------------------------------------------------------- Answer (§11)


class TestAnswer:
    def test_exact_fields(self) -> None:
        assert set(Answer.model_fields) == {
            "answer",
            "sources",
            "abstained",
            "tool_used",
            "needs_human_review",
        }

    def test_all_fields_required(self) -> None:
        with pytest.raises(ValidationError):
            Answer(answer="x", sources=[], abstained=False, needs_human_review=False)  # type: ignore[call-arg]

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            Answer(
                answer="x",
                sources=[],
                abstained=False,
                tool_used=None,
                needs_human_review=False,
                confidence=0.9,  # type: ignore[call-arg]
            )

    def test_roundtrip_json(self) -> None:
        ans = Answer(
            answer="Vaya a Inventario > Recepciones.",
            sources=[
                SourceRef(
                    source="odoo-18.0-inventario.pdf",
                    page=12,
                    chunk_id="odoo-18.0-inventario::p0012::c000",
                )
            ],
            abstained=False,
            tool_used=None,
            needs_human_review=False,
        )
        dumped = ans.model_dump()
        assert list(dumped) == ["answer", "sources", "abstained", "tool_used", "needs_human_review"]
        assert Answer.model_validate_json(ans.model_dump_json()) == ans

    def test_source_ref_validation(self) -> None:
        with pytest.raises(ValidationError):
            SourceRef(source="a.pdf", page=0, chunk_id="a::p0001::c000")
        with pytest.raises(ValidationError):
            SourceRef(source="a.pdf", page=1, chunk_id="not-a-chunk-id")


# --------------------------------------------------------------------------- Tickets


class TestTickets:
    def _draft(self, **overrides: Any) -> TicketDraft:
        base: dict[str, Any] = {
            "summary": "Error al validar recepción parcial",
            "description": "Al validar aparece un error de stock negativo.",
            "category": "error_funcional",
            "priority": "alta",
            "module": "inventario",
        }
        base.update(overrides)
        return TicketDraft(**base)

    def test_valid_draft(self) -> None:
        draft = self._draft(summary="  Error al validar recepción parcial  ")
        assert draft.summary == "Error al validar recepción parcial"
        assert draft.account_id is None

    @pytest.mark.parametrize("summary", ["corto", "x" * 201])
    def test_summary_bounds(self, summary: str) -> None:
        with pytest.raises(ValidationError):
            self._draft(summary=summary)

    def test_description_max(self) -> None:
        with pytest.raises(ValidationError):
            self._draft(description="x" * 2001)

    @pytest.mark.parametrize(
        ("field", "value"),
        [("category", "otro"), ("priority", "urgente"), ("module", "rrhh")],
    )
    def test_closed_vocabularies(self, field: str, value: str) -> None:
        with pytest.raises(ValidationError):
            self._draft(**{field: value})

    def test_pending_action(self) -> None:
        pa = PendingAction(
            action="create_ticket",
            draft=self._draft(),
            confirmation_token="tok_" + "a" * 40,
            expires_at="2026-11-17T08:10:00Z",
        )
        assert pa.action == "create_ticket"
        with pytest.raises(ValidationError):
            PendingAction(
                action="delete_ticket",  # type: ignore[arg-type]
                draft=self._draft(),
                confirmation_token="tok_" + "a" * 40,
                expires_at="2026-11-17T08:10:00Z",
            )

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("TCK-00042", True), ("TCK-0042", False), ("tck-00042", False), ("TCK-000420", False)],
    )
    def test_ticket_id_pattern(self, value: str, expected: bool) -> None:
        assert is_ticket_id(value) is expected


# --------------------------------------------------------------------------- Telemetría


class TestTelemetry:
    def test_usage_add(self) -> None:
        total = Usage(llm_calls=1, input_tokens=100, output_tokens=20, cost_usd=0.01) + Usage(
            llm_calls=1, input_tokens=50, output_tokens=10, cost_usd=0.02
        )
        assert total.llm_calls == 2
        assert total.total_tokens == 180
        assert total.cost_usd == pytest.approx(0.03)

    def test_usage_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            Usage(input_tokens=-1)

    def test_trace_event(self) -> None:
        ev = TraceEvent(stage="retrieval.bm25", status="ok", duration_ms=3.2)
        assert ev.data == {}
        with pytest.raises(ValidationError):
            TraceEvent(stage="router", status="done", duration_ms=1.0)  # type: ignore[arg-type]

    def test_query_response(self) -> None:
        resp = QueryResponse(
            trace_id="trc_123",
            intent="out_of_domain",
            response=Answer(
                answer=ABSTENTION_MESSAGE,
                sources=[],
                abstained=True,
                tool_used=None,
                needs_human_review=False,
            ),
            latency_ms=12.5,
        )
        assert resp.blocked is False
        assert resp.usage == Usage()
        assert resp.events == []
        with pytest.raises(ValidationError):
            QueryResponse.model_validate({**resp.model_dump(), "intent": "chitchat"})


def test_constants() -> None:
    assert CANARY_TOKEN == "DOCUERP-CANARY-7f3a"
    assert ABSTENTION_MESSAGE
