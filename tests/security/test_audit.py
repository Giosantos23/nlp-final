"""Auditoría: redacción de PII y secretos, registro por consulta y rotación por tamaño."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docuerp.schemas import (
    CANARY_TOKEN,
    Answer,
    QueryRequest,
    QueryResponse,
    SourceRef,
    TraceEvent,
    Usage,
)
from docuerp.security.audit import (
    REDACTED,
    AuditLogger,
    build_record,
    hash_identifier,
    sanitize,
)

QUERY = "Soy ana@volcanes.com, tel. +502 2345-6789: ¿cómo valido la recepción de TCK-00042?"


def _request() -> QueryRequest:
    return QueryRequest(query=QUERY, role="n1", user_id="USR-0101", session_id="sess-12345678")


def _response(trace_id: str = "trace-001") -> QueryResponse:
    return QueryResponse(
        trace_id=trace_id,
        intent="doc_query",
        response=Answer(
            answer=f"Llame a +502 2345-6789. {CANARY_TOKEN}",
            sources=[
                SourceRef(
                    source="odoo-18.0-inventario.pdf",
                    page=12,
                    chunk_id="odoo-18.0-inventario::p0012::c000",
                )
            ],
            abstained=False,
            tool_used="search_documents",
            needs_human_review=False,
        ),
        latency_ms=1234.5,
        usage=Usage(llm_calls=1, input_tokens=900, output_tokens=120, cost_usd=0.0005),
        events=[
            TraceEvent(
                stage="input_guard",
                status="ok",
                duration_ms=3.2,
                data={"allowed": True, "risk_score": 0.1, "flags": ["pii:email"]},
            ),
            TraceEvent(
                stage="rerank",
                status="ok",
                duration_ms=80.0,
                data={
                    "candidates": [
                        {"chunk_id": f"odoo::p{i:04d}::c000", "score": 1 - i / 100, "text": "x"}
                        for i in range(30)
                    ],
                    "filters": {"status": "vigente"},
                },
            ),
            TraceEvent(
                stage="tool.create_ticket",
                status="ok",
                duration_ms=5.0,
                data={"decision": "needs_confirmation", "confirmation_token": "abc.def.ghi"},
            ),
        ],
    )


# --- sanitización ---------------------------------------------------------------------


def test_sanitize_redacts_secret_keys_and_keeps_metrics() -> None:
    record = sanitize(
        {
            "api_key": "sk-123",
            "Authorization": "Bearer x",
            "confirmation_token": "abc",
            "password": "hunter2",
            "nested": {"hmac_secret": "s", "input_tokens": 10, "total_tokens": 99},
            "token_count": 5,
            "missing_token": None,
        }
    )
    assert record["api_key"] == REDACTED
    assert record["Authorization"] == REDACTED
    assert record["confirmation_token"] == REDACTED
    assert record["password"] == REDACTED
    assert record["nested"]["hmac_secret"] == REDACTED
    assert record["nested"]["input_tokens"] == 10
    assert record["nested"]["total_tokens"] == 99
    assert record["token_count"] == 5
    assert record["missing_token"] is None


def test_sanitize_masks_pii_removes_canary_and_truncates() -> None:
    out = sanitize(["ana@volcanes.com", f"x {CANARY_TOKEN}", "a" * 3000, 3, True, ("t",)])
    assert out[0] == "[EMAIL]"
    assert CANARY_TOKEN not in out[1]
    assert "[CANARY]" in out[1]
    assert out[2].endswith("…[recortado]")
    assert len(out[2]) < 2100
    assert out[3:] == [3, True, ["t"]]


def test_hash_identifier_is_stable_and_keyed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOCUERP_HMAC_SECRET", raising=False)
    first = hash_identifier("USR-0101")
    assert first == hash_identifier("USR-0101")
    assert first is not None
    assert len(first) == 16
    assert "USR" not in first
    monkeypatch.setenv("DOCUERP_HMAC_SECRET", "otra-clave")
    assert hash_identifier("USR-0101") != first
    assert hash_identifier(None) is None


# --- registro -----------------------------------------------------------------------


def test_build_record_summarizes_trace() -> None:
    record = build_record(_response(), request=_request())
    assert record["intent"] == "doc_query"
    assert record["role"] == "n1"
    assert record["filters"] == {"role": "n1", "erp_version": "18.0", "status": "vigente"}
    assert len(record["candidates"]["rerank"]) == 20  # máximo por etapa
    assert set(record["candidates"]["rerank"][0]) == {"chunk_id", "score"}  # sin texto
    assert record["guards"]["input_guard"]["risk_score"] == 0.1
    assert record["guards"]["tool.create_ticket"]["decision"] == "needs_confirmation"
    assert record["latencies_ms"]["total"] == 1234.5
    assert record["sources"] == ["odoo-18.0-inventario::p0012::c000"]


def test_logger_writes_sanitized_line(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path / "audit.jsonl")
    written = logger.log(_response(), request=_request())
    raw = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    for secret in ("ana@volcanes.com", "2345-6789", "USR-0101", "sess-12345678", CANARY_TOKEN):
        assert secret not in raw
    assert "[EMAIL]" in written["query_masked"]
    assert "[TELEFONO]" in written["answer_masked"]
    assert "TCK-00042" in written["query_masked"]
    assert json.loads(raw.strip()) == written
    assert logger.find("trace-001") == written
    assert logger.find("nope") is None


def test_logger_accepts_dicts_and_rejects_other_types(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path / "a.jsonl")
    record = logger.log({"trace_id": "d1", "request": {"role": "n2", "query": "hola"}})
    assert record["role"] == "n2"
    with pytest.raises(TypeError):
        logger.log("no soy una traza")


def test_iter_records_skips_corrupt_lines(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    logger = AuditLogger(path)
    logger.log(_response("t1"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{esto no es json\n\n")
    logger.log(_response("t2"))
    assert [r["trace_id"] for r in logger.iter_records()] == ["t1", "t2"]
    assert [r["trace_id"] for r in logger.tail(1)] == ["t2"]
    assert logger.tail(0) == []


def test_missing_file_yields_nothing(tmp_path: Path) -> None:
    assert list(AuditLogger(tmp_path / "nada.jsonl").iter_records()) == []


# --- rotación -----------------------------------------------------------------------


def test_rotation_keeps_n_backups(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path, max_bytes=3000, backups=2)
    for i in range(8):
        logger.log(_response(f"t{i}"))
    assert path.exists()
    assert (tmp_path / "audit.jsonl.1").exists()
    assert (tmp_path / "audit.jsonl.2").exists()
    assert not (tmp_path / "audit.jsonl.3").exists()
    assert path.stat().st_size <= 3000
    # El registro más reciente queda en el archivo vigente.
    assert logger.tail(1)[0]["trace_id"] == "t7"


def test_rotation_without_backups_truncates(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    logger = AuditLogger(path, max_bytes=3000, backups=0)
    for i in range(4):
        logger.log(_response(f"t{i}"))
    assert [r["trace_id"] for r in logger.iter_records()] == ["t3"]
    assert not (tmp_path / "audit.jsonl.1").exists()


def test_invalid_rotation_params() -> None:
    with pytest.raises(ValueError):
        AuditLogger(Path("x.jsonl"), max_bytes=0)
    with pytest.raises(ValueError):
        AuditLogger(Path("x.jsonl"), backups=-1)
