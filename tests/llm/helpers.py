"""Utilidades compartidas de las pruebas de ``llm`` y ``generation``."""

from __future__ import annotations

from typing import Any

from claude_agent_sdk import ResultMessage

from docuerp.generation.demo import make_hit
from docuerp.schemas import ScoredChunk


def result_message(**overrides: Any) -> ResultMessage:
    """``ResultMessage`` válido de ejemplo; ``overrides`` reemplaza cualquier campo."""
    fields: dict[str, Any] = {
        "subtype": "success",
        "duration_ms": 2500,
        "duration_api_ms": 1200,
        "is_error": False,
        "num_turns": 1,
        "session_id": "sesion-de-prueba",
        "total_cost_usd": 0.0004,
        "usage": {
            "input_tokens": 2,
            "output_tokens": 50,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 1400,
        },
        "result": "pong",
        "structured_output": None,
    }
    fields.update(overrides)
    return ResultMessage(**fields)


def sample_hits() -> list[ScoredChunk]:
    """Dos chunks de ejemplo: uno relevante (recepciones) y uno de ventas."""
    return [
        make_hit(
            rank=1,
            doc_id="odoo-18.0-inventario",
            source="odoo-18.0-inventario.pdf",
            page=12,
            section_path="Inventario > Operaciones > Recepciones",
            text=(
                "Inventario > Operaciones > Recepciones\n"
                "Para validar una recepción abra Inventario y seleccione Recepciones. "
                "Haga clic en Validar para confirmar la recepción y actualizar el stock."
            ),
        ),
        make_hit(
            rank=2,
            doc_id="odoo-18.0-ventas",
            source="odoo-18.0-ventas.pdf",
            page=40,
            section_path="Ventas > Cotizaciones",
            score=0.5,
            text="Ventas > Cotizaciones\nUna cotización se confirma con el botón Confirmar.",
        ),
    ]


def good_answer(hits: list[ScoredChunk], **overrides: Any) -> dict[str, Any]:
    """Respuesta contractual válida que cita el primer chunk de ``hits``."""
    chunk = hits[0].chunk
    payload: dict[str, Any] = {
        "answer": "1. Abra Inventario > Recepciones. 2. Haga clic en Validar.",
        "sources": [{"source": chunk.source, "page": chunk.page, "chunk_id": chunk.chunk_id}],
        "abstained": False,
        "tool_used": None,
        "needs_human_review": False,
    }
    payload.update(overrides)
    return payload
