"""Registros que la evaluación escribe y lee de ``eval/results/<run_id>/``.

* :class:`PredictionRecord`: una línea de ``predictions.jsonl`` (un caso × una repetición).
* :class:`Judgment`: una línea de ``judgments.jsonl`` (veredicto del LLM-as-judge).
* :class:`RunConfig`: contenido de ``config.json``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from docuerp.schemas import Intent, SourceRef, Usage


class PredictionRecord(BaseModel):
    """Predicción del sistema para un caso en una repetición (``run``)."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    run: int = Field(ge=0)
    config: str
    question: str
    answer: str = ""
    sources: list[SourceRef] = Field(default_factory=list)
    abstained: bool = False
    tool_used: str | None = None
    needs_human_review: bool = False
    intent: Intent | None = None
    blocked: bool = False
    block_reason: str | None = None
    pending_action: bool = False
    latency_ms: float = Field(default=0.0, ge=0.0)
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    """Suma de ``duration_ms`` de los eventos de traza, por etapa (``stage``)."""
    stage_status: dict[str, str] = Field(default_factory=dict)
    """Último estado (``ok``, ``blocked``, ``error``…) de cada etapa."""
    api_latency_ms: float | None = None
    usage: Usage = Field(default_factory=Usage)
    trace_id: str | None = None
    error: str | None = None
    """Mensaje si el servicio falló para este caso (la evaluación continúa)."""

    @property
    def key(self) -> tuple[str, int]:
        """Identificador único ``(case_id, run)`` usado para reanudar."""
        return (self.case_id, self.run)


ClaimVerdictLabel = Literal["supported", "unsupported", "contradicted"]


class ClaimVerdict(BaseModel):
    """Veredicto del juez sobre una afirmación atómica de la respuesta."""

    model_config = ConfigDict(extra="forbid")

    claim: str
    verdict: ClaimVerdictLabel
    evidence_chunk_id: str | None = None
    reason: str = ""


class Judgment(BaseModel):
    """Veredicto completo del juez para una predicción."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    run: int
    judge_model: str
    faithfulness: float | None = None
    """Afirmaciones soportadas / afirmaciones totales (``None`` si no aplica)."""
    claims: list[ClaimVerdict] = Field(default_factory=list)
    correctness: int | None = Field(default=None, ge=0, le=2)
    correctness_rationale: str = ""
    coverage: float | None = None
    """Hechos clave cubiertos / hechos clave esperados (``None`` si no aplica)."""
    key_facts: list[dict[str, Any]] = Field(default_factory=list)
    citation_support: dict[str, bool] = Field(default_factory=dict)
    """``chunk_id → ¿el chunk citado respalda la respuesta?``"""
    notes: list[str] = Field(default_factory=list)
    judge_cost_usd: float = 0.0
    judge_calls: int = 0

    @property
    def key(self) -> tuple[str, int]:
        """Identificador único ``(case_id, run)``."""
        return (self.case_id, self.run)


class RunConfig(BaseModel):
    """Contenido de ``config.json``: qué se ejecutó y con qué parámetros."""

    model_config = ConfigDict(extra="allow")

    kind: Literal["e2e", "retrieval"]
    run_id: str
    name: str
    """``baseline`` / ``final`` (e2e) o etiqueta libre (retrieval)."""
    created_at: str
    split: str | None = None
    runs: int = 1
    seed: int = 42
    generator_model: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    configs: list[str] = Field(default_factory=list)
    """Para ``retrieval``: configuraciones evaluadas (``modo/chunker/embedder``)."""
    case_ids: list[str] = Field(default_factory=list)
