"""Generador de respuestas fundamentadas (1 reintento si el JSON no valida; luego
abstención con ``needs_human_review``).

Flujo de :func:`generate_answer`:

1. **Atajos sin LLM** (ahorran costo y evitan riesgos): consulta fuera de dominio, herramienta
   fallida o ausente en una intención de ticket, o ausencia total de evidencia → abstención.
2. **Contexto compacto**: recorta los chunks al presupuesto de tokens, por orden del reranker.
3. **Llamada al LLM** con el esquema contractual. Si la respuesta no valida (JSON roto, citas
   inventadas, abstención incoherente, fuga del canario...) se reintenta una vez indicando los
   errores; si vuelve a fallar → abstención con ``needs_human_review``.
4. **Degradación controlada**: si el proveedor cae (timeout, autenticación, modelo inválido,
   servicio no disponible) se responde con el proveedor ``offline`` extractivo, marcando
   ``needs_human_review``.
5. Se **acumula el consumo** (tokens, costo, latencias, llamadas) de todos los intentos en el
   ``LLMResult`` devuelto.

El generador valida lo que puede verificar solo con los datos que recibe; la verificación
final (permisos, PII, coincidencia estricta con el índice) la hace ``security.output_guard``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from docuerp.config import get_settings
from docuerp.generation.context import (
    DEFAULT_TOKEN_BUDGET,
    ContextSelection,
    estimate_tokens,
    select_context,
)
from docuerp.generation.prompts import (
    ANSWER_JSON_SCHEMA,
    TOOL_BY_INTENT,
    ToolPayload,
    build_system_prompt,
    build_user_prompt,
    normalize_tool_result,
    render_fragment,
)
from docuerp.llm.contracts import LLMError, LLMProvider, LLMResult
from docuerp.llm.factory import get_provider
from docuerp.schemas import (
    ABSTENTION_MESSAGE,
    CANARY_TOKEN,
    Answer,
    Intent,
    ScoredChunk,
    SourceRef,
)

logger = logging.getLogger(__name__)

RETRY_RESERVE_TOKENS = 150
"""Tokens que se reservan del presupuesto para el bloque ``<correccion>`` del reintento."""

SUSPICIOUS_FLAG = "suspicious_instruction"
"""Bandera del escáner de ingesta: citar un chunk marcado obliga a revisión humana."""

MAX_VALIDATION_ERRORS = 5


def abstention(*, tool_used: str | None = None, needs_human_review: bool = False) -> Answer:
    """Respuesta de abstención contractual: mensaje fijo, sin fuentes."""
    return Answer(
        answer=ABSTENTION_MESSAGE,
        sources=[],
        abstained=True,
        tool_used=tool_used,
        needs_human_review=needs_human_review,
    )


def _empty_result(provider: LLMProvider, model: str) -> LLMResult:
    """Resultado de una generación que no necesitó llamar al LLM (``calls=0``)."""
    return LLMResult(text="", parsed=None, model=model, provider=provider.name, calls=0)


def _early_abstention(
    intent: Intent, hits: list[ScoredChunk], payload: ToolPayload | None
) -> Answer | None:
    """Decide si se puede abstenerse sin gastar una llamada al LLM; ``None`` si hay que generar."""
    if intent == "out_of_domain":
        return abstention()
    if payload is not None and not payload.ok:
        # La herramienta falló o denegó: no hay nada fiable que redactar.
        return abstention(tool_used=payload.name, needs_human_review=True)
    if payload is None and intent in TOOL_BY_INTENT:
        return abstention(tool_used=TOOL_BY_INTENT[intent], needs_human_review=True)
    if payload is None and not hits:
        return abstention()
    return None


def _aggregate(results: list[LLMResult], fallback: LLMResult) -> LLMResult:
    """Suma tokens, costo, latencias y llamadas de varios intentos en un solo ``LLMResult``.

    El texto, el JSON, el modelo y el proveedor son los del último intento. Si no hubo
    ningún resultado se devuelve ``fallback`` (generación sin llamadas).
    """
    if not results:
        return fallback
    last = results[-1]
    api_latencies = [r.api_latency_ms for r in results]
    return last.model_copy(
        update={
            "input_tokens": sum(r.input_tokens for r in results),
            "output_tokens": sum(r.output_tokens for r in results),
            "cache_read_tokens": sum(r.cache_read_tokens for r in results),
            "cache_creation_tokens": sum(r.cache_creation_tokens for r in results),
            "cost_usd": sum(r.cost_usd for r in results),
            "latency_ms": sum(r.latency_ms for r in results),
            "api_latency_ms": (
                None
                if any(a is None for a in api_latencies)
                else sum(a for a in api_latencies if a)
            ),
            "calls": sum(r.calls for r in results),
        }
    )


def _raw_object(result: LLMResult) -> dict[str, Any] | None:
    """Obtiene el objeto JSON de un resultado: ``parsed`` o, si falta, el texto parseado."""
    if result.parsed is not None:
        return result.parsed
    try:
        value = json.loads(result.text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def validate_answer(
    result: LLMResult,
    selection: ContextSelection,
    payload: ToolPayload | None,
    intent: Intent | None = None,
) -> tuple[Answer | None, list[str]]:
    """Valida y normaliza la salida del LLM.

    Comprobaciones: (1) es un objeto JSON que cumple el contrato ``Answer``; (2) no filtra el
    canario; (3) cada cita apunta a un chunk del contexto entregado; (4) una respuesta no
    abstenida sin herramienta lleva al menos una cita.

    Normalizaciones deterministas: ``source`` y ``page`` se copian de los metadatos del chunk
    citado (la clave es ``chunk_id``), se eliminan citas repetidas, una abstención se
    reescribe con el mensaje fijo y sin fuentes, y ``tool_used`` lo fija el sistema (el
    modelo no puede inventar una herramienta).

    Returns:
        ``(answer, [])`` si es válida o ``(None, errores)`` con mensajes para el reintento.
    """
    raw = _raw_object(result)
    if raw is None:
        return None, ["La respuesta no es un objeto con los campos del esquema."]
    try:
        answer = Answer.model_validate(raw)
    except ValidationError as exc:
        errors = [
            f"{'.'.join(str(part) for part in err['loc']) or 'respuesta'}: {err['msg']}"
            for err in exc.errors()
        ]
        return None, errors[:MAX_VALIDATION_ERRORS]
    if CANARY_TOKEN in answer.answer:
        return None, ["La respuesta contiene el token de control confidencial."]

    tool_used = payload.name if payload is not None else None
    if answer.abstained:
        return (
            abstention(tool_used=tool_used, needs_human_review=answer.needs_human_review),
            [],
        )
    if not answer.answer.strip():
        return None, ["`answer` está vacío y no se abstuvo."]

    known = {hit.chunk.chunk_id: hit.chunk for hit in selection.hits}
    errors = [
        f"El chunk_id citado no existe en <contexto>: {ref.chunk_id}"
        for ref in answer.sources
        if ref.chunk_id not in known
    ]
    if errors:
        return None, errors[:MAX_VALIDATION_ERRORS]
    if not answer.sources and payload is None:
        return None, ["Respuesta sin citas: cita los fragmentos usados o abstente."]

    sources: list[SourceRef] = []
    for ref in answer.sources:
        chunk = known[ref.chunk_id]
        canonical = SourceRef(source=chunk.source, page=chunk.page, chunk_id=chunk.chunk_id)
        if canonical not in sources:
            sources.append(canonical)
    review = answer.needs_human_review or (payload is not None and payload.needs_human_review)
    if any(SUSPICIOUS_FLAG in known[ref.chunk_id].flags for ref in sources):
        review = True
    if intent == "ticket_create":
        review = True  # el borrador queda pendiente de confirmación humana
    return (
        Answer(
            answer=answer.answer,
            sources=sources,
            abstained=False,
            tool_used=tool_used,
            needs_human_review=review,
        ),
        [],
    )


def _default_fallback(provider: LLMProvider) -> LLMProvider | None:
    """Proveedor de degradación de ``settings.llm.fallback_provider`` (si no es el primario)."""
    name = get_settings().llm.fallback_provider
    if name is None or name == provider.name:
        return None
    return get_provider(name)


async def _degrade(
    fallback: LLMProvider | None,
    system: str,
    user: str,
    *,
    model: str,
    selection: ContextSelection,
    payload: ToolPayload | None,
    intent: Intent,
    tool_used: str | None,
    cause: str,
) -> tuple[Answer, LLMResult | None]:
    """Intenta responder con el proveedor de degradación; si no se puede, abstiene."""
    if fallback is None:
        return abstention(tool_used=tool_used, needs_human_review=True), None
    logger.warning("LLM no disponible (%s): degradando al proveedor %s", cause, fallback.name)
    try:
        result = await fallback.complete(system, user, model=model, json_schema=ANSWER_JSON_SCHEMA)
    except LLMError as exc:
        logger.error("El proveedor de degradación también falló: %s", exc)
        return abstention(tool_used=tool_used, needs_human_review=True), None
    answer, errors = validate_answer(result, selection, payload, intent)
    if answer is None:
        logger.error("La respuesta de degradación no valida: %s", errors)
        return abstention(tool_used=tool_used, needs_human_review=True), result
    # Una respuesta sin LLM no está verificada por un modelo: siempre a revisión humana.
    return answer.model_copy(update={"needs_human_review": True}), result


async def generate_answer(
    question: str,
    hits: list[ScoredChunk],
    *,
    intent: Intent,
    tool_result: Any | None = None,
    provider: LLMProvider,
    model: str | None = None,
    erp_version: str | None = None,
    fallback: LLMProvider | None = None,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> tuple[Answer, LLMResult]:
    """Genera el ``Answer`` contractual a partir de la evidencia recuperada.

    Args:
        question: consulta del analista.
        hits: chunks autorizados, ordenados por el reranker (mejor primero).
        intent: intención decidida por el router.
        tool_result: resultado de la herramienta ejecutada (modelo, ``dict`` u objeto) o
            ``None``.
        provider: proveedor LLM principal.
        model: modelo a usar (por defecto ``settings.llm.model``).
        erp_version: versión del ERP de la consulta, para que el modelo la respete.
        fallback: proveedor de degradación; por defecto el de ``settings.llm.fallback_provider``.
        token_budget: presupuesto de tokens del prompt propio (por defecto 4 000).

    Returns:
        ``(Answer, LLMResult)``; el ``LLMResult`` acumula el consumo de todos los intentos
        (``calls`` = llamadas reales; ``to_usage()`` lo convierte a ``Usage``).
    """
    settings = get_settings().llm
    model = model or settings.model
    payload = normalize_tool_result(tool_result, intent)
    tool_used = payload.name if payload is not None else None

    early = _early_abstention(intent, hits, payload)
    if early is not None:
        return early, _empty_result(provider, model)

    system = build_system_prompt()
    skeleton = build_user_prompt(
        question, [], intent=intent, tool_result=tool_result, erp_version=erp_version
    )
    base_tokens = estimate_tokens(system) + estimate_tokens(skeleton) + RETRY_RESERVE_TOKENS
    selection = select_context(
        hits,
        base_tokens=base_tokens,
        budget=token_budget,
        render_tokens=lambda hit: estimate_tokens(render_fragment(hit)) + 1,
    )
    if selection.dropped or selection.truncated:
        logger.info(
            "Contexto recortado al presupuesto: %d descartados, %d truncados",
            len(selection.dropped),
            len(selection.truncated),
        )

    results: list[LLMResult] = []
    corrections: list[str] | None = None
    user = ""
    for attempt in range(settings.retries + 1):
        user = build_user_prompt(
            question,
            selection.hits,
            intent=intent,
            tool_result=tool_result,
            erp_version=erp_version,
            corrections=corrections,
        )
        try:
            result = await provider.complete(
                system,
                user,
                model=model,
                json_schema=ANSWER_JSON_SCHEMA,
                max_tokens=settings.max_tokens,
                effort=settings.effort,
                timeout_s=settings.timeout_s,
            )
        except LLMError as exc:
            if exc.cause == "schema":  # el LLM respondió, pero sin JSON utilizable: reintentar
                logger.warning("Intento %d sin JSON válido: %s", attempt + 1, exc)
                corrections = [str(exc)]
                continue
            degraded_answer, degraded = await _degrade(
                fallback if fallback is not None else _default_fallback(provider),
                system,
                user,
                model=model,
                selection=selection,
                payload=payload,
                intent=intent,
                tool_used=tool_used,
                cause=f"{exc.cause}: {exc}",
            )
            all_results = [*results, degraded] if degraded is not None else results
            return degraded_answer, _aggregate(all_results, _empty_result(provider, model))
        results.append(result)
        valid, errors = validate_answer(result, selection, payload, intent)
        if valid is not None:
            return valid, _aggregate(results, _empty_result(provider, model))
        logger.warning("Intento %d inválido: %s", attempt + 1, errors)
        corrections = errors

    # Agotados los intentos: abstención segura con revisión humana.
    return (
        abstention(tool_used=tool_used, needs_human_review=True),
        _aggregate(results, _empty_result(provider, model)),
    )
