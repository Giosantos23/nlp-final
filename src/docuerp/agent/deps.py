"""Dependencias inyectables del agente.

El grafo no importa directamente el recuperador, el LLM ni los guards: los recibe en un
:class:`AgentDeps`. Así las pruebas pasan *fakes* deterministas y la producción usa
:func:`build_default_deps`, que conecta las implementaciones reales de las otras ramas
(``docuerp.retrieval``, ``docuerp.llm``, ``docuerp.generation``, ``docuerp.security``).

Todo lo pesado es **perezoso**: construir las dependencias no carga modelos, no abre la BD
y no lanza el CLI del LLM; eso ocurre en la primera consulta que lo necesita.
"""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from docuerp.agent.router import IntentRouter, RouteDecision
from docuerp.agent.tools.base import ToolRegistry, default_registry
from docuerp.agent.tools.create_ticket import TicketDraftService, get_ticket_service
from docuerp.llm.contracts import LLMProvider, LLMResult, ProviderName
from docuerp.schemas import Answer, Intent, QueryRequest, ScoredChunk, UserRole
from docuerp.security.input_guard import GuardVerdict, SanitizedContext
from docuerp.security.output_guard import OutputVerdict

if TYPE_CHECKING:
    from docuerp.config import Settings


class RouterLike(Protocol):
    """Lo único que el grafo necesita del router."""

    def decide(self, text: str) -> RouteDecision:
        """Devuelve la decisión de intención para ``text``."""
        ...


class AuditSink(Protocol):
    """Destino de auditoría (``docuerp.security.audit.AuditLogger``)."""

    def log(self, trace: Any, *, request: QueryRequest | None = None) -> None:
        """Registra la traza sanitizada (con la solicitud: rol, usuario y consulta, que el
        logger enmascara o resume con HMAC)."""
        ...


InputGuardFn = Callable[[QueryRequest], GuardVerdict]
"""Firma de ``docuerp.security.input_guard.check_input``."""

OutputGuardFn = Callable[..., OutputVerdict]
"""Firma de ``check_output(answer, retrieved, *, tool_result=None, role=None)``."""

FilterAuthorizedFn = Callable[[UserRole, list[ScoredChunk]], tuple[list[ScoredChunk], int]]
"""Firma de ``docuerp.security.permissions.filter_authorized``."""

ContextGuardFn = Callable[[list[ScoredChunk]], SanitizedContext]
"""Firma de ``docuerp.security.input_guard.sanitize_context``: contexto sin la inyección
indirecta (oraciones retiradas o chunks en cuarentena)."""


class GenerateFn(Protocol):
    """Firma de ``docuerp.generation.generator.generate_answer``."""

    def __call__(
        self,
        question: str,
        hits: list[ScoredChunk],
        *,
        intent: Intent,
        tool_result: Any | None = None,
        provider: LLMProvider,
        model: str | None = None,
        erp_version: str | None = None,
    ) -> Awaitable[tuple[Answer, LLMResult]]:
        """Genera el ``Answer`` contractual y el ``LLMResult`` de consumo."""
        ...


class LazyProvider:
    """Proveedor LLM que se resuelve con ``docuerp.llm.get_provider`` en la primera llamada."""

    def __init__(self, name: ProviderName | None = None) -> None:
        self._name = name
        self._provider: LLMProvider | None = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Nombre configurado (o ``default`` si se usa el de ``settings``)."""
        return self._name or "default"

    def _resolve(self) -> LLMProvider:
        with self._lock:
            if self._provider is None:
                from docuerp.llm import get_provider

                self._provider = get_provider(self._name)
            return self._provider

    async def complete(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
        effort: str | None = None,
        timeout_s: float = 45.0,
    ) -> LLMResult:
        """Delega en el proveedor real."""
        return await self._resolve().complete(
            system,
            user,
            model=model,
            json_schema=json_schema,
            max_tokens=max_tokens,
            effort=effort,
            timeout_s=timeout_s,
        )


@dataclass
class AgentDeps:
    """Todo lo que el grafo necesita del mundo exterior."""

    settings: Settings
    router: RouterLike
    registry: ToolRegistry
    ticket_service: TicketDraftService
    input_guard: InputGuardFn
    output_guard: OutputGuardFn
    filter_authorized: FilterAuthorizedFn
    generator: GenerateFn
    provider: LLMProvider
    fallback_provider: LLMProvider | None
    audit: AuditSink
    context_guard: ContextGuardFn | None = None
    """Saneamiento del contexto ante inyección indirecta (``None`` = sin escaneo, p. ej.
    pruebas)."""


def _input_guard(req: QueryRequest) -> GuardVerdict:
    """Adaptador perezoso a ``docuerp.security.input_guard.check_input``."""
    from docuerp.security.input_guard import check_input

    return check_input(req)


def _output_guard(
    answer: Answer,
    retrieved: list[ScoredChunk],
    *,
    tool_result: Any | None = None,
    role: UserRole | None = None,
) -> OutputVerdict:
    """Adaptador perezoso a ``docuerp.security.output_guard.check_output`` (con el rol, cada
    cita se revalida contra el RBAC: defensa en profundidad)."""
    from docuerp.generation.prompts import build_system_prompt
    from docuerp.security.output_guard import check_output

    return check_output(
        answer,
        retrieved,
        tool_result=tool_result,
        role=role,
        system_prompt=build_system_prompt(),
    )


def _filter_authorized(role: UserRole, hits: list[ScoredChunk]) -> tuple[list[ScoredChunk], int]:
    """Adaptador perezoso a ``docuerp.security.permissions.filter_authorized``."""
    from docuerp.security.permissions import filter_authorized

    return filter_authorized(role, hits)


def _context_guard(hits: list[ScoredChunk]) -> SanitizedContext:
    """Adaptador perezoso a ``docuerp.security.input_guard.sanitize_context``."""
    from docuerp.security.input_guard import sanitize_context

    return sanitize_context(hits)


async def _generate(
    question: str,
    hits: list[ScoredChunk],
    *,
    intent: Intent,
    tool_result: Any | None = None,
    provider: LLMProvider,
    model: str | None = None,
    erp_version: str | None = None,
) -> tuple[Answer, LLMResult]:
    """Adaptador perezoso a ``docuerp.generation.generator.generate_answer``."""
    from docuerp.generation.generator import generate_answer

    return await generate_answer(
        question,
        hits,
        intent=intent,
        tool_result=tool_result,
        provider=provider,
        model=model,
        erp_version=erp_version,
    )


class _LazyAudit:
    """Auditoría real (``logs/audit.jsonl``) creada en el primer registro."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._logger: AuditSink | None = None

    def log(self, trace: Any, *, request: QueryRequest | None = None) -> None:
        if self._logger is None:
            from docuerp.security.audit import AuditLogger

            self._logger = AuditLogger(self._settings.paths.resolve("audit_log"))
        self._logger.log(trace, request=request)


def build_default_deps(settings: Settings | None = None) -> AgentDeps:
    """Dependencias reales (perezosas) según ``settings``."""
    from docuerp.config import get_settings

    active = settings or get_settings()
    ticket_service = get_ticket_service()
    fallback_name = active.llm.fallback_provider
    return AgentDeps(
        settings=active,
        router=IntentRouter(),
        registry=default_registry(ticket_service=ticket_service, settings=active),
        ticket_service=ticket_service,
        input_guard=_input_guard,
        output_guard=_output_guard,
        filter_authorized=_filter_authorized,
        generator=_generate,
        provider=LazyProvider(active.llm.provider),
        fallback_provider=LazyProvider(fallback_name) if fallback_name else None,
        audit=_LazyAudit(active),
        context_guard=_context_guard,
    )
