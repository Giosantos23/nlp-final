"""Contrato y registro de herramientas (roles, límites, timeouts, degradación).

Cada herramienta se declara con un :class:`ToolSpec`: su contrato de argumentos (un modelo
Pydantic con ``extra="forbid"``), los roles que pueden usarla, cuántas veces puede llamarse
por consulta y por sesión, y su timeout. El :class:`ToolRegistry` es el único punto por
donde el agente invoca herramientas, así que ahí se concentran todos los controles:

1. la herramienta existe;
2. el rol está autorizado (RBAC);
3. no se superó el límite por consulta ni por sesión;
4. los argumentos cumplen el contrato;
5. la herramienta no está marcada como caída (simulación de indisponibilidad);
6. se ejecuta con timeout y cualquier excepción se convierte en ``ToolResult(ok=False)``.

``invoke`` **nunca lanza**: el grafo decide qué hacer con un resultado fallido
(degradación controlada) en lugar de romper la consulta completa.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import Awaitable, Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from docuerp.schemas import ROLE_LEVEL, ErpVersion, UserRole

if TYPE_CHECKING:
    from docuerp.agent.tools.classify import TicketClassifier
    from docuerp.agent.tools.create_ticket import TicketDraftService
    from docuerp.agent.tools.ticket_status import TicketReader
    from docuerp.config import Settings
    from docuerp.retrieval.contracts import Retriever

log = logging.getLogger(__name__)

UNAVAILABLE_ENV = "DOCUERP_TOOLS_UNAVAILABLE"
"""Variable de entorno para simular herramientas caídas: nombres separados por coma o
``all``. Ejemplo: ``DOCUERP_TOOLS_UNAVAILABLE=get_ticket_status``."""

UNAVAILABLE_MESSAGE = "La herramienta no está disponible en este momento."
"""Mensaje de error (en español) cuando la herramienta está caída o simulada como caída."""


class ToolResult(BaseModel):
    """Resultado de una herramienta. En degradación: ``ok=False`` y ``error`` descriptivo.

    ``data`` siempre es serializable a JSON (para la API y la traza). Convenciones de
    claves especiales que el grafo interpreta:

    * ``denied=True``: el registro rechazó la llamada por permisos (no es una caída).
    * ``invalid_args=True``: los argumentos no cumplen el contrato.
    * ``limit_exceeded=True``: se superó un límite de llamadas.
    * ``unavailable=True``: la herramienta está caída o hizo timeout.
    """

    model_config = ConfigDict(extra="forbid")

    tool: str
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    needs_human_review: bool = False
    latency_ms: float = Field(default=0.0, ge=0.0)


@dataclass(frozen=True)
class ToolContext:
    """Identidad y contexto de quien invoca la herramienta (lo fija el agente, no el LLM).

    Las herramientas usan este contexto —y no los argumentos— para decidir permisos, de
    modo que un argumento manipulado (p. ej. ``role="admin"``) no escala privilegios.
    """

    role: UserRole
    user_id: str
    session_id: str | None = None
    erp_version: ErpVersion = "18.0"

    @property
    def session_key(self) -> str:
        """Clave para contar llamadas por sesión (ver :func:`session_key_for`)."""
        return session_key_for(self.user_id, self.session_id)


def session_key_for(user_id: str, session_id: str | None) -> str:
    """Clave para contar por sesión; sin ``session_id`` se agrupa por usuario."""
    return session_id or f"user:{user_id}"


ToolHandler = Callable[[Any, ToolContext], Awaitable[ToolResult]]
"""Firma de un *handler*: recibe los argumentos ya validados y el contexto."""


class ToolSpec(BaseModel):
    """Declaración de una herramienta: contrato de entrada, roles y límites."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    description: str
    input_model: type[BaseModel]
    allowed_roles: list[UserRole]
    writes: bool = False
    requires_confirmation: bool = False
    timeout_s: float = Field(default=10.0, gt=0.0)
    max_calls_per_query: int = Field(default=1, ge=1)
    max_calls_per_session: int | None = Field(default=None, ge=1)
    handler: Callable[..., Any]


ESCALATED_SUFFIX = "_escalated"
"""En ``configs/roles.yaml``, ``<tool>_escalated`` lista roles que pueden pedir la
herramienta pero cuyo resultado se escala a revisión humana (p. ej. ``portal_user`` en
``create_ticket``)."""


@lru_cache(maxsize=1)
def _roles_policy() -> dict[str, list[str]]:
    """Sección ``tools`` de ``configs/roles.yaml`` (única fuente de verdad del RBAC)."""
    from docuerp import paths

    with paths.ROLES_FILE.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    tools = data.get("tools")
    if not isinstance(tools, dict):
        raise ValueError(f"{paths.ROLES_FILE} no define la sección 'tools'")
    return {str(name): [str(role) for role in roles] for name, roles in tools.items()}


def tool_roles(name: str) -> list[UserRole]:
    """Roles autorizados para ``name`` según ``configs/roles.yaml`` (incluye escalados).

    Raises:
        KeyError: si la herramienta no figura en la política.
        ValueError: si la política nombra un rol desconocido.
    """
    policy = _roles_policy()
    if name not in policy:
        raise KeyError(f"La herramienta {name!r} no está en configs/roles.yaml")
    roles = {*policy[name], *policy.get(f"{name}{ESCALATED_SUFFIX}", [])}
    unknown = sorted(roles - set(ROLE_LEVEL))
    if unknown:
        raise ValueError(f"Roles desconocidos para {name}: {unknown}")
    # Orden jerárquico (portal_user → admin) para que el listado sea legible.
    return sorted(roles, key=ROLE_LEVEL.__getitem__)  # type: ignore[return-value]


_SIMULATED_UNAVAILABLE: ContextVar[frozenset[str]] = ContextVar(
    "docuerp_simulated_unavailable", default=frozenset()
)
"""Caídas simuladas para la tarea o hilo actual (la evaluación las fija por caso)."""


@contextmanager
def simulate_unavailable(names: Iterable[str]) -> Iterator[None]:
    """Simula caídas solo dentro del bloque y del contexto actual.

    A diferencia de :data:`UNAVAILABLE_ENV`, no afecta a consultas concurrentes: las tareas
    de ``asyncio`` y ``asyncio.to_thread`` heredan una copia del contexto en que se crean.
    """
    token = _SIMULATED_UNAVAILABLE.set(frozenset(n.strip() for n in names if n.strip()))
    try:
        yield
    finally:
        _SIMULATED_UNAVAILABLE.reset(token)


def _env_unavailable() -> set[str]:
    """Lee :data:`UNAVAILABLE_ENV` y devuelve el conjunto de herramientas simuladas caídas."""
    raw = os.environ.get(UNAVAILABLE_ENV, "")
    return {name.strip() for name in raw.split(",") if name.strip()}


class ToolRegistry:
    """Registro de herramientas disponible para el agente.

    Lleva en memoria el conteo de llamadas por sesión (vive lo que vive el proceso). El
    conteo por consulta lo aporta quien invoca (el grafo) con ``calls_in_query``, porque el
    registro se comparte entre consultas concurrentes.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._session_calls: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()
        self.unavailable: set[str] = set()
        """Herramientas forzadas como caídas desde código (pruebas/demos)."""

    def register(self, spec: ToolSpec) -> None:
        """Registra una herramienta (``ValueError`` si el nombre ya existe)."""
        if spec.name in self._tools:
            raise ValueError(f"La herramienta {spec.name!r} ya está registrada")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        """Devuelve la especificación de ``name`` (``KeyError`` si no existe)."""
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Herramienta desconocida: {name!r}") from exc

    def names(self) -> list[str]:
        """Nombres registrados."""
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        """Especificaciones registradas, ordenadas por nombre."""
        return [self._tools[name] for name in self.names()]

    def is_unavailable(self, name: str) -> bool:
        """``True`` si la herramienta está simulada como caída (código o entorno)."""
        flagged = self.unavailable | _env_unavailable() | _SIMULATED_UNAVAILABLE.get()
        return name in flagged or "all" in flagged

    def session_calls(self, session_key: str, name: str) -> int:
        """Llamadas aceptadas de ``name`` en la sesión ``session_key``."""
        with self._lock:
            return self._session_calls.get((session_key, name), 0)

    def _reserve_session_call(self, spec: ToolSpec, session_key: str) -> bool:
        """Reserva un cupo de sesión de forma atómica; ``False`` si ya no quedan."""
        with self._lock:
            key = (session_key, spec.name)
            used = self._session_calls.get(key, 0)
            if spec.max_calls_per_session is not None and used >= spec.max_calls_per_session:
                return False
            self._session_calls[key] = used + 1
            return True

    async def invoke(
        self,
        name: str,
        args: dict[str, Any],
        *,
        role: UserRole,
        user_id: str,
        session_id: str | None = None,
        erp_version: ErpVersion = "18.0",
        calls_in_query: int = 0,
    ) -> ToolResult:
        """Valida argumentos y rol, aplica límites y timeout; devuelve ``ToolResult``.

        Nunca lanza: cualquier problema se reporta como ``ok=False`` con ``error`` en
        español y una clave en ``data`` que explica la causa (ver :class:`ToolResult`).

        Args:
            name: herramienta a invocar.
            args: argumentos crudos (se validan contra ``input_model``).
            role: rol del usuario autenticado (no el que diga ``args``).
            user_id: identificador del usuario.
            session_id: sesión, para el límite por sesión.
            erp_version: versión del ERP de la consulta.
            calls_in_query: cuántas veces ya se llamó ``name`` en esta consulta.
        """
        started = time.perf_counter()

        def _fail(error: str, **flags: Any) -> ToolResult:
            elapsed = (time.perf_counter() - started) * 1000
            return ToolResult(tool=name, ok=False, error=error, data=flags, latency_ms=elapsed)

        spec = self._tools.get(name)
        if spec is None:
            return _fail(f"Herramienta desconocida: {name}.", unknown_tool=True)
        if role not in spec.allowed_roles:
            return _fail(
                f"El rol {role} no tiene permiso para usar {name}.", denied=True, role=role
            )
        if calls_in_query >= spec.max_calls_per_query:
            return _fail(
                f"Se alcanzó el límite de {spec.max_calls_per_query} llamada(s) a {name} "
                "por consulta.",
                limit_exceeded=True,
                scope="query",
            )
        try:
            parsed = spec.input_model.model_validate(args)
        except ValidationError as exc:
            fields = sorted({".".join(str(p) for p in err["loc"]) for err in exc.errors()})
            return _fail(
                f"Argumentos inválidos para {name}: {', '.join(fields) or 'entrada'}.",
                invalid_args=True,
                fields=fields,
            )
        ctx = ToolContext(
            role=role, user_id=user_id, session_id=session_id, erp_version=erp_version
        )
        if self.is_unavailable(name):
            return _fail(UNAVAILABLE_MESSAGE, unavailable=True, simulated=True)
        if not self._reserve_session_call(spec, ctx.session_key):
            return _fail(
                f"Se alcanzó el límite de {spec.max_calls_per_session} llamada(s) a {name} "
                "por sesión.",
                limit_exceeded=True,
                scope="session",
            )
        try:
            result = await asyncio.wait_for(spec.handler(parsed, ctx), timeout=spec.timeout_s)
        except TimeoutError:
            log.warning("Timeout de %s tras %.1f s", name, spec.timeout_s)
            return _fail(
                f"La herramienta {name} no respondió en {spec.timeout_s:g} s.",
                unavailable=True,
                timeout=True,
            )
        except Exception as exc:  # degradación: ninguna herramienta tumba la consulta
            log.exception("Fallo inesperado en la herramienta %s", name)
            return _fail(UNAVAILABLE_MESSAGE, unavailable=True, error_type=type(exc).__name__)
        if not isinstance(result, ToolResult):
            return _fail(f"{name} devolvió un resultado inválido.", unavailable=True)
        elapsed = (time.perf_counter() - started) * 1000
        return result.model_copy(update={"latency_ms": elapsed, "tool": name})


def default_registry(
    *,
    retriever: Retriever | None = None,
    ticket_reader: TicketReader | None = None,
    classifier: TicketClassifier | None = None,
    ticket_service: TicketDraftService | None = None,
    settings: Settings | None = None,
) -> ToolRegistry:
    """Registro con las cuatro herramientas del proyecto.

    Cada dependencia se puede inyectar (pruebas con *fakes*); si se omite, se usa la
    implementación real, que se construye de forma **perezosa** (los modelos y la BD solo
    se tocan en la primera llamada). Importar o construir el registro no tiene efectos.
    ``settings`` fija timeouts y límites (por defecto, la configuración global).
    """
    from docuerp.agent.tools.classify import build_classify_spec, get_default_classifier
    from docuerp.agent.tools.create_ticket import build_create_ticket_spec, get_ticket_service
    from docuerp.agent.tools.search_documents import LazyRetriever, build_search_spec
    from docuerp.agent.tools.ticket_status import TicketReader, build_ticket_status_spec
    from docuerp.config import get_settings

    settings = settings or get_settings()
    registry = ToolRegistry()
    registry.register(build_search_spec(retriever or LazyRetriever(), settings=settings))
    registry.register(build_ticket_status_spec(ticket_reader or TicketReader(), settings=settings))
    registry.register(
        build_classify_spec(classifier or get_default_classifier(), settings=settings)
    )
    registry.register(
        build_create_ticket_spec(ticket_service or get_ticket_service(), settings=settings)
    )
    return registry
