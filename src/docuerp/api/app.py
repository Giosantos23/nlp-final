"""Aplicación FastAPI en ``:8000`` con CORS para ``:5173``.

Rutas: ``POST /api/query``, ``POST /api/query/stream`` (SSE: un evento por ``TraceEvent`` y
al final ``QueryResponse``), ``POST /api/actions/confirm``, ``GET /api/trace/{trace_id}``,
``GET /api/documents``, ``GET /api/documents/{source}/file``, ``GET /api/chunks/{chunk_id}``,
``GET /api/eval/summary``, ``GET /api/eval/runs``, ``GET /api/security/summary``,
``GET /api/users``, ``GET /api/health``.

Principios:

* La API es una capa delgada: valida con los modelos de ``schemas`` y delega el trabajo en un
  ``AgentService`` (SPEC §11) y en un ``DataSource`` de solo lectura. Ambos se inyectan, de
  modo que las pruebas y el modo mock (``DOCUERP_API_MOCK=1``) no necesitan modelos.
* Todo error responde ``{error, detail, trace_id}``; los fallos internos no filtran detalles.
* Control de abuso: límite de tasa por ``user_id`` (30/min por defecto) en las rutas que
  invocan al agente. El log nunca incluye el texto de la consulta (sin PII).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import FastAPI, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from docuerp import __version__
from docuerp.api.datasources import ApiPaths, DataSource, FileDataSource
from docuerp.api.errors import (
    ApiError,
    error_body,
    install_error_handlers,
    new_request_id,
    request_id_of,
)
from docuerp.api.mock import MockDataSource, MockService
from docuerp.api.models import (
    ConfirmRequest,
    DocumentList,
    HealthResponse,
    TraceLookup,
    UserList,
)
from docuerp.api.ratelimit import SlidingWindowRateLimiter
from docuerp.api.service import (
    AgentFacade,
    AgentService,
    ConfirmationRejectedError,
    ServiceUnavailableError,
)
from docuerp.api.traces import TraceStore
from docuerp.api.warmup import Warmup
from docuerp.config import Settings, get_settings
from docuerp.schemas import QueryRequest, QueryResponse, TraceEvent, UserRole

logger = logging.getLogger("docuerp.api")

TRACE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
"""Forma aceptada de un ``trace_id`` en la URL."""

DEFAULT_RATE_LIMIT = 30
"""Consultas por minuto y por ``user_id``."""

_TRUTHY = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    """Interpreta una variable de entorno booleana (1/true/yes/on)."""
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _env_number(name: str, default: float) -> float:
    """Lee un número de una variable de entorno; si es inválido usa ``default``."""
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


class RequestContextMiddleware:
    """Middleware ASGI: asigna un id a cada petición, registra su resultado y añade cabeceras.

    Se escribe como ASGI puro (y no con ``BaseHTTPMiddleware``) para no interferir con el
    streaming SSE. Registra solo método, ruta, estado y duración: nunca cuerpos ni consultas.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = new_request_id()
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-Id"] = request_id
                headers["X-Content-Type-Options"] = "nosniff"
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            logger.info(
                "%s %s -> %s en %.0f ms [%s]",
                scope["method"],
                scope["path"],
                status,
                (time.perf_counter() - started) * 1000,
                request_id,
            )


def _map_exception(exc: Exception) -> ApiError:
    """Traduce una excepción del servicio al error HTTP uniforme (sin filtrar detalles)."""
    if isinstance(exc, ApiError):
        return exc
    if isinstance(exc, ServiceUnavailableError):
        return ApiError(503, "service_unavailable", str(exc))
    if isinstance(exc, ConfirmationRejectedError):
        return ApiError(409, "confirmation_rejected", str(exc))
    if isinstance(exc, PermissionError):
        return ApiError(403, "forbidden", "No tienes permiso para realizar esta acción.")
    if isinstance(exc, TimeoutError):
        return ApiError(504, "gateway_timeout", "El agente tardó demasiado en responder.")
    logger.error("fallo del servicio tipo=%s", type(exc).__name__)
    return ApiError(500, "internal_error", "Error interno al procesar la consulta.")


def create_app(
    *,
    service: AgentService | None = None,
    data: DataSource | None = None,
    settings: Settings | None = None,
    mock: bool | None = None,
    rate_limit_per_min: int | None = None,
    trace_capacity: int = 500,
    request_timeout_s: float = 120.0,
    warmup: bool | None = None,
) -> FastAPI:
    """Construye la aplicación ``FastAPI`` con todas las rutas registradas.

    Args:
        service: servicio del agente (por defecto :class:`AgentFacade`, o el mock).
        data: fuente de datos de solo lectura (por defecto archivos reales, o el mock).
        settings: configuración (por defecto ``get_settings()``).
        mock: fuerza el modo mock; si es ``None`` se lee ``DOCUERP_API_MOCK``.
        rate_limit_per_min: consultas por minuto y usuario (``DOCUERP_API_RATE_LIMIT`` o 30).
        trace_capacity: tamaño de la caché LRU de trazas.
        request_timeout_s: tiempo máximo que se espera al agente por consulta o por evento.
        warmup: precarga modelos e índices en segundo plano al arrancar. Por defecto solo con
            el agente real (sin ``service`` inyectado, fuera del mock) y si
            ``DOCUERP_API_WARMUP`` no vale ``0``.
    """
    settings = settings or get_settings()
    is_mock = _env_flag("DOCUERP_API_MOCK") if mock is None else mock
    if warmup is None:
        flag = os.environ.get("DOCUERP_API_WARMUP", "1").strip().lower()
        warmup = service is None and not is_mock and flag not in {"0", "false", "no", "off"}
    warm = Warmup(enabled=warmup)
    if service is None:
        service = (
            MockService(delay_scale=_env_number("DOCUERP_API_MOCK_DELAY_SCALE", 1.0))
            if is_mock
            else AgentFacade()
        )
    if data is None:
        data = MockDataSource() if is_mock else FileDataSource(ApiPaths.from_settings(settings))
    limit = rate_limit_per_min or int(_env_number("DOCUERP_API_RATE_LIMIT", DEFAULT_RATE_LIMIT))
    limiter = SlidingWindowRateLimiter(limit=limit, window_s=60.0)
    traces = TraceStore(capacity=trace_capacity)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Lanza el precalentamiento sin bloquear el arranque (la API responde mientras)."""
        task = asyncio.create_task(warm.run()) if warm.enabled else None
        try:
            yield
        finally:
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(
        title="DocuERP API",
        version=__version__,
        description="Asistente RAG agéntico para soporte funcional de Odoo.",
        lifespan=lifespan,
    )
    app.state.warmup = warm
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        expose_headers=["X-Request-Id", "Retry-After", "X-Rendition-Of"],
    )
    app.add_middleware(RequestContextMiddleware)
    install_error_handlers(app)

    def throttle(user_id: str) -> None:
        """Aplica el límite de tasa; si se excede, responde 429 con ``Retry-After``."""
        retry_after = limiter.hit(user_id)
        if retry_after is not None:
            raise ApiError(
                429,
                "rate_limited",
                f"Demasiadas consultas. Reintenta en {retry_after} s.",
                headers={"Retry-After": str(retry_after)},
            )

    def remember(response: QueryResponse, user_id: str) -> QueryResponse:
        """Guarda la traza en la caché y registra un resumen sin PII."""
        traces.put(response, user_id)
        logger.info(
            "consulta trace_id=%s intent=%s bloqueada=%s abstuvo=%s latencia_ms=%.0f",
            response.trace_id,
            response.intent,
            response.blocked,
            response.response.abstained,
            response.latency_ms,
        )
        return response

    # -- consultas ------------------------------------------------------------------------

    @app.post("/api/query", response_model=QueryResponse)
    async def query(req: QueryRequest) -> QueryResponse:
        """Procesa una consulta de extremo a extremo."""
        throttle(req.user_id)
        try:
            response = await asyncio.wait_for(service.answer_query(req), request_timeout_s)
        except Exception as exc:
            raise _map_exception(exc) from exc
        return remember(response, req.user_id)

    @app.post("/api/query/stream")
    async def query_stream(req: QueryRequest, request: Request) -> EventSourceResponse:
        """Como ``/api/query`` pero con SSE: ``event: trace`` por etapa y ``event: final``."""
        throttle(req.user_id)
        iterator = service.answer_query_stream(req).__aiter__()

        async def next_item() -> TraceEvent | QueryResponse | None:
            """Siguiente elemento del agente (``None`` al terminar), con tiempo límite."""
            try:
                return await asyncio.wait_for(iterator.__anext__(), request_timeout_s)
            except StopAsyncIteration:
                return None

        try:
            first = await next_item()
        except Exception as exc:
            raise _map_exception(exc) from exc
        if first is None:
            raise ApiError(502, "empty_stream", "El agente no produjo ninguna respuesta.")
        trace_id = request_id_of(request)

        async def events() -> AsyncIterator[dict[str, str]]:
            item: TraceEvent | QueryResponse | None = first
            try:
                while item is not None:
                    if isinstance(item, TraceEvent):
                        yield {"event": "trace", "data": item.model_dump_json()}
                    elif isinstance(item, QueryResponse):
                        yield {
                            "event": "final",
                            "data": remember(item, req.user_id).model_dump_json(),
                        }
                    item = await next_item()
            except Exception as exc:
                api_error = _map_exception(exc)
                body = error_body(api_error.error, api_error.detail, trace_id)
                yield {"event": "error", "data": json.dumps(body, ensure_ascii=False)}
            finally:
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    await aclose()

        return EventSourceResponse(events(), ping=15)

    @app.post("/api/actions/confirm", response_model=QueryResponse)
    async def confirm(body: ConfirmRequest) -> QueryResponse:
        """Confirma (HITL) un borrador pendiente con su token de un solo uso."""
        throttle(body.user_id)
        req = QueryRequest(
            query="Confirmar acción pendiente",
            role=body.role,
            user_id=body.user_id,
            session_id=body.session_id,
            confirmation_token=body.confirmation_token,
        )
        try:
            response = await asyncio.wait_for(
                service.confirm_action(body.confirmation_token, req), request_timeout_s
            )
        except Exception as exc:
            raise _map_exception(exc) from exc
        return remember(response, body.user_id)

    @app.get("/api/trace/{trace_id}", response_model=TraceLookup)
    def get_trace(
        trace_id: str,
        user_id: str | None = Query(default=None, pattern=r"^USR-\d{4}$"),
    ) -> TraceLookup:
        """Devuelve una traza de la caché LRU o, si ya salió de ella, de ``audit.jsonl``.

        Si se pasa ``user_id`` solo se entrega la traza de ese usuario (si no, 404).
        """
        if not TRACE_ID_RE.fullmatch(trace_id):
            raise ApiError(400, "invalid_trace_id", "Identificador de traza no válido.")
        stored = traces.get(trace_id)
        if stored is not None:
            if user_id is not None and stored.user_id != user_id:
                raise ApiError(404, "trace_not_found", "Traza no encontrada.")
            return TraceLookup(trace_id=trace_id, origin="memory", response=stored.response)
        record = data.audit_record(trace_id)
        if record is None or (user_id is not None and record.get("user_id") not in (None, user_id)):
            raise ApiError(404, "trace_not_found", "Traza no encontrada.")
        return TraceLookup(trace_id=trace_id, origin="audit_log", audit=record)

    # -- corpus ---------------------------------------------------------------------------

    @app.get("/api/documents", response_model=DocumentList)
    def list_documents(role: UserRole = "portal_user") -> DocumentList:
        """Manifiesto del corpus visible para ``role`` (por defecto el de menor privilegio)."""
        documents = data.documents(role)
        return DocumentList(
            count=len(documents), documents=documents, stats=data.corpus_stats(documents)
        )

    @app.get("/api/documents/{source}/file")
    def get_document_file(
        source: str,
        role: UserRole = "portal_user",
        format: Literal["auto", "original"] = "auto",
    ) -> Response:
        """Sirve el archivo de ``source`` (nombre validado de forma estricta, con RBAC).

        Un ``.docx`` se entrega como su PDF equivalente (misma paginación) para el visor;
        ``format=original`` entrega el DOCX. La cabecera ``X-Rendition-Of`` lo indica.
        """
        found = data.document_file(source, role, original=format == "original")
        headers = {"Cache-Control": "private, max-age=300"}
        if found.rendition_of:
            headers["X-Rendition-Of"] = found.rendition_of
        if found.path is not None:
            return FileResponse(
                found.path,
                media_type=found.media_type,
                filename=found.filename,
                content_disposition_type="inline",
                headers=headers,
            )
        return Response(
            content=found.content or b"",
            media_type=found.media_type,
            headers={**headers, "Content-Disposition": f'inline; filename="{found.filename}"'},
        )

    @app.get("/api/chunks/{chunk_id}")
    def get_chunk(chunk_id: str, role: UserRole = "portal_user") -> dict[str, Any]:
        """Devuelve un chunk completo si ``role`` está autorizado a leerlo."""
        return data.chunk(chunk_id, role)

    # -- evaluación y seguridad -----------------------------------------------------------

    @app.get("/api/eval/summary")
    def eval_summary(run_id: str | None = None) -> dict[str, Any]:
        """Resumen de la corrida de evaluación más reciente (o de ``run_id``)."""
        return data.eval_summary(run_id)

    @app.get("/api/eval/runs")
    def eval_runs() -> dict[str, Any]:
        """Todas las corridas de evaluación con sus métricas."""
        return data.eval_runs()

    @app.get("/api/security/summary")
    def security_summary() -> dict[str, Any]:
        """Matriz red-team: resultados por vector y por caso."""
        return data.security_summary()

    # -- soporte --------------------------------------------------------------------------

    @app.get("/api/users", response_model=UserList)
    def list_users() -> UserList:
        """Usuarios demo (desde la BD de tickets; lista fija si aún no existe)."""
        return UserList(users=data.users())

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Estado de índices, BD de tickets, proveedor LLM y modelos configurados."""
        ready = data.readiness()
        return HealthResponse(
            status="ok" if is_mock or all(ready.values()) else "degraded",
            version=__version__,
            mock=is_mock,
            llm_provider="mock" if is_mock else settings.llm.provider,
            models={
                "llm": settings.llm.model,
                "embedder": settings.retrieval.embedder,
                "reranker": settings.models.reranker,
            },
            index_ready=ready["index_ready"],
            tickets_db_ready=ready["tickets_db_ready"],
            traces_cached=len(traces),
            warmup=warm.as_dict(),
        )

    return app
