"""Errores de la API con un formato JSON consistente: ``{error, detail, trace_id}``.

Toda respuesta de error, venga de una validación, de una ruta inexistente, de una regla de
negocio o de un fallo inesperado, usa el mismo cuerpo para que el frontend lo maneje con una
sola rutina. Los fallos inesperados nunca exponen el mensaje interno (podría contener datos
sensibles): se registra en el log del servidor y al cliente solo se le entrega el ``trace_id``.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("docuerp.api")

_STATUS_CODES: dict[int, str] = {
    400: "bad_request",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    502: "bad_gateway",
    503: "service_unavailable",
    504: "gateway_timeout",
}
"""Código de error estable por estado HTTP (cuando la ruta no define uno propio)."""


def new_request_id() -> str:
    """Identificador corto para correlacionar un error con el log del servidor."""
    return f"req-{uuid.uuid4().hex[:12]}"


class ApiError(Exception):
    """Error de negocio con estado HTTP, código estable y mensaje en español."""

    def __init__(
        self,
        status_code: int,
        error: str,
        detail: Any,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(f"{error}: {detail}")
        self.status_code = status_code
        self.error = error
        self.detail = detail
        self.headers = headers or {}


def request_id_of(request: Request) -> str:
    """Devuelve el id de la petición (lo asigna el middleware; se crea uno si falta)."""
    request_id = getattr(request.state, "request_id", None)
    if not request_id:
        request_id = new_request_id()
        request.state.request_id = request_id
    return str(request_id)


def error_body(error: str, detail: Any, trace_id: str) -> dict[str, Any]:
    """Cuerpo JSON uniforme de error."""
    return {"error": error, "detail": detail, "trace_id": trace_id}


def _validation_detail(exc: RequestValidationError) -> list[dict[str, str]]:
    """Resume los errores de validación SIN repetir el valor recibido (puede ser PII)."""
    return [
        {"loc": ".".join(str(p) for p in err.get("loc", ())), "msg": str(err.get("msg", ""))}
        for err in exc.errors()
    ]


def install_error_handlers(app: FastAPI) -> None:
    """Registra los manejadores de excepciones que producen el formato uniforme."""

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            error_body(exc.error, exc.detail, request_id_of(request)),
            status_code=exc.status_code,
            headers=exc.headers,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_CODES.get(exc.status_code, "http_error")
        return JSONResponse(
            error_body(code, str(exc.detail), request_id_of(request)),
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            error_body("validation_error", _validation_detail(exc), request_id_of(request)),
            status_code=422,
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        trace_id = request_id_of(request)
        logger.error("error inesperado trace_id=%s tipo=%s", trace_id, type(exc).__name__)
        return JSONResponse(
            error_body("internal_error", "Error interno del servidor.", trace_id),
            status_code=500,
        )
