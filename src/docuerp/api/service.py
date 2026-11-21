"""Fachada del agente que usa la API y errores de servicio.

La API depende solo del protocolo :class:`AgentService` (SPEC §11). En producción se usa
:class:`AgentFacade`, que delega en ``docuerp.agent.service`` importándolo de forma perezosa
(así la API arranca aunque el agente o sus modelos no estén disponibles, y devuelve 503 en
lugar de fallar al importar). En pruebas y en modo mock se inyecta otra implementación.
"""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterator
from types import ModuleType
from typing import Protocol

from docuerp.schemas import QueryRequest, QueryResponse, TraceEvent


class ServiceUnavailableError(Exception):
    """El agente no puede atender la petición (no implementado o dependencia ausente)."""


class ConfirmationRejectedError(Exception):
    """El token de confirmación es inválido, expiró, ya se usó o no corresponde al usuario."""


class AgentService(Protocol):
    """Contrato mínimo que la API necesita del agente."""

    async def answer_query(self, req: QueryRequest) -> QueryResponse:
        """Procesa una consulta completa."""
        ...

    def answer_query_stream(self, req: QueryRequest) -> AsyncIterator[TraceEvent | QueryResponse]:
        """Emite cada ``TraceEvent`` y, al final, el ``QueryResponse``."""
        ...

    async def confirm_action(self, token: str, req: QueryRequest) -> QueryResponse:
        """Confirma un borrador pendiente (HITL)."""
        ...


class AgentFacade:
    """Adaptador de ``docuerp.agent.service`` con importación perezosa."""

    @staticmethod
    def _module() -> ModuleType:
        """Importa el servicio del agente; si falla, informa indisponibilidad."""
        try:
            return importlib.import_module("docuerp.agent.service")
        except ImportError as exc:
            raise ServiceUnavailableError("El servicio del agente no está disponible.") from exc

    async def answer_query(self, req: QueryRequest) -> QueryResponse:
        """Delega en ``docuerp.agent.service.answer_query``."""
        module = self._module()
        try:
            response: QueryResponse = await module.answer_query(req)
        except NotImplementedError as exc:
            raise ServiceUnavailableError("El agente aún no está implementado.") from exc
        return response

    async def answer_query_stream(
        self, req: QueryRequest
    ) -> AsyncIterator[TraceEvent | QueryResponse]:
        """Usa ``answer_query_stream`` del agente; si no existe, simula el flujo.

        El respaldo ejecuta la consulta completa y reproduce los eventos de la respuesta
        (sin tiempo real, pero con el mismo formato) para que el cliente SSE siga funcionando.
        """
        module = self._module()
        stream = getattr(module, "answer_query_stream", None)
        try:
            if stream is None:
                response: QueryResponse = await module.answer_query(req)
                for event in response.events:
                    yield event
                yield response
                return
            async for item in stream(req):
                yield item
        except NotImplementedError as exc:
            raise ServiceUnavailableError("El agente aún no está implementado.") from exc

    async def confirm_action(self, token: str, req: QueryRequest) -> QueryResponse:
        """Delega en ``docuerp.agent.service.confirm_action``.

        El agente informa un token rechazado (inválido, vencido, reutilizado o de otro
        usuario) como respuesta bloqueada con ``block_reason="confirmacion_rechazada"``; aquí
        se traduce a :class:`ConfirmationRejectedError` para cumplir el contrato HTTP 409
        de la API. La auditoría ya quedó escrita por el agente.
        """
        module = self._module()
        try:
            response: QueryResponse = await module.confirm_action(token, req)
        except NotImplementedError as exc:
            raise ServiceUnavailableError("El agente aún no está implementado.") from exc
        if response.blocked and response.block_reason == REJECTED_REASON:
            raise ConfirmationRejectedError(response.response.answer)
        return response


REJECTED_REASON = "confirmacion_rechazada"
"""``block_reason`` con que el agente marca un token de confirmación rechazado."""
