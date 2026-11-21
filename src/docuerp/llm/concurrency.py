"""Semáforo de concurrencia seguro entre bucles de eventos.

``asyncio.Semaphore`` queda ligado al bucle de eventos donde se usa por primera vez bajo
contención. En la CLI, en pruebas y en la API se crean varios bucles (``asyncio.run``), así
que se mantiene un semáforo **por bucle** y se limita igual el número de llamadas
simultáneas dentro de cada uno.
"""

from __future__ import annotations

import asyncio
import os
import weakref

DEFAULT_MAX_CONCURRENCY = 6
"""Valor medido en el sondeo: 3 a 8 consultas simultáneas sin errores ni 429."""


def int_from_env(variable: str, default: int) -> int:
    """Lee un entero positivo del entorno; si falta o es inválido devuelve ``default``."""
    raw = os.environ.get(variable, "").strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 1 else default


class LoopSemaphore:
    """Semáforo asíncrono que crea una instancia de ``asyncio.Semaphore`` por bucle."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("el límite de concurrencia debe ser ≥ 1")
        self.limit = limit
        self._by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )

    def _current(self) -> asyncio.Semaphore:
        """Devuelve (creándolo si hace falta) el semáforo del bucle en ejecución."""
        loop = asyncio.get_running_loop()
        semaphore = self._by_loop.get(loop)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self.limit)
            self._by_loop[loop] = semaphore
        return semaphore

    async def __aenter__(self) -> None:
        await self._current().acquire()

    async def __aexit__(self, *exc_info: object) -> None:
        self._current().release()
