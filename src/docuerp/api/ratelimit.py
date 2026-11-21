"""Límite de tasa simple por usuario (ventana deslizante en memoria).

Es un control de abuso básico (SPEC §5.6): un mismo ``user_id`` no puede lanzar más de
``limit`` consultas por ``window_s`` segundos. No reemplaza a un limitador distribuido; basta
para una demostración de un solo proceso.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable

MAX_TRACKED_KEYS = 10_000
"""Tope de usuarios rastreados; evita crecimiento sin límite de memoria."""


class SlidingWindowRateLimiter:
    """Cuenta peticiones por clave en una ventana de tiempo deslizante."""

    def __init__(
        self,
        limit: int,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit < 1:
            raise ValueError("limit debe ser >= 1")
        if window_s <= 0:
            raise ValueError("window_s debe ser > 0")
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> int | None:
        """Registra una petición de ``key``.

        Returns:
            ``None`` si se permite; si se excede el límite, los segundos (enteros, ≥ 1) que
            faltan para que se libere un cupo (valor para el encabezado ``Retry-After``).
        """
        now = self._clock()
        with self._lock:
            window = self._hits.setdefault(key, deque())
            while window and now - window[0] >= self.window_s:
                window.popleft()
            if len(window) >= self.limit:
                return max(1, math.ceil(self.window_s - (now - window[0])))
            window.append(now)
            if len(self._hits) > MAX_TRACKED_KEYS:
                self._prune(now)
            return None

    def _prune(self, now: float) -> None:
        """Elimina las claves sin actividad dentro de la ventana (se llama con el lock)."""
        stale = [k for k, w in self._hits.items() if not w or now - w[-1] >= self.window_s]
        for key in stale:
            del self._hits[key]
