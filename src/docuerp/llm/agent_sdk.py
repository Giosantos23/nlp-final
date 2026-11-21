"""Proveedor ``agent_sdk``: ``claude-agent-sdk`` (≥0.2.183) con autenticación por suscripción.

Exporta ``CLAUDE_CODE_OAUTH_TOKEN`` desde ``.env`` al entorno del subproceso y elimina
``CLAUDECODE`` antes de lanzar el CLI. Opciones: ``tools=[]``, ``setting_sources=[]``,
``permission_mode="dontAsk"``, ``max_turns`` mínimo y ``output_format`` JSON Schema si se pide.
Lee ``ResultMessage`` (``usage``, ``total_cost_usd``, ``duration_api_ms``).

Hallazgos medidos del sondeo (``docs/research/agent_sdk.md``) que condicionan el diseño:

* No existe ``max_tokens``: se limita con ``CLAUDE_CODE_MAX_OUTPUT_TOKENS`` y un valor menor
  de 2048 hace fallar respuestas largas, por eso se usa ``max(max_tokens, 2048)``.
* Con ``output_format`` el JSON validado llega en ``ResultMessage.structured_output``.
* Un modelo inválido lanza ``ResultError`` (no devuelve ``ResultMessage``); un timeout se
  corta con ``asyncio.timeout`` y el transporte cierra el subproceso.
* Opcional (bandera ``use_client_pool``): reutilizar ``ClaudeSDKClient`` y aislar cada consulta
  con ``/clear`` ahorra el arranque del CLI. Medido aquí con Haiku, effort low, contexto de
  ~3k tokens y 6 consultas seguidas: pared P50 4.24 s con ``query()`` por llamada frente a
  2.74 s con el pool (-35 %); la API P50 queda igual (~2.7 s). Tras ``/clear`` el modelo no
  recuerda la consulta anterior (verificado). Ver :class:`_ClientPool`.

Hallazgo adicional: con salida estructurada Haiku falla entre ~12 % y ~35 % de las llamadas
pegando el parámetro ``sources`` dentro de ``answer`` (``</answer><parameter name=...>``);
el CLI lo rechaza y repite el mismo error en cada turno. Se usa ``max_turns=1`` para fallar
rápido y :func:`repair_structured_output` recupera el JSON (24/24 llamadas válidas tras la
reparación); lo irreparable se tipa como ``schema`` y el generador reintenta.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import time
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ClaudeSDKError,
    CLIConnectionError,
    CLINotFoundError,
    ResultError,
    ResultMessage,
    ToolUseBlock,
    query,
)
from dotenv import dotenv_values

from docuerp import paths
from docuerp.llm.concurrency import DEFAULT_MAX_CONCURRENCY, LoopSemaphore, int_from_env
from docuerp.llm.contracts import LLMError, LLMErrorCause, LLMResult, LLMTimeout
from docuerp.llm.pricing import estimate_cost

logger = logging.getLogger(__name__)

TOKEN_VARIABLE = "CLAUDE_CODE_OAUTH_TOKEN"
"""Variable con el token OAuth de la suscripción (se lee de ``.env``; nunca se imprime)."""

MIN_OUTPUT_TOKENS = 2048
"""Mínimo medido para ``CLAUDE_CODE_MAX_OUTPUT_TOKENS``: por debajo, una respuesta JSON
larga aborta con un error duro del CLI (con 1 024 Haiku falló con 1 083 tokens de salida)."""

MAX_TURNS_STRUCTURED = 1
"""Turnos máximos con salida estructurada (ver ``_build_options``)."""

STRUCTURED_OUTPUT_TOOL = "StructuredOutput"
"""Herramienta interna con la que el CLI recibe la salida estructurada del modelo."""

VALID_EFFORTS: frozenset[str] = frozenset({"low", "medium", "high", "xhigh", "max"})
"""Valores de ``effort`` aceptados por el CLI."""

CONCURRENCY_VARIABLE = "DOCUERP_LLM_MAX_CONCURRENCY"
POOL_VARIABLE = "DOCUERP_LLM_CLIENT_POOL"

_CLEAR_TIMEOUT_S = 10.0
_STDERR_LINES_KEPT = 20

# Fragmentos (en minúscula) del texto de error que permiten clasificar la causa.
_INVALID_MODEL_HINTS = ("selected model", "unrecognized_model", "may not exist", "model_not_found")
_AUTH_HINTS = (
    "authentication",
    "oauth",
    "not logged in",
    "/login",
    "invalid api key",
    "unauthorized",
    "credential",
    "token has expired",
    "invalid bearer",
)
_SCHEMA_HINTS = (
    "structured output",
    "structured_output",
    "required property",
    "maximum number of turns",
    "error_max_turns",
)
_UNAVAILABLE_HINTS = ("overloaded", "rate limit", "connection", "econnreset", "timed out")
_UNAVAILABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})


def classify_failure(message: str, status: int | None = None) -> LLMErrorCause:
    """Deduce la causa tipada de una falla a partir de su texto y del estado HTTP (si hay).

    El modelo inválido se evalúa antes que la autenticación porque su mensaje ("you may not
    have access to it") se parece al de una credencial rechazada.
    """
    text = message.lower()
    if status == 404 or any(hint in text for hint in _INVALID_MODEL_HINTS):
        return "invalid_model"
    if status in (401, 403) or any(hint in text for hint in _AUTH_HINTS):
        return "auth"
    if any(hint in text for hint in _SCHEMA_HINTS):
        return "schema"
    if status in _UNAVAILABLE_STATUS or any(hint in text for hint in _UNAVAILABLE_HINTS):
        return "unavailable"
    return "unknown"


def _scrub(text: str) -> str:
    """Elimina del texto cualquier aparición del token OAuth (defensa en profundidad)."""
    token = os.environ.get(TOKEN_VARIABLE)
    return text.replace(token, "[REDACTADO]") if token else text


def _extract_object(res: ResultMessage) -> dict[str, Any] | None:
    """Obtiene el objeto JSON de un ``ResultMessage``: ``structured_output`` o, en su
    defecto, el texto del resultado parseado como JSON."""
    if isinstance(res.structured_output, dict):
        return res.structured_output
    try:
        value = json.loads(res.result or "")
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


@dataclass(frozen=True)
class _Job:
    """Un trabajo para un trabajador del pool: el prompt y dónde dejar el resultado."""

    prompt: str
    future: asyncio.Future[ResultMessage]


_LEAKED_PARAMETER_RE = re.compile(r'<parameter name="(\w+)">')
_LEAKED_CLOSERS_RE = re.compile(r"</parameter>.*$|</invoke>.*$|</function_calls>.*$", re.DOTALL)


def repair_structured_output(
    tool_input: dict[str, Any], required: Sequence[str]
) -> dict[str, Any] | None:
    """Repara la fuga de formato de llamada a herramienta que a veces produce Haiku.

    Síntoma medido: el modelo escribe en ``answer`` el texto correcto seguido de
    ``</answer>\n<parameter name="sources">[...]`` y deja el campo ``sources`` fuera, de modo
    que el CLI rechaza la salida (``must have required property 'sources'``) y los
    reintentos internos repiten el mismo error. Aquí se separa el texto real del resto y se
    recuperan los parámetros escapados leyéndolos como JSON.

    Es una reparación puramente sintáctica: no inventa contenido. El resultado se revalida
    después con Pydantic y las citas contra el contexto.

    Returns:
        El objeto reparado con todos los campos de ``required``, o ``None`` si no hay fuga
        que reparar o la reparación no basta.
    """
    answer = tool_input.get("answer")
    if not isinstance(answer, str) or '<parameter name="' not in answer:
        return None
    head, _, tail = answer.partition("</answer>")
    pieces = _LEAKED_PARAMETER_RE.split(tail)  # ["", nombre1, valor1, nombre2, valor2, ...]
    repaired = {**tool_input, "answer": head.strip()}
    for name, raw in zip(pieces[1::2], pieces[2::2], strict=False):
        try:
            repaired[name] = json.loads(_LEAKED_CLOSERS_RE.sub("", raw).strip())
        except json.JSONDecodeError:
            return None
    return repaired if all(field in repaired for field in required) else None


def _required_fields(options: ClaudeAgentOptions) -> Sequence[str]:
    """Campos obligatorios del esquema de salida estructurada de ``options`` (o vacío)."""
    output_format = options.output_format or {}
    return output_format.get("schema", {}).get("required", [])


async def _final_result(messages: Any, required: Sequence[str] = ()) -> ResultMessage | None:
    """Consume un flujo de mensajes del SDK y devuelve el ``ResultMessage`` final.

    Si el CLI falló porque la salida estructurada no cumplía el esquema y la última llamada
    a la herramienta interna se puede reparar (:func:`repair_structured_output`), el
    resultado se corrige y se marca como exitoso. Con ``query()`` el SDK además lanza
    ``ResultError`` al terminar un resultado con error; se ignora solo si hubo reparación.
    """
    result: ResultMessage | None = None
    last_tool_input: dict[str, Any] | None = None
    try:
        async for message in messages:
            if isinstance(message, ResultMessage):
                result = message
            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock) and block.name == STRUCTURED_OUTPUT_TOOL:
                        last_tool_input = block.input
    except ResultError:
        if not _salvage(result, last_tool_input, required):
            raise
        return result
    _salvage(result, last_tool_input, required)
    return result


def _salvage(
    result: ResultMessage | None, tool_input: dict[str, Any] | None, required: Sequence[str]
) -> bool:
    """Aplica la reparación sobre un ``ResultMessage`` con error; devuelve si lo corrigió."""
    if result is None or not result.is_error or tool_input is None or not required:
        return False
    repaired = repair_structured_output(tool_input, required)
    if repaired is None:
        return False
    logger.info("Salida estructurada reparada (fuga de formato de herramienta del modelo)")
    result.is_error = False
    result.subtype = "success"
    result.structured_output = repaired
    result.result = json.dumps(repaired, ensure_ascii=False)
    return True


class _PooledWorker:
    """Un ``ClaudeSDKClient`` vivo atendido por una tarea propia (modelo de actor).

    ``ClaudeSDKClient`` usa grupos de tareas de ``anyio``: conectar y desconectar deben
    ocurrir en la **misma** tarea. Por eso el cliente vive dentro de ``_run`` y las demás
    corrutinas solo le envían trabajos por una cola. Tras cada consulta se envía ``/clear``
    para que la conversación anterior no se arrastre (verificado en el sondeo).
    """

    def __init__(self, options: ClaudeAgentOptions) -> None:
        self._options = options
        self._jobs: asyncio.Queue[_Job | None] = asyncio.Queue()
        self._task = asyncio.get_running_loop().create_task(self._run())

    @property
    def alive(self) -> bool:
        """Indica si la tarea del trabajador sigue en ejecución."""
        return not self._task.done()

    async def submit(self, prompt: str) -> ResultMessage:
        """Envía ``prompt`` al cliente y espera su ``ResultMessage``."""
        future: asyncio.Future[ResultMessage] = asyncio.get_running_loop().create_future()
        await self._jobs.put(_Job(prompt, future))
        return await future

    async def close(self) -> None:
        """Pide al trabajador que cierre su cliente y espera a que termine."""
        if self.alive:
            await self._jobs.put(None)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(self._task), timeout=_CLEAR_TIMEOUT_S)
        self.kill()

    def kill(self) -> None:
        """Cancela la tarea (el ``async with`` del cliente cierra el subproceso)."""
        if not self._task.done():
            self._task.cancel()

    async def _run(self) -> None:
        """Bucle del trabajador: atiende trabajos hasta recibir ``None`` o fallar."""
        try:
            async with ClaudeSDKClient(self._options) as client:
                while (job := await self._jobs.get()) is not None:
                    if job.future.cancelled():
                        break  # el que esperaba expiró: el cliente puede quedar a medias
                    try:
                        await client.query(job.prompt)
                        result = await _final_result(
                            client.receive_response(), _required_fields(self._options)
                        )
                        if result is None:
                            raise LLMError("el cliente no devolvió ResultMessage", "unknown")
                    except Exception as exc:
                        if not job.future.done():
                            job.future.set_exception(exc)
                        break  # un cliente con error no se reutiliza
                    try:
                        # Aísla la siguiente consulta: borra la conversación conservando
                        # el system prompt y el formato de salida.
                        async with asyncio.timeout(_CLEAR_TIMEOUT_S):
                            await client.query("/clear")
                            await _final_result(client.receive_response())
                    except Exception:
                        logger.warning("Falló /clear: el cliente del pool se descarta")
                        if not job.future.done():
                            job.future.set_result(result)
                        break
                    if not job.future.done():
                        job.future.set_result(result)
        except Exception as exc:
            logger.warning("Trabajador del pool terminó con error: %s", type(exc).__name__)
            self._fail_pending(exc)
        finally:
            self._fail_pending(LLMError("el cliente del pool se cerró", "unavailable"))

    def _fail_pending(self, exc: BaseException) -> None:
        """Rechaza los trabajos que quedaron en la cola para que nadie espere en vano."""
        while not self._jobs.empty():
            job = self._jobs.get_nowait()
            if job is not None and not job.future.done():
                job.future.set_exception(exc)


class _ClientPool:
    """Pool de trabajadores por configuración ``(modelo, effort, system, schema)``.

    Un trabajador atiende una consulta a la vez; si todos están ocupados se crea otro (el
    número total lo acota el semáforo del proveedor). Un pool pertenece a un bucle de
    eventos: ``WeakKeyDictionary`` en :class:`AgentSDKProvider` mantiene uno por bucle.
    """

    def __init__(self) -> None:
        self._idle: dict[tuple[Any, ...], list[_PooledWorker]] = {}

    async def run(
        self, key: tuple[Any, ...], options: ClaudeAgentOptions, prompt: str
    ) -> ResultMessage:
        """Ejecuta ``prompt`` en un trabajador libre (o nuevo) de la configuración ``key``."""
        worker = self._take(key) or _PooledWorker(options)
        try:
            result = await worker.submit(prompt)
        except BaseException:
            worker.kill()  # tras un error o cancelación el cliente no es de fiar
            raise
        self._idle.setdefault(key, []).append(worker)
        return result

    def _take(self, key: tuple[Any, ...]) -> _PooledWorker | None:
        """Saca un trabajador vivo de la lista de libres de ``key``."""
        workers = self._idle.get(key, [])
        while workers:
            worker = workers.pop()
            if worker.alive:
                return worker
        return None

    async def aclose(self) -> None:
        """Cierra todos los trabajadores libres."""
        workers = [w for group in self._idle.values() for w in group]
        self._idle.clear()
        await asyncio.gather(*(w.close() for w in workers), return_exceptions=True)


class AgentSDKProvider:
    """Implementación de ``LLMProvider`` sobre ``claude_agent_sdk.query``.

    Args:
        default_model: modelo cuando ``complete`` no recibe ``model``.
        env_path: archivo ``.env`` del que leer el token (por defecto el de la raíz del repo).
        max_concurrency: llamadas simultáneas máximas (por defecto la variable de entorno
            ``DOCUERP_LLM_MAX_CONCURRENCY`` o 6, valor medido sin errores en el sondeo).
        use_client_pool: reutiliza clientes ``ClaudeSDKClient`` con ``/clear`` (por defecto
            la variable ``DOCUERP_LLM_CLIENT_POOL=1``; apagado: ``query()`` por llamada).
    """

    name = "agent_sdk"

    def __init__(
        self,
        default_model: str = "claude-haiku-5-5",
        *,
        env_path: Path | None = None,
        max_concurrency: int | None = None,
        use_client_pool: bool | None = None,
    ) -> None:
        self.default_model = default_model
        self._env_path = env_path or paths.DOTENV
        limit = max_concurrency or int_from_env(CONCURRENCY_VARIABLE, DEFAULT_MAX_CONCURRENCY)
        self._semaphore = LoopSemaphore(limit)
        if use_client_pool is None:
            use_client_pool = os.environ.get(POOL_VARIABLE, "") in {"1", "true", "True"}
        self.use_client_pool = use_client_pool
        self._pools: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _ClientPool] = (
            weakref.WeakKeyDictionary()
        )

    # ------------------------------------------------------------------
    # Autenticación y opciones
    # ------------------------------------------------------------------
    def _prepare_environment(self) -> None:
        """Deja el entorno listo para lanzar el CLI.

        * Quita ``CLAUDECODE``: sin eso el CLI cree que está anidado en otra sesión.
        * Exporta ``CLAUDE_CODE_OAUTH_TOKEN`` desde ``.env`` solo si aún no está en el
          entorno (``dotenv_values`` lee el archivo sin tocar ``os.environ`` ni imprimir).

        Raises:
            LLMError: con causa ``auth`` si no hay token en el entorno ni en ``.env``.
        """
        os.environ.pop("CLAUDECODE", None)
        if os.environ.get(TOKEN_VARIABLE):
            return
        token = (
            dotenv_values(self._env_path).get(TOKEN_VARIABLE) if self._env_path.is_file() else None
        )
        if not token:
            raise LLMError(
                f"Falta {TOKEN_VARIABLE}: defínelo en .env o en el entorno "
                "(se obtiene con `claude setup-token`).",
                cause="auth",
            )
        os.environ[TOKEN_VARIABLE] = token

    def _build_options(
        self,
        system: str,
        model: str,
        json_schema: dict[str, Any] | None,
        max_tokens: int,
        effort: str | None,
        stderr_lines: list[str],
    ) -> ClaudeAgentOptions:
        """Construye las opciones del SDK: sin herramientas, sin configuración externa."""
        if effort is not None and effort not in VALID_EFFORTS:
            raise ValueError(f"effort inválido: {effort!r}; use uno de {sorted(VALID_EFFORTS)}")

        def keep_stderr(line: str) -> None:
            """Guarda las últimas líneas de stderr (para clasificar errores) sin imprimirlas."""
            stderr_lines.append(line)
            del stderr_lines[:-_STDERR_LINES_KEPT]

        return ClaudeAgentOptions(
            model=model,
            effort=effort,  # type: ignore[arg-type]  # validado arriba contra VALID_EFFORTS
            system_prompt=system,
            tools=[],  # ninguna herramienta: el LLM solo redacta
            setting_sources=[],  # ignora CLAUDE.md/hooks/ajustes del usuario
            permission_mode="dontAsk",
            # La salida estructurada se emite con una herramienta interna que valida contra el
            # esquema. Con 1 turno una salida inválida falla rápido (en vez de repetir el mismo
            # error 2-3 veces, medido) y ``repair_structured_output`` o el reintento del
            # generador se encargan del resto.
            max_turns=MAX_TURNS_STRUCTURED if json_schema is not None else 1,
            output_format={"type": "json_schema", "schema": json_schema}
            if json_schema is not None
            else None,
            env={
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(max(max_tokens, MIN_OUTPUT_TOKENS)),
                # Fuerza el uso de la suscripción aunque haya una clave de API en el entorno.
                "ANTHROPIC_API_KEY": "",
            },
            stderr=keep_stderr,
        )

    # ------------------------------------------------------------------
    # Llamada
    # ------------------------------------------------------------------
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
        """Una sola vuelta sin herramientas; respeta ``timeout_s``.

        Raises:
            LLMError: con causa ``auth``, ``invalid_model``, ``schema`` (se pidió JSON y no
                llegó), ``unavailable`` o ``unknown``; :class:`LLMTimeout` si se agota el
                tiempo.
            ValueError: si ``effort`` no es un valor válido.
        """
        model = model or self.default_model
        self._prepare_environment()
        stderr_lines: list[str] = []
        options = self._build_options(system, model, json_schema, max_tokens, effort, stderr_lines)
        started = time.perf_counter()
        result: ResultMessage | None
        try:
            # El timeout envuelve también la espera del semáforo: acota la latencia total.
            async with asyncio.timeout(timeout_s), self._semaphore:
                if self.use_client_pool:
                    result = await self._pool().run(
                        (
                            model,
                            effort,
                            max_tokens,
                            system,
                            json.dumps(json_schema, sort_keys=True),
                        ),
                        options,
                        user,
                    )
                else:
                    result = await _final_result(
                        query(prompt=user, options=options), _required_fields(options)
                    )
        except TimeoutError as exc:
            raise LLMTimeout(f"Tiempo agotado ({timeout_s:g} s) con el modelo {model}.") from exc
        except LLMError:
            raise
        except ClaudeSDKError as exc:  # ResultError, ProcessError, CLINotFoundError...
            raise self._error_from_exception(exc, model, stderr_lines) from exc
        except Exception as exc:  # p. ej. OSError al lanzar el subproceso
            raise LLMError(
                f"Falla inesperada del proveedor agent_sdk: {type(exc).__name__}", "unknown"
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        return self._to_llm_result(result, model, json_schema, latency_ms, stderr_lines)

    def _pool(self) -> _ClientPool:
        """Devuelve el pool del bucle de eventos actual (creándolo si hace falta)."""
        loop = asyncio.get_running_loop()
        pool = self._pools.get(loop)
        if pool is None:
            pool = self._pools[loop] = _ClientPool()
        return pool

    async def aclose(self) -> None:
        """Cierra los clientes del pool del bucle actual (sin efecto si no se usa el pool)."""
        pool = self._pools.pop(asyncio.get_running_loop(), None)
        if pool is not None:
            await pool.aclose()

    # ------------------------------------------------------------------
    # Conversión de resultados y errores
    # ------------------------------------------------------------------
    def _error_from_exception(
        self, exc: ClaudeSDKError, model: str, stderr_lines: list[str]
    ) -> LLMError:
        """Traduce una excepción del SDK a :class:`LLMError` con causa tipada."""
        status = getattr(exc, "api_error_status", None)
        result_text = getattr(exc, "result", None) or ""
        detail = " ".join([str(exc), result_text, *getattr(exc, "errors", []), *stderr_lines[-5:]])
        if isinstance(exc, CLINotFoundError | CLIConnectionError):
            cause: LLMErrorCause = "unavailable"
        else:
            cause = classify_failure(detail, status)
        reason = {
            "auth": "credencial de la suscripción ausente, vencida o rechazada",
            "invalid_model": f"el modelo {model!r} no existe o no está disponible",
            "unavailable": "el CLI o el servicio no respondió",
            "unknown": "falla no clasificada",
        }.get(cause, "falla del proveedor")
        return LLMError(
            f"Error del proveedor agent_sdk ({reason}): {_scrub(result_text or str(exc))[:300]}",
            cause,
        )

    def _to_llm_result(
        self,
        res: ResultMessage | None,
        model: str,
        json_schema: dict[str, Any] | None,
        latency_ms: float,
        stderr_lines: list[str],
    ) -> LLMResult:
        """Convierte el ``ResultMessage`` final en :class:`LLMResult`."""
        if res is None:
            raise LLMError("El SDK terminó sin devolver un ResultMessage.", "unavailable")
        if res.is_error:
            detail = " ".join([res.result or "", *(res.errors or []), *stderr_lines[-5:]])
            raise LLMError(
                f"El modelo devolvió un error (estado={res.api_error_status}, "
                f"subtipo={res.subtype}): {_scrub(res.result or '')[:300]}",
                classify_failure(detail, res.api_error_status),
            )
        parsed = _extract_object(res) if json_schema is not None else None
        if json_schema is not None and parsed is None:
            raise LLMError("Se pidió JSON estructurado pero la respuesta no lo trae.", "schema")
        usage = res.usage or {}
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        cache_read = int(usage.get("cache_read_input_tokens", 0))
        cache_creation = int(usage.get("cache_creation_input_tokens", 0))
        cost = res.total_cost_usd
        if cost is None:
            cost = estimate_cost(model, input_tokens, output_tokens, cache_read, cache_creation)
        return LLMResult(
            text=res.result or (json.dumps(parsed, ensure_ascii=False) if parsed else ""),
            parsed=parsed,
            model=model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
            cost_usd=cost,
            latency_ms=latency_ms,
            api_latency_ms=float(res.duration_api_ms),
        )
