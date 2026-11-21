/**
 * Cliente tipado de la API de DocuERP (SPEC §5.6) con modo demo offline.
 *
 * Modos de conexión:
 * - `live`:    la API responde en `/api` (Vite la redirige a http://127.0.0.1:8000).
 * - `offline`: la API no respondió; se usan los fixtures de `src/mocks` y la UI lo indica.
 * - `mock`:    forzado con `VITE_MOCK=1` (demo sin backend, p. ej. para capturas).
 *
 * El modo se expone como un pequeño "store" observable para que React lo muestre en la
 * barra lateral sin depender de un contexto global.
 */
import { EvalFormatError, normalizeEvalSummary, normalizeSecuritySummary, normalizeTraceLookup } from "./normalize";
import { SseParser } from "./sse";
import type {
  Chunk,
  ConfirmRequest,
  CorpusStats,
  DemoUser,
  DocumentsResponse,
  EvalSummary,
  HealthResponse,
  QueryRequest,
  QueryResponse,
  SecuritySummary,
  SourceEntry,
  TraceEvent,
  UserRole,
} from "./types";

export type ConnectionMode = "live" | "offline" | "mock";

/** Error de la API con un mensaje listo para mostrar al usuario (en español). */
export class ApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

const FORCED_MOCK = import.meta.env.VITE_MOCK === "1";
const HEALTH_TIMEOUT_MS = 2500;

/* ------------------------------------------------------------------------------------ */
/* Estado de conexión observable                                                         */
/* ------------------------------------------------------------------------------------ */

let mode: ConnectionMode = FORCED_MOCK ? "mock" : "live";
const listeners = new Set<() => void>();

function setMode(next: ConnectionMode): void {
  if (mode === next) return;
  mode = next;
  for (const fn of listeners) fn();
}

export const connection = {
  get mode(): ConnectionMode {
    return mode;
  },
  subscribe: (fn: () => void): (() => void) => {
    listeners.add(fn);
    return () => {
      listeners.delete(fn);
    };
  },
  /** `true` si la UI está usando fixtures (offline o mock forzado). */
  get usesFixtures(): boolean {
    return mode !== "live";
  },
};

/** Carga diferida de los fixtures: el bundle de producción en modo live no los necesita al inicio. */
const loadMocks = () => import("../mocks");

/* ------------------------------------------------------------------------------------ */
/* Transporte HTTP                                                                       */
/* ------------------------------------------------------------------------------------ */

/** Extrae un mensaje legible del cuerpo de error de FastAPI (`detail` puede ser lista). */
async function errorMessage(res: Response): Promise<string> {
  try {
    const body: unknown = await res.json();
    if (body && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string") return detail;
      if (Array.isArray(detail)) {
        return detail
          .map((d: unknown) => {
            const item = asRecord(d);
            return item && typeof item.msg === "string" ? item.msg : JSON.stringify(d);
          })
          .join("; ");
      }
    }
  } catch {
    // El cuerpo no es JSON: usamos el texto de estado HTTP.
  }
  return `La API respondió ${res.status} ${res.statusText}`.trim();
}

/** `true` si el error indica que la API no está disponible (y conviene pasar a offline). */
function isUnreachable(error: unknown, status?: number): boolean {
  if (error instanceof TypeError) return true; // fallo de red / conexión rechazada
  return status === 502 || status === 503 || status === 504;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      ...init,
      headers: { Accept: "application/json", ...(init?.body ? { "Content-Type": "application/json" } : {}), ...init?.headers },
    });
  } catch (error) {
    if (isUnreachable(error)) setMode("offline");
    throw new ApiError("No se pudo contactar la API de DocuERP.", 0);
  }
  if (!res.ok) {
    if (isUnreachable(null, res.status)) setMode("offline");
    throw new ApiError(await errorMessage(res), res.status);
  }
  return (await res.json()) as T;
}

/** Ejecuta `live` si hay API y, si no, `fixture`. Un fallo de red en `live` cae a `fixture`. */
async function withFallback<T>(live: () => Promise<T>, fixture: () => Promise<T>): Promise<T> {
  if (connection.usesFixtures) return fixture();
  try {
    return await live();
  } catch (error) {
    if (connection.usesFixtures) return fixture();
    throw error;
  }
}

/* ------------------------------------------------------------------------------------ */
/* Normalizadores tolerantes de endpoints auxiliares                                     */
/* ------------------------------------------------------------------------------------ */

function asRecord(v: unknown): Record<string, unknown> | null {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

/** Acepta `[...]` o `{users: [...]}` y completa `name` con el id si falta. */
export function normalizeUsers(raw: unknown): DemoUser[] {
  const list = Array.isArray(raw) ? raw : (asRecord(raw)?.users as unknown[] | undefined) ?? [];
  const roles: UserRole[] = ["portal_user", "n1", "n2", "admin"];
  return list.flatMap((item) => {
    const r = asRecord(item);
    if (!r || typeof r.user_id !== "string" || !roles.includes(r.role as UserRole)) return [];
    const name =
      (typeof r.name === "string" && r.name) ||
      (typeof r.display_name === "string" && r.display_name) ||
      (typeof r.full_name === "string" && r.full_name) ||
      r.user_id;
    const account =
      (typeof r.account === "string" && r.account) ||
      (typeof r.account_name === "string" && r.account_name) ||
      null;
    return [{ user_id: r.user_id, name, role: r.role as UserRole, account }];
  });
}

/** Calcula estadísticas mínimas cuando el endpoint solo devuelve el manifiesto. */
export function statsFromSources(sources: SourceEntry[]): CorpusStats {
  const byModule = new Map<SourceEntry["module"], number>();
  let pages = 0;
  let chunks = 0;
  let tokens = 0;
  for (const s of sources) {
    pages += s.pages ?? 0;
    chunks += s.chunks ?? 0;
    tokens += s.tokens ?? 0;
    byModule.set(s.module, (byModule.get(s.module) ?? 0) + (s.chunks ?? 0));
  }
  return {
    documents: sources.length,
    pages,
    chunks,
    tokens,
    flagged_chunks: 0,
    duplicates: 0,
    chunker: "structure",
    embedder: "bge-m3",
    by_module: [...byModule.entries()].map(([module, n]) => ({ module, chunks: n })),
    token_histogram: [],
  };
}

/** Acepta `[...]`, `{sources: [...]}` o `{documents: [...], stats}`. */
export function normalizeDocuments(raw: unknown): DocumentsResponse {
  const r = asRecord(raw);
  const list = (Array.isArray(raw) ? raw : (r?.sources ?? r?.documents ?? [])) as SourceEntry[];
  const stats = asRecord(r?.stats) as CorpusStats | null;
  return { sources: list, stats: stats ?? statsFromSources(list) };
}

/** Convierte un error de formato del normalizador en un `ApiError` mostrable. */
function formatGuard<T>(fn: () => T): T {
  try {
    return fn();
  } catch (error) {
    if (error instanceof EvalFormatError) throw new ApiError(error.message, 422);
    throw error;
  }
}

/* ------------------------------------------------------------------------------------ */
/* Simulación de streaming en modo demo                                                  */
/* ------------------------------------------------------------------------------------ */

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Consulta cancelada", "AbortError"));
      return;
    }
    const t = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(t);
        reject(new DOMException("Consulta cancelada", "AbortError"));
      },
      { once: true },
    );
  });
}

/** Retardo visual de cada etapa en la demo: proporcional a su duración real, comprimido. */
export function demoDelay(event: TraceEvent): number {
  if (event.status === "skipped") return 70;
  return Math.round(Math.min(900, Math.max(110, event.duration_ms * 0.28)));
}

/* ------------------------------------------------------------------------------------ */
/* API pública                                                                           */
/* ------------------------------------------------------------------------------------ */

export const api = {
  /** Sondea `/api/health`; si no responde a tiempo se pasa a modo offline. */
  async probe(): Promise<ConnectionMode> {
    if (FORCED_MOCK) return "mock";
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), HEALTH_TIMEOUT_MS);
    try {
      const res = await fetch("/api/health", { signal: ctrl.signal });
      const ok = res.ok && (res.headers.get("content-type") ?? "").includes("json");
      setMode(ok ? "live" : "offline");
    } catch {
      setMode("offline");
    } finally {
      clearTimeout(timer);
    }
    return mode;
  },

  /** Estado de la API, incluido el precalentamiento de modelos (`warmup`). */
  async health(): Promise<HealthResponse | null> {
    if (connection.usesFixtures) return null;
    try {
      return await request<HealthResponse>("/api/health");
    } catch {
      return null;
    }
  },

  users(): Promise<DemoUser[]> {
    return withFallback(
      async () => normalizeUsers(await request<unknown>("/api/users")),
      async () => (await loadMocks()).mockUsers(),
    );
  },

  /** Manifiesto visible para `role` (la API filtra por RBAC; por defecto, portal_user). */
  documents(role?: UserRole): Promise<DocumentsResponse> {
    const query = role ? `?role=${encodeURIComponent(role)}` : "";
    return withFallback(
      async () => normalizeDocuments(await request<unknown>(`/api/documents${query}`)),
      async () => (await loadMocks()).MOCK_DOCUMENTS,
    );
  },

  /** Chunk citado; `role` es el de la consulta que lo citó (la API revalida el RBAC). */
  chunk(chunkId: string, role?: UserRole): Promise<Chunk> {
    const query = role ? `?role=${encodeURIComponent(role)}` : "";
    return withFallback(
      () => request<Chunk>(`/api/chunks/${encodeURIComponent(chunkId)}${query}`),
      async () => {
        const found = (await loadMocks()).MOCK_CHUNKS[chunkId];
        if (!found) throw new ApiError(`El fragmento ${chunkId} no está en la demo offline.`, 404);
        return found;
      },
    );
  },

  evalSummary(): Promise<EvalSummary> {
    return withFallback(
      async () => {
        const raw = await request<unknown>("/api/eval/summary");
        return formatGuard(() => normalizeEvalSummary(raw));
      },
      async () => (await loadMocks()).MOCK_EVAL,
    );
  },

  securitySummary(): Promise<SecuritySummary> {
    return withFallback(
      async () => {
        const raw = await request<unknown>("/api/security/summary");
        return formatGuard(() => normalizeSecuritySummary(raw));
      },
      async () => (await loadMocks()).MOCK_SECURITY,
    );
  },

  /** Traza registrada en el servidor (`GET /api/trace/{trace_id}`). */
  trace(traceId: string): Promise<QueryResponse> {
    return withFallback(
      async () => {
        const raw = await request<unknown>(`/api/trace/${encodeURIComponent(traceId)}`);
        return formatGuard(() => normalizeTraceLookup(raw));
      },
      () => Promise.reject(new ApiError("En la demo offline solo están disponibles las trazas de esta sesión.", 404)),
    );
  },

  /**
   * URL del archivo fuente para el visor PDF (en demo, los PDF de `public/mock-docs`).
   * Para un `.docx` la API entrega su PDF equivalente, con la misma paginación; con
   * `original: true` se pide el DOCX tal cual (descarga).
   */
  documentFileUrl(source: string, role?: UserRole, opts?: { original?: boolean }): string {
    if (connection.usesFixtures) return `/mock-docs/${encodeURIComponent(source)}`;
    const params = new URLSearchParams();
    if (role) params.set("role", role);
    if (opts?.original) params.set("format", "original");
    const query = params.toString();
    return `/api/documents/${encodeURIComponent(source)}/file${query ? `?${query}` : ""}`;
  },

  /** `true` si el visor puede mostrar la fuente (PDF, o DOCX convertido por la API). */
  canPreview(source: string): boolean {
    return !/\.docx?$/i.test(source) || !connection.usesFixtures;
  },

  /** URL alternativa de demo para una fuente, si existe un PDF local equivalente. */
  async fallbackFileUrl(source: string): Promise<string | null> {
    const { MOCK_PDF_SOURCES } = await loadMocks();
    return MOCK_PDF_SOURCES.has(source) ? `/mock-docs/${encodeURIComponent(source)}` : null;
  },

  query(req: QueryRequest): Promise<QueryResponse> {
    return withFallback(
      () => request<QueryResponse>("/api/query", { method: "POST", body: JSON.stringify(req) }),
      async () => (await loadMocks()).buildMockResponse(req),
    );
  },

  /**
   * Consulta con streaming SSE. Llama a `onEvent` por cada etapa completada y resuelve con
   * el `QueryResponse` final. En modo demo reproduce la traza del escenario con retardos.
   */
  async streamQuery(
    req: QueryRequest,
    onEvent: (event: TraceEvent) => void,
    signal?: AbortSignal,
  ): Promise<QueryResponse> {
    const replay = async (): Promise<QueryResponse> => {
      const response = (await loadMocks()).buildMockResponse(req);
      for (const event of response.events) {
        await sleep(demoDelay(event), signal);
        onEvent(event);
      }
      await sleep(120, signal);
      return response;
    };
    if (connection.usesFixtures) return replay();

    let res: Response;
    try {
      res = await fetch("/api/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify(req),
        signal,
      });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") throw error;
      setMode("offline");
      return replay();
    }
    if (!res.ok || !res.body) {
      if (isUnreachable(null, res.status)) {
        setMode("offline");
        return replay();
      }
      throw new ApiError(await errorMessage(res), res.status);
    }

    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    const parser = new SseParser();
    // Contenedor mutable: el cierre `handle` asigna la respuesta final al llegar `event: final`.
    const result: { final: QueryResponse | null } = { final: null };
    const handle = (messages: ReturnType<SseParser["push"]>) => {
      for (const msg of messages) {
        if (msg.event === "trace") onEvent(JSON.parse(msg.data) as TraceEvent);
        else if (msg.event === "final") result.final = JSON.parse(msg.data) as QueryResponse;
        else if (msg.event === "error") {
          const parsed = asRecord(safeJson(msg.data));
          const detail = typeof parsed?.detail === "string" ? parsed.detail : typeof parsed?.message === "string" ? parsed.message : msg.data;
          throw new ApiError(detail, 500);
        }
      }
    };
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      handle(parser.push(value));
    }
    handle(parser.flush());
    if (!result.final) throw new ApiError("El flujo terminó sin la respuesta final del agente.", 500);
    return result.final;
  },

  confirm(body: ConfirmRequest): Promise<QueryResponse> {
    return withFallback(
      () => request<QueryResponse>("/api/actions/confirm", { method: "POST", body: JSON.stringify(body) }),
      async () => (await loadMocks()).confirmMockAction(body.confirmation_token, body.role),
    );
  },
};

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

/** Identificador de sesión del navegador (cumple `^[A-Za-z0-9_-]{8,64}$`). */
export function newSessionId(): string {
  const bytes = new Uint8Array(12);
  crypto.getRandomValues(bytes);
  return `ses_${[...bytes].map((b) => b.toString(16).padStart(2, "0")).join("")}`;
}
