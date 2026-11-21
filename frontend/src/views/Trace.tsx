/**
 * Inspector de traza: waterfall de latencias por etapa, listas BM25 / denso / RRF / rerank
 * con scores y cambio de posición, gate τ, filtros aplicados, consumo y JSON contractual.
 */
import { motion } from "framer-motion";
import { ArrowDown, ArrowLeft, ArrowUp, Filter, Gauge, Sparkles, Waypoints } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { ApiError, api } from "../api/client";
import type { QueryRequest, QueryResponse, TraceEvent } from "../api/types";
import { JsonView } from "../components/JsonView";
import { Badge, Button, EmptyState, Panel, SectionHeader, Skeleton } from "../components/ui";
import { cx } from "../lib/cx";
import { INTENT_LABEL, ROLE_SHORT, formatInt, formatMs, formatScore, formatUsd, labelOf, plural } from "../lib/format";
import { OUTCOME_LABEL, OUTCOME_TONE, TONE_DOT, TONE_HEX, type Tone } from "../lib/tone";
import {
  findEvent,
  gateInfo,
  outcomeOf,
  retrievalColumns,
  stageLabel,
  waterfall,
  type RankedHit,
  type RetrievalColumn,
} from "../lib/trace";
import { useSession } from "../state/session";

const STATUS_TONE: Record<TraceEvent["status"], Tone> = {
  ok: "accent",
  blocked: "danger",
  error: "danger",
  retry: "warning",
  skipped: "neutral",
};

/* ------------------------------------------------------------------------------------ */
/* Waterfall                                                                             */
/* ------------------------------------------------------------------------------------ */

function Waterfall({ events }: { events: TraceEvent[] }) {
  const { bars, totalMs } = useMemo(() => waterfall(events), [events]);
  const [open, setOpen] = useState<number | null>(null);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const scale = totalMs || 1;
  return (
    <div>
      <div className="mb-2 grid grid-cols-[148px_1fr_72px] gap-3 text-2xs text-faint">
        <span>Etapa</span>
        <div className="relative h-4">
          {ticks.map((t) => (
            <span
              key={t}
              className="absolute top-0 -translate-x-1/2 whitespace-nowrap font-mono first:translate-x-0 last:-translate-x-full"
              style={{ left: `${t * 100}%` }}
            >
              {formatMs(t * totalMs)}
            </span>
          ))}
        </div>
        <span className="text-right">Duración</span>
      </div>
      <ul className="space-y-px">
        {bars.map((bar, i) => {
          const tone = STATUS_TONE[bar.status];
          const left = (bar.startMs / scale) * 100;
          const width = Math.max(0.4, (bar.durationMs / scale) * 100);
          const event = events[i];
          return (
            <li key={`${bar.stage}-${i}`}>
              <button
                type="button"
                onClick={() => setOpen(open === i ? null : i)}
                aria-expanded={open === i}
                className="grid w-full grid-cols-[148px_1fr_72px] items-center gap-3 rounded-[8px] px-1 py-1.5 text-left transition-colors duration-150 hover:bg-white/[0.03]"
              >
                <span className="flex min-w-0 items-center gap-2">
                  <span className={cx("h-1.5 w-1.5 shrink-0 rounded-full", TONE_DOT[tone])} aria-hidden />
                  <span className={cx("truncate text-xs", bar.status === "skipped" ? "text-faint" : "text-text")}>
                    {stageLabel(bar.stage)}
                  </span>
                </span>
                <div className="relative h-5">
                  {ticks.slice(1, -1).map((t) => (
                    <span key={t} className="absolute inset-y-0 w-px bg-white/[0.04]" style={{ left: `${t * 100}%` }} aria-hidden />
                  ))}
                  <motion.span
                    className="absolute top-1/2 h-2.5 -translate-y-1/2 rounded-[3px]"
                    style={{
                      left: `${left}%`,
                      width: `${width}%`,
                      background: bar.status === "skipped" ? "rgba(255,255,255,0.08)" : TONE_HEX[tone],
                      boxShadow: bar.status === "ok" && bar.durationMs / scale > 0.2 ? `0 0 14px ${TONE_HEX[tone]}55` : undefined,
                      transformOrigin: "left center",
                    }}
                    initial={{ opacity: 0, transform: "translateY(-50%) scaleX(0.6)" }}
                    animate={{ opacity: 1, transform: "translateY(-50%) scaleX(1)" }}
                    transition={{ duration: 0.24, delay: i * 0.02, ease: [0.23, 1, 0.32, 1] }}
                  />
                </div>
                <span className="text-right font-mono text-2xs text-muted">{formatMs(bar.durationMs)}</span>
              </button>
              {open === i && event && (
                <div className="mb-2 ml-[156px] mt-1">
                  <JsonView value={event.data} maxHeight={240} />
                </div>
              )}
            </li>
          );
        })}
      </ul>
      <p className="mt-3 text-2xs text-faint">Haz clic en una etapa para ver su `data` sanitizado.</p>
    </div>
  );
}

/* ------------------------------------------------------------------------------------ */
/* Listas de recuperación                                                                */
/* ------------------------------------------------------------------------------------ */

function Delta({ hit, column }: { hit: RankedHit; column: RetrievalColumn["key"] }) {
  if (column === "bm25") return <span className="w-7" />;
  if (hit.delta === null)
    return <span className="w-7 text-right font-mono text-[10px] text-info">nuevo</span>;
  if (hit.delta === 0) return <span className="w-7 text-right font-mono text-[10px] text-faint">=</span>;
  const up = hit.delta > 0;
  return (
    <span className={cx("flex w-7 items-center justify-end gap-px font-mono text-[10px]", up ? "text-accent-2" : "text-danger/80")}>
      {up ? <ArrowUp size={10} strokeWidth={2.2} aria-hidden /> : <ArrowDown size={10} strokeWidth={2.2} aria-hidden />}
      {Math.abs(hit.delta)}
    </span>
  );
}

function RetrievalLists({ events, cited }: { events: TraceEvent[]; cited: Set<string> }) {
  const columns = useMemo(() => retrievalColumns(events), [events]);
  const [hovered, setHovered] = useState<string | null>(null);
  if (!columns.length) {
    return (
      <EmptyState
        icon={Waypoints}
        title="Sin recuperación en esta traza"
        description="La consulta se resolvió con una herramienta o se detuvo antes de buscar en el índice."
        className="py-10"
      />
    );
  }
  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
      {columns.map((col) => {
        const max = Math.max(...col.hits.map((h) => h.score), 1e-9);
        return (
          <div key={col.key} className="min-w-0 rounded-[12px] bg-black/20 ring-1 ring-inset ring-white/[0.05]">
            <div className="flex items-baseline justify-between border-b hairline px-3 py-2.5">
              <span className="text-xs font-medium text-text">{col.label}</span>
              <span className="truncate pl-2 text-2xs text-faint">{col.caption}</span>
            </div>
            <ol className="p-1.5">
              {col.hits.map((hit) => {
                const isHover = hovered === hit.chunk_id;
                const isCited = cited.has(hit.chunk_id);
                return (
                  <li
                    key={hit.chunk_id}
                    onMouseEnter={() => setHovered(hit.chunk_id)}
                    onMouseLeave={() => setHovered(null)}
                    title={hit.chunk_id}
                    className={cx(
                      "rounded-[8px] px-2 py-1.5 transition-colors duration-150",
                      isHover ? "bg-accent/[0.08]" : hovered ? "opacity-55" : "",
                    )}
                  >
                    <div className="flex items-center gap-2">
                      <span className="w-4 shrink-0 text-right font-mono text-[10.5px] text-faint">{hit.rank}</span>
                      <span className="min-w-0 flex-1 truncate text-[11.5px] text-text">
                        {hit.source.replace(/\.pdf$/, "")}
                        {hit.page !== null && <span className="text-faint"> · p.{hit.page}</span>}
                      </span>
                      {isCited && <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-accent" title="Citado en la respuesta" />}
                      <Delta hit={hit} column={col.key} />
                    </div>
                    <div className="mt-1 flex items-center gap-2 pl-6">
                      <div className="h-1 flex-1 rounded-full bg-white/[0.05]">
                        <div
                          className={cx("h-full rounded-full", col.key === "rerank" ? "bg-accent" : "bg-white/25")}
                          style={{ width: `${(hit.score / max) * 100}%` }}
                        />
                      </div>
                      <span className="w-12 text-right font-mono text-[10px] text-muted">
                        {formatScore(hit.score, col.key === "fusion" ? 4 : col.key === "bm25" ? 2 : 3)}
                      </span>
                    </div>
                  </li>
                );
              })}
            </ol>
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------------------------ */
/* Gate τ y filtros                                                                      */
/* ------------------------------------------------------------------------------------ */

function GateRule({ events }: { events: TraceEvent[] }) {
  const gate = gateInfo(events);
  if (!gate) return <p className="text-xs text-faint">Esta traza no pasó por el gate de evidencia.</p>;
  const pct = (v: number) => `${Math.max(0, Math.min(1, v)) * 100}%`;
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <div>
          <div className="font-mono text-xl text-text">{formatScore(gate.top, 2)}</div>
          <div className="text-2xs text-faint">puntaje máximo del reranker</div>
        </div>
        <Badge tone={gate.ok ? "accent" : "warning"}>{gate.ok ? "Evidencia suficiente" : "Abstención"}</Badge>
      </div>
      <div className="relative mt-6 h-2 rounded-full bg-gradient-to-r from-warning/25 via-white/[0.06] to-accent/25">
        <div className="absolute inset-y-0 left-0 rounded-l-full bg-warning/20" style={{ width: pct(gate.tau) }} />
        <div className="absolute -top-5 -translate-x-1/2 font-mono text-[10px] text-muted" style={{ left: pct(gate.tau) }}>
          τ {formatScore(gate.tau, 2)}
        </div>
        <div className="absolute -top-1.5 h-5 w-px bg-text/70" style={{ left: pct(gate.tau) }} aria-hidden />
        <motion.div
          className={cx(
            "absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-bg",
            gate.ok ? "bg-accent shadow-glow" : "bg-warning",
          )}
          style={{ left: pct(gate.top) }}
          initial={{ opacity: 0, scale: 0.7 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.22, ease: [0.23, 1, 0.32, 1] }}
        />
      </div>
      <div className="mt-2 flex justify-between font-mono text-[10px] text-faint">
        <span>0</span>
        <span>1</span>
      </div>
    </div>
  );
}

/**
 * Tokens de entrada del prompt. Con caché de prompt el proveedor separa lo no cacheado
 * (`input_tokens`), lo leído de caché y lo escrito en caché; los tres forman el contexto.
 */
function promptTokens(res: QueryResponse): number {
  return res.usage.input_tokens + res.usage.cache_read_tokens + res.usage.cache_creation_tokens;
}

function Filters({ res, request }: { res: QueryResponse; request: QueryRequest | null }) {
  const bm25 = findEvent(res.events, "retrieval.bm25");
  const perm = findEvent(res.events, "permission_check");
  const search = findEvent(res.events, "tool.search_documents");
  // El agente no repite los filtros en la traza (evita datos de más): se toman de la petición.
  const filters = (bm25?.data.filters ??
    (request && search
      ? { role: request.role, erp_version: [request.erp_version, "any"], status: "vigente", include_obsolete: false }
      : {})) as Record<string, unknown>;
  const rows: [string, string][] = [];
  if (typeof filters.role === "string") rows.push(["Rol", ROLE_SHORT[filters.role as keyof typeof ROLE_SHORT] ?? filters.role]);
  if (filters.erp_version !== undefined)
    rows.push([
      "Versión ERP",
      Array.isArray(filters.erp_version) ? filters.erp_version.map(String).join(" + ") : JSON.stringify(filters.erp_version).replace(/"/g, ""),
    ]);
  if (filters.erp_version !== undefined) rows.push(["Docs internos", "erp_version = any"]);
  if (typeof filters.status === "string") rows.push(["Vigencia", filters.status]);
  if (typeof filters.include_obsolete === "boolean") rows.push(["Incluye obsoletos", filters.include_obsolete ? "sí" : "no"]);
  const filteredRaw = bm25?.data.filtered_out ?? search?.data.filtered_out;
  const filtered = typeof filteredRaw === "number" ? filteredRaw : null;
  if (filtered !== null) rows.push(["Descartados por RBAC", String(filtered)]);
  if (perm) rows.push(["Revalidación", perm.status === "ok" ? "aprobada" : "denegada"]);
  if (!rows.length) return <p className="text-xs text-faint">Sin filtros de recuperación en esta traza.</p>;
  return (
    <dl className="divide-y divide-white/[0.05]">
      {rows.map(([k, v]) => (
        <div key={k} className="flex items-center justify-between gap-4 py-2 text-xs">
          <dt className="text-muted">{k}</dt>
          <dd className="font-mono text-text">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

/* ------------------------------------------------------------------------------------ */
/* Vista                                                                                 */
/* ------------------------------------------------------------------------------------ */

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="min-w-0 rounded-[12px] bg-white/[0.02] px-4 py-3 ring-1 ring-inset ring-white/[0.05]">
      <div className="text-2xs text-faint">{label}</div>
      <div className="mt-1 truncate font-mono text-[17px] tracking-tight text-text" title={value}>
        {value}
      </div>
      {sub && <div className="mt-0.5 truncate text-2xs text-muted">{sub}</div>}
    </div>
  );
}

type Loaded = { status: "idle" } | { status: "loading" } | { status: "ok"; res: QueryResponse } | { status: "error"; message: string };

export default function TraceView({ traceId, onBack }: { traceId: string | null; onBack: () => void }) {
  const { turns, focusedTurn, ask } = useSession();
  const [remote, setRemote] = useState<Loaded>({ status: "idle" });
  const [tab, setTab] = useState<"answer" | "full">("answer");

  const sessionTraces = turns.filter((t) => t.response).map((t) => t.response as QueryResponse);
  const local = traceId
    ? sessionTraces.find((r) => r.trace_id === traceId)
    : (focusedTurn?.response ?? sessionTraces[sessionTraces.length - 1]);

  useEffect(() => {
    if (!traceId || local) return;
    let alive = true;
    setRemote({ status: "loading" });
    api
      .trace(traceId)
      .then((res) => alive && setRemote({ status: "ok", res }))
      .catch((e: unknown) =>
        alive && setRemote({ status: "error", message: e instanceof ApiError ? e.message : "No se pudo cargar la traza." }),
      );
    return () => {
      alive = false;
    };
  }, [traceId, local]);

  const res = local ?? (remote.status === "ok" ? remote.res : null);
  const cited = useMemo(() => new Set(res?.response.sources.map((s) => s.chunk_id) ?? []), [res]);

  if (!res) {
    if (remote.status === "loading") {
      return (
        <div className="mx-auto max-w-[1180px] space-y-4 px-6 py-8">
          <Skeleton className="h-24" />
          <Skeleton className="h-80" />
        </div>
      );
    }
    return (
      <div className="mx-auto max-w-[1180px] px-6 py-16">
        <EmptyState
          icon={Waypoints}
          title={remote.status === "error" ? "Traza no disponible" : "Aún no hay trazas en esta sesión"}
          description={
            remote.status === "error"
              ? remote.message
              : "Ejecuta una consulta en la consola: cada respuesta deja una traza con latencias, rankings y el contrato JSON."
          }
          action={
            <div className="flex gap-2">
              <Button icon={ArrowLeft} variant="secondary" onClick={onBack}>
                Ir a la consola
              </Button>
              <Button
                icon={Sparkles}
                variant="primary"
                onClick={() => {
                  onBack();
                  ask("¿Cómo configuro la recepción en dos pasos en Inventario?");
                }}
              >
                Ejecutar ejemplo
              </Button>
            </div>
          }
        />
      </div>
    );
  }

  const outcome = outcomeOf(res);
  const gen = findEvent(res.events, "generation");
  const apiMs = typeof gen?.data.api_latency_ms === "number" ? gen.data.api_latency_ms : null;
  const model = typeof gen?.data.model === "string" ? gen.data.model : null;

  return (
    <div className="mx-auto max-w-[1180px] space-y-6 px-4 py-8 md:px-6">
      <div className="flex flex-wrap items-center gap-3">
        <span className="font-mono text-sm text-text">{res.trace_id}</span>
        <Badge tone={OUTCOME_TONE[outcome]}>{OUTCOME_LABEL[outcome]}</Badge>
        <Badge>{labelOf(INTENT_LABEL, res.intent)}</Badge>
        {sessionTraces.length > 1 && (
          <label className="ml-auto flex items-center gap-2 text-2xs text-faint">
            Traza
            <select
              value={res.trace_id}
              onChange={(e) => {
                window.location.hash = `/traza/${encodeURIComponent(e.target.value)}`;
              }}
              className="h-8 max-w-[260px] rounded-[9px] bg-white/[0.04] px-2 font-mono text-xs text-text ring-1 ring-inset ring-white/[0.08] focus:outline-none"
            >
              {sessionTraces.map((r, i) => (
                <option key={r.trace_id} value={r.trace_id}>
                  #{i + 1} · {r.trace_id} · {labelOf(INTENT_LABEL, r.intent)}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
        <Stat label="Latencia total" value={formatMs(res.latency_ms)} sub={apiMs !== null ? `API LLM ${formatMs(apiMs)}` : "sin LLM"} />
        <Stat label="Etapas" value={String(res.events.length)} sub={plural(res.events.filter((e) => e.status === "skipped").length, "omitida")} />
        <Stat
          label="Tokens"
          value={formatInt(promptTokens(res) + res.usage.output_tokens)}
          sub={`${formatInt(promptTokens(res))} in · ${formatInt(res.usage.output_tokens)} out`}
        />
        <Stat label="Caché de prompt" value={formatInt(res.usage.cache_read_tokens)} sub="tokens leídos" />
        <Stat label="Costo estimado" value={formatUsd(res.usage.cost_usd)} sub={`${plural(res.usage.llm_calls, "llamada")} al LLM`} />
        <Stat label="Modelo" value={model ? model.replace(/^claude-/, "") : "—"} sub={res.response.tool_used ? `tool: ${res.response.tool_used}` : "sin herramienta"} />
      </div>

      <Panel coreClassName="p-5">
        <SectionHeader
          eyebrow="Línea de tiempo"
          title="Latencia por etapa"
          description="Waterfall secuencial del grafo: cada barra empieza donde termina la anterior."
          className="mb-5"
        />
        <Waterfall events={res.events} />
      </Panel>

      <Panel coreClassName="p-5">
        <SectionHeader
          eyebrow="Recuperación"
          title="BM25 → denso → RRF → rerank"
          description="Pasa el cursor sobre un fragmento para seguirlo entre listas. Las flechas muestran el cambio de posición respecto a la lista anterior; el punto verde marca lo citado."
          className="mb-5"
        />
        <RetrievalLists events={res.events} cited={cited} />
      </Panel>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Panel coreClassName="p-5">
          <SectionHeader eyebrow="Abstención" title="Gate de evidencia τ" className="mb-6" actions={<Gauge size={16} className="text-faint" />} />
          <GateRule events={res.events} />
        </Panel>
        <Panel coreClassName="p-5">
          <SectionHeader eyebrow="RBAC y vigencia" title="Filtros aplicados" className="mb-3" actions={<Filter size={16} className="text-faint" />} />
          <Filters res={res} request={turns.find((t) => t.response?.trace_id === res.trace_id)?.request ?? null} />
        </Panel>
      </div>

      <Panel coreClassName="p-5">
        <SectionHeader
          eyebrow="Contrato de salida"
          title="JSON"
          description="El objeto `Answer` contiene exactamente los cinco campos del §11; el `QueryResponse` agrega traza, uso y acción pendiente."
          className="mb-4"
          actions={
            <div role="tablist" className="flex rounded-[9px] bg-white/[0.03] p-0.5 ring-1 ring-inset ring-white/[0.07]">
              {(
                [
                  ["answer", "Answer §11"],
                  ["full", "QueryResponse"],
                ] as const
              ).map(([key, label]) => (
                <button
                  key={key}
                  role="tab"
                  type="button"
                  aria-selected={tab === key}
                  onClick={() => setTab(key)}
                  className={cx(
                    "h-7 rounded-[7px] px-2.5 text-xs transition-colors duration-150",
                    tab === key ? "bg-white/[0.07] text-text" : "text-faint hover:text-muted",
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
          }
        />
        <JsonView value={tab === "answer" ? res.response : res} maxHeight={tab === "answer" ? 360 : 520} />
      </Panel>
    </div>
  );
}
