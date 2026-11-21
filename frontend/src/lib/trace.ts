/**
 * Interpretación de la traza del agente (`TraceEvent[]`, SPEC §4 y §11) para la interfaz.
 *
 * Funciones puras (sin React) para poder probarlas: agrupan eventos en pasos visibles del
 * pipeline, extraen las listas de recuperación con sus cambios de posición, construyen el
 * waterfall de latencias y deciden el estado visual de una respuesta.
 */
import type { Intent, QueryResponse, TraceEvent, TraceStatus } from "../api/types";
import { INTENT_LABEL, formatMs, formatScore, labelOf } from "./format";

/* ------------------------------------------------------------------------------------ */
/* Pasos del pipeline                                                                    */
/* ------------------------------------------------------------------------------------ */

export type StepStatus = TraceStatus | "pending" | "running";

export interface StepDef {
  key: string;
  label: string;
  /** Etapas de `TraceEvent.stage` que pertenecen a este paso ("x." = prefijo). */
  match: string[];
  /** Nº de sub-etapas que completan el paso (p. ej. recuperación = BM25 + denso + RRF). */
  expected?: number;
  /** Texto en gerundio mientras el paso está en curso ("Recuperando evidencia…"). */
  running: string;
}

const STEP_INPUT: StepDef = {
  key: "input_guard",
  label: "Guard de entrada",
  match: ["input_guard"],
  running: "Verificando la entrada…",
};
const STEP_ROUTER: StepDef = { key: "router", label: "Enrutador", match: ["router"], running: "Detectando la intención…" };
const STEP_RETRIEVAL: StepDef = {
  key: "retrieval",
  label: "Recuperación híbrida",
  match: ["retrieval."],
  expected: 3,
  running: "Recuperando evidencia…",
};
const STEP_RERANK: StepDef = { key: "rerank", label: "Rerank", match: ["rerank", "mmr"], running: "Reordenando candidatos…" };
const STEP_GATE: StepDef = {
  key: "evidence_gate",
  label: "Gate de evidencia",
  match: ["evidence_gate"],
  running: "Evaluando la evidencia…",
};
const STEP_PERM: StepDef = {
  key: "permission_check",
  label: "Permisos",
  match: ["permission_check"],
  running: "Verificando permisos…",
};
const STEP_TOOL: StepDef = { key: "tool", label: "Herramienta", match: ["tool."], running: "Ejecutando la herramienta…" };
const STEP_GEN: StepDef = { key: "generation", label: "Generación", match: ["generation"], running: "Generando la respuesta…" };
const STEP_OUT: StepDef = {
  key: "output_guard",
  label: "Guard de salida",
  match: ["output_guard"],
  running: "Validando la salida…",
};

/** Plan de pasos según la intención (SPEC §5.4: rama documental o rama de herramienta). */
export function planFor(intent: Intent | null): StepDef[] {
  if (intent === "ticket_status" || intent === "ticket_classify" || intent === "ticket_create") {
    return [STEP_INPUT, STEP_ROUTER, STEP_TOOL, STEP_PERM, STEP_GEN, STEP_OUT]; // orden del grafo: tool → permission_check
  }
  if (intent === "out_of_domain") return [STEP_INPUT, STEP_ROUTER];
  return [STEP_INPUT, STEP_ROUTER, STEP_RETRIEVAL, STEP_RERANK, STEP_GATE, STEP_PERM, STEP_GEN, STEP_OUT];
}

export interface PipelineStep {
  def: StepDef;
  status: StepStatus;
  durationMs: number;
  detail: string | null;
  events: TraceEvent[];
}

function matches(def: StepDef, stage: string): boolean {
  if (stage === RETRIEVAL_TOTAL) return false; // agregado: su tiempo ya está en las sub-etapas
  return def.match.some((m) => (m.endsWith(".") ? stage.startsWith(m) : stage === m));
}

/** Evento agregado de la recuperación (suma de BM25 + denso + RRF + rerank). */
export const RETRIEVAL_TOTAL = "retrieval.total";

/**
 * Eventos "contenedor": su duración incluye la de otros eventos de la traza. El agente
 * emite `retrieval.total` y `tool.search_documents` además de cada etapa de recuperación;
 * sumarlos duplicaría el tiempo en el waterfall.
 */
export function isContainerEvent(event: TraceEvent, events: TraceEvent[]): boolean {
  if (event.stage === RETRIEVAL_TOTAL) return true;
  return event.stage === "tool.search_documents" && events.some((e) => e.stage.startsWith("retrieval."));
}

/** Prioridad para resumir varios estados en uno (el peor gana). */
const SEVERITY: Record<TraceStatus, number> = { blocked: 4, error: 3, retry: 2, ok: 1, skipped: 0 };

function worst(events: TraceEvent[]): TraceStatus {
  return events.reduce<TraceStatus>(
    (acc, e) => (SEVERITY[e.status] > SEVERITY[acc] ? e.status : acc),
    events[0]?.status ?? "skipped",
  );
}

/** Intención anunciada por el router dentro de la traza (si ya llegó). */
export function intentFromEvents(events: TraceEvent[]): Intent | null {
  const router = events.find((e) => e.stage === "router");
  const value = router?.data.intent;
  return typeof value === "string" ? (value as Intent) : null;
}

/**
 * Agrupa los eventos en pasos visibles. Con `done = false` (streaming) marca como
 * `running` el paso en curso; con `done = true` los pasos sin eventos quedan `skipped`.
 */
export function buildPipeline(
  events: TraceEvent[],
  done: boolean,
  intentHint: Intent | null = null,
): PipelineStep[] {
  // Si el guard de entrada bloqueó, ninguna etapa posterior se ejecuta: se muestra el plan
  // documental completo para que se vea todo lo que el bloqueo evitó.
  const halted = events.some((e) => e.status === "blocked" && e.stage === "input_guard");
  const plan = planFor(halted ? null : (intentFromEvents(events) ?? intentHint));
  let runningAssigned = false;
  return plan.map((def, i) => {
    const own = events.filter((e) => matches(def, e.stage));
    const later = plan.slice(i + 1).some((d) => events.some((e) => matches(d, e.stage)));
    let status: StepStatus;
    if (own.length > 0) {
      const incomplete = def.expected !== undefined && own.length < def.expected;
      status = !done && !later && incomplete ? "running" : worst(own);
    } else if (done || halted) {
      status = "skipped";
    } else if (!runningAssigned) {
      status = "running";
    } else {
      status = "pending";
    }
    if (status === "running") runningAssigned = true;
    return {
      def,
      status,
      durationMs: own.reduce((s, e) => s + e.duration_ms, 0),
      detail: own.length ? stepDetail(def.key, own) : null,
      events: own,
    };
  });
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Pasos del plan documental en espera (carril vacío antes de la primera consulta). */
export function idlePipeline(): PipelineStep[] {
  return planFor(null).map((def) => ({ def, status: "pending", durationMs: 0, detail: null, events: [] }));
}

/** Línea de detalle breve para cada paso, extraída de `data` (formato tolerante). */
export function stepDetail(key: string, events: TraceEvent[]): string | null {
  const last = events[events.length - 1];
  if (!last) return null;
  const d = last.data;
  if (last.status === "skipped") return "Omitido";
  switch (key) {
    case "input_guard": {
      const risk = num(d.risk_score);
      const flags = Array.isArray(d.flags) ? d.flags.length : 0;
      const riskText = risk !== null ? risk.toFixed(2) : "—";
      if (last.status === "blocked") return `Bloqueado · riesgo ${riskText}`;
      return `Riesgo ${riskText} · ${flags ? `${flags} señales` : "sin señales"}`;
    }
    case "router": {
      const intent = typeof d.intent === "string" ? labelOf(INTENT_LABEL, d.intent) : "—";
      const conf = num(d.confidence);
      return conf !== null ? `${intent} · ${conf.toFixed(2)}` : intent;
    }
    case "retrieval": {
      const candidates = num(events[0]?.data.candidates);
      const fusion = events.find((e) => e.stage === "retrieval.fusion");
      const k = num(fusion?.data.k);
      return [
        candidates !== null ? `${candidates} candidatos` : null,
        "BM25 + denso",
        k !== null ? `RRF k=${k}` : null,
      ]
        .filter(Boolean)
        .join(" · ");
    }
    case "rerank": {
      const hits = extractHits(last);
      const top = hits[0]?.score;
      return `Top ${hits.length}${top !== undefined ? ` · máx ${formatScore(top, 2)}` : ""}`;
    }
    case "evidence_gate": {
      const top = num(d.top_score);
      const tau = num(d.tau);
      if (top === null || tau === null) return last.status === "ok" ? "Evidencia suficiente" : "Sin evidencia";
      return `${top.toFixed(2)} ${top >= tau ? "≥" : "<"} τ ${tau.toFixed(2)}`;
    }
    case "permission_check": {
      if (d.in_scope === false) return "Fuera del alcance del usuario";
      if (last.status === "blocked" || d.denied === true) return "Acceso denegado para el rol";
      const auth = num(d.authorized) ?? num(d.kept);
      const filtered = num(d.filtered_out) ?? num(d.dropped);
      if (auth !== null) return `${auth} autorizados${filtered ? ` · ${filtered} filtrados` : ""}`;
      return d.tool_allowed === true || d.allowed === true ? "Herramienta permitida" : "Verificado";
    }
    case "tool":
      return last.stage.replace(/^tool\./, "");
    case "generation": {
      const model = typeof d.model === "string" ? d.model : null;
      if (d.mode === "template") return "Plantilla determinista, sin LLM";
      const tout = num(d.output_tokens);
      // Con caché de prompt, input_tokens solo cuenta lo no cacheado: se informa la salida.
      const tokens = tout !== null ? `${tout} tok de salida` : null;
      return [model?.replace(/^claude-/, ""), tokens].filter(Boolean).join(" · ") || null;
    }
    case "output_guard": {
      const valid = num(d.citations_valid);
      const checked = num(d.citations_checked);
      if (valid !== null && checked !== null && checked > 0) return `${valid}/${checked} citas válidas`;
      if (last.status === "blocked") return "Respuesta corregida";
      return d.schema_valid === true || last.status === "ok" ? "Citas y esquema válidos" : null;
    }
    default:
      return null;
  }
}

/* ------------------------------------------------------------------------------------ */
/* Listas de recuperación                                                                */
/* ------------------------------------------------------------------------------------ */

export interface RankedHit {
  chunk_id: string;
  source: string;
  page: number | null;
  section_path: string | null;
  score: number;
  rank: number;
  /** Cambio de posición respecto a la lista de referencia (+ sube, − baja, null = nuevo). */
  delta: number | null;
}

/**
 * Lee los hits de un evento de recuperación. Acepta `data.hits`, `data.top`,
 * `data.results` o `data.ranking`, con filas planas o `ScoredChunk` completos.
 */
export function extractHits(event: TraceEvent | undefined): RankedHit[] {
  if (!event) return [];
  const d = event.data;
  const raw = [d.hits, d.top, d.results, d.ranking].find(Array.isArray) as unknown[] | undefined;
  if (!raw) return [];
  const out: RankedHit[] = [];
  raw.forEach((item, i) => {
    if (!item || typeof item !== "object") return;
    const r = item as Record<string, unknown>;
    const chunk = (r.chunk && typeof r.chunk === "object" ? r.chunk : r) as Record<string, unknown>;
    const id = typeof chunk.chunk_id === "string" ? chunk.chunk_id : null;
    const score = num(r.score);
    if (!id || score === null) return;
    out.push({
      chunk_id: id,
      source: typeof chunk.source === "string" ? chunk.source : (id.split("::")[0] ?? id),
      page: num(chunk.page),
      section_path: typeof chunk.section_path === "string" ? chunk.section_path : null,
      score,
      rank: num(r.rank) ?? i + 1,
      delta: null,
    });
  });
  return out.sort((a, b) => a.rank - b.rank);
}

export type RetrievalKey = "bm25" | "dense" | "fusion" | "rerank" | "mmr";

export interface RetrievalColumn {
  key: RetrievalKey;
  label: string;
  caption: string;
  hits: RankedHit[];
}

const COLUMNS: { key: RetrievalKey; stage: string; label: string; caption: string }[] = [
  { key: "bm25", stage: "retrieval.bm25", label: "BM25", caption: "léxico" },
  { key: "dense", stage: "retrieval.dense", label: "Denso", caption: "bge-m3 · coseno" },
  { key: "fusion", stage: "retrieval.fusion", label: "RRF", caption: "k = 60 · 0.5 / 0.5" },
  { key: "rerank", stage: "rerank", label: "Rerank", caption: "cross-encoder · sigmoide" },
  { key: "mmr", stage: "mmr", label: "MMR", caption: "λ = 0.7" },
];

/** Lista contra la que se mide el cambio de posición de cada columna. */
const REFERENCE: Record<RetrievalKey, RetrievalKey | null> = {
  bm25: null,
  dense: "bm25",
  fusion: "bm25",
  rerank: "fusion",
  mmr: "rerank",
};

/** Columnas BM25 → denso → RRF → rerank (→ MMR) con el cambio de posición de cada chunk. */
export function retrievalColumns(events: TraceEvent[]): RetrievalColumn[] {
  const cols: RetrievalColumn[] = [];
  for (const c of COLUMNS) {
    const event = events.find((e) => e.stage === c.stage);
    if (!event) continue;
    cols.push({ key: c.key, label: c.label, caption: c.caption, hits: extractHits(event) });
  }
  const byKey = new Map(cols.map((c) => [c.key, c]));
  for (const col of cols) {
    const refKey = REFERENCE[col.key];
    const ref = refKey ? byKey.get(refKey) : undefined;
    if (!ref) continue;
    const refRank = new Map(ref.hits.map((h) => [h.chunk_id, h.rank]));
    col.hits = col.hits.map((h) => {
      const prev = refRank.get(h.chunk_id);
      return { ...h, delta: prev === undefined ? null : prev - h.rank };
    });
  }
  return cols;
}

/* ------------------------------------------------------------------------------------ */
/* Waterfall                                                                             */
/* ------------------------------------------------------------------------------------ */

export interface WaterfallBar {
  stage: string;
  status: TraceStatus;
  startMs: number;
  durationMs: number;
}

/** Barras secuenciales (el grafo ejecuta las etapas en orden) con su inicio acumulado. */
export function waterfall(events: TraceEvent[]): { bars: WaterfallBar[]; totalMs: number } {
  let cursor = 0;
  const bars = events.filter((e) => !isContainerEvent(e, events)).map((e) => {
    const bar = { stage: e.stage, status: e.status, startMs: cursor, durationMs: e.duration_ms };
    cursor += e.duration_ms;
    return bar;
  });
  return { bars, totalMs: cursor };
}

const STAGE_LABEL: Record<string, string> = {
  input_guard: "Guard de entrada",
  router: "Enrutador",
  "retrieval.bm25": "BM25",
  "retrieval.dense": "Denso",
  "retrieval.fusion": "Fusión RRF",
  rerank: "Rerank",
  mmr: "MMR",
  evidence_gate: "Gate τ",
  permission_check: "Permisos",
  generation: "Generación",
  output_guard: "Guard de salida",
  respond: "Respuesta",
  error: "Error",
};

/** Nombre legible de una etapa (`tool.get_ticket_status` → "get_ticket_status"). */
export function stageLabel(stage: string): string {
  if (stage.startsWith("tool.")) return stage.slice(5);
  return STAGE_LABEL[stage] ?? stage;
}

/* ------------------------------------------------------------------------------------ */
/* Estado visual de una respuesta                                                        */
/* ------------------------------------------------------------------------------------ */

export type Outcome =
  | "answered"
  | "abstained"
  | "blocked"
  | "denied"
  | "review"
  | "tool"
  | "pending_confirmation";

/** Decide cómo se presenta la respuesta. El orden refleja la gravedad del estado. */
export function outcomeOf(res: QueryResponse): Outcome {
  if (res.blocked) return "blocked";
  const perm = findEvent(res.events, "permission_check");
  if (perm && (perm.status === "blocked" || perm.data.denied === true)) return "denied";
  if (res.pending_action) return "pending_confirmation";
  if (res.response.abstained) return "abstained";
  if (res.response.needs_human_review) return "review";
  // search_documents es la recuperación de la rama documental, no una "herramienta" de tickets.
  if (res.response.tool_used && res.response.tool_used !== "search_documents") return "tool";
  return "answered";
}

/** Evento de una etapa concreta (el último si se repite por reintentos; "x." = prefijo). */
export function findEvent(events: TraceEvent[], stage: string): TraceEvent | undefined {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const e = events[i];
    if (e && (e.stage === stage || (stage.endsWith(".") && e.stage.startsWith(stage)))) return e;
  }
  return undefined;
}

/** Datos del gate τ para la regla visual del inspector. */
export function gateInfo(events: TraceEvent[]): { tau: number; top: number; ok: boolean } | null {
  const gate = findEvent(events, "evidence_gate");
  if (!gate) return null;
  const tau = num(gate.data.tau);
  const top = num(gate.data.top_score);
  if (tau === null || top === null) return null;
  return { tau, top, ok: gate.status === "ok" };
}

/** Resumen legible de la latencia total: "4.42 s · API 2.63 s". */
export function latencySummary(res: QueryResponse): string {
  const gen = findEvent(res.events, "generation");
  const apiMs = num(gen?.data.api_latency_ms);
  return apiMs !== null ? `${formatMs(res.latency_ms)} · API ${formatMs(apiMs)}` : formatMs(res.latency_ms);
}
