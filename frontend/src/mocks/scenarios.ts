/**
 * Escenarios precargados del modo demo offline (SPEC §5.6 y §11).
 *
 * Cada escenario construye un `QueryResponse` completo y realista (con su traza de
 * etapas) a partir de la consulta, el rol y la versión. Cubren: consulta nominal con citas,
 * abstención, denegación por rol, estado de ticket, clasificación, creación con borrador y
 * confirmación humana (HITL), inyección bloqueada, comparación 17.0 vs 18.0 y fuera de
 * dominio. Las listas de recuperación se calculan con la misma fórmula RRF del backend
 * (k = 60, pesos 0.5/0.5) para que el inspector muestre números coherentes.
 */
import {
  ABSTENTION_MESSAGE,
  type DemoUser,
  type ErpVersion,
  type Intent,
  type PendingAction,
  type QueryRequest,
  type QueryResponse,
  type SourceRef,
  type TicketDraft,
  type TraceEvent,
  type Usage,
  type UserRole,
} from "../api/types";
import { MOCK_CHUNKS } from "./corpus";
import { MOCK_USERS } from "./users";

/* ------------------------------------------------------------------------------------ */
/* Utilidades                                                                            */
/* ------------------------------------------------------------------------------------ */

const ROLE_LEVEL: Record<UserRole, number> = { portal_user: 0, n1: 1, n2: 2, admin: 3 };
const TAU = 0.3;
const RRF_K = 60;

/** Fila resumida de una lista de recuperación, tal como viaja en `TraceEvent.data.hits`. */
export interface HitRow {
  chunk_id: string;
  source: string;
  page: number;
  section_path: string;
  score: number;
  rank: number;
}

/** Metadatos de chunks "distractores" que aparecen en las listas pero no se citan. */
const DISTRACTORS: Record<string, Omit<HitRow, "score" | "rank" | "chunk_id">> = {
  "odoo-18.0-inventario::p0009::c000": {
    source: "odoo-18.0-inventario.pdf",
    page: 9,
    section_path: "Inventario > Almacenes > Rutas de varios pasos",
  },
  "odoo-18.0-inventario::p0011::c001": {
    source: "odoo-18.0-inventario.pdf",
    page: 11,
    section_path: "Inventario > Ubicaciones > Sububicaciones",
  },
  "odoo-18.0-inventario::p0016::c000": {
    source: "odoo-18.0-inventario.pdf",
    page: 16,
    section_path: "Inventario > Recepciones > Tres pasos con control de calidad",
  },
  "odoo-18.0-compras::p0004::c001": {
    source: "odoo-18.0-compras.pdf",
    page: 4,
    section_path: "Compras > Órdenes de compra > Recepción de productos",
  },
  "odoo-18.0-compras::p0007::c000": {
    source: "odoo-18.0-compras.pdf",
    page: 7,
    section_path: "Compras > Proveedores > Acuerdos de compra",
  },
  "odoo-17.0-inventario::p0008::c002": {
    source: "odoo-17.0-inventario.pdf",
    page: 8,
    section_path: "Inventario > Almacenes > Rutas",
  },
  "odoo-17.0-inventario::p0010::c000": {
    source: "odoo-17.0-inventario.pdf",
    page: 10,
    section_path: "Inventario > Ubicaciones",
  },
  "odoo-18.0-facturacion::p0019::c000": {
    source: "odoo-18.0-facturacion.pdf",
    page: 19,
    section_path: "Facturación > Diarios > Diario de ventas",
  },
  "RB-N1-007@v1.0::p0002::c001": {
    source: "RB-N1-007_accesos_portal_v1.0.docx",
    page: 2,
    section_path: "RB-N1-007 > Restablecer acceso",
  },
  "odoo-18.0-ventas::p0006::c000": {
    source: "odoo-18.0-ventas.pdf",
    page: 6,
    section_path: "Ventas > Cotizaciones > Integraciones de comercio electrónico",
  },
};

function hitMeta(chunkId: string): Omit<HitRow, "score" | "rank"> {
  const chunk = MOCK_CHUNKS[chunkId];
  if (chunk) {
    return {
      chunk_id: chunkId,
      source: chunk.source,
      page: chunk.page,
      section_path: chunk.section_path,
    };
  }
  const d = DISTRACTORS[chunkId];
  if (!d) throw new Error(`Chunk de demo desconocido: ${chunkId}`);
  return { chunk_id: chunkId, ...d };
}

/** Lista ordenada a partir de pares `[chunk_id, score]` (ya en orden de rango). */
function ranked(pairs: [string, number][]): HitRow[] {
  return pairs.map(([id, score], i) => ({ ...hitMeta(id), score, rank: i + 1 }));
}

/**
 * Fusión Reciprocal Rank Fusion: `score = Σ w_i / (k + rank_i)` con k = 60 (SPEC §5.2).
 * Un chunk ausente de una lista no aporta nada desde esa lista.
 */
export function rrfFuse(lists: HitRow[][], weights: number[], k = RRF_K): HitRow[] {
  const acc = new Map<string, { meta: Omit<HitRow, "score" | "rank">; score: number }>();
  lists.forEach((list, li) => {
    const w = weights[li] ?? 1;
    for (const hit of list) {
      const prev = acc.get(hit.chunk_id);
      const add = w / (k + hit.rank);
      if (prev) prev.score += add;
      else {
        const meta = {
          chunk_id: hit.chunk_id,
          source: hit.source,
          page: hit.page,
          section_path: hit.section_path,
        };
        acc.set(hit.chunk_id, { meta, score: add });
      }
    }
  });
  return [...acc.values()]
    .sort((a, b) => b.score - a.score)
    .map((v, i) => ({ ...v.meta, score: Number(v.score.toFixed(5)), rank: i + 1 }));
}

function sourceRef(chunkId: string): SourceRef {
  const meta = hitMeta(chunkId);
  return { source: meta.source, page: meta.page, chunk_id: chunkId };
}

let traceCounter = 0;
/** Identificador de traza legible y único dentro de la sesión. */
function newTraceId(): string {
  traceCounter += 1;
  const rand = Math.random().toString(16).slice(2, 10).padEnd(8, "0");
  return `trc_${rand}${traceCounter.toString(16).padStart(4, "0")}`;
}

function usage(input: number, output: number, cacheRead = 0, calls = 1): Usage {
  // Tarifa ilustrativa (USD por millón de tokens) calibrada para que una consulta típica
  // cueste ≈ US$0.0005, el costo medido con Haiku en el sondeo del Agent SDK (SPEC §12).
  const cost = (input * 0.12 + output * 0.6 + cacheRead * 0.012) / 1_000_000;
  return {
    llm_calls: calls,
    input_tokens: input,
    output_tokens: output,
    cache_read_tokens: cacheRead,
    cache_creation_tokens: 0,
    cost_usd: Number(cost.toFixed(6)),
  };
}

function ev(
  stage: string,
  status: TraceEvent["status"],
  duration: number,
  data: Record<string, unknown> = {},
): TraceEvent {
  return { stage, status, duration_ms: duration, data };
}

function totalLatency(events: TraceEvent[]): number {
  return Number(events.reduce((s, e) => s + e.duration_ms, 0).toFixed(1));
}

function finish(
  partial: Omit<QueryResponse, "latency_ms" | "trace_id" | "pending_action" | "blocked" | "block_reason"> &
    Partial<Pick<QueryResponse, "pending_action" | "blocked" | "block_reason">>,
): QueryResponse {
  return {
    trace_id: newTraceId(),
    pending_action: null,
    blocked: false,
    block_reason: null,
    ...partial,
    latency_ms: totalLatency(partial.events),
  };
}

function inputGuardOk(extra: Record<string, unknown> = {}): TraceEvent {
  return ev("input_guard", "ok", 7.8, {
    risk_score: 0.04,
    flags: [],
    pii_masked: 0,
    normalized: true,
    ...extra,
  });
}

function router(intent: Intent, confidence: number, rationale: string): TraceEvent {
  return ev("router", "ok", 41.6, { intent, confidence, rationale });
}

/* ------------------------------------------------------------------------------------ */
/* Pipeline documental                                                                   */
/* ------------------------------------------------------------------------------------ */

interface DocPipelineSpec {
  role: UserRole;
  erpVersions: ErpVersion[];
  bm25: [string, number][];
  dense: [string, number][];
  /** Score del reranker (sigmoide, [0,1]) por chunk; se ordena de mayor a menor. */
  rerank: Record<string, number>;
  filteredOut: number;
  permission?: { status: TraceEvent["status"]; data: Record<string, unknown> };
}

interface DocPipelineResult {
  events: TraceEvent[];
  topScore: number;
  evidenceOk: boolean;
}

/** Construye los eventos de recuperación → rerank → gate → permisos de una consulta. */
function docPipeline(spec: DocPipelineSpec): DocPipelineResult {
  const bm25 = ranked(spec.bm25);
  const dense = ranked(spec.dense);
  const fused = rrfFuse([bm25, dense], [0.5, 0.5]);
  const reranked = fused
    .slice(0, 20)
    .filter((h) => spec.rerank[h.chunk_id] !== undefined)
    .map((h) => ({ ...h, score: spec.rerank[h.chunk_id] ?? 0 }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)
    .map((h, i) => ({ ...h, rank: i + 1 }));
  const topScore = reranked[0]?.score ?? 0;
  const evidenceOk = topScore >= TAU && reranked.length > 0;
  const filters = {
    role: spec.role,
    erp_version: spec.erpVersions.length === 1 ? spec.erpVersions[0] : spec.erpVersions,
    status: "vigente",
    include_obsolete: false,
  };
  const events: TraceEvent[] = [
    ev("retrieval.bm25", "ok", 11.4, {
      candidates: 20,
      filters,
      filtered_out: spec.filteredOut,
      hits: bm25,
    }),
    ev("retrieval.dense", "ok", 63.9, {
      candidates: 20,
      embedder: "bge-m3",
      filters,
      hits: dense,
    }),
    ev("retrieval.fusion", "ok", 0.9, { method: "rrf", k: RRF_K, weights: [0.5, 0.5], hits: fused.slice(0, 10) }),
    ev("rerank", "ok", 382.5, {
      model: "BAAI/bge-reranker-v2-m3",
      input: Math.min(20, fused.length),
      top_n: 5,
      hits: reranked,
    }),
    ev("evidence_gate", evidenceOk ? "ok" : "blocked", 0.3, {
      tau: TAU,
      top_score: topScore,
      evidence_ok: evidenceOk,
      authorized_hits: reranked.length,
    }),
    ev(
      "permission_check",
      spec.permission?.status ?? "ok",
      0.4,
      spec.permission?.data ?? { role: spec.role, authorized: reranked.length, filtered_out: 0 },
    ),
  ];
  return { events, topScore, evidenceOk };
}

function generation(input: number, output: number, latency = 3870): TraceEvent {
  return ev("generation", "ok", latency, {
    provider: "agent_sdk",
    model: "claude-haiku-5-5",
    effort: "low",
    input_tokens: input,
    output_tokens: output,
    api_latency_ms: Math.round(latency * 0.68),
    json_valid: true,
  });
}

function outputGuard(citations: number): TraceEvent {
  return ev("output_guard", "ok", 5.6, {
    schema_valid: true,
    citations_checked: citations,
    citations_valid: citations,
    pii_found: 0,
    canary_leak: false,
  });
}

const respond = (): TraceEvent => ev("respond", "ok", 0.8, {});

/* ------------------------------------------------------------------------------------ */
/* Escenarios                                                                            */
/* ------------------------------------------------------------------------------------ */

const INV18_C1 = "odoo-18.0-inventario::p0014::c001";
const INV18_C2 = "odoo-18.0-inventario::p0014::c002";
const INV18_P15 = "odoo-18.0-inventario::p0015::c000";
const INV17_C1 = "odoo-17.0-inventario::p0012::c001";
const INV17_P13 = "odoo-17.0-inventario::p0013::c000";
const RB_N1 = "RB-N1-004@v1.2::p0002::c000";
const RB_N2 = "RB-N2-003@v1.1::p0003::c000";
const POL_SLA = "POL-SLA-001@v2.0::p0003::c000";

/** Consulta nominal: recepción en dos pasos (respuesta dependiente de versión y rol). */
function nominalReceipts(req: QueryRequest): QueryResponse {
  const canRunbook = ROLE_LEVEL[req.role] >= 1;
  const is17 = req.erp_version === "17.0";
  const main = is17 ? INV17_C1 : INV18_C1;
  const second = is17 ? INV17_P13 : INV18_C2;
  const processing = is17 ? INV17_P13 : INV18_P15;
  const d1 = is17 ? "odoo-17.0-inventario::p0008::c002" : "odoo-18.0-inventario::p0009::c000";
  const d2 = is17 ? "odoo-17.0-inventario::p0010::c000" : "odoo-18.0-inventario::p0011::c001";

  const bm25: [string, number][] = is17
    ? [
        [main, 17.42],
        [d1, 13.05],
        [second, 12.61],
        ...(canRunbook ? ([[RB_N1, 9.87]] as [string, number][]) : []),
        [d2, 8.12],
        ["odoo-18.0-compras::p0004::c001", 6.4],
      ]
    : [
        [second, 16.84],
        [main, 15.97],
        [d1, 13.12],
        [processing, 11.5],
        [d2, 9.03],
        ...(canRunbook ? ([[RB_N1, 8.76]] as [string, number][]) : []),
        ["odoo-18.0-compras::p0004::c001", 6.91],
        ["odoo-18.0-inventario::p0016::c000", 6.22],
      ];
  const dense: [string, number][] = is17
    ? [
        [main, 0.812],
        ...(canRunbook ? ([[RB_N1, 0.764]] as [string, number][]) : []),
        [second, 0.741],
        [d1, 0.688],
        [d2, 0.602],
      ]
    : [
        [main, 0.823],
        [second, 0.807],
        ...(canRunbook ? ([[RB_N1, 0.771]] as [string, number][]) : []),
        [processing, 0.733],
        ["odoo-18.0-inventario::p0016::c000", 0.689],
        [d1, 0.664],
        [d2, 0.597],
      ];
  const rerank: Record<string, number> = is17
    ? { [main]: 0.93, [second]: 0.81, [d1]: 0.37, [d2]: 0.22, ...(canRunbook ? { [RB_N1]: 0.86 } : {}) }
    : {
        [main]: 0.94,
        [second]: 0.91,
        [processing]: 0.62,
        [d1]: 0.41,
        ["odoo-18.0-inventario::p0016::c000"]: 0.33,
        ...(canRunbook ? { [RB_N1]: 0.88 } : {}),
      };

  const pipe = docPipeline({
    role: req.role,
    erpVersions: [req.erp_version],
    bm25,
    dense,
    rerank,
    filteredOut: canRunbook ? 0 : 3,
    permission: {
      status: "ok",
      data: { role: req.role, authorized: 5, filtered_out: canRunbook ? 0 : 3, revalidated: true },
    },
  });

  const cited = canRunbook ? [main, second, RB_N1] : [main, second];
  const option = is17
    ? "*Recibir mercancía en entrada y luego existencias (2 pasos)*"
    : "*Recibir y luego almacenar (2 pasos)*";
  const settings = is17
    ? "activa **Rutas de varios pasos** en la sección *Almacén* (Ubicaciones de almacenamiento se activa sola) [1]."
    : "en la sección *Almacén* activa **Ubicaciones de almacenamiento** y **Rutas de varios pasos**, y guarda [1].";
  const step3 = is17
    ? "3. Valida la recepción desde **Inventario › Operaciones › Recepciones**; el traslado interno queda *En espera* hasta entonces y luego se valida en **Operaciones › Internas** [2]."
    : "3. Odoo genera automáticamente la recepción y un traslado interno de `WH/Input` a `WH/Stock`; valida ambos para completar el flujo [2].";
  const lines = [
    `Para activar la recepción en dos pasos en Odoo ${req.erp_version}:`,
    "",
    `1. Ve a **Inventario › Configuración › Ajustes** y ${settings}`,
    `2. Abre **Inventario › Configuración › Almacenes**, selecciona el almacén y en **Envíos entrantes** elige ${option} [${is17 ? 1 : 2}].`,
    step3,
  ];
  if (canRunbook) {
    lines.push(
      "",
      "Antes del cambio confirma que no queden recepciones en estado *Listo* y, al terminar, pide al cliente una recepción de prueba y registra el cambio con la etiqueta `ruta-almacen` [3].",
    );
  }
  const events = [
    inputGuardOk(),
    router("doc_query", 0.96, "Pregunta procedimental sobre configuración de Inventario"),
    ...pipe.events,
    generation(2140 + cited.length * 180, 286),
    outputGuard(cited.length),
    respond(),
  ];
  return finish({
    intent: "doc_query",
    response: {
      answer: lines.join("\n"),
      sources: cited.map(sourceRef),
      abstained: false,
      tool_used: null,
      needs_human_review: false,
    },
    usage: usage(2140 + cited.length * 180, 286, 1680),
    events,
  });
}

/** Comparación explícita entre versiones (se consultan ambas versiones del manual). */
function versionComparison(req: QueryRequest): QueryResponse {
  const pipe = docPipeline({
    role: req.role,
    erpVersions: ["17.0", "18.0"],
    bm25: [
      [INV17_C1, 16.9],
      [INV18_C1, 16.2],
      [INV18_C2, 14.8],
      ["odoo-17.0-inventario::p0008::c002", 11.7],
      ["odoo-18.0-inventario::p0009::c000", 11.1],
      [INV17_P13, 9.4],
    ],
    dense: [
      [INV18_C1, 0.834],
      [INV17_C1, 0.829],
      [INV18_C2, 0.792],
      [INV17_P13, 0.701],
      ["odoo-18.0-inventario::p0009::c000", 0.655],
    ],
    rerank: {
      [INV17_C1]: 0.92,
      [INV18_C1]: 0.92,
      [INV18_C2]: 0.87,
      [INV17_P13]: 0.58,
      ["odoo-18.0-inventario::p0009::c000"]: 0.36,
    },
    filteredOut: 0,
  });
  const answer = [
    "El flujo es el mismo en ambas versiones, pero cambian la activación y el nombre de la opción:",
    "",
    "- **Odoo 17.0:** basta con activar *Rutas de varios pasos*; *Ubicaciones de almacenamiento* se habilita automáticamente. La opción del almacén se llama *Recibir mercancía en entrada y luego existencias (2 pasos)* [1].",
    "- **Odoo 18.0:** en *Ajustes › Almacén* se activan explícitamente *Ubicaciones de almacenamiento* y *Rutas de varios pasos* [2], y la opción pasa a llamarse *Recibir y luego almacenar (2 pasos)* [3].",
    "",
    "Si el cliente migra de 17.0 a 18.0, la ruta configurada se conserva; solo cambia la etiqueta en la interfaz.",
  ].join("\n");
  const cited = [INV17_C1, INV18_C1, INV18_C2];
  return finish({
    intent: "doc_query",
    response: {
      answer,
      sources: cited.map(sourceRef),
      abstained: false,
      tool_used: null,
      needs_human_review: false,
    },
    usage: usage(2980, 241, 1680),
    events: [
      inputGuardOk(),
      router("doc_query", 0.93, "Consulta comparativa entre versiones: se consultan 17.0 y 18.0"),
      ...pipe.events,
      generation(2980, 241, 4210),
      outputGuard(3),
      respond(),
    ],
  });
}

/** Abstención: no hay evidencia suficiente (top score del reranker < τ). */
function abstention(req: QueryRequest): QueryResponse {
  const pipe = docPipeline({
    role: req.role,
    erpVersions: [req.erp_version],
    bm25: [
      ["odoo-18.0-compras::p0004::c001", 7.12],
      ["odoo-18.0-compras::p0007::c000", 6.58],
      ["odoo-18.0-ventas::p0006::c000", 5.03],
      ["odoo-18.0-inventario::p0009::c000", 3.9],
    ],
    dense: [
      ["odoo-18.0-ventas::p0006::c000", 0.512],
      ["odoo-18.0-compras::p0007::c000", 0.497],
      ["odoo-18.0-compras::p0004::c001", 0.471],
    ],
    rerank: {
      ["odoo-18.0-ventas::p0006::c000"]: 0.19,
      ["odoo-18.0-compras::p0007::c000"]: 0.14,
      ["odoo-18.0-compras::p0004::c001"]: 0.09,
      ["odoo-18.0-inventario::p0009::c000"]: 0.03,
    },
    filteredOut: 0,
  });
  return finish({
    intent: "doc_query",
    response: {
      answer: ABSTENTION_MESSAGE,
      sources: [],
      abstained: true,
      tool_used: null,
      needs_human_review: false,
    },
    usage: usage(0, 0, 0, 0),
    events: [
      inputGuardOk(),
      router("doc_query", 0.71, "Pregunta sobre integración con un sistema externo"),
      ...pipe.events,
      ev("generation", "skipped", 0, { reason: "evidence_gate: top_score < τ" }),
      ev("output_guard", "ok", 1.2, { schema_valid: true, abstention_coherent: true }),
      respond(),
    ],
  });
}

/** Runbook N2 restringido: se responde a n2/admin y se deniega a portal/n1. */
function restrictedRunbook(req: QueryRequest): QueryResponse {
  const allowed = ROLE_LEVEL[req.role] >= 2;
  if (allowed) {
    const pipe = docPipeline({
      role: req.role,
      erpVersions: [req.erp_version],
      bm25: [
        [RB_N2, 15.3],
        ["odoo-18.0-inventario::p0011::c001", 4.2],
      ],
      dense: [
        [RB_N2, 0.842],
        ["odoo-18.0-facturacion::p0019::c000", 0.41],
      ],
      rerank: { [RB_N2]: 0.89, ["odoo-18.0-facturacion::p0019::c000"]: 0.08 },
      filteredOut: 0,
    });
    return finish({
      intent: "doc_query",
      response: {
        answer: [
          "La reindexación de producción solo puede ejecutarse dentro de la **ventana de mantenimiento aprobada por el cliente** y con un **respaldo verificado del día** [1].",
          "",
          "1. Notifica al administrador de la cuenta.",
          "2. Detén los trabajos programados.",
          "3. Registra la hora de inicio en el ticket antes de comenzar [1].",
        ].join("\n"),
        sources: [sourceRef(RB_N2)],
        abstained: false,
        tool_used: null,
        needs_human_review: false,
      },
      usage: usage(1870, 162, 1680),
      events: [
        inputGuardOk(),
        router("doc_query", 0.94, "Consulta de procedimiento interno de plataforma"),
        ...pipe.events,
        generation(1870, 162, 3320),
        outputGuard(1),
        respond(),
      ],
    });
  }
  const pipe = docPipeline({
    role: req.role,
    erpVersions: [req.erp_version],
    bm25: [
      ["odoo-18.0-inventario::p0011::c001", 4.2],
      ["odoo-18.0-facturacion::p0019::c000", 3.1],
    ],
    dense: [
      ["odoo-18.0-facturacion::p0019::c000", 0.41],
      ["odoo-18.0-inventario::p0011::c001", 0.33],
    ],
    rerank: {
      ["odoo-18.0-facturacion::p0019::c000"]: 0.08,
      ["odoo-18.0-inventario::p0011::c001"]: 0.05,
    },
    filteredOut: 6,
    permission: {
      status: "blocked",
      data: {
        role: req.role,
        authorized: 0,
        filtered_out: 6,
        required_role: "n2",
        denied: true,
        reason: "La evidencia relevante requiere un rol superior al del usuario",
      },
    },
  });
  return finish({
    intent: "doc_query",
    response: {
      answer: ABSTENTION_MESSAGE,
      sources: [],
      abstained: true,
      tool_used: null,
      needs_human_review: true,
    },
    usage: usage(0, 0, 0, 0),
    events: [
      inputGuardOk(),
      router("doc_query", 0.92, "Consulta de procedimiento interno de plataforma"),
      ...pipe.events,
      ev("generation", "skipped", 0, { reason: "permission_check: sin evidencia autorizada" }),
      ev("output_guard", "ok", 1.1, { schema_valid: true, abstention_coherent: true }),
      respond(),
    ],
  });
}

/** Ticket demo con alcance por rol (SPEC §3). */
interface MockTicket {
  ticket_id: string;
  summary: string;
  status: "nuevo" | "en_progreso" | "en_espera" | "resuelto" | "cerrado";
  priority: "baja" | "media" | "alta" | "critica";
  category: string;
  module: string;
  account: string;
  reported_by: string;
  assigned_to: string;
  created_at: string;
  updated_at: string;
  sla: {
    policy: string;
    response_due_at: string;
    resolution_due_at: string;
    paused: boolean;
    breached: boolean;
    elapsed_ratio: number;
  };
}

function hoursFromNow(h: number): string {
  return new Date(Date.now() + h * 3_600_000).toISOString();
}

function ticketFixture(id: string): MockTicket {
  if (id === "TCK-00042") {
    return {
      ticket_id: id,
      summary: "Error al validar recepción con lote duplicado",
      status: "en_progreso",
      priority: "alta",
      category: "error_funcional",
      module: "inventario",
      account: "Comercial Los Volcanes",
      reported_by: "USR-0001",
      assigned_to: "Andrea Castillo (N1)",
      created_at: hoursFromNow(-9.5),
      updated_at: hoursFromNow(-1.2),
      sla: {
        policy: "POL-SLA-001 · prioridad alta",
        response_due_at: hoursFromNow(-5.5),
        resolution_due_at: hoursFromNow(14.4),
        paused: false,
        breached: false,
        elapsed_ratio: 0.4,
      },
    };
  }
  return {
    ticket_id: id,
    summary: "Diferencia de costo promedio tras ajuste de inventario",
    status: "en_espera",
    priority: "media",
    category: "configuracion",
    module: "inventario",
    account: "Distribuidora Atitlán",
    reported_by: "USR-0007",
    assigned_to: "Diego Herrera (N2)",
    created_at: hoursFromNow(-52),
    updated_at: hoursFromNow(-20),
    sla: {
      policy: "POL-SLA-001 · prioridad media",
      response_due_at: hoursFromNow(-28),
      resolution_due_at: hoursFromNow(68),
      paused: true,
      breached: false,
      elapsed_ratio: 0.43,
    },
  };
}

const STATUS_LABEL: Record<MockTicket["status"], string> = {
  nuevo: "Nuevo",
  en_progreso: "En progreso",
  en_espera: "En espera del cliente",
  resuelto: "Resuelto",
  cerrado: "Cerrado",
};

function ticketStatus(req: QueryRequest, ticketId: string): QueryResponse {
  const ticket = ticketFixture(ticketId);
  const inScope = req.role !== "portal_user" || ticket.reported_by === req.user_id;
  const toolEvents: TraceEvent[] = [
    inputGuardOk(),
    router("ticket_status", 0.99, `Regla determinista: identificador ${ticketId}`),
  ];
  if (!inScope) {
    toolEvents.push(
      ev("tool.get_ticket_status", "ok", 3.1, {
        ticket_id: ticketId,
        found: true,
        read_only: true,
      }),
      ev("permission_check", "blocked", 0.3, {
        role: req.role,
        scope: "solo tickets reportados por el usuario",
        denied: true,
        reason: "El ticket no pertenece al alcance del rol",
      }),
      ev("generation", "skipped", 0, { reason: "permission_check" }),
      ev("output_guard", "ok", 0.9, { schema_valid: true }),
      respond(),
    );
    return finish({
      intent: "ticket_status",
      response: {
        answer:
          "No puedo mostrar ese ticket: solo puedes consultar los tickets que tú reportaste. Si crees que es un error, contacta a tu analista asignado.",
        sources: [],
        abstained: true,
        tool_used: "get_ticket_status",
        needs_human_review: false,
      },
      usage: usage(0, 0, 0, 0),
      events: toolEvents,
    });
  }
  const canPolicy = ROLE_LEVEL[req.role] >= 1;
  const remaining = new Date(ticket.sla.resolution_due_at).getTime() - Date.now();
  const hours = Math.max(0, Math.round(remaining / 3_600_000));
  const sources = canPolicy ? [sourceRef(POL_SLA)] : [];
  const slaText = ticket.sla.paused
    ? "El contador de SLA está **pausado** mientras se espera respuesta del cliente"
    : `Quedan **${hours} h** para el vencimiento del SLA de resolución`;
  const answer = [
    `El ticket **${ticket.ticket_id}** («${ticket.summary}», cuenta ${ticket.account}) está **${STATUS_LABEL[ticket.status]}**, asignado a ${ticket.assigned_to}, con prioridad **${ticket.priority}**.`,
    "",
    `${slaText}${canPolicy ? " según la política de niveles de servicio [1]." : "."}`,
  ].join("\n");
  toolEvents.push(
    ev("permission_check", "ok", 0.3, {
      role: req.role,
      scope:
        req.role === "portal_user"
          ? "tickets reportados"
          : req.role === "n1"
            ? "tickets asignados"
            : req.role === "n2"
              ? "cartera de cuentas"
              : "todos",
      tool_allowed: true,
    }),
    ev("tool.get_ticket_status", "ok", 3.4, {
      read_only: true,
      sql: "SELECT … FROM tickets WHERE ticket_id = ? (parametrizado, mode=ro)",
      ticket,
    }),
    generation(980, 118, 2610),
    outputGuard(sources.length),
    respond(),
  );
  return finish({
    intent: "ticket_status",
    response: {
      answer,
      sources,
      abstained: false,
      tool_used: "get_ticket_status",
      needs_human_review: false,
    },
    usage: usage(980, 118, 840),
    events: toolEvents,
  });
}

function classify(req: QueryRequest): QueryResponse {
  const text = req.query.replace(/^\s*clasifica(r)?( este ticket)?\s*:?\s*/i, "");
  const vague = text.split(/\s+/).filter(Boolean).length < 7;
  const result = vague
    ? {
        category: "configuracion",
        module: "general",
        priority: "media",
        confidence: 0.41,
        threshold: 0.6,
        model: "bge-m3 + regresión logística",
        probabilities: [
          { label: "configuracion", p: 0.41 },
          { label: "capacitacion", p: 0.27 },
          { label: "error_funcional", p: 0.19 },
          { label: "acceso_permisos", p: 0.08 },
          { label: "integracion", p: 0.05 },
        ],
      }
    : {
        category: "error_funcional",
        module: "facturacion",
        priority: "alta",
        confidence: 0.87,
        threshold: 0.6,
        model: "bge-m3 + regresión logística",
        probabilities: [
          { label: "error_funcional", p: 0.87 },
          { label: "configuracion", p: 0.07 },
          { label: "integracion", p: 0.03 },
          { label: "migracion", p: 0.02 },
          { label: "acceso_permisos", p: 0.01 },
        ],
      };
  const review = result.confidence < result.threshold;
  // La documentación oficial es pública: cualquier rol puede recibir la cita de apoyo.
  const sources = vague ? [] : [sourceRef("odoo-18.0-facturacion::p0021::c002")];
  const answer = review
    ? `Clasificación tentativa: **${label(result.category)}** · módulo **${label(result.module)}** · prioridad **${result.priority}**. La confianza (${result.confidence.toFixed(2)}) está bajo el umbral de ${result.threshold.toFixed(2)}, así que la marco para **revisión humana**.`
    : [
        `Clasificación sugerida: **${label(result.category)}** · módulo **${label(result.module)}** · prioridad **${result.priority}** (confianza ${result.confidence.toFixed(2)}).`,
        "",
        "Pista para el analista: Odoo bloquea la confirmación de la factura si falta la cuenta de ingresos en el producto o su categoría [1].",
      ].join("\n");
  return finish({
    intent: "ticket_classify",
    response: {
      answer,
      sources,
      abstained: false,
      tool_used: "classify_ticket",
      needs_human_review: review,
    },
    usage: usage(vague ? 640 : 1520, vague ? 92 : 131, 840),
    events: [
      inputGuardOk(),
      router("ticket_classify", 0.97, "Regla determinista: verbo «clasifica»"),
      ev("permission_check", "ok", 0.2, { role: req.role, tool_allowed: true }),
      ev("tool.classify_ticket", "ok", 48.7, { ...result, needs_human_review: review }),
      generation(vague ? 640 : 1520, vague ? 92 : 131, 2240),
      outputGuard(sources.length),
      respond(),
    ],
  });
}

const CATEGORIES: TicketDraft["category"][] = [
  "configuracion",
  "error_funcional",
  "acceso_permisos",
  "rendimiento",
  "migracion",
  "capacitacion",
  "integracion",
];
const PRIORITIES: TicketDraft["priority"][] = ["baja", "media", "alta", "critica"];
const MODULES: TicketDraft["module"][] = [
  "inventario",
  "compras",
  "ventas",
  "facturacion",
  "accesos",
  "actualizacion",
  "general",
];

/** Lee una pista estructurada `[clave=valor; …]` si el valor pertenece al vocabulario. */
function hint<T extends string>(hints: string, key: string, allowed: T[]): T | null {
  const m = new RegExp(`${key}\\s*=\\s*([a-z_]+)`, "i").exec(hints);
  const value = m?.[1]?.toLowerCase() as T | undefined;
  return value && allowed.includes(value) ? value : null;
}

/**
 * Construye el borrador a partir del texto. Acepta pistas `[categoría=…; prioridad=…;
 * módulo=…]` (las envía la consola al regenerar un borrador editado) y un separador " — "
 * entre resumen y descripción.
 */
export function draftFromQuery(query: string): TicketDraft {
  const raw = query.replace(/^\s*(crea|crear|abre|abrir|registra|levanta)( un)? ticket\s*:?\s*/i, "").trim();
  const hintsMatch = /\[([^\]]*)\]\s*$/.exec(raw);
  const hints = hintsMatch?.[1] ?? "";
  const body = hintsMatch ? raw.slice(0, hintsMatch.index).trim() : raw;
  const [head = "", ...tail] = body.split(" — ");
  const sentence = head.charAt(0).toUpperCase() + head.slice(1);
  const summary = sentence.length > 90 ? `${sentence.slice(0, 87).trimEnd()}…` : sentence;
  const detail = tail.join(" — ").trim();
  const lower = body.toLowerCase();
  const integration = /sincroniz|tienda|e-?commerce|api|integraci/.test(lower);
  return {
    summary: summary.length >= 10 ? summary : `Solicitud de soporte: ${summary}`,
    description:
      detail ||
      `${sentence}${sentence.endsWith(".") ? "" : "."} Reportado desde la consola DocuERP; requiere diagnóstico del analista asignado.`,
    category: hint(hints, "categor[ií]a", CATEGORIES) ?? (integration ? "integracion" : "error_funcional"),
    priority: hint(hints, "prioridad", PRIORITIES) ?? (/urgente|cr[ií]tic|detenid|no puede/.test(lower) ? "critica" : "alta"),
    module:
      hint(hints, "m[oó]dulo", MODULES) ??
      (/factur/.test(lower) ? "facturacion" : /compra/.test(lower) ? "compras" : "inventario"),
    account_id: "ACC-0007",
  };
}

function randomToken(): string {
  const bytes = new Uint8Array(24);
  if (typeof crypto !== "undefined" && "getRandomValues" in crypto) crypto.getRandomValues(bytes);
  else for (let i = 0; i < bytes.length; i += 1) bytes[i] = Math.floor(Math.random() * 256);
  return `cft_${[...bytes].map((b) => b.toString(16).padStart(2, "0")).join("")}`;
}

/** Tokens emitidos por el mock: borrador y caducidad, para validar la confirmación. */
/** Etiquetas legibles para el texto de las respuestas simuladas. */
const LABEL: Record<string, string> = {
  error_funcional: "error funcional",
  configuracion: "configuración",
  integracion: "integración",
  capacitacion: "capacitación",
  acceso_permisos: "accesos y permisos",
  migracion: "migración",
  rendimiento: "rendimiento",
  facturacion: "Facturación",
  inventario: "Inventario",
  compras: "Compras",
  ventas: "Ventas",
  accesos: "Accesos",
  actualizacion: "Actualización",
  general: "General",
};
const label = (key: string): string => LABEL[key] ?? key;

const issuedTokens = new Map<string, { draft: TicketDraft; expiresAt: number; used: boolean; role: UserRole }>();

function createTicketDraft(req: QueryRequest): QueryResponse {
  const draft = draftFromQuery(req.query);
  const token = randomToken();
  const expiresAt = Date.now() + 10 * 60_000;
  issuedTokens.set(token, { draft, expiresAt, used: false, role: req.role });
  const pending: PendingAction = {
    action: "create_ticket",
    draft,
    confirmation_token: token,
    expires_at: new Date(expiresAt).toISOString(),
  };
  const portal = req.role === "portal_user";
  return finish({
    intent: "ticket_create",
    response: {
      answer: portal
        ? "Preparé un borrador de ticket con tu reporte. Revísalo y confírmalo: un analista lo validará antes de asignarlo."
        : "Preparé un borrador de ticket. Revísalo y confírmalo: no se escribe nada en la base de tickets hasta tu confirmación.",
      sources: [],
      abstained: false,
      tool_used: "create_ticket",
      needs_human_review: true,
    },
    pending_action: pending,
    usage: usage(1210, 204, 840),
    events: [
      inputGuardOk(),
      router("ticket_create", 0.98, "Regla determinista: verbo de creación «crea un ticket»"),
      ev("permission_check", "ok", 0.2, {
        role: req.role,
        tool_allowed: true,
        escalated: portal,
        session_quota: "1 de 3",
      }),
      generation(1210, 204, 2980),
      ev("tool.create_ticket", "ok", 2.7, {
        mode: "draft",
        written: false,
        token_ttl_s: 600,
        single_use: true,
        signature: "HMAC-SHA256",
      }),
      ev("output_guard", "ok", 2.1, { schema_valid: true, pii_found: 0 }),
      respond(),
    ],
  });
}

/** Ejecuta la confirmación HITL de un borrador emitido por este mock. */
export function confirmMockAction(token: string, role: UserRole): QueryResponse {
  const issued = issuedTokens.get(token);
  const baseEvents = [ev("input_guard", "ok", 1.9, { confirmation: true })];
  if (!issued || issued.used || issued.expiresAt < Date.now() || issued.role !== role) {
    const reason = !issued
      ? "token desconocido"
      : issued.used
        ? "token ya utilizado"
        : issued.expiresAt < Date.now()
          ? "token expirado"
          : "el rol no coincide con el del borrador";
    return finish({
      intent: "ticket_create",
      response: {
        answer: `No se pudo confirmar la acción (${reason}). Genera un borrador nuevo.`,
        sources: [],
        abstained: false,
        tool_used: "create_ticket",
        needs_human_review: true,
      },
      usage: usage(0, 0, 0, 0),
      events: [
        ...baseEvents,
        ev("tool.create_ticket", "blocked", 0.6, { mode: "confirm", reason }),
        respond(),
      ],
    });
  }
  issued.used = true;
  const ticketId = `TCK-${String(318 + Math.floor(Math.random() * 40)).padStart(5, "0")}`;
  return finish({
    intent: "ticket_create",
    response: {
      answer: `Listo: creé el ticket **${ticketId}** («${issued.draft.summary}») con prioridad **${issued.draft.priority}** en la cola de ${label(issued.draft.module)}. ${role === "portal_user" ? "Un analista lo revisará antes de asignarlo." : "Quedó asignado a la cola N1."}`,
      sources: [],
      abstained: false,
      tool_used: "create_ticket",
      needs_human_review: role === "portal_user",
    },
    usage: usage(0, 0, 0, 0),
    events: [
      ...baseEvents,
      ev("permission_check", "ok", 0.3, { role, tool_allowed: true, token_valid: true }),
      ev("tool.create_ticket", "ok", 9.8, {
        mode: "confirm",
        written: true,
        ticket_id: ticketId,
        idempotency_key: token.slice(4, 20),
      }),
      respond(),
    ],
  });
}

function injectionBlocked(): QueryResponse {
  return finish({
    intent: "out_of_domain",
    response: {
      answer:
        "La consulta fue bloqueada por el filtro de seguridad de entrada. Si necesitas ayuda, reformúlala sin instrucciones dirigidas al asistente.",
      sources: [],
      abstained: false,
      tool_used: null,
      needs_human_review: false,
    },
    blocked: true,
    block_reason:
      "Intento de inyección de instrucciones: se pidió ignorar las reglas y revelar el prompt de sistema.",
    usage: usage(0, 0, 0, 0),
    events: [
      ev("input_guard", "blocked", 14.2, {
        risk_score: 0.97,
        flags: ["prompt_injection", "system_prompt_exfiltration"],
        matched_patterns: ["ignora (todas )?las instrucciones", "prompt de sistema"],
        semantic_similarity: 0.91,
        decoded: [],
      }),
      ev("router", "skipped", 0, {}),
      ev("respond", "ok", 0.6, {}),
    ],
  });
}

function outOfDomain(req: QueryRequest): QueryResponse {
  return finish({
    intent: "out_of_domain",
    response: {
      answer: ABSTENTION_MESSAGE,
      sources: [],
      abstained: true,
      tool_used: null,
      needs_human_review: false,
    },
    usage: usage(0, 0, 0, 0),
    events: [
      inputGuardOk(),
      router(
        "out_of_domain",
        0.88,
        `Similitud máxima al corpus 0.21 < 0.35 (consulta ajena al soporte ERP; rol ${req.role})`,
      ),
      ev("respond", "ok", 0.7, {}),
    ],
  });
}

/* ------------------------------------------------------------------------------------ */
/* Selección de escenario                                                                */
/* ------------------------------------------------------------------------------------ */

export type ScenarioId =
  | "injection"
  | "ticket_status"
  | "ticket_create"
  | "ticket_classify"
  | "version_compare"
  | "restricted"
  | "abstention"
  | "nominal"
  | "out_of_domain";

/** Normaliza para comparar sin mayúsculas ni acentos. */
function fold(s: string): string {
  return s
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase();
}

/**
 * Decide qué escenario corresponde a una consulta. El orden importa: la seguridad se
 * evalúa primero (igual que `input_guard` en el grafo real) y luego las reglas del router.
 */
export function matchScenario(query: string): ScenarioId {
  const q = fold(query);
  if (/ignora|olvida|ignore|system prompt|prompt de sistema|jailbreak|modo desarrollador/.test(q))
    return "injection";
  if (/\b(crea|crear|abre|abrir|registra|levanta)\b( un)? ticket/.test(q)) return "ticket_create";
  if (/\bclasifica/.test(q)) return "ticket_classify";
  if (/tck-\d{5}/.test(q)) return "ticket_status";
  if (/17(\.0)?\b.*\b(vs|frente|y|contra|comparad?)\b.*18|diferencias? entre/.test(q))
    return "version_compare";
  if (/reindex|base de datos de produccion|mantenimiento de base/.test(q)) return "restricted";
  if (/ariba|sap|salesforce|shopify/.test(q)) return "abstention";
  if (/recepci|dos pasos|2 pasos|entrada y luego|almacenar/.test(q)) return "nominal";
  return "out_of_domain";
}

/** Construye la respuesta mock completa para una solicitud. */
export function buildMockResponse(req: QueryRequest): QueryResponse {
  const scenario = matchScenario(req.query);
  switch (scenario) {
    case "injection":
      return injectionBlocked();
    case "ticket_create":
      return createTicketDraft(req);
    case "ticket_classify":
      return classify(req);
    case "ticket_status": {
      const id = /TCK-\d{5}/i.exec(req.query)?.[0]?.toUpperCase() ?? "TCK-00042";
      return ticketStatus(req, id);
    }
    case "version_compare":
      return versionComparison(req);
    case "restricted":
      return restrictedRunbook(req);
    case "abstention":
      return abstention(req);
    case "nominal":
      return nominalReceipts(req);
    case "out_of_domain":
      return outOfDomain(req);
  }
}

/** Usuarios para `GET /api/users` en modo demo. */
export function mockUsers(): DemoUser[] {
  return MOCK_USERS;
}
