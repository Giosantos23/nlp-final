/**
 * Normalizadores de los endpoints de lectura hacia los tipos que usa la interfaz.
 *
 * La API real entrega los resultados tal como los escriben los módulos de Python:
 *
 * - `/api/eval/summary`: el `summary.json` de `docuerp eval report` (`format: "report"`, con
 *   `targets`, `retrieval`, `e2e` y `paired`) o una corrida suelta de `eval/results/<run_id>/`
 *   (`format: "e2e" | "retrieval"`, con su `metrics.json` sin detalle por caso).
 * - `/api/security/summary`: los casos del runner red-team (`security/results/<run_id>.jsonl`)
 *   con `observed` y `passed` (`null` = no aplica al modo de ejecución).
 * - `/api/trace/{id}`: `{trace_id, origin, response?, audit?}`.
 *
 * Aquí se convierten a `EvalSummary`, `SecuritySummary` y `QueryResponse`. Los datos ya
 * normalizados (fixtures de la demo) pasan sin cambios.
 */
import type {
  AblationRow,
  CategoryRow,
  EvalSummary,
  FailureRow,
  MetricValue,
  QueryResponse,
  RedteamCase,
  RiskLevel,
  SecuritySummary,
  TraceEvent,
} from "./types";

type Rec = Record<string, unknown>;

const rec = (v: unknown): Rec | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Rec) : null);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const str = (v: unknown): string | null => (typeof v === "string" && v ? v : null);

/** Media de un resumen estadístico `{mean, ci_low, ci_high, n}` (o un número suelto). */
function meanOf(v: unknown): number | null {
  return num(v) ?? num(rec(v)?.mean);
}

function ciOf(v: unknown): [number, number] | null {
  const r = rec(v);
  const lo = num(r?.ci_low);
  const hi = num(r?.ci_high);
  return lo !== null && hi !== null ? [lo, hi] : null;
}

/* ------------------------------------------------------------------------------------ */
/* Evaluación                                                                            */
/* ------------------------------------------------------------------------------------ */

export class EvalFormatError extends Error {}

const UNIT: Record<string, MetricValue["unit"]> = { ms: "ms", tokens: "tokens", usd: "usd" };

/** Configuración de recuperación "final" (la del sistema), con la misma regla que Python. */
function pickFinal(configs: Rec, preferred: string | null): string | null {
  if (preferred && configs[preferred]) return preferred;
  if (configs["hybrid_rerank/structure/bge-m3"]) return "hybrid_rerank/structure/bge-m3";
  let best: string | null = null;
  let bestRecall = -1;
  for (const [label, cfg] of Object.entries(configs)) {
    const recall = meanOf(rec(rec(cfg)?.aggregate)?.["recall@5"]) ?? 0;
    if (recall > bestRecall) {
      best = label;
      bestRecall = recall;
    }
  }
  return best;
}

function ablationsFrom(configs: Rec, final: string | null): AblationRow[] {
  if (!final) return [];
  const [fMode, fChunker, fEmbedder] = final.split("/");
  const rows: AblationRow[] = [];
  for (const [label, raw] of Object.entries(configs)) {
    const cfg = rec(raw);
    const agg = rec(cfg?.aggregate);
    if (!cfg || !agg) continue;
    const [mode, chunker, embedder] = label.split("/");
    const base = {
      recall_at_5: meanOf(agg["recall@5"]) ?? 0,
      mrr: meanOf(agg.mrr) ?? 0,
      ndcg_at_5: meanOf(agg["ndcg@5"]) ?? 0,
      latency_ms: num(rec(cfg.latency_ms)?.p50) ?? 0,
      selected: label === final,
    };
    if (chunker === fChunker && embedder === fEmbedder) rows.push({ group: "retriever", config: mode ?? label, ...base });
    if (mode === fMode && embedder === fEmbedder) rows.push({ group: "chunker", config: chunker ?? label, ...base });
    if (mode === fMode && chunker === fChunker) rows.push({ group: "embedder", config: embedder ?? label, ...base });
  }
  return rows;
}

/** Valor de baseline de una meta (misma fuente que el valor final). */
function baselineFor(key: string, source: "retrieval" | "e2e" | "operation", baseline: Rec | null, configs: Rec | null): number | null {
  if (source === "retrieval") {
    const bm25 = rec(configs?.["bm25/structure/bge-m3"]) ?? rec(configs?.["bm25/structure/e5-base"]);
    return meanOf(rec(bm25?.aggregate)?.[key]);
  }
  if (!baseline) return null;
  if (source === "e2e") return meanOf(rec(baseline.aggregate)?.[key]);
  const op = rec(baseline.operation);
  if (key === "latency_p50") return num(rec(op?.latency_ms)?.p50);
  if (key === "latency_p95") return num(rec(op?.latency_ms)?.p95);
  return num(rec(op?.tokens)?.mean);
}

const RETRIEVAL_KEYS = new Set(["recall@5", "mrr", "redundancy@5", "ndcg@5"]);
const OPERATION_KEYS = new Set(["latency_p50", "latency_p95", "tokens_mean"]);

function sourceOf(key: string): "retrieval" | "e2e" | "operation" {
  if (RETRIEVAL_KEYS.has(key)) return "retrieval";
  if (OPERATION_KEYS.has(key)) return "operation";
  return "e2e";
}

function metricsFrom(targets: unknown[], final: Rec | null, baseline: Rec | null, configs: Rec | null, finalCfg: string | null): MetricValue[] {
  const finalAgg = rec(rec(configs?.[finalCfg ?? ""])?.aggregate);
  const out: MetricValue[] = [];
  for (const raw of targets) {
    const t = rec(raw);
    const key = str(t?.key);
    const value = num(t?.value);
    if (!t || !key || value === null) continue; // meta sin datos en esta corrida
    const source = sourceOf(key);
    const summary = source === "retrieval" ? finalAgg?.[key] : source === "e2e" ? rec(final?.aggregate)?.[key] : null;
    const baseSummary = source === "e2e" ? rec(baseline?.aggregate)?.[key] : null;
    out.push({
      key,
      label: str(t.label) ?? key,
      value,
      ci: ciOf(summary),
      baseline: baselineFor(key, source, baseline, configs),
      baseline_ci: ciOf(baseSummary),
      target: num(t.threshold),
      direction: t.comparator === "le" ? "lte" : "gte",
      unit: UNIT[str(t.unit) ?? ""] ?? "ratio",
    });
  }
  // Referencias sin meta: latencia de API del LLM y costo por consulta.
  const op = rec(final?.operation);
  const apiP50 = num(rec(op?.api_latency_ms)?.p50);
  if (apiP50 !== null) {
    out.push({
      key: "api_latency_p50",
      label: "Latencia de API (P50)",
      value: apiP50,
      baseline: num(rec(rec(baseline?.operation)?.api_latency_ms)?.p50),
      target: null,
      direction: "lte",
      unit: "ms",
      note: "Solo el tiempo de la API del LLM, sin el arranque del CLI del Agent SDK.",
    });
  }
  const cost = num(rec(op?.cost_usd)?.mean);
  if (cost !== null) {
    out.push({
      key: "cost",
      label: "Costo por consulta",
      value: cost,
      baseline: num(rec(rec(baseline?.operation)?.cost_usd)?.mean),
      target: null,
      direction: "lte",
      unit: "usd",
      note: "Precio de lista estimado; el anteproyecto no fija meta de costo.",
    });
  }
  return out;
}

function categoriesFrom(final: Rec | null, configs: Rec | null, finalCfg: string | null): CategoryRow[] {
  const byCat = rec(final?.by_category);
  const retrievalByCat = rec(rec(configs?.[finalCfg ?? ""])?.by_category);
  if (!byCat) {
    // Corrida solo de recuperación: una fila por categoría con Recall@5.
    return Object.entries(retrievalByCat ?? {}).map(([category, raw]) => {
      const m = rec(raw);
      const n = num(rec(m?.["recall@5"])?.n) ?? 0;
      return { category, n, recall_at_5: meanOf(m?.["recall@5"]) ?? Number.NaN, faithfulness: Number.NaN, citation_precision: Number.NaN, passed: Number.NaN };
    });
  }
  return Object.entries(byCat).map(([category, raw]) => {
    const m = rec(raw);
    const behavior = rec(m?.behavior_correct);
    const n = num(behavior?.n) ?? 0;
    const passed = meanOf(behavior);
    return {
      category,
      n,
      recall_at_5: meanOf(rec(retrievalByCat?.[category])?.["recall@5"]) ?? Number.NaN,
      faithfulness: meanOf(m?.faithfulness) ?? Number.NaN,
      citation_precision: meanOf(m?.citation_precision) ?? Number.NaN,
      passed: passed !== null ? Math.round(passed * n) : Number.NaN,
    };
  });
}

const RISK: RiskLevel[] = ["low", "medium", "high", "critical"];

function failuresFrom(final: Rec | null): FailureRow[] {
  const list = Array.isArray(final?.failures) ? (final.failures as unknown[]) : [];
  return list.flatMap((raw) => {
    const f = rec(raw);
    const id = str(f?.case_id);
    if (!f || !id) return [];
    const risk = str(f.risk_level) as RiskLevel | null;
    const reasons = Array.isArray(f.reasons) ? f.reasons.filter((r): r is string => typeof r === "string") : [];
    const observed = f.error
      ? `Error: ${str(f.error) ?? "error del servicio"}`
      : f.blocked
        ? "Bloqueada"
        : f.abstained
          ? "Se abstuvo"
          : (str(f.answer_excerpt) ?? "—");
    return [
      {
        case_id: id,
        category: str(f.category) ?? "—",
        question: str(f.question) ?? "",
        expected: str(f.expected_behavior) ?? "—",
        observed,
        cause: reasons.join("; ") || "—",
        severity: risk && RISK.includes(risk) ? risk : "medium",
      },
    ];
  });
}

function pairedFrom(paired: unknown): EvalSummary["paired_test"] {
  const r = rec(paired);
  if (!r) return undefined;
  const rows = Object.entries(r).flatMap(([metric, raw]) => {
    const p = rec(raw);
    const pValue = num(p?.p_value);
    const delta = num(p?.delta) ?? (num(p?.final) !== null && num(p?.baseline) !== null ? (p?.final as number) - (p?.baseline as number) : null);
    return pValue !== null && delta !== null ? [{ metric, p_value: pValue, delta }] : [];
  });
  return rows.length ? rows : undefined;
}

function runsPerCase(final: Rec | null): number {
  const counts = rec(final?.runs_per_case);
  const keys = Object.keys(counts ?? {}).map(Number).filter(Number.isFinite);
  return keys.length ? Math.max(...keys) : 1;
}

/** Convierte un reporte (`format: report`) o una corrida suelta a `EvalSummary`. */
export function normalizeEvalSummary(raw: unknown): EvalSummary {
  const r = rec(raw);
  if (!r) throw new EvalFormatError("El resumen de evaluación no es un objeto.");
  if (Array.isArray(r.metrics)) {
    return {
      ablations: [],
      categories: [],
      failures: [],
      runs: 1,
      n_cases: 0,
      baseline_run_id: "",
      created_at: "",
      run_id: "",
      ...(r as Partial<EvalSummary>),
    } as EvalSummary;
  }

  let configs: Rec | null = null;
  let final: Rec | null = null;
  let baseline: Rec | null = null;
  let preferred: string | null = null;
  let runId = str(r.run_id) ?? "";
  let baselineRunId = "";
  if (r.format === "report" || rec(r.e2e) || rec(r.retrieval)) {
    configs = rec(r.retrieval);
    final = rec(rec(r.e2e)?.final);
    baseline = rec(rec(r.e2e)?.baseline);
    preferred = str(r.final_retrieval_config);
    const runs = rec(r.runs);
    runId = str(runs?.final) ?? str(runs?.retrieval) ?? runId;
    baselineRunId = str(runs?.baseline) ?? "";
  } else {
    const metrics = rec(r.metrics);
    if (!metrics) throw new EvalFormatError("El resumen de evaluación no trae métricas.");
    if (metrics.kind === "retrieval") configs = rec(metrics.configs);
    else if (metrics.kind === "e2e") final = metrics;
    else throw new EvalFormatError("Formato de métricas de evaluación no reconocido.");
  }
  const finalCfg = configs ? pickFinal(configs, preferred) : null;
  const targets = Array.isArray(r.targets) ? r.targets : [];
  return {
    run_id: runId,
    baseline_run_id: baselineRunId,
    created_at: str(r.generated_at) ?? "",
    n_cases: num(final?.n_cases) ?? num(rec(configs?.[finalCfg ?? ""])?.n_cases) ?? 0,
    runs: runsPerCase(final),
    metrics: metricsFrom(targets, final, baseline, configs, finalCfg),
    ablations: configs ? ablationsFrom(configs, finalCfg) : [],
    categories: categoriesFrom(final, configs, finalCfg),
    failures: failuresFrom(final),
    paired_test: pairedFrom(r.paired),
  };
}

/* ------------------------------------------------------------------------------------ */
/* Seguridad                                                                             */
/* ------------------------------------------------------------------------------------ */

/** Casos red-team aplicables (los `passed: null` no cuentan) con los campos que pinta la UI. */
export function normalizeSecuritySummary(raw: unknown): SecuritySummary & { not_applicable: number; mode: string | null } {
  const r = rec(raw);
  const list = Array.isArray(raw) ? raw : r?.cases;
  if (!Array.isArray(list)) throw new EvalFormatError("El resumen de seguridad no trae `cases`.");
  let skipped = 0;
  const cases = list.flatMap((item): RedteamCase[] => {
    const c = rec(item);
    if (!c || typeof c.id !== "string") return [];
    if (c.passed !== true && c.passed !== false) {
      skipped += 1;
      return [];
    }
    return [
      {
        id: c.id,
        vector: c.vector as RedteamCase["vector"],
        attack: str(c.attack) ?? "",
        role: (str(c.role) ?? "n1") as RedteamCase["role"],
        threat: str(c.threat) ?? "",
        component: str(c.component) ?? "",
        expected_behavior: str(c.expected_behavior) ?? "",
        expected_outcome: (str(c.expected_outcome) ?? "blocked") as RedteamCase["expected_outcome"],
        observed: str(c.observed) ?? "—",
        passed: c.passed,
        mitigation: str(c.mitigation) ?? "",
        in_official_bench: c.in_official_bench === true,
      },
    ];
  });
  return {
    run_id: str(r?.run_id) ?? "",
    created_at: str(r?.created_at) ?? str(r?.generated_at) ?? "",
    cases,
    not_applicable: num(r?.not_applicable) ?? skipped,
    mode: str(r?.mode),
  };
}

/* ------------------------------------------------------------------------------------ */
/* Trazas                                                                                */
/* ------------------------------------------------------------------------------------ */

/**
 * `GET /api/trace/{id}` devuelve la respuesta completa si sigue en memoria; si ya solo está
 * en la bitácora de auditoría se reconstruye una traza mínima (sin listas de recuperación).
 */
export function normalizeTraceLookup(raw: unknown): QueryResponse {
  const r = rec(raw);
  if (!r) throw new EvalFormatError("Traza con formato inesperado.");
  const inner = rec(r.response);
  if (inner && rec(inner.response)) return inner as unknown as QueryResponse; // origin: memory
  if (inner && typeof inner.answer === "string" && Array.isArray(r.events)) {
    return r as unknown as QueryResponse; // ya es un QueryResponse
  }
  const audit = rec(r.audit);
  if (!audit) throw new EvalFormatError("La traza no trae respuesta ni registro de auditoría.");
  const latencies = rec(audit.latencies_ms) ?? {};
  const guards = rec(audit.guards) ?? {};
  const events: TraceEvent[] = Object.entries(latencies)
    .filter(([stage]) => stage !== "total")
    .map(([stage, ms]) => ({
      stage,
      status: (str(rec(guards[stage])?.status) ?? "ok") as TraceEvent["status"],
      duration_ms: num(ms) ?? 0,
      data: { from_audit: true },
    }));
  const usage = rec(audit.usage) ?? {};
  return {
    trace_id: str(r.trace_id) ?? "",
    intent: (str(audit.intent) ?? "out_of_domain") as QueryResponse["intent"],
    response: {
      answer: str(audit.answer_masked) ?? "",
      sources: Array.isArray(audit.sources) ? (audit.sources as QueryResponse["response"]["sources"]) : [],
      abstained: audit.abstained === true,
      tool_used: str(audit.tool_used),
      needs_human_review: audit.needs_human_review === true,
    },
    pending_action: null,
    blocked: audit.blocked === true,
    block_reason: str(audit.block_reason),
    latency_ms: num(latencies.total) ?? events.reduce((s, e) => s + e.duration_ms, 0),
    usage: {
      llm_calls: num(usage.llm_calls) ?? 0,
      input_tokens: num(usage.input_tokens) ?? 0,
      output_tokens: num(usage.output_tokens) ?? 0,
      cache_read_tokens: num(usage.cache_read_tokens) ?? 0,
      cache_creation_tokens: num(usage.cache_creation_tokens) ?? 0,
      cost_usd: num(usage.cost_usd) ?? 0,
    },
    events,
  };
}
