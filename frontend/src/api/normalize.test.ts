/**
 * Normalizadores contra las formas que producen los módulos reales de Python
 * (`docuerp.eval.report.build_summary`, `docuerp.eval.aggregate`, `security/redteam.py`,
 * `docuerp.api.models.TraceLookup`) y contra eventos reales del agente.
 */
import { describe, expect, it } from "vitest";
import { classificationFromEvent } from "../lib/toolData";
import { buildPipeline, isContainerEvent, outcomeOf, stepDetail, waterfall } from "../lib/trace";
import { normalizeEvalSummary, normalizeSecuritySummary, normalizeTraceLookup } from "./normalize";
import type { TraceEvent } from "./types";

const s = (mean: number, n = 10, lo = mean - 0.05, hi = mean + 0.05) => ({ mean, ci_low: lo, ci_high: hi, n });

const retrievalCfg = (recall: number, p50: number) => ({
  mode: "x",
  chunker: "structure",
  embedder: "bge-m3",
  n_cases: 20,
  aggregate: { "recall@5": s(recall, 20), mrr: s(recall - 0.1, 20), "ndcg@5": s(recall - 0.05, 20), "redundancy@5": s(0.1, 20) },
  by_category: { direct: { "recall@5": s(1, 5) } },
  errors: 0,
  latency_ms: { mean: p50, p50, p95: p50 * 2 },
});

const e2eRun = (behavior: number) => ({
  kind: "e2e",
  n_cases: 42,
  runs_per_case: { "3": 42 },
  aggregate: { behavior_correct: s(behavior, 42), faithfulness: s(0.93, 20), citation_precision: s(0.96, 20) },
  by_category: {
    direct: { behavior_correct: s(1, 5), faithfulness: s(0.95, 5), citation_precision: s(1, 5) },
    tool_nominal: { behavior_correct: s(0.5, 2), faithfulness: { mean: null, ci_low: null, ci_high: null, n: 0 } },
  },
  operation: {
    latency_ms: { mean: 6000, p50: 5200, p95: 9100, n: 126 },
    api_latency_ms: { mean: 3500, p50: 3300, p95: 6000, n: 60 },
    tokens: { mean: 3100, p95: 3900, input_mean: 2000, output_mean: 1100 },
    cost_usd: { mean: 0.0016, total: 0.2 },
  },
  failures: [
    {
      case_id: "EV-017",
      category: "multipart",
      risk_level: "high",
      question: "¿Cómo...?",
      expected_behavior: "answer",
      severity: 3,
      reasons: ["citas que no apuntan a la evidencia esperada"],
      answer_excerpt: "Para...",
      abstained: false,
      blocked: false,
      tool_used: "search_documents",
      n_sources: 2,
      error: null,
    },
  ],
});

const REPORT = {
  format: "report",
  generated_at: "2026-11-17T10:00:00+00:00",
  runs: { retrieval: "20261117-0900_retrieval", baseline: "20261117-0910_baseline", final: "20261117-0930_final" },
  targets: [
    { key: "recall@5", label: "Recall@5", comparator: "ge", threshold: 0.85, unit: "", value: 0.9, meets: true },
    { key: "faithfulness", label: "Fidelidad (juez)", comparator: "ge", threshold: 0.9, unit: "", value: 0.93, meets: true },
    { key: "latency_p50", label: "Latencia P50", comparator: "le", threshold: 4000, unit: "ms", value: 5200, meets: false },
    { key: "attack_blocked", label: "Ataques bloqueados", comparator: "ge", threshold: 0.95, unit: "", value: null, meets: null },
  ],
  final_retrieval_config: "hybrid_rerank/structure/bge-m3",
  retrieval: {
    "bm25/structure/bge-m3": retrievalCfg(0.7, 3),
    "hybrid_rerank/structure/bge-m3": retrievalCfg(0.9, 1700),
    "hybrid_rerank/fixed256/bge-m3": retrievalCfg(0.8, 1600),
  },
  e2e: { baseline: e2eRun(0.7), final: e2eRun(0.9) },
  paired: { behavior_correct: { baseline: 0.7, final: 0.9, delta: 0.2, ci_low: 0.1, ci_high: 0.3, p_value: 0.004 } },
  figures: {},
};

describe("normalizeEvalSummary", () => {
  it("convierte el summary.json de `docuerp eval report`", () => {
    const out = normalizeEvalSummary(REPORT);
    expect(out.run_id).toBe("20261117-0930_final");
    expect(out.baseline_run_id).toBe("20261117-0910_baseline");
    expect(out.n_cases).toBe(42);
    expect(out.runs).toBe(3);
    // La meta sin datos (attack_blocked: null) se omite; se agregan dos referencias sin meta.
    expect(out.metrics.map((m) => m.key)).toEqual(["recall@5", "faithfulness", "latency_p50", "api_latency_p50", "cost"]);
    const recall = out.metrics[0];
    expect(recall).toMatchObject({ value: 0.9, target: 0.85, direction: "gte", unit: "ratio", baseline: 0.7 });
    expect(recall?.ci?.[0]).toBeCloseTo(0.85);
    const latency = out.metrics.find((m) => m.key === "latency_p50");
    expect(latency).toMatchObject({ unit: "ms", direction: "lte", baseline: 5200 });
    expect(out.metrics.find((m) => m.key === "faithfulness")?.baseline_ci).not.toBeNull();
    const selected = out.ablations.filter((a) => a.selected).map((a) => a.group);
    expect(selected.sort()).toEqual(["chunker", "embedder", "retriever"]);
    expect(out.ablations.find((a) => a.group === "chunker" && a.config === "fixed256")?.recall_at_5).toBeCloseTo(0.8);
    const direct = out.categories.find((c) => c.category === "direct");
    expect(direct).toMatchObject({ n: 5, passed: 5, recall_at_5: 1 });
    expect(Number.isNaN(out.categories.find((c) => c.category === "tool_nominal")?.faithfulness)).toBe(true);
    expect(out.failures[0]).toMatchObject({ case_id: "EV-017", severity: "high", observed: "Para..." });
    expect(out.paired_test).toEqual([{ metric: "behavior_correct", p_value: 0.004, delta: 0.2 }]);
  });

  it("acepta una corrida e2e suelta de eval/results/<run_id>", () => {
    const out = normalizeEvalSummary({
      run_id: "20261117-0930_final",
      format: "e2e",
      generated_at: "2026-11-17T09:40:00Z",
      metrics: e2eRun(0.88),
      targets: [{ key: "faithfulness", label: "Fidelidad", comparator: "ge", threshold: 0.9, unit: "", value: 0.93 }],
    });
    expect(out.run_id).toBe("20261117-0930_final");
    expect(out.metrics[0]?.key).toBe("faithfulness");
    expect(out.ablations).toEqual([]);
    expect(out.failures).toHaveLength(1);
  });

  it("acepta una corrida solo de recuperación", () => {
    const out = normalizeEvalSummary({
      run_id: "20261117-0900_retrieval",
      metrics: { kind: "retrieval", configs: REPORT.retrieval },
      targets: [{ key: "recall@5", label: "Recall@5", comparator: "ge", threshold: 0.85, unit: "", value: 0.9 }],
    });
    expect(out.n_cases).toBe(20);
    expect(out.ablations.length).toBeGreaterThan(0);
    expect(out.categories[0]).toMatchObject({ category: "direct", recall_at_5: 1 });
  });

  it("deja pasar un resumen ya normalizado y rechaza formatos desconocidos", () => {
    expect(normalizeEvalSummary({ run_id: "x", metrics: [] }).run_id).toBe("x");
    expect(() => normalizeEvalSummary({ metrics: { retrieval: { "recall@5": 1 } } })).toThrow();
  });
});

describe("normalizeSecuritySummary", () => {
  it("omite los casos que no aplican al modo y conserva los campos del banco", () => {
    const out = normalizeSecuritySummary({
      run_id: "20261117-1000_guards",
      mode: "guards_only",
      generated_at: "2026-11-17T10:00:00+00:00",
      not_applicable: 1,
      cases: [
        { id: "RT-001", vector: "injection_direct", attack: "Ignora...", role: "n1", expected_outcome: "blocked", observed: "blocked", passed: true, in_official_bench: true },
        { id: "RT-020", vector: "critical_action", attack: "Crea...", role: "n1", expected_outcome: "needs_confirmation", observed: "n/a", passed: null },
      ],
    });
    expect(out.cases.map((c) => c.id)).toEqual(["RT-001"]);
    expect(out.not_applicable).toBe(1);
    expect(out.mode).toBe("guards_only");
    expect(out.created_at).toBe("2026-11-17T10:00:00+00:00");
  });
});

describe("normalizeTraceLookup", () => {
  const response = {
    trace_id: "abc",
    intent: "ticket_status",
    response: { answer: "ok", sources: [], abstained: false, tool_used: "get_ticket_status", needs_human_review: false },
    pending_action: null,
    blocked: false,
    block_reason: null,
    latency_ms: 10,
    usage: { llm_calls: 0, input_tokens: 0, output_tokens: 0, cache_read_tokens: 0, cache_creation_tokens: 0, cost_usd: 0 },
    events: [],
  };

  it("devuelve la respuesta completa si la traza sigue en memoria", () => {
    expect(normalizeTraceLookup({ trace_id: "abc", origin: "memory", response }).intent).toBe("ticket_status");
  });

  it("reconstruye una traza mínima desde la bitácora de auditoría", () => {
    const out = normalizeTraceLookup({
      trace_id: "abc",
      origin: "audit_log",
      audit: {
        intent: "ticket_create",
        guards: { "tool.create_ticket": { status: "blocked" } },
        blocked: true,
        block_reason: "confirmacion_rechazada",
        abstained: true,
        latencies_ms: { "tool.create_ticket": 0.1, total: 0.2 },
        answer_masked: "No se registró ningún ticket.",
        tool_used: "create_ticket",
        sources: [],
        usage: { llm_calls: 0 },
      },
    });
    expect(out.blocked).toBe(true);
    expect(out.latency_ms).toBe(0.2);
    expect(out.events).toEqual([{ stage: "tool.create_ticket", status: "blocked", duration_ms: 0.1, data: { from_audit: true } }]);
  });
});

/* ------------------------------------------------------------------------------------ */
/* Eventos reales del agente (recortados de una consulta documental en vivo)             */
/* ------------------------------------------------------------------------------------ */

const ev = (stage: string, duration_ms: number, data: Record<string, unknown> = {}, status: TraceEvent["status"] = "ok"): TraceEvent => ({
  stage,
  status,
  duration_ms,
  data,
});

const LIVE_DOC: TraceEvent[] = [
  ev("input_guard", 80, { allowed: true, risk_score: 0, flags: [] }),
  ev("router", 26.4, { intent: "doc_query", confidence: 0.9776, method: "classifier" }),
  ev("retrieval.bm25", 1.5, { top: [] }),
  ev("retrieval.dense", 31.2, { top: [] }),
  ev("retrieval.fusion", 0.1, { top: [] }),
  ev("rerank", 1922.7, { top: [{ chunk_id: "a::p0001::c000", source: "a.pdf", page: 1, rank: 1, score: 0.96 }] }),
  ev("retrieval.total", 1955.5, { top: [] }),
  ev("tool.search_documents", 1955.7, { ok: true, evidence_ok: true, filtered_out: 35 }),
  ev("evidence_gate", 0, { evidence_ok: true, top_score: 0.96, tau: 0.3, n_hits: 5 }),
  ev("permission_check", 0, { kept: 5, dropped: 0 }),
  ev("generation", 7148.8, { mode: "llm", model: "claude-haiku-5-5", input_tokens: 2, output_tokens: 1280 }),
  ev("output_guard", 1.4),
  ev("respond", 1),
];

describe("traza real del agente", () => {
  it("el waterfall no suma dos veces la recuperación", () => {
    const { bars, totalMs } = waterfall(LIVE_DOC);
    expect(bars.map((b) => b.stage)).not.toContain("retrieval.total");
    expect(bars.map((b) => b.stage)).not.toContain("tool.search_documents");
    expect(totalMs).toBeCloseTo(80 + 26.4 + 1.5 + 31.2 + 0.1 + 1922.7 + 7148.8 + 1.4 + 1);
    expect(isContainerEvent(ev("tool.search_documents", 1), [ev("tool.search_documents", 1)])).toBe(false);
  });

  it("el paso de recuperación agrupa BM25 + denso + RRF sin el total", () => {
    const step = buildPipeline(LIVE_DOC, true).find((s) => s.def.key === "retrieval");
    expect(step?.events.map((e) => e.stage)).toEqual(["retrieval.bm25", "retrieval.dense", "retrieval.fusion"]);
    expect(step?.durationMs).toBeCloseTo(32.8);
  });

  it("lee kept/dropped de permission_check y los tokens de salida de generation", () => {
    expect(stepDetail("permission_check", [ev("permission_check", 0, { kept: 5, dropped: 2 })])).toBe("5 autorizados · 2 filtrados");
    expect(stepDetail("generation", [LIVE_DOC[10] as TraceEvent])).toBe("haiku-5-5 · 1280 tok de salida");
    expect(stepDetail("generation", [ev("generation", 0, { mode: "template" })])).toBe("Plantilla determinista, sin LLM");
    expect(
      stepDetail("permission_check", [ev("permission_check", 0, { tool: "get_ticket_status", allowed: true, in_scope: false }, "blocked")]),
    ).toBe("Fuera del alcance del usuario");
  });

  it("una respuesta documental con tool_used=search_documents cuenta como respondida", () => {
    const res = normalizeTraceLookup({
      trace_id: "t",
      origin: "memory",
      response: {
        trace_id: "t",
        intent: "doc_query",
        response: { answer: "ok", sources: [], abstained: false, tool_used: "search_documents", needs_human_review: false },
        pending_action: null,
        blocked: false,
        block_reason: null,
        latency_ms: 9000,
        usage: { llm_calls: 1, input_tokens: 2, output_tokens: 1280, cache_read_tokens: 2080, cache_creation_tokens: 4800, cost_usd: 0.0016 },
        events: LIVE_DOC,
      },
    });
    expect(outcomeOf(res)).toBe("answered");
    expect(outcomeOf({ ...res, response: { ...res.response, tool_used: "get_ticket_status" } })).toBe("tool");
  });

  it("interpreta el resultado real de classify_ticket", () => {
    const view = classificationFromEvent(
      ev("tool.classify_ticket", 53, {
        kind: "embeddings",
        category: "error_funcional",
        category_confidence: 0.6795,
        category_alternatives: [
          { label: "configuracion", confidence: 0.1153 },
          { label: "rendimiento", confidence: 0.0654 },
        ],
        priority: "critica",
        module: "facturacion",
      }),
    );
    expect(view).toMatchObject({ category: "error_funcional", confidence: 0.6795, module: "facturacion", model: "bge-m3 + regresión logística" });
    expect(view?.probabilities.map((p) => p.label)).toEqual(["error_funcional", "configuracion", "rendimiento"]);
  });
});
