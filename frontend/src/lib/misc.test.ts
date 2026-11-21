import { describe, expect, it } from "vitest";
import { normalizeDocuments, normalizeUsers, statsFromSources } from "../api/client";
import type { MetricValue, RedteamCase, SourceEntry } from "../api/types";
import { parseHash } from "../hooks/useHashRoute";
import { MOCK_SECURITY } from "../mocks/security";
import { buildDraftQuery, buildEscalationQuery } from "./drafts";
import { formatCountdown, formatMs, formatRemaining, formatUsd, shortHash } from "./format";
import { tokenizeJson } from "./jsonTokens";
import { formatTarget, improvement, metricStatus, securityKpis, summarizeVectors } from "./metrics";

const metric = (over: Partial<MetricValue>): MetricValue => ({
  key: "k",
  label: "K",
  value: 0.8,
  target: 0.85,
  direction: "gte",
  unit: "ratio",
  ...over,
});

describe("métricas", () => {
  it("cumple/no cumple según la dirección y referencia sin meta", () => {
    expect(metricStatus(metric({ value: 0.85 }))).toBe("pass");
    expect(metricStatus(metric({ value: 0.84 }))).toBe("fail");
    expect(metricStatus(metric({ direction: "lte", unit: "ms", value: 5240, target: 4000 }))).toBe("fail");
    expect(metricStatus(metric({ target: null }))).toBe("reference");
  });

  it("la mejora invierte el signo para métricas lte", () => {
    expect(improvement(metric({ baseline: 0.7, value: 0.8 }))).toBeCloseTo(0.1);
    expect(improvement(metric({ direction: "lte", baseline: 5000, value: 4000 }))).toBe(1000);
    expect(improvement(metric({ baseline: null }))).toBeNull();
  });

  it("formatTarget muestra la dirección", () => {
    expect(formatTarget(metric({}))).toBe("≥ 0.85");
    expect(formatTarget(metric({ direction: "lte", unit: "ms", target: 4000 }))).toBe("≤ 4.00 s");
    expect(formatTarget(metric({ target: null }))).toBe("sin meta");
  });
});

describe("seguridad", () => {
  it("resume por vector y respeta las metas", () => {
    const vectors = summarizeVectors(MOCK_SECURITY.cases);
    const indirect = vectors.find((v) => v.vector === "injection_indirect");
    expect(indirect).toMatchObject({ total: 3, passed: 2 });
    const kpis = securityKpis(MOCK_SECURITY.cases);
    expect(kpis[0]?.total).toBe(MOCK_SECURITY.cases.length);
  });

  it("un banco vacío no divide entre cero", () => {
    const empty: RedteamCase[] = [];
    expect(summarizeVectors(empty)).toEqual([]);
    expect(securityKpis(empty).every((k) => k.value === 0)).toBe(true);
  });
});

describe("formatos", () => {
  it("formatea latencias, costos, cuentas regresivas y hashes", () => {
    expect(formatMs(0)).toBe("0 ms");
    expect(formatMs(0.4)).toBe("0.4 ms");
    expect(formatMs(382.5)).toBe("383 ms");
    expect(formatMs(4390)).toBe("4.39 s");
    expect(formatUsd(0)).toBe("US$0");
    expect(formatUsd(0.00043)).toBe("US$0.0004");
    expect(formatCountdown(65_000)).toBe("01:05");
    expect(formatCountdown(-5)).toBe("00:00");
    expect(formatRemaining(26 * 3_600_000)).toBe("1 d 2 h");
    expect(formatRemaining(-90_000)).toBe("vencido hace 1 min");
    expect(shortHash("abcdef0123456789", 4, 2)).toBe("abcd…89");
  });
});

describe("enrutador por hash", () => {
  it("interpreta vista y parámetro, con consola por defecto", () => {
    expect(parseHash("#/traza/trc_123")).toEqual({ view: "traza", param: "trc_123" });
    expect(parseHash("#/evaluacion")).toEqual({ view: "evaluacion", param: null });
    expect(parseHash("")).toEqual({ view: "consola", param: null });
    expect(parseHash("#/inexistente")).toEqual({ view: "consola", param: null });
  });
});

describe("normalizadores del cliente", () => {
  it("normalizeUsers acepta listas u objetos y descarta roles inválidos", () => {
    expect(
      normalizeUsers({
        users: [
          { user_id: "USR-0101", role: "n1", display_name: "Ana" },
          { user_id: "USR-9", role: "root" },
        ],
      }),
    ).toEqual([{ user_id: "USR-0101", name: "Ana", role: "n1", account: null }]);
    expect(normalizeUsers([{ user_id: "USR-0301", role: "admin" }])[0]?.name).toBe("USR-0301");
    expect(normalizeUsers("basura")).toEqual([]);
  });

  it("normalizeDocuments calcula estadísticas si el endpoint solo da el manifiesto", () => {
    const src = { id: "x", module: "ventas", pages: 3, chunks: 7, tokens: 100 } as SourceEntry;
    const out = normalizeDocuments([src]);
    expect(out.stats).toMatchObject({ documents: 1, pages: 3, chunks: 7, tokens: 100 });
    expect(statsFromSources([]).documents).toBe(0);
  });
});

describe("JSON y consultas generadas", () => {
  it("tokenizeJson distingue claves, cadenas, números, booleanos y null", () => {
    const kinds = tokenizeJson(JSON.stringify({ a: "x", b: 1.5, c: true, d: null })).map((t) => t.kind);
    expect(kinds).toEqual(expect.arrayContaining(["key", "string", "number", "boolean", "null", "punct"]));
    expect(tokenizeJson('"a": "b"').filter((t) => t.kind === "key")).toHaveLength(1);
  });

  it("buildDraftQuery incluye las pistas que el agente debe respetar", () => {
    expect(
      buildDraftQuery({ summary: "Error al facturar", description: "", category: "error_funcional", priority: "alta", module: "facturacion" }),
    ).toBe("Crea un ticket: Error al facturar [categoría=error_funcional; prioridad=alta; módulo=facturacion]");
    expect(buildEscalationQuery("x".repeat(200)).length).toBeLessThan(220);
  });
});
