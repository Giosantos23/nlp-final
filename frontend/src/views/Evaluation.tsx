/**
 * Evaluación: KPIs con las metas del anteproyecto (cumple / no cumple), baseline vs final
 * con intervalos de confianza bootstrap 95 %, ablaciones, desglose por categoría y fallas.
 */
import { AlertTriangle, CheckCircle2, CircleDashed, FlaskConical, RefreshCw, XCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ApiError, api } from "../api/client";
import type { AblationRow, EvalSummary, MetricValue } from "../api/types";
import { ChartTooltip, Legend } from "../components/charts";
import { PendingResults } from "../components/PendingResults";
import { Badge, Button, EmptyState, Panel, SectionHeader, Skeleton, TableScroll } from "../components/ui";
import { useConnection } from "../hooks/useConnection";
import { CHART, axisProps } from "../lib/chartTheme";
import { cx } from "../lib/cx";
import { formatDateTime } from "../lib/format";
import { formatMetric, formatTarget, improvement, metricStatus, type MetricStatus } from "../lib/metrics";
import type { Tone } from "../lib/tone";

type State =
  | { status: "loading" }
  | { status: "ok"; data: EvalSummary }
  | { status: "empty" }
  | { status: "error"; message: string };

const EVAL_STEPS = [
  { label: "Recuperación: Recall@k, MRR y nDCG por configuración (sin LLM)", command: "make eval-retrieval" },
  { label: "Extremo a extremo: baseline y sistema final, 3 corridas por caso", command: "make eval-e2e" },
  { label: "Juez LLM (fidelidad y cobertura) y reporte con IC bootstrap", command: "make eval-judge eval-report" },
];

const STATUS_COPY: Record<MetricStatus, { label: string; tone: Tone; icon: typeof CheckCircle2 }> = {
  pass: { label: "Cumple", tone: "accent", icon: CheckCircle2 },
  fail: { label: "No cumple", tone: "danger", icon: XCircle },
  reference: { label: "Referencia", tone: "neutral", icon: CircleDashed },
};

function KpiTile({ m }: { m: MetricValue }) {
  const status = metricStatus(m);
  const copy = STATUS_COPY[status];
  const delta = improvement(m);
  return (
    <div
      className={cx(
        "flex min-w-0 flex-col rounded-[14px] bg-surface p-4 shadow-inner-hl ring-1 ring-inset",
        status === "fail" ? "ring-danger/25" : "ring-white/[0.06]",
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="text-xs text-muted">{m.label}</span>
        <Badge tone={copy.tone} icon={copy.icon}>
          {copy.label}
        </Badge>
      </div>
      <div className="mt-3 font-mono text-[26px] leading-none tracking-[-0.03em] text-text">{formatMetric(m.value, m.unit)}</div>
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-2xs">
        <span className="text-faint">{m.target === null ? "sin meta" : `meta ${formatTarget(m)}`}</span>
        {delta !== null && m.baseline !== null && m.baseline !== undefined && (
          <span className={cx("font-mono", delta >= 0 ? "text-accent-2" : "text-danger")}>
            {delta >= 0 ? "▲" : "▼"} vs baseline {formatMetric(m.baseline, m.unit)}
          </span>
        )}
      </div>
      {m.ci && (
        <div className="mt-1 font-mono text-[10.5px] text-faint">
          IC 95 % [{formatMetric(m.ci[0], m.unit)}, {formatMetric(m.ci[1], m.unit)}]
        </div>
      )}
      {m.note && <p className="mt-2 border-t hairline pt-2 text-2xs leading-4 text-muted">{m.note}</p>}
    </div>
  );
}

/** Filas baseline vs final con IC y la meta como marca vertical (escala común 0.5–1). */
function IntervalChart({ metrics }: { metrics: MetricValue[] }) {
  const rows = metrics.filter((m) => m.unit === "ratio" && m.direction === "gte" && m.baseline != null);
  const lo = 0.4;
  const pos = (v: number) => `${((Math.max(lo, Math.min(1, v)) - lo) / (1 - lo)) * 100}%`;
  const [hover, setHover] = useState<string | null>(null);
  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <Legend
          items={[
            { label: "Baseline (BM25)", color: CHART.baseline },
            { label: "Final (híbrido + rerank)", color: CHART.final },
          ]}
        />
        <span className="flex items-center gap-1.5 text-2xs text-muted">
          <span className="h-3 w-px bg-text/70" aria-hidden /> meta del anteproyecto · bigotes = IC 95 %
        </span>
      </div>
      <div className="grid grid-cols-[150px_1fr] gap-x-4 text-2xs text-faint">
        <span />
        <div className="relative mb-1 h-4">
          {[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1].map((t) => (
            <span key={t} className="absolute -translate-x-1/2 font-mono" style={{ left: pos(t) }}>
              {t.toFixed(1)}
            </span>
          ))}
        </div>
      </div>
      <ul className="space-y-1">
        {rows.map((m) => {
          const series = [
            { key: "b", value: m.baseline ?? 0, ci: m.baseline_ci, color: CHART.baseline, label: "Baseline" },
            { key: "f", value: m.value, ci: m.ci, color: CHART.final, label: "Final" },
          ];
          const isHover = hover === m.key;
          return (
            <li
              key={m.key}
              className={cx(
                "grid grid-cols-[150px_1fr] items-center gap-x-4 rounded-[8px] py-1.5 transition-colors duration-150",
                isHover && "bg-white/[0.025]",
              )}
              onMouseEnter={() => setHover(m.key)}
              onMouseLeave={() => setHover(null)}
              tabIndex={0}
              onFocus={() => setHover(m.key)}
              onBlur={() => setHover(null)}
              aria-label={`${m.label}: baseline ${m.baseline?.toFixed(2)}, final ${m.value.toFixed(2)}, meta ${m.target?.toFixed(2)}`}
            >
              <span className="truncate pl-2 text-xs text-text">{m.label}</span>
              <div className="relative h-9">
                {[0.5, 0.6, 0.7, 0.8, 0.9].map((t) => (
                  <span key={t} className="absolute inset-y-0 w-px bg-white/[0.04]" style={{ left: pos(t) }} aria-hidden />
                ))}
                {series.map((s, i) => (
                  <div key={s.key} className="absolute left-0 right-0" style={{ top: i === 0 ? 7 : 21, height: 8 }}>
                    <div
                      className="absolute left-0 h-2 rounded-r-[4px]"
                      style={{ width: pos(s.value), background: s.color, opacity: i === 0 ? 0.9 : 1 }}
                    />
                    {s.ci && (
                      <div
                        className="absolute top-1/2 h-px -translate-y-1/2 bg-text/60"
                        style={{ left: pos(s.ci[0]), width: `calc(${pos(s.ci[1])} - ${pos(s.ci[0])})` }}
                        aria-hidden
                      >
                        <span className="absolute -top-[3px] left-0 h-[7px] w-px bg-text/60" />
                        <span className="absolute -top-[3px] right-0 h-[7px] w-px bg-text/60" />
                      </div>
                    )}
                  </div>
                ))}
                {m.target !== null && (
                  <span className="absolute -top-0.5 h-10 w-px bg-text/80" style={{ left: pos(m.target) }} aria-hidden />
                )}
                {isHover && (
                  <div className="pointer-events-none absolute -top-1 right-0 z-[1] rounded-[8px] bg-surface-2 px-2.5 py-1.5 text-2xs shadow-float">
                    <span className="font-mono font-medium text-text">{m.value.toFixed(2)}</span>
                    <span className="text-muted"> final · </span>
                    <span className="font-mono font-medium text-text">{m.baseline?.toFixed(2)}</span>
                    <span className="text-muted"> baseline · meta {formatTarget(m)}</span>
                  </div>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

const GROUP_LABEL: Record<AblationRow["group"], string> = {
  retriever: "Recuperador",
  chunker: "Chunker",
  embedder: "Embedder",
};

/** Etiquetas cortas para que los nombres de configuración no se encimen en el eje. */
const CONFIG_SHORT: Record<string, string> = {
  hybrid: "híbrido",
  hybrid_rerank: "+rerank",
  hybrid_rerank_mmr: "+MMR",
};

function AblationChart({ rows, group }: { rows: AblationRow[]; group: AblationRow["group"] }) {
  const data = rows
    .filter((r) => r.group === group)
    .map((r) => {
      const short = CONFIG_SHORT[r.config] ?? r.config;
      return { ...r, name: r.selected ? `${short} ✓` : short };
    });
  return (
    <div className="min-w-0 rounded-[12px] bg-black/20 p-4 ring-1 ring-inset ring-white/[0.05]">
      <div className="mb-3 flex items-baseline justify-between">
        <span className="text-xs font-medium text-text">{GROUP_LABEL[group]}</span>
        <span className="text-2xs text-faint">✓ = configuración final</span>
      </div>
      <div className="h-[210px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 16, right: 4, left: -18, bottom: 0 }} barGap={2} barCategoryGap="22%">
            <CartesianGrid stroke={CHART.grid} vertical={false} />
            <XAxis dataKey="name" {...axisProps} interval={0} tick={{ fill: CHART.axis, fontSize: 10 }} />
            <YAxis domain={[0.4, 1]} ticks={[0.4, 0.6, 0.8, 1]} {...axisProps} />
            <Tooltip cursor={{ fill: "rgba(255,255,255,0.03)" }} content={<ChartTooltip />} />
            <Bar dataKey="recall_at_5" name="Recall@5" fill={CHART.final} radius={[4, 4, 0, 0]} maxBarSize={18}>
              <LabelList
                dataKey="recall_at_5"
                position="top"
                formatter={(v: number) => v.toFixed(2)}
                style={{ fill: "#E7EAEE", fontSize: 10, fontFamily: "Geist Mono Variable, monospace" }}
              />
            </Bar>
            <Bar dataKey="mrr" name="MRR" fill={CHART.second} radius={[4, 4, 0, 0]} maxBarSize={18} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function CategoryTable({ data }: { data: EvalSummary }) {
  const cell = (v: number) =>
    Number.isFinite(v) ? (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-16 rounded-full bg-white/[0.05]">
        <div className={cx("h-full rounded-full", v >= 0.9 ? "bg-accent" : v >= 0.8 ? "bg-accent/60" : "bg-warning")} style={{ width: `${v * 100}%` }} />
      </div>
      <span className="font-mono text-2xs text-muted">{v.toFixed(2)}</span>
    </div>
    ) : (
      // La métrica no aplica a la categoría (p. ej. Recall@5 en casos de herramientas).
      <span className="font-mono text-2xs text-faint">—</span>
    );
  return (
    <TableScroll>
      <table className="w-full min-w-[640px] text-left text-xs">
        <thead className="text-2xs text-faint">
          <tr className="border-b hairline">
            <th className="py-2 pr-3 font-normal">Categoría</th>
            <th className="py-2 pr-3 text-right font-normal">Casos</th>
            <th className="py-2 pr-3 font-normal">Recall@5</th>
            <th className="py-2 pr-3 font-normal">Fidelidad</th>
            <th className="py-2 pr-3 font-normal">Precisión de citas</th>
            <th className="py-2 text-right font-normal">Aprobados</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-white/[0.04]">
          {data.categories.map((c) => (
            <tr key={c.category} className="transition-colors duration-150 hover:bg-white/[0.02]">
              <td className="py-2.5 pr-3 font-mono text-text">{c.category}</td>
              <td className="py-2.5 pr-3 text-right font-mono text-muted">{c.n}</td>
              <td className="py-2.5 pr-3">{cell(c.recall_at_5)}</td>
              <td className="py-2.5 pr-3">{cell(c.faithfulness)}</td>
              <td className="py-2.5 pr-3">{cell(c.citation_precision)}</td>
              <td className="py-2.5 text-right font-mono">
                <span className={c.passed === c.n ? "text-accent-2" : "text-warning"}>
                  {Number.isFinite(c.passed) ? `${c.passed}/${c.n}` : `—/${c.n}`}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </TableScroll>
  );
}

const SEVERITY_TONE: Record<string, Tone> = { low: "neutral", medium: "warning", high: "danger", critical: "danger" };
const SEVERITY_LABEL: Record<string, string> = { low: "baja", medium: "media", high: "alta", critical: "crítica" };

export default function EvaluationView() {
  const [state, setState] = useState<State>({ status: "loading" });
  const [reload, setReload] = useState(0);
  const mode = useConnection();

  useEffect(() => {
    let alive = true;
    setState({ status: "loading" });
    api
      .evalSummary()
      .then((data) => alive && setState({ status: "ok", data }))
      .catch((e: unknown) =>
        alive &&
        setState(
          e instanceof ApiError && e.status === 404
            ? { status: "empty" }
            : { status: "error", message: e instanceof ApiError ? e.message : "No se pudo cargar la evaluación." },
        ),
      );
    return () => {
      alive = false;
    };
  }, [reload]);

  const passed = useMemo(() => {
    if (state.status !== "ok") return null;
    const withTarget = state.data.metrics.filter((m) => m.target !== null);
    return { ok: withTarget.filter((m) => metricStatus(m) === "pass").length, total: withTarget.length };
  }, [state]);

  if (state.status === "loading") {
    return (
      <div className="mx-auto max-w-[1180px] space-y-4 px-6 py-8">
        <Skeleton className="h-6 w-64" />
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          {Array.from({ length: 8 }, (_, i) => (
            <Skeleton key={i} className="h-32" />
          ))}
        </div>
      </div>
    );
  }
  if (state.status === "empty") {
    return (
      <PendingResults
        icon={FlaskConical}
        eyebrow="Evaluación · banco de 42 casos"
        title="Todavía no hay una corrida de evaluación"
        description="La API está en vivo, pero aún no existe eval/results. Cuando corras la evaluación, esta vista mostrará las metas del anteproyecto, baseline contra final con intervalos de confianza, ablaciones, desglose por categoría y fallas."
        steps={EVAL_STEPS}
        path="eval/results/latest/summary.json"
        onRetry={() => setReload((n) => n + 1)}
      />
    );
  }
  if (state.status === "error") {
    return (
      <div className="mx-auto max-w-[1180px] px-6 py-16">
        <EmptyState
          icon={AlertTriangle}
          tone="warning"
          title="No se pudo leer la evaluación"
          description={state.message}
          action={
            <Button icon={RefreshCw} onClick={() => setReload((n) => n + 1)}>
              Reintentar
            </Button>
          }
        />
      </div>
    );
  }

  const data = state.data;
  return (
    <div className="mx-auto max-w-[1180px] space-y-6 px-4 py-8 md:px-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="eyebrow mb-1.5">Corrida {data.run_id}</div>
          <h2 className="text-xl font-semibold tracking-[-0.03em] text-text">
            {passed ? `${passed.ok} de ${passed.total} metas cumplidas` : "Resultados"}
          </h2>
          <p className="mt-1 text-sm text-muted">
            {data.n_cases} casos × {data.runs} corridas · baseline {data.baseline_run_id || "—"}
            {data.created_at ? ` · ${formatDateTime(data.created_at)}` : ""}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {mode !== "live" && (
            <Badge tone="violet" icon={FlaskConical}>
              Datos de demostración
            </Badge>
          )}
          {data.paired_test?.map((t) => (
            <Badge key={t.metric} mono tone={t.p_value < 0.05 ? "accent" : "neutral"}>
              {t.metric} Δ+{t.delta.toFixed(2)} · p={t.p_value.toFixed(3)}
            </Badge>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {data.metrics.map((m) => (
          <KpiTile key={m.key} m={m} />
        ))}
      </div>

      <Panel coreClassName="p-5">
        <SectionHeader
          eyebrow="Baseline vs final"
          title="Mejora con intervalo de confianza"
          description="Bootstrap 95 % sobre los casos del banco; la línea vertical es la meta del anteproyecto."
          className="mb-5"
        />
        <IntervalChart metrics={data.metrics} />
      </Panel>

      {data.ablations.length > 0 && (
        <Panel coreClassName="p-5">
          <SectionHeader
            eyebrow="Ablaciones"
            title="Qué aporta cada decisión"
            description="Recall@5 y MRR por recuperador, estrategia de chunking y modelo de embeddings (el resto de parámetros fijo)."
            className="mb-4"
            actions={
              <Legend
                items={[
                  { label: "Recall@5", color: CHART.final },
                  { label: "MRR", color: CHART.second },
                ]}
              />
            }
          />
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
            {(["retriever", "chunker", "embedder"] as const).map((g) => (
              <AblationChart key={g} rows={data.ablations} group={g} />
            ))}
          </div>
        </Panel>
      )}

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1.15fr_1fr]">
        <Panel coreClassName="p-5">
          <SectionHeader eyebrow="Desglose" title="Por categoría del banco" className="mb-3" />
          <CategoryTable data={data} />
        </Panel>
        <Panel coreClassName="p-5">
          <SectionHeader
            eyebrow="Análisis de errores"
            title={`Fallas (${data.failures.length})`}
            description="Casos que no cumplieron el comportamiento esperado y su causa raíz."
            className="mb-3"
          />
          <ul className="divide-y divide-white/[0.05]">
            {data.failures.map((f) => (
              <li key={f.case_id} className="py-3">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs text-text">{f.case_id}</span>
                  <Badge mono>{f.category}</Badge>
                  <Badge tone={SEVERITY_TONE[f.severity] ?? "neutral"} className="ml-auto">
                    severidad {SEVERITY_LABEL[f.severity] ?? f.severity}
                  </Badge>
                </div>
                <p className="mt-1.5 text-xs text-text/90">{f.question}</p>
                <dl className="mt-2 grid grid-cols-[72px_1fr] gap-x-3 gap-y-1 text-2xs">
                  <dt className="text-faint">Esperado</dt>
                  <dd className="text-muted">{f.expected}</dd>
                  <dt className="text-faint">Observado</dt>
                  <dd className="text-muted">{f.observed}</dd>
                  <dt className="text-faint">Causa</dt>
                  <dd className="text-warning/90">{f.cause}</dd>
                </dl>
              </li>
            ))}
          </ul>
        </Panel>
      </div>

      <p className="text-2xs text-faint">
        Las latencias son de pared e incluyen el arranque del CLI del Agent SDK (SPEC §12); la latencia de API se reporta
        aparte en el inspector de traza.
      </p>
    </div>
  );
}
