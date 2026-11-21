/**
 * Seguridad: KPIs del anteproyecto, matriz red-team por vector × resultado esperado y casos
 * con amenaza, componente, esperado, observado y mitigación (SPEC §5.5 y §10).
 */
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, ChevronDown, FlaskConical, RefreshCw, ShieldCheck, XCircle } from "lucide-react";
import { Fragment, useEffect, useMemo, useState } from "react";
import { ApiError, api } from "../api/client";
import type { RedteamCase, RedteamOutcome, RedteamVector, SecuritySummary } from "../api/types";
import { PendingResults } from "../components/PendingResults";
import { Badge, Button, EmptyState, Panel, SectionHeader, Skeleton, TableScroll } from "../components/ui";
import { useConnection } from "../hooks/useConnection";
import { cx } from "../lib/cx";
import { ROLE_SHORT, formatDateTime, formatPct } from "../lib/format";
import { OUTCOME_LABEL, VECTOR_LABEL, rateText, securityKpis, summarizeVectors } from "../lib/metrics";

type State =
  | { status: "loading" }
  | { status: "ok"; data: SecuritySummary & { not_applicable?: number; mode?: string | null } }
  | { status: "empty" }
  | { status: "error"; message: string };

const REDTEAM_STEPS = [
  { label: "Capa de guards contra el banco red-team (sin LLM, segundos)", command: "make redteam" },
  { label: "Agente completo: herramientas, permisos y HITL de extremo a extremo", command: "make redteam-e2e" },
];

const OUTCOMES: RedteamOutcome[] = ["blocked", "denied", "abstained", "needs_confirmation", "masked"];

function Matrix({
  cases,
  selected,
  onSelect,
}: {
  cases: RedteamCase[];
  selected: RedteamVector | null;
  onSelect: (v: RedteamVector | null) => void;
}) {
  const vectors = useMemo(() => summarizeVectors(cases), [cases]);
  const failedCell = (v: RedteamVector, o: RedteamOutcome) =>
    cases.some((c) => c.vector === v && c.expected_outcome === o && !c.passed);
  return (
    <TableScroll>
      <table className="w-full min-w-[760px] border-separate border-spacing-[3px] text-xs">
        <thead>
          <tr className="text-2xs text-faint">
            <th className="px-2 pb-1 text-left font-normal">Vector</th>
            {OUTCOMES.map((o) => (
              <th key={o} className="px-1 pb-1 text-center font-normal">
                {OUTCOME_LABEL[o]}
              </th>
            ))}
            <th className="px-2 pb-1 text-left font-normal">Mitigados</th>
          </tr>
        </thead>
        <tbody>
          {vectors.map((v) => {
            const active = selected === v.vector;
            return (
              <tr
                key={v.vector}
                onClick={() => onSelect(active ? null : v.vector)}
                className={cx("cursor-pointer", selected && !active && "opacity-50")}
                aria-selected={active}
              >
                <td
                  className={cx(
                    "rounded-[8px] px-2 py-2 transition-colors duration-150",
                    active ? "bg-white/[0.06] text-text" : "text-text hover:bg-white/[0.03]",
                  )}
                >
                  {VECTOR_LABEL[v.vector]}
                </td>
                {OUTCOMES.map((o) => {
                  const n = v.byOutcome[o] ?? 0;
                  const failed = failedCell(v.vector, o);
                  return (
                    <td key={o} className="p-0">
                      <div
                        className={cx(
                          "flex h-9 items-center justify-center rounded-[8px] font-mono text-xs ring-1 ring-inset",
                          n === 0 && "text-faint/50 ring-white/[0.03]",
                          n > 0 && !failed && "text-text ring-accent/20",
                          failed && "text-danger ring-danger/40",
                        )}
                        style={n > 0 && !failed ? { background: `rgba(16,185,129,${0.08 + Math.min(n, 4) * 0.06})` } : failed ? { background: "rgba(248,113,113,0.1)" } : undefined}
                        title={n ? `${n} caso(s)${failed ? " · con falla" : ""}` : "sin casos"}
                      >
                        {n || "·"}
                      </div>
                    </td>
                  );
                })}
                <td className="px-2">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-20 rounded-full bg-white/[0.05]">
                      <div
                        className={cx("h-full rounded-full", v.rate === 1 ? "bg-accent" : "bg-warning")}
                        style={{ width: `${v.rate * 100}%` }}
                      />
                    </div>
                    <span className={cx("font-mono text-2xs", v.rate === 1 ? "text-muted" : "text-warning")}>
                      {v.passed}/{v.total}
                    </span>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </TableScroll>
  );
}

function CaseRows({ cases }: { cases: RedteamCase[] }) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <TableScroll>
      <table className="w-full min-w-[920px] text-left text-xs">
        <thead className="text-2xs text-faint">
          <tr className="border-b hairline">
            <th className="w-6 py-2 font-normal" />
            <th className="py-2 pr-3 font-normal">Caso</th>
            <th className="py-2 pr-3 font-normal">Ataque</th>
            <th className="py-2 pr-3 font-normal">Componente</th>
            <th className="py-2 pr-3 font-normal">Esperado</th>
            <th className="py-2 pr-3 font-normal">Observado</th>
            <th className="py-2 text-right font-normal">Resultado</th>
          </tr>
        </thead>
        <tbody>
          {cases.map((c) => {
            const isOpen = open === c.id;
            return (
              <Fragment key={c.id}>
                <tr
                  onClick={() => setOpen(isOpen ? null : c.id)}
                  className="cursor-pointer border-b border-white/[0.04] transition-colors duration-150 hover:bg-white/[0.02]"
                  aria-expanded={isOpen}
                >
                  <td className="py-2.5">
                    <ChevronDown
                      size={14}
                      strokeWidth={1.6}
                      className={cx("text-faint transition-transform duration-200 ease-out-strong", isOpen && "rotate-180")}
                      aria-hidden
                    />
                  </td>
                  <td className="py-2.5 pr-3">
                    <div className="font-mono text-text">{c.id}</div>
                    <div className="text-2xs text-faint">
                      {ROLE_SHORT[c.role]}
                      {c.in_official_bench ? " · banco oficial" : ""}
                    </div>
                  </td>
                  <td className="max-w-[300px] py-2.5 pr-3">
                    <div className="truncate font-mono text-[11.5px] text-muted" title={c.attack}>
                      {c.attack}
                    </div>
                  </td>
                  <td className="py-2.5 pr-3 font-mono text-[11px] text-muted">{c.component}</td>
                  <td className="py-2.5 pr-3">
                    <Badge>{OUTCOME_LABEL[c.expected_outcome]}</Badge>
                  </td>
                  <td className="max-w-[240px] py-2.5 pr-3">
                    <div className="truncate text-muted" title={c.observed}>
                      {c.observed}
                    </div>
                  </td>
                  <td className="py-2.5 text-right">
                    {c.passed ? (
                      <Badge tone="accent" icon={CheckCircle2}>
                        Mitigado
                      </Badge>
                    ) : (
                      <Badge tone="danger" icon={XCircle}>
                        Falla
                      </Badge>
                    )}
                  </td>
                </tr>
                <AnimatePresence initial={false}>
                  {isOpen && (
                    <tr>
                      <td colSpan={7} className="p-0">
                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0, transition: { duration: 0.1 } }}
                          transition={{ duration: 0.18 }}
                          className="grid gap-4 border-b border-white/[0.04] bg-white/[0.015] px-8 py-4 md:grid-cols-2"
                        >
                          <div>
                            <div className="text-2xs text-faint">Ataque</div>
                            <p className="mt-1 font-mono text-[11.5px] leading-relaxed text-text/90">{c.attack}</p>
                            <div className="mt-3 text-2xs text-faint">Amenaza</div>
                            <p className="mt-1 text-xs text-muted">{c.threat}</p>
                          </div>
                          <div>
                            <div className="text-2xs text-faint">Comportamiento esperado</div>
                            <p className="mt-1 text-xs text-muted">{c.expected_behavior}</p>
                            <div className="mt-3 text-2xs text-faint">Observado</div>
                            <p className={cx("mt-1 text-xs", c.passed ? "text-muted" : "text-danger")}>{c.observed}</p>
                            <div className="mt-3 text-2xs text-faint">Mitigación</div>
                            <p className="mt-1 text-xs text-accent-2/90">{c.mitigation}</p>
                          </div>
                        </motion.div>
                      </td>
                    </tr>
                  )}
                </AnimatePresence>
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </TableScroll>
  );
}

export default function SecurityView() {
  const [state, setState] = useState<State>({ status: "loading" });
  const [reload, setReload] = useState(0);
  const [vector, setVector] = useState<RedteamVector | null>(null);
  const [onlyFailures, setOnlyFailures] = useState(false);
  const mode = useConnection();

  useEffect(() => {
    let alive = true;
    setState({ status: "loading" });
    api
      .securitySummary()
      .then((data) => alive && setState({ status: "ok", data }))
      .catch((e: unknown) =>
        alive &&
        setState(
          e instanceof ApiError && e.status === 404
            ? { status: "empty" }
            : { status: "error", message: e instanceof ApiError ? e.message : "No se pudo cargar el banco red-team." },
        ),
      );
    return () => {
      alive = false;
    };
  }, [reload]);

  if (state.status === "loading") {
    return (
      <div className="mx-auto max-w-[1180px] space-y-4 px-6 py-8">
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          {Array.from({ length: 4 }, (_, i) => (
            <Skeleton key={i} className="h-28" />
          ))}
        </div>
        <Skeleton className="h-80" />
      </div>
    );
  }
  if (state.status === "empty") {
    return (
      <PendingResults
        icon={ShieldCheck}
        eyebrow="Seguridad · banco red-team"
        title="Todavía no hay resultados red-team"
        description="La API está en vivo, pero security/results está vacío. Al ejecutar el banco, esta vista mostrará cada ataque con el componente que lo detiene, el resultado esperado contra el observado y la mitigación."
        steps={REDTEAM_STEPS}
        path="security/results/latest.json"
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
          title="Sin resultados red-team"
          description={`${state.message} Ejecuta \`make redteam\` para generarlos.`}
          action={
            <Button icon={RefreshCw} onClick={() => setReload((n) => n + 1)}>
              Reintentar
            </Button>
          }
        />
      </div>
    );
  }

  const cases = state.data.cases;
  const kpis = securityKpis(cases);
  const visible = cases.filter((c) => (!vector || c.vector === vector) && (!onlyFailures || !c.passed));
  const failures = cases.filter((c) => !c.passed).length;

  return (
    <div className="mx-auto max-w-[1180px] space-y-6 px-4 py-8 md:px-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="eyebrow mb-1.5">Banco red-team {state.data.run_id && `· ${state.data.run_id}`}</div>
          <h2 className="text-xl font-semibold tracking-[-0.03em] text-text">
            {cases.length - failures} de {cases.length} ataques mitigados
          </h2>
          <p className="mt-1 text-sm text-muted">
            {summarizeVectors(cases).length} vectores · {cases.filter((c) => c.in_official_bench).length} casos en el banco oficial
            {state.data.not_applicable ? ` · ${state.data.not_applicable} no aplican en modo ${state.data.mode === "guards_only" ? "solo guards" : (state.data.mode ?? "actual")}` : ""}
            {state.data.created_at ? ` · ${formatDateTime(state.data.created_at)}` : ""}
          </p>
        </div>
        {mode !== "live" && (
          <Badge tone="violet" icon={FlaskConical}>
            Datos de demostración
          </Badge>
        )}
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {kpis.map((k) => {
          const ok = k.value >= k.target;
          return (
            <div
              key={k.label}
              className={cx(
                "rounded-[14px] bg-surface p-4 shadow-inner-hl ring-1 ring-inset",
                ok ? "ring-white/[0.06]" : "ring-danger/25",
              )}
            >
              <div className="flex items-start justify-between gap-2">
                <span className="text-xs text-muted">{k.label}</span>
                <Badge tone={ok ? "accent" : "danger"} icon={ok ? CheckCircle2 : XCircle}>
                  {ok ? "Cumple" : "No cumple"}
                </Badge>
              </div>
              <div className="mt-3 font-mono text-[26px] leading-none tracking-[-0.03em] text-text">{formatPct(k.value)}</div>
              <div className="mt-2 text-2xs text-faint">
                meta ≥ {formatPct(k.target, 0)} · {rateText(k.passed, k.total).split(" · ")[0]} casos
              </div>
            </div>
          );
        })}
      </div>

      <Panel coreClassName="p-5">
        <SectionHeader
          eyebrow="Matriz"
          title="Vector de ataque × resultado esperado"
          description="Cada celda cuenta los casos; el rojo marca dónde hubo una falla. Haz clic en un vector para filtrar los casos."
          className="mb-4"
          actions={<ShieldCheck size={16} className="text-faint" aria-hidden />}
        />
        <Matrix cases={cases} selected={vector} onSelect={setVector} />
      </Panel>

      <Panel coreClassName="p-5">
        <SectionHeader
          eyebrow="Casos"
          title={vector ? VECTOR_LABEL[vector] : "Todos los casos"}
          description="Amenaza, componente que la mitiga, comportamiento esperado, resultado observado y mitigación."
          className="mb-4"
          actions={
            <div className="flex items-center gap-2">
              {vector && (
                <Button size="sm" variant="ghost" onClick={() => setVector(null)}>
                  Quitar filtro
                </Button>
              )}
              <Button
                size="sm"
                variant={onlyFailures ? "danger" : "secondary"}
                onClick={() => setOnlyFailures((v) => !v)}
                aria-pressed={onlyFailures}
              >
                Solo fallas ({failures})
              </Button>
            </div>
          }
        />
        {visible.length ? (
          <CaseRows cases={visible} />
        ) : (
          <EmptyState icon={ShieldCheck} tone="accent" title="Sin fallas en este filtro" className="py-10" />
        )}
      </Panel>
    </div>
  );
}
