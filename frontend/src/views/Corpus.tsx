/**
 * Corpus y gobernanza: manifiesto de fuentes (origen, licencia, versión, vigencia, roles,
 * sensibilidad, hash) y estadísticas de chunks del índice (SPEC §5.1).
 */
import { AlertTriangle, Check, Copy, FlaskConical, RefreshCw, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ApiError, api } from "../api/client";
import type { DocType, DocumentsResponse, SourceEntry } from "../api/types";
import { ChartTooltip } from "../components/charts";
import { Badge, EmptyState, Panel, SectionHeader, Skeleton, TableScroll, Button } from "../components/ui";
import { useConnection } from "../hooks/useConnection";
import { useCopy } from "../hooks/useCopy";
import { CHART, axisProps } from "../lib/chartTheme";
import { cx } from "../lib/cx";
import { MODULE_LABEL, ROLE_SHORT, formatCompact, formatDate, formatInt, labelOf, shortHash } from "../lib/format";
import type { Tone } from "../lib/tone";
import { useSession } from "../state/session";

type State = { status: "loading" } | { status: "ok"; data: DocumentsResponse } | { status: "error"; message: string };

const DOC_TYPE_LABEL: Record<DocType, string> = { odoo_doc: "Odoo oficial", runbook: "Runbook", policy: "Política" };
const SENSITIVITY_TONE: Record<string, Tone> = {
  publica: "neutral",
  interna: "info",
  confidencial: "warning",
  restringida: "danger",
};

function HashCell({ hash }: { hash: string }) {
  const [copied, copy] = useCopy(1200);
  return (
    <button
      type="button"
      title={hash}
      onClick={() => copy(hash)}
      className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-md px-1 py-0.5 font-mono text-[11px] text-muted transition-colors duration-150 hover:text-text"
      aria-label={`Copiar sha256 ${hash}`}
    >
      {shortHash(hash, 6, 4)}
      {copied ? (
        <Check size={11} strokeWidth={2} className="text-accent-2" aria-hidden />
      ) : (
        <Copy size={11} strokeWidth={1.8} className="opacity-60" aria-hidden />
      )}
    </button>
  );
}

function StatTile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-[14px] bg-surface p-4 shadow-inner-hl ring-1 ring-inset ring-white/[0.06]">
      <div className="text-xs text-muted">{label}</div>
      <div className="mt-2.5 font-mono text-[24px] leading-none tracking-[-0.03em] text-text">{value}</div>
      {sub && <div className="mt-2 truncate text-2xs text-faint">{sub}</div>}
    </div>
  );
}

function ManifestTable({ sources }: { sources: SourceEntry[] }) {
  const [q, setQ] = useState("");
  const [type, setType] = useState<DocType | "all">("all");
  const [showObsolete, setShowObsolete] = useState(true);
  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return sources.filter(
      (s) =>
        (type === "all" || s.doc_type === type) &&
        (showObsolete || s.status === "vigente") &&
        (!needle || `${s.id} ${s.title} ${s.source_file} ${s.origin}`.toLowerCase().includes(needle)),
    );
  }, [sources, q, type, showObsolete]);

  return (
    <>
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <label className="relative flex h-8 min-w-[220px] flex-1 items-center sm:max-w-[300px]">
          <Search size={14} strokeWidth={1.6} className="absolute left-2.5 text-faint" aria-hidden />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Buscar por id, título u origen…"
            aria-label="Buscar en el manifiesto"
            className="h-8 w-full rounded-[9px] bg-white/[0.03] pl-8 pr-3 text-xs text-text ring-1 ring-inset ring-white/[0.07] placeholder:text-faint focus:outline-none focus:ring-accent/40"
          />
        </label>
        <div role="radiogroup" aria-label="Tipo de documento" className="flex rounded-[9px] bg-white/[0.03] p-0.5 ring-1 ring-inset ring-white/[0.07]">
          {(["all", "odoo_doc", "runbook", "policy"] as const).map((t) => (
            <button
              key={t}
              type="button"
              role="radio"
              aria-checked={type === t}
              onClick={() => setType(t)}
              className={cx(
                "h-7 rounded-[7px] px-2.5 text-xs transition-colors duration-150",
                type === t ? "bg-white/[0.07] text-text" : "text-faint hover:text-muted",
              )}
            >
              {t === "all" ? "Todos" : DOC_TYPE_LABEL[t]}
            </button>
          ))}
        </div>
        <label className="ml-auto flex cursor-pointer items-center gap-2 text-xs text-muted">
          <input
            type="checkbox"
            checked={showObsolete}
            onChange={(e) => setShowObsolete(e.target.checked)}
            className="h-3.5 w-3.5 accent-[#10B981]"
          />
          Mostrar obsoletos
        </label>
      </div>
      <TableScroll>
        <table className="w-full min-w-[1080px] text-left text-xs">
          <thead className="text-2xs text-faint">
            <tr className="border-b hairline">
              <th className="py-2 pr-3 font-normal">Documento</th>
              <th className="py-2 pr-3 font-normal">Origen · licencia</th>
              <th className="py-2 pr-3 font-normal">Versión</th>
              <th className="py-2 pr-3 font-normal">Vigencia</th>
              <th className="py-2 pr-3 font-normal">Roles</th>
              <th className="py-2 pr-3 font-normal">Sensibilidad</th>
              <th className="py-2 pr-3 text-right font-normal">Chunks</th>
              <th className="py-2 font-normal">sha256</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/[0.04]">
            {filtered.map((s) => (
              <tr key={s.id} className={cx("transition-colors duration-150 hover:bg-white/[0.02]", s.status === "obsoleto" && "opacity-60")}>
                <td className="max-w-[250px] py-3 pr-3">
                  <div className="truncate text-text">{s.title}</div>
                  <div className="mt-0.5 flex items-center gap-1.5">
                    <span className="truncate font-mono text-[10.5px] text-faint">{s.source_file}</span>
                  </div>
                </td>
                <td className="max-w-[190px] py-3 pr-3">
                  <div className="truncate text-muted">{s.origin}</div>
                  <div className="text-2xs text-faint">
                    {s.license} · {s.owner}
                  </div>
                </td>
                <td className="py-3 pr-3">
                  <div className="font-mono text-[11px] text-muted">{s.version}</div>
                  <div className="text-2xs text-faint">{s.erp_version === "any" ? "todas" : `Odoo ${s.erp_version}`}</div>
                </td>
                <td className="py-3 pr-3">
                  <Badge tone={s.status === "vigente" ? "accent" : "warning"}>{s.status}</Badge>
                  <div className="mt-1 whitespace-nowrap text-2xs text-faint" title={`Actualización ${s.update_frequency}`}>
                    {formatDate(s.date)}
                  </div>
                </td>
                <td className="py-3 pr-3">
                  <div className="flex gap-1">
                    {s.allowed_roles.map((r) => (
                      <span key={r} className="rounded-[5px] bg-white/[0.05] px-1.5 py-px font-mono text-[10px] text-muted">
                        {ROLE_SHORT[r]}
                      </span>
                    ))}
                  </div>
                </td>
                <td className="py-3 pr-3">
                  <Badge tone={SENSITIVITY_TONE[s.sensitivity] ?? "neutral"}>{s.sensitivity}</Badge>
                </td>
                <td className="py-3 pr-3 text-right font-mono text-muted">{s.chunks !== undefined ? formatInt(s.chunks) : "—"}</td>
                <td className="py-3">
                  <HashCell hash={s.sha256} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </TableScroll>
      {filtered.length === 0 && <EmptyState icon={Search} title="Sin documentos para ese filtro" className="py-10" />}
    </>
  );
}

export default function CorpusView() {
  // El manifiesto se filtra por RBAC en la API: se pide con el rol de la identidad activa.
  const role = useSession().user.role;
  const [state, setState] = useState<State>({ status: "loading" });
  const [reload, setReload] = useState(0);
  const mode = useConnection();

  useEffect(() => {
    let alive = true;
    setState({ status: "loading" });
    api
      .documents(role)
      .then((data) => alive && setState({ status: "ok", data }))
      .catch((e: unknown) =>
        alive && setState({ status: "error", message: e instanceof ApiError ? e.message : "No se pudo cargar el manifiesto." }),
      );
    return () => {
      alive = false;
    };
  }, [reload, role]);

  if (state.status === "loading") {
    return (
      <div className="mx-auto max-w-[1180px] space-y-4 px-6 py-8">
        <div className="grid grid-cols-3 gap-3 lg:grid-cols-6">
          {Array.from({ length: 6 }, (_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
        <Skeleton className="h-96" />
      </div>
    );
  }
  if (state.status === "error") {
    return (
      <div className="mx-auto max-w-[1180px] px-6 py-16">
        <EmptyState
          icon={AlertTriangle}
          tone="warning"
          title="Manifiesto no disponible"
          description={`${state.message} Ejecuta \`make ingest\` para construir el índice.`}
          action={
            <Button icon={RefreshCw} onClick={() => setReload((n) => n + 1)}>
              Reintentar
            </Button>
          }
        />
      </div>
    );
  }

  const { sources, stats } = state.data;
  const byModule = stats.by_module.map((m) => ({ name: labelOf(MODULE_LABEL, m.module), chunks: m.chunks }));
  const vigentes = sources.filter((s) => s.status === "vigente").length;

  return (
    <div className="mx-auto max-w-[1180px] space-y-6 px-4 py-8 md:px-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="eyebrow mb-1.5">Índice {stats.chunker} · {stats.embedder}</div>
          <h2 className="text-xl font-semibold tracking-[-0.03em] text-text">
            {stats.documents} fuentes · {vigentes} vigentes
          </h2>
          <p className="mt-1 text-sm text-muted">
            Documentación oficial de Odoo 17.0/18.0 (CC BY-SA 4.0) y runbooks y políticas sintéticos de Altamira Soluciones ERP.
          </p>
        </div>
        {mode !== "live" && (
          <Badge tone="violet" icon={FlaskConical}>
            Datos de demostración
          </Badge>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
        <StatTile label="Documentos" value={formatInt(stats.documents)} sub={`${vigentes} vigentes`} />
        <StatTile label="Páginas" value={formatInt(stats.pages)} />
        <StatTile label="Chunks" value={formatInt(stats.chunks)} sub="chunk_id determinista" />
        <StatTile label="Tokens" value={formatCompact(stats.tokens)} sub="tokenizador bge-m3" />
        <StatTile label="Instrucciones sospechosas" value={formatInt(stats.flagged_chunks)} sub="marcadas en la ingesta" />
        <StatTile label="Duplicados por hash" value={formatInt(stats.duplicates)} sub="descartados" />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Panel coreClassName="p-5">
          <SectionHeader eyebrow="Distribución" title="Chunks por módulo" className="mb-4" />
          <div className="h-[240px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={byModule} layout="vertical" margin={{ top: 0, right: 16, left: 8, bottom: 0 }} barCategoryGap="28%">
                <CartesianGrid stroke={CHART.grid} horizontal={false} />
                <XAxis type="number" {...axisProps} />
                <YAxis type="category" dataKey="name" width={92} {...axisProps} />
                <Tooltip cursor={{ fill: "rgba(255,255,255,0.03)" }} content={<ChartTooltip format={(v) => formatInt(v)} />} />
                <Bar dataKey="chunks" name="chunks" fill={CHART.final} radius={[0, 4, 4, 0]} maxBarSize={16} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
        <Panel coreClassName="p-5">
          <SectionHeader
            eyebrow="Tamaño"
            title="Tokens por chunk"
            description="StructureChunker con objetivo de 450 tokens y 15 % de solape."
            className="mb-4"
          />
          {stats.token_histogram.length ? (
            <div className="h-[212px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={stats.token_histogram} margin={{ top: 4, right: 4, left: -16, bottom: 0 }} barCategoryGap="14%">
                  <CartesianGrid stroke={CHART.grid} vertical={false} />
                  <XAxis dataKey="bucket" {...axisProps} />
                  <YAxis {...axisProps} />
                  <Tooltip cursor={{ fill: "rgba(255,255,255,0.03)" }} content={<ChartTooltip format={(v) => formatInt(v)} />} />
                  <Bar dataKey="chunks" name="chunks" fill={CHART.final} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="text-xs text-faint">El endpoint no reporta histograma de tokens.</p>
          )}
        </Panel>
      </div>

      <Panel coreClassName="p-5">
        <SectionHeader
          eyebrow="Gobernanza"
          title="Manifiesto de fuentes"
          description="Cada fuente declara origen, licencia, versión, vigencia, roles autorizados y sensibilidad; el sha256 permite verificar que el índice corresponde al archivo."
          className="mb-4"
        />
        <ManifestTable sources={sources} />
      </Panel>
    </div>
  );
}
