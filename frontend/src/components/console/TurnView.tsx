/**
 * Un turno de la consola: la consulta del usuario y la tarjeta de respuesta con su estado
 * (respondida, abstención, bloqueo, acceso restringido, revisión humana, herramienta).
 */
import { motion } from "framer-motion";
import {
  ArrowUpRight,
  CircleSlash,
  Copy,
  FileText,
  GitCompareArrows,
  Lock,
  ShieldAlert,
  UserCheck,
  Wrench,
} from "lucide-react";
import { useMemo, type ReactNode } from "react";
import type { QueryResponse, SourceRef } from "../../api/types";
import { useCopy } from "../../hooks/useCopy";
import { cx } from "../../lib/cx";
import { INTENT_LABEL, ROLE_SHORT, formatScore } from "../../lib/format";
import { OUTCOME_LABEL, OUTCOME_TONE, TONE_TEXT, type Tone } from "../../lib/tone";
import { classificationFromEvent, ticketFromEvent } from "../../lib/toolData";
import { buildPipeline, findEvent, gateInfo, latencySummary, outcomeOf, type Outcome } from "../../lib/trace";
import { useSession, type Turn } from "../../state/session";
import { Badge, Button, IconButton, Skeleton } from "../ui";
import { AnswerBody } from "./AnswerBody";
import { ClassificationCard, PendingDraftCard, TicketStatusCard } from "./ToolCards";

const OUTCOME_ICON: Record<Outcome, typeof FileText> = {
  answered: FileText,
  abstained: CircleSlash,
  blocked: ShieldAlert,
  denied: Lock,
  review: UserCheck,
  tool: Wrench,
  pending_confirmation: UserCheck,
};

function Banner({ tone, icon: Icon, title, children }: { tone: Tone; icon: typeof FileText; title: string; children: ReactNode }) {
  const ring: Record<Tone, string> = {
    danger: "bg-danger/[0.06] ring-danger/20",
    warning: "bg-warning/[0.06] ring-warning/20",
    violet: "bg-violet/[0.06] ring-violet/20",
    accent: "bg-accent/[0.06] ring-accent/20",
    info: "bg-info/[0.06] ring-info/20",
    neutral: "bg-white/[0.03] ring-white/10",
  };
  return (
    <div className={cx("rounded-[12px] p-4 ring-1 ring-inset", ring[tone])}>
      <div className="flex items-start gap-3">
        <Icon size={18} strokeWidth={1.6} className={cx("mt-0.5 shrink-0", TONE_TEXT[tone])} aria-hidden />
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium text-text">{title}</div>
          <div className="mt-1 text-[13px] leading-relaxed text-muted">{children}</div>
        </div>
      </div>
    </div>
  );
}

function CitationChips({ sources, onOpen }: { sources: SourceRef[]; onOpen: (i: number) => void }) {
  if (!sources.length) return null;
  return (
    <div>
      <div className="eyebrow mb-2">Fuentes citadas · {sources.length}</div>
      <div className="flex flex-wrap gap-2">
        {sources.map((s, i) => (
          <button
            key={`${s.chunk_id}-${i}`}
            type="button"
            onClick={() => onOpen(i)}
            aria-label={`Abrir cita ${i + 1}: ${s.source}, página ${s.page}`}
            className={cx(
              "group flex max-w-full items-center gap-2 rounded-[10px] bg-white/[0.03] py-1.5 pl-1.5 pr-3 text-left ring-1 ring-inset ring-white/[0.07]",
              "transition-[background-color,box-shadow,transform] duration-150 ease-out-strong active:scale-[0.98]",
              "hover:bg-accent/[0.07] hover:ring-accent/30",
            )}
          >
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-[7px] bg-accent/15 font-mono text-[11px] text-accent-2 ring-1 ring-inset ring-accent/30">
              {i + 1}
            </span>
            <span className="min-w-0">
              <span className="block truncate text-xs text-text">{s.source}</span>
              <span className="block font-mono text-[10.5px] text-faint">
                pág. {s.page} · {s.chunk_id.split("::").slice(1).join("::")}
              </span>
            </span>
            <ArrowUpRight
              size={13}
              strokeWidth={1.8}
              className="shrink-0 text-faint transition-[transform,color] duration-150 group-hover:-translate-y-px group-hover:translate-x-px group-hover:text-accent-2"
              aria-hidden
            />
          </button>
        ))}
      </div>
    </div>
  );
}

/** Asegura punto final en un texto que viene del servidor. */
function sentence(text: string): string {
  const t = text.trim();
  return /[.!?…]$/.test(t) ? t : `${t}.`;
}

function StatusBanner({ res, outcome, onEscalate }: { res: QueryResponse; outcome: Outcome; onEscalate: () => void }) {
  if (outcome === "blocked") {
    const guard = findEvent(res.events, "input_guard");
    const flags = Array.isArray(guard?.data.flags) ? (guard?.data.flags as string[]) : [];
    const risk = typeof guard?.data.risk_score === "number" ? guard.data.risk_score : null;
    return (
      <Banner tone="danger" icon={ShieldAlert} title="Bloqueada por el guard de entrada">
        {res.block_reason ?? "La consulta infringe la política de seguridad."}
        {(flags.length > 0 || risk !== null) && (
          <div className="mt-3 flex flex-wrap items-center gap-1.5">
            {risk !== null && (
              <Badge tone="danger" mono>
                riesgo {risk.toFixed(2)}
              </Badge>
            )}
            {flags.map((f) => (
              <Badge key={f} tone="neutral" mono>
                {f}
              </Badge>
            ))}
          </div>
        )}
        <div className="mt-3 text-2xs text-faint">No se consultó el índice ni el LLM. El intento quedó en la auditoría.</div>
      </Banner>
    );
  }
  if (outcome === "denied") {
    const perm = findEvent(res.events, "permission_check");
    const required = typeof perm?.data.required_role === "string" ? perm.data.required_role : null;
    const filtered =
      typeof perm?.data.filtered_out === "number"
        ? perm.data.filtered_out
        : typeof perm?.data.dropped === "number"
          ? perm.data.dropped
          : null;
    // En la rama de herramientas la denegación viene de la herramienta (p. ej. un ticket ajeno):
    // la respuesta del agente ya explica el motivo sin revelar si el recurso existe.
    const toolDenial = typeof perm?.data.tool === "string";
    return (
      <Banner tone="danger" icon={Lock} title={toolDenial ? "Fuera de tu alcance de acceso" : "Fuera del alcance de tu rol"}>
        {toolDenial
          ? sentence(res.response.answer)
          : sentence(typeof perm?.data.reason === "string" ? perm.data.reason : "No tienes permiso para ver la evidencia necesaria")}
        {required ? ` Requiere rol ${ROLE_SHORT[required as keyof typeof ROLE_SHORT] ?? required} o superior.` : ""}
        {toolDenial ? "" : " No se muestra ningún contenido restringido."}
        <div className="mt-3 flex flex-wrap items-center gap-2">
          {filtered !== null && filtered > 0 && (
            <Badge tone="neutral" mono>
              {filtered} fragmentos filtrados por RBAC
            </Badge>
          )}
          <Button size="sm" variant="secondary" onClick={onEscalate}>
            Solicitar escalamiento
          </Button>
        </div>
      </Banner>
    );
  }
  if (outcome === "abstained") {
    const gate = gateInfo(res.events);
    const ood = res.intent === "out_of_domain";
    return (
      <Banner tone="warning" icon={CircleSlash} title={ood ? "Fuera del dominio de soporte ERP" : "Sin evidencia suficiente"}>
        {res.response.answer}
        <div className="mt-3 flex flex-wrap items-center gap-2">
          {gate && (
            <Badge tone="warning" mono>
              máx {formatScore(gate.top, 2)} &lt; τ {formatScore(gate.tau, 2)}
            </Badge>
          )}
          {!ood && (
            <Button size="sm" variant="secondary" onClick={onEscalate}>
              Escalar a N2
            </Button>
          )}
        </div>
      </Banner>
    );
  }
  return null;
}

export function TurnView({
  turn,
  focused,
  onFocus,
  onOpenCitation,
  onInspect,
  onCompare,
  onEscalate,
  onReviewDraft,
}: {
  turn: Turn;
  focused: boolean;
  onFocus: () => void;
  onOpenCitation: (index: number) => void;
  onInspect: (traceId: string) => void;
  onCompare: () => void;
  onEscalate: () => void;
  onReviewDraft: () => void;
}) {
  const res = turn.response;
  const { replayConfirm } = useSession();
  const outcome = res ? outcomeOf(res) : null;
  // Tras la decisión humana, la insignia refleja el resultado final de la acción.
  const decided =
    outcome === "pending_confirmation" && turn.confirm.status === "done"
      ? { label: "Ticket creado", tone: "accent" as Tone }
      : outcome === "pending_confirmation" && turn.confirm.status === "cancelled"
        ? { label: "Acción cancelada", tone: "neutral" as Tone }
        : null;
  const tone: Tone = decided?.tone ?? (outcome ? OUTCOME_TONE[outcome] : "neutral");
  const [copied, copy] = useCopy();
  const running = useMemo(
    () => buildPipeline(turn.events, false).find((s) => s.status === "running")?.def.running ?? "Iniciando…",
    [turn.events],
  );
  const rawTicket = res ? ticketFromEvent(findEvent(res.events, "tool.get_ticket_status")) : null;
  // La traza omite el resumen del ticket (texto libre); la plantilla del agente lo trae al final.
  const ticket =
    rawTicket && !rawTicket.summary && res
      ? { ...rawTicket, summary: /Resumen: (.+)$/.exec(res.response.answer.trim())?.[1] ?? "" }
      : rawTicket;
  const classification = res ? classificationFromEvent(findEvent(res.events, "tool.classify_ticket")) : null;
  const OutcomeIcon = outcome ? OUTCOME_ICON[outcome] : FileText;
  const showAnswer = res && outcome !== "blocked" && outcome !== "abstained" && outcome !== "denied";
  const canCompare = res?.intent === "doc_query" && outcome === "answered" && !turn.compareOf;

  return (
    <motion.article
      initial={{ opacity: 0, transform: "translateY(8px)" }}
      animate={{ opacity: 1, transform: "translateY(0px)" }}
      transition={{ duration: 0.24, ease: [0.23, 1, 0.32, 1] }}
      className="scroll-mt-24"
      aria-busy={turn.status === "streaming"}
    >
      {/* Consulta */}
      <div className="mb-3 flex items-start gap-3 px-1">
        <div className="min-w-0 flex-1">
          <div className="mb-1 flex flex-wrap items-center gap-1.5 text-2xs text-faint">
            {turn.compareOf && (
              <Badge tone="info" icon={GitCompareArrows}>
                Comparación
              </Badge>
            )}
            <span className="font-mono">{turn.request.user_id}</span>
            <span aria-hidden>·</span>
            <span>{ROLE_SHORT[turn.request.role]}</span>
            <span aria-hidden>·</span>
            <span className="font-mono">Odoo {turn.request.erp_version}</span>
          </div>
          <p className="text-[15px] leading-relaxed text-text">{turn.request.query}</p>
        </div>
      </div>

      {/* Respuesta */}
      <div
        onClick={onFocus}
        className={cx(
          "bezel transition-[box-shadow] duration-200",
          focused && turn.status !== "streaming" && "ring-white/[0.1]",
        )}
      >
        <div className="bezel-core overflow-hidden">
          <div className="flex items-center gap-2.5 border-b hairline px-5 py-3">
            {turn.status === "streaming" ? (
              <Badge tone="accent">
                <span className="live-dot h-1.5 w-1.5 rounded-full bg-accent-2" aria-hidden />
                {running}
              </Badge>
            ) : turn.status === "error" ? (
              <Badge tone="danger">Error</Badge>
            ) : (
              outcome && (
                <Badge tone={tone} icon={OutcomeIcon}>
                  {decided?.label ?? OUTCOME_LABEL[outcome]}
                </Badge>
              )
            )}
            {res && !res.blocked && <span className="hidden text-xs text-faint sm:inline">{INTENT_LABEL[res.intent]}</span>}
            {res?.response.tool_used && (
              <Badge tone="info" icon={Wrench} mono>
                {res.response.tool_used}
              </Badge>
            )}
            {res?.response.needs_human_review && outcome !== "pending_confirmation" && outcome !== "review" && (
              <Badge tone="violet" icon={UserCheck}>
                Revisión humana
              </Badge>
            )}
            <span className="ml-auto font-mono text-2xs text-faint">{res ? latencySummary(res) : ""}</span>
          </div>

          <div className="space-y-5 px-5 py-5">
            {turn.status === "streaming" && (
              <div className="space-y-2.5" aria-label="Generando respuesta">
                <Skeleton className="h-3.5 w-[92%]" />
                <Skeleton className="h-3.5 w-[78%]" />
                <Skeleton className="h-3.5 w-[85%]" />
                <Skeleton className="h-3.5 w-[40%]" />
              </div>
            )}
            {turn.status === "error" && (
              <Banner tone="danger" icon={ShieldAlert} title="No se pudo completar la consulta">
                {turn.error}
              </Banner>
            )}
            {res && outcome && (
              <>
                <StatusBanner res={res} outcome={outcome} onEscalate={onEscalate} />
                {showAnswer && (
                  <AnswerBody text={res.response.answer} sources={res.response.sources} onOpenCitation={onOpenCitation} />
                )}
                {ticket && <TicketStatusCard ticket={ticket} />}
                {classification && <ClassificationCard result={classification} />}
                {res.pending_action && (
                  <PendingDraftCard
                    pending={res.pending_action}
                    confirm={turn.confirm}
                    onReview={onReviewDraft}
                    onReplay={() => void replayConfirm(turn.id)}
                  />
                )}
                <CitationChips sources={res.response.sources} onOpen={onOpenCitation} />
              </>
            )}
          </div>

          {res && (
            <div className="flex flex-wrap items-center gap-2 border-t hairline px-4 py-2.5">
              <span className="pl-1 font-mono text-2xs text-faint">{res.trace_id}</span>
              <div className="ml-auto flex items-center gap-1">
                {canCompare && (
                  <Button size="sm" variant="ghost" icon={GitCompareArrows} onClick={onCompare}>
                    Comparar con Odoo {turn.request.erp_version === "18.0" ? "17.0" : "18.0"}
                  </Button>
                )}
                <IconButton icon={Copy} label={copied ? "Copiado" : "Copiar JSON contractual (§11)"} onClick={() => res && copy(JSON.stringify(res.response, null, 2))} />
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => onInspect(res.trace_id)}
                  trailing={<ArrowUpRight size={13} strokeWidth={1.8} aria-hidden />}
                >
                  Inspeccionar traza
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </motion.article>
  );
}
