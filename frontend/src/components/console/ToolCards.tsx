/**
 * Tarjetas de resultado de herramientas: estado de ticket con SLA, clasificación con
 * confianza y borrador pendiente de confirmación humana (HITL).
 */
import {
  AlertTriangle,
  CheckCircle2,
  Clock3,
  FilePenLine,
  KeyRound,
  Loader2,
  PauseCircle,
  RotateCcw,
  ShieldCheck,
  Tags,
  Ticket,
  XCircle,
} from "lucide-react";
import type { ReactNode } from "react";
import type { PendingAction, QueryResponse } from "../../api/types";
import { useNow } from "../../hooks/useNow";
import { cx } from "../../lib/cx";
import {
  CATEGORY_LABEL,
  MODULE_LABEL,
  PRIORITY_LABEL,
  formatCountdown,
  formatDateTime,
  formatRemaining,
  labelOf,
} from "../../lib/format";
import type { ClassificationView, TicketView } from "../../lib/toolData";
import type { ConfirmState, ReplayState } from "../../state/session";
import { Badge, Button, Meter } from "../ui";
import { AnswerBody } from "./AnswerBody";

const STATUS_FLOW = ["nuevo", "en_progreso", "en_espera", "resuelto", "cerrado"] as const;
const STATUS_TEXT: Record<string, string> = {
  nuevo: "Nuevo",
  en_progreso: "En progreso",
  en_espera: "En espera",
  resuelto: "Resuelto",
  cerrado: "Cerrado",
};

function Field({ label, value, mono = false }: { label: string; value: string | null; mono?: boolean }) {
  return (
    <div className="min-w-0">
      <div className="text-2xs text-faint">{label}</div>
      <div className={cx("truncate text-sm text-text", mono && "font-mono text-xs")}>{value ?? "—"}</div>
    </div>
  );
}

function ToolFrame({
  icon: Icon,
  title,
  meta,
  children,
}: {
  icon: typeof Ticket;
  title: string;
  meta?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="overflow-hidden rounded-[12px] bg-surface-2/70 ring-1 ring-inset ring-white/[0.06]">
      <div className="flex items-center gap-2 border-b hairline px-4 py-2.5">
        <Icon size={14} strokeWidth={1.6} className="text-info" aria-hidden />
        <span className="font-mono text-xs text-muted">{title}</span>
        <div className="ml-auto flex items-center gap-2">{meta}</div>
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
}

/* ------------------------------------------------------------------------------------ */

export function TicketStatusCard({ ticket }: { ticket: TicketView }) {
  const now = useNow(30_000);
  const currentIndex = STATUS_FLOW.indexOf(ticket.status as (typeof STATUS_FLOW)[number]);
  const due = ticket.sla?.resolution_due_at ? new Date(ticket.sla.resolution_due_at).getTime() : null;
  const remaining = due !== null ? due - now : null;
  const breached = ticket.sla?.breached || (remaining !== null && remaining < 0);
  const slaTone = breached ? "danger" : ticket.sla?.paused ? "neutral" : remaining !== null && remaining < 4 * 3_600_000 ? "warning" : "accent";
  return (
    <ToolFrame
      icon={Ticket}
      title="get_ticket_status · solo lectura"
      meta={ticket.priority && <Badge tone={ticket.priority === "critica" || ticket.priority === "alta" ? "warning" : "neutral"}>Prioridad {labelOf(PRIORITY_LABEL, ticket.priority).toLowerCase()}</Badge>}
    >
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="font-mono text-sm text-accent-2">{ticket.ticket_id}</span>
        <span className="text-sm text-text">{ticket.summary}</span>
      </div>

      <ol className="mt-4 grid grid-cols-5 gap-1.5" aria-label="Flujo de estados del ticket">
        {STATUS_FLOW.map((s, i) => {
          const reached = currentIndex >= 0 && i <= currentIndex;
          const current = i === currentIndex;
          return (
            <li key={s} className="min-w-0" aria-current={current ? "step" : undefined}>
              <div className={cx("h-1 rounded-full", reached ? (current ? "bg-info" : "bg-info/40") : "bg-white/[0.06]")} />
              <div className={cx("mt-1.5 truncate text-2xs", current ? "text-text" : "text-faint")}>{STATUS_TEXT[s]}</div>
            </li>
          );
        })}
      </ol>

      <div className="mt-4 grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-4">
        <Field label="Cuenta" value={ticket.account} />
        <Field label="Asignado a" value={ticket.assigned_to} />
        <Field label="Categoría" value={ticket.category ? labelOf(CATEGORY_LABEL, ticket.category) : null} />
        <Field label="Actualizado" value={ticket.updated_at ? formatDateTime(ticket.updated_at) : null} />
      </div>

      {ticket.sla && (
        <div className="mt-4 rounded-[10px] bg-black/20 p-3 ring-1 ring-inset ring-white/[0.05]">
          <div className="flex items-center gap-2 text-xs">
            {ticket.sla.paused ? (
              <PauseCircle size={14} strokeWidth={1.6} className="text-muted" aria-hidden />
            ) : breached ? (
              <AlertTriangle size={14} strokeWidth={1.6} className="text-danger" aria-hidden />
            ) : (
              <Clock3 size={14} strokeWidth={1.6} className="text-accent-2" aria-hidden />
            )}
            <span className="text-text">
              {ticket.sla.paused
                ? "SLA pausado · en espera del cliente"
                : remaining !== null
                  ? remaining < 0
                    ? `SLA ${formatRemaining(remaining)}`
                    : `Resolución en ${formatRemaining(remaining)}`
                  : "SLA sin vencimiento registrado"}
            </span>
            {ticket.sla.policy && <span className="ml-auto truncate text-2xs text-faint">{ticket.sla.policy}</span>}
          </div>
          {ticket.sla.elapsed_ratio !== null && (
            <Meter
              className="mt-2.5"
              value={ticket.sla.elapsed_ratio}
              tone={slaTone}
              label="Proporción del SLA consumida"
            />
          )}
          {ticket.sla.resolution_due_at && (
            <div className="mt-1.5 text-2xs text-faint">Vence {formatDateTime(ticket.sla.resolution_due_at)}</div>
          )}
        </div>
      )}
    </ToolFrame>
  );
}

/* ------------------------------------------------------------------------------------ */

export function ClassificationCard({ result }: { result: ClassificationView }) {
  const low = result.threshold !== null && result.confidence < result.threshold;
  return (
    <ToolFrame
      icon={Tags}
      title="classify_ticket"
      meta={result.model && <span className="hidden text-2xs text-faint sm:inline">{result.model}</span>}
    >
      <div className="grid gap-5 sm:grid-cols-[1fr_1.1fr]">
        <div>
          <div className="text-2xs text-faint">Categoría sugerida</div>
          <div className="mt-0.5 text-lg font-medium text-text">{labelOf(CATEGORY_LABEL, result.category)}</div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {result.module && <Badge>{labelOf(MODULE_LABEL, result.module)}</Badge>}
            {result.priority && <Badge>Prioridad {labelOf(PRIORITY_LABEL, result.priority).toLowerCase()}</Badge>}
          </div>
          <div className="mt-4">
            <div className="mb-1.5 flex items-baseline justify-between text-2xs">
              <span className="text-faint">Confianza</span>
              <span className={cx("font-mono text-xs", low ? "text-violet" : "text-accent-2")}>
                {result.confidence.toFixed(2)}
                {result.threshold !== null && <span className="text-faint"> / umbral {result.threshold.toFixed(2)}</span>}
              </span>
            </div>
            <Meter
              value={result.confidence}
              tone={low ? "violet" : "accent"}
              marker={result.threshold ?? undefined}
              label="Confianza del clasificador"
            />
            {low && <p className="mt-2 text-2xs text-violet">Bajo el umbral: se marca para revisión humana.</p>}
          </div>
        </div>
        {result.probabilities.length > 0 && (
          <div>
            <div className="mb-2 text-2xs text-faint">Distribución (top {result.probabilities.length})</div>
            <ul className="space-y-1.5">
              {result.probabilities.map((p, i) => (
                <li key={p.label} className="grid grid-cols-[112px_1fr_40px] items-center gap-2.5">
                  <span className={cx("truncate text-xs", i === 0 ? "text-text" : "text-muted")}>
                    {labelOf(CATEGORY_LABEL, p.label)}
                  </span>
                  <div className="h-1.5 rounded-full bg-white/[0.05]">
                    <div
                      className={cx("h-full rounded-full", i === 0 ? (low ? "bg-violet" : "bg-accent") : "bg-white/20")}
                      style={{ width: `${Math.max(2, p.p * 100)}%` }}
                    />
                  </div>
                  <span className="text-right font-mono text-2xs text-muted">{p.p.toFixed(2)}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </ToolFrame>
  );
}

/* ------------------------------------------------------------------------------------ */

export function PendingDraftCard({
  pending,
  confirm,
  onReview,
  onReplay,
}: {
  pending: PendingAction;
  confirm: ConfirmState;
  onReview: () => void;
  /** Reenvía el token ya consumido para comprobar que es de un solo uso. */
  onReplay?: () => void;
}) {
  const now = useNow(1000, confirm.status === "idle" || confirm.status === "error");
  const remaining = new Date(pending.expires_at).getTime() - now;
  const expired = remaining <= 0;
  const d = pending.draft;

  if (confirm.status === "done") {
    return <CreatedTicketCard response={confirm.response} replay={confirm.replay} onReplay={onReplay} />;
  }

  return (
    <ToolFrame
      icon={FilePenLine}
      title="create_ticket · borrador"
      meta={
        confirm.status === "cancelled" ? (
          <Badge tone="neutral" icon={XCircle}>
            Cancelado
          </Badge>
        ) : expired ? (
          <Badge tone="danger">Expirado</Badge>
        ) : (
          <Badge tone="violet" icon={KeyRound} mono>
            {formatCountdown(remaining)}
          </Badge>
        )
      }
    >
      <div className="text-sm text-text">{d.summary}</div>
      {d.description && <p className="mt-1 line-clamp-2 text-xs text-muted">{d.description}</p>}
      <div className="mt-3 flex flex-wrap gap-1.5">
        <Badge>{labelOf(CATEGORY_LABEL, d.category)}</Badge>
        <Badge>Prioridad {labelOf(PRIORITY_LABEL, d.priority).toLowerCase()}</Badge>
        <Badge>{labelOf(MODULE_LABEL, d.module)}</Badge>
        {d.account_id && <Badge mono>{d.account_id}</Badge>}
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        {confirm.status === "cancelled" ? (
          <p className="text-xs text-muted">No se escribió nada en la base de tickets.</p>
        ) : (
          <>
            <Button
              variant="primary"
              size="sm"
              icon={confirm.status === "confirming" ? Loader2 : CheckCircle2}
              disabled={expired || confirm.status === "confirming"}
              onClick={onReview}
              className={confirm.status === "confirming" ? "[&_svg]:animate-spin" : undefined}
            >
              {confirm.status === "confirming" ? "Confirmando…" : "Revisar y confirmar"}
            </Button>
            <span className="text-2xs text-faint">Token HMAC de un solo uso · nada se escribe sin tu confirmación</span>
          </>
        )}
      </div>
      {confirm.status === "error" && <p className="mt-3 text-xs text-danger">{confirm.message}</p>}
    </ToolFrame>
  );
}

function CreatedTicketCard({
  response,
  replay,
  onReplay,
}: {
  response: QueryResponse;
  replay?: ReplayState;
  onReplay?: () => void;
}) {
  const ok = !response.events.some((e) => e.stage === "tool.create_ticket" && e.status !== "ok");
  return (
    <ToolFrame
      icon={ok ? CheckCircle2 : XCircle}
      title={ok ? "create_ticket · confirmado" : "create_ticket · rechazado"}
      meta={<Badge tone={ok ? "accent" : "danger"}>{ok ? "Escrito en la BD" : "Rechazado"}</Badge>}
    >
      <AnswerBody text={response.response.answer} sources={[]} onOpenCitation={() => undefined} />
      {ok && onReplay && (
        <div className="mt-4 border-t hairline pt-3">
          {!replay || replay.status === "sending" ? (
            <div className="flex flex-wrap items-center gap-3">
              <Button
                size="sm"
                variant="secondary"
                icon={replay?.status === "sending" ? Loader2 : RotateCcw}
                className={replay?.status === "sending" ? "[&_svg]:animate-spin" : undefined}
                disabled={replay?.status === "sending"}
                onClick={onReplay}
              >
                Reenviar el mismo token
              </Button>
              <span className="text-2xs text-faint">Prueba de un solo uso: el servidor debe rechazarlo.</span>
            </div>
          ) : replay.status === "rejected" ? (
            <div className="flex items-start gap-2 text-xs">
              <ShieldCheck size={14} strokeWidth={1.6} className="mt-px shrink-0 text-accent-2" aria-hidden />
              <span className="text-muted">
                <span className="text-text">Reutilización rechazada (409).</span> {replay.message}
              </span>
            </div>
          ) : (
            <p className="text-xs text-danger">El servidor aceptó un token ya usado. Revisa la configuración HITL.</p>
          )}
        </div>
      )}
    </ToolFrame>
  );
}
