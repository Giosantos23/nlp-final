/**
 * Modal de confirmación humana (HITL) para `create_ticket` (SPEC §5.4).
 *
 * El servidor firma el borrador con un token HMAC de un solo uso que expira en 10 minutos.
 * Si el analista edita el borrador, el token deja de corresponder al contenido firmado:
 * por eso editar no "parcha" el borrador, sino que genera uno nuevo para firmar otra vez.
 * Así nunca se escribe en la base algo distinto de lo que el servidor validó.
 */
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown, KeyRound, Loader2, RefreshCw, ShieldCheck, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { ErpModule, PendingAction, TicketCategory, TicketDraft, TicketPriority } from "../../api/types";
import { useNow } from "../../hooks/useNow";
import { cx } from "../../lib/cx";
import { CATEGORY_LABEL, MODULE_LABEL, PRIORITY_LABEL, formatCountdown } from "../../lib/format";
import type { ConfirmState } from "../../state/session";
import { Button, IconButton } from "../ui";

const CATEGORIES = Object.keys(CATEGORY_LABEL) as TicketCategory[];
const PRIORITIES = Object.keys(PRIORITY_LABEL) as TicketPriority[];
const MODULES = Object.keys(MODULE_LABEL) as ErpModule[];
const TTL_MS = 10 * 60_000;

function CountdownRing({ remaining }: { remaining: number }) {
  const r = 17;
  const circumference = 2 * Math.PI * r;
  const ratio = Math.max(0, Math.min(1, remaining / TTL_MS));
  const color = remaining <= 0 ? "#F87171" : remaining < 60_000 ? "#F5B544" : "#A78BFA";
  return (
    <div className="relative h-11 w-11" role="timer" aria-label={`El token expira en ${formatCountdown(remaining)}`}>
      <svg viewBox="0 0 40 40" className="h-11 w-11 -rotate-90">
        <circle cx="20" cy="20" r={r} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="2.5" />
        <circle
          cx="20"
          cy="20"
          r={r}
          fill="none"
          stroke={color}
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - ratio)}
          style={{ transition: "stroke-dashoffset 1s linear, stroke 200ms ease" }}
        />
      </svg>
      <KeyRound size={14} strokeWidth={1.7} className="absolute inset-0 m-auto" style={{ color }} aria-hidden />
    </div>
  );
}

const inputCls =
  "w-full rounded-[10px] bg-black/25 px-3 py-2 text-sm text-text ring-1 ring-inset ring-white/[0.08] placeholder:text-faint transition-shadow duration-150 focus:outline-none focus:ring-accent/50";

/** Select nativo (accesible) con flecha propia para no depender del estilo del sistema. */
function Select({ value, onChange, children }: { value: string; onChange: (v: string) => void; children: ReactNode }) {
  return (
    <span className="relative block">
      <select value={value} onChange={(e) => onChange(e.target.value)} className={cx(inputCls, "appearance-none pr-8")}>
        {children}
      </select>
      <ChevronDown
        size={14}
        strokeWidth={1.6}
        className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-faint"
        aria-hidden
      />
    </span>
  );
}

function Labeled({ label, hint, error, children }: { label: string; hint?: string; error?: string | null; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 flex items-baseline justify-between text-2xs">
        <span className="text-muted">{label}</span>
        {error ? <span className="text-danger">{error}</span> : hint && <span className="font-mono text-faint">{hint}</span>}
      </span>
      {children}
    </label>
  );
}

export function HitlModal({
  open,
  pending,
  confirm,
  onClose,
  onConfirm,
  onCancelAction,
  onRegenerate,
}: {
  open: boolean;
  pending: PendingAction | null;
  confirm: ConfirmState;
  onClose: () => void;
  onConfirm: () => void;
  onCancelAction: () => void;
  onRegenerate: (draft: TicketDraft) => void;
}) {
  const [draft, setDraft] = useState<TicketDraft | null>(pending?.draft ?? null);
  const first = useRef<HTMLInputElement>(null);
  const now = useNow(1000, open);

  useEffect(() => {
    if (open && pending) {
      setDraft(pending.draft);
      requestAnimationFrame(() => first.current?.focus());
    }
  }, [open, pending]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const edited = useMemo(
    () => !!pending && !!draft && JSON.stringify(draft) !== JSON.stringify(pending.draft),
    [draft, pending],
  );

  const remaining = pending ? new Date(pending.expires_at).getTime() - now : 0;
  const expired = remaining <= 0;
  const summaryError =
    draft && (draft.summary.trim().length < 10 || draft.summary.trim().length > 200) ? "10 a 200 caracteres" : null;
  const descError = draft && draft.description.length > 2000 ? "máx. 2000" : null;
  const confirming = confirm.status === "confirming";

  const set = <K extends keyof TicketDraft>(key: K, value: TicketDraft[K]) =>
    setDraft((d) => (d ? { ...d, [key]: value } : d));

  return (
    <AnimatePresence>
      {open && pending && draft && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <motion.div
            className="absolute inset-0 bg-black/65 backdrop-blur-[2px]"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, transition: { duration: 0.15 } }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
            aria-hidden
          />
          <motion.div
            role="dialog"
            aria-modal="true"
            aria-labelledby="hitl-title"
            className="relative w-full max-w-[560px]"
            initial={{ opacity: 0, transform: "scale(0.96) translateY(6px)" }}
            animate={{ opacity: 1, transform: "scale(1) translateY(0px)" }}
            exit={{ opacity: 0, transform: "scale(0.98)", transition: { duration: 0.14 } }}
            transition={{ duration: 0.22, ease: [0.23, 1, 0.32, 1] }}
          >
            <div className="bezel">
              <div className="bezel-core overflow-hidden bg-surface-2">
                <div className="flex items-start gap-4 border-b hairline px-5 py-4">
                  <CountdownRing remaining={remaining} />
                  <div className="min-w-0 flex-1">
                    <div className="eyebrow text-violet/80">Acción crítica · confirmación humana</div>
                    <h2 id="hitl-title" className="mt-1 text-md font-medium text-text">
                      Confirmar creación de ticket
                    </h2>
                    <p className="mt-0.5 text-xs text-muted">
                      {expired ? (
                        <span className="text-danger">El token expiró. Genera un borrador nuevo.</span>
                      ) : (
                        <>
                          El token expira en <span className="font-mono text-text">{formatCountdown(remaining)}</span>. Nada se
                          escribe hasta que confirmes.
                        </>
                      )}
                    </p>
                  </div>
                  <IconButton icon={X} label="Cerrar (el borrador sigue pendiente)" onClick={onClose} />
                </div>

                <div className="space-y-4 px-5 py-5">
                  <Labeled label="Resumen" hint={`${draft.summary.length}/200`} error={summaryError}>
                    <input
                      ref={first}
                      value={draft.summary}
                      onChange={(e) => set("summary", e.target.value)}
                      className={inputCls}
                      maxLength={220}
                    />
                  </Labeled>
                  <Labeled label="Descripción" hint={`${draft.description.length}/2000`} error={descError}>
                    <textarea
                      value={draft.description}
                      onChange={(e) => set("description", e.target.value)}
                      rows={3}
                      className={cx(inputCls, "resize-none leading-relaxed")}
                    />
                  </Labeled>
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                    <Labeled label="Categoría">
                      <Select value={draft.category} onChange={(v) => set("category", v as TicketCategory)}>
                        {CATEGORIES.map((c) => (
                          <option key={c} value={c}>
                            {CATEGORY_LABEL[c]}
                          </option>
                        ))}
                      </Select>
                    </Labeled>
                    <Labeled label="Prioridad">
                      <Select value={draft.priority} onChange={(v) => set("priority", v as TicketPriority)}>
                        {PRIORITIES.map((p) => (
                          <option key={p} value={p}>
                            {PRIORITY_LABEL[p]}
                          </option>
                        ))}
                      </Select>
                    </Labeled>
                    <Labeled label="Módulo">
                      <Select value={draft.module} onChange={(v) => set("module", v as ErpModule)}>
                        {MODULES.map((m) => (
                          <option key={m} value={m}>
                            {MODULE_LABEL[m]}
                          </option>
                        ))}
                      </Select>
                    </Labeled>
                  </div>

                  <div className="flex items-start gap-2.5 rounded-[10px] bg-white/[0.025] px-3 py-2.5 text-2xs leading-4 text-muted ring-1 ring-inset ring-white/[0.06]">
                    <ShieldCheck size={14} strokeWidth={1.6} className="mt-px shrink-0 text-accent-2" aria-hidden />
                    <span>
                      {edited
                        ? "Editaste el borrador: el token firmado ya no corresponde. Se generará un borrador nuevo para que el servidor lo valide y firme otra vez."
                        : "Token HMAC de un solo uso, ligado a tu sesión y rol. Clave de idempotencia: confirmar dos veces no duplica el ticket."}
                    </span>
                  </div>
                  {confirm.status === "error" && <p className="text-xs text-danger">{confirm.message}</p>}
                </div>

                <div className="flex flex-wrap items-center gap-2 border-t hairline bg-black/10 px-5 py-3.5">
                  <Button variant="ghost" onClick={onCancelAction} disabled={confirming}>
                    Cancelar acción
                  </Button>
                  <div className="ml-auto flex gap-2">
                    {edited ? (
                      <Button
                        variant="primary"
                        icon={RefreshCw}
                        disabled={!!summaryError || !!descError}
                        onClick={() => onRegenerate(draft)}
                      >
                        Regenerar borrador
                      </Button>
                    ) : expired ? (
                      <Button variant="primary" icon={RefreshCw} onClick={() => onRegenerate(draft)}>
                        Generar borrador nuevo
                      </Button>
                    ) : (
                      <Button
                        variant="primary"
                        icon={confirming ? Loader2 : undefined}
                        className={confirming ? "[&_svg]:animate-spin" : undefined}
                        disabled={confirming}
                        onClick={onConfirm}
                      >
                        {confirming ? "Creando…" : "Confirmar y crear ticket"}
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
