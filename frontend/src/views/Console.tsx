/**
 * Vista Consola: barra de consulta, hilo de turnos con respuestas citadas, carril de
 * pipeline en vivo, panel de citas con el PDF y modal de confirmación humana.
 */
import { motion } from "framer-motion";
import {
  ArrowRight,
  BookOpenCheck,
  CircleSlash,
  FilePlus2,
  GitCompareArrows,
  Lock,
  ShieldAlert,
  Tags,
  Ticket,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { Composer, type ComposerHandle } from "../components/console/Composer";
import { CitationDrawer } from "../components/console/CitationDrawer";
import { HitlModal } from "../components/console/HitlModal";
import { PipelineRail } from "../components/console/PipelineRail";
import { TurnView } from "../components/console/TurnView";
import { Button } from "../components/ui";
import { buildDraftQuery, buildEscalationQuery } from "../lib/drafts";
import { EXAMPLE_QUERIES, type ExampleKind } from "../lib/examples";
import { useSession } from "../state/session";

const EXAMPLE_ICON: Record<ExampleKind, typeof Ticket> = {
  nominal: BookOpenCheck,
  version_compare: GitCompareArrows,
  ticket_status: Ticket,
  ticket_classify: Tags,
  ticket_create: FilePlus2,
  restricted: Lock,
  abstention: CircleSlash,
  injection: ShieldAlert,
};

function EmptyHero({ composer, onPick }: { composer: ReactNode; onPick: (q: string) => void }) {
  return (
    <div className="mx-auto w-full max-w-[780px] px-4 pb-16 pt-[9vh] md:px-6">
      <motion.div
        initial={{ opacity: 0, transform: "translateY(10px)" }}
        animate={{ opacity: 1, transform: "translateY(0px)" }}
        transition={{ duration: 0.4, ease: [0.23, 1, 0.32, 1] }}
      >
        <div className="mb-5 inline-flex items-center gap-2 rounded-full bg-white/[0.03] py-1 pl-1 pr-3 ring-1 ring-inset ring-white/[0.07]">
          <span className="rounded-full bg-accent/15 px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.14em] text-accent-2">
            RAG agéntico
          </span>
          <span className="text-2xs text-muted">Odoo 17.0 · 18.0 · runbooks · tickets</span>
        </div>
        <h2 className="max-w-[640px] text-[34px] font-semibold leading-[1.08] tracking-[-0.035em] text-text md:text-[42px]">
          Respuestas de soporte con{" "}
          <span className="bg-gradient-to-r from-accent-2 to-accent bg-clip-text text-transparent">evidencia verificable</span>.
        </h2>
        <p className="mt-4 max-w-[560px] text-[15px] leading-relaxed text-muted">
          Cada respuesta cita fuente, página y fragmento. Si no hay evidencia o tu rol no alcanza, DocuERP se abstiene; si la
          acción es crítica, te pide confirmación.
        </p>
      </motion.div>

      <motion.div
        className="mt-8"
        initial={{ opacity: 0, transform: "translateY(10px)" }}
        animate={{ opacity: 1, transform: "translateY(0px)" }}
        transition={{ duration: 0.4, delay: 0.06, ease: [0.23, 1, 0.32, 1] }}
      >
        {composer}
      </motion.div>

      <div className="mt-10">
        <div className="eyebrow mb-3">Escenarios de ejemplo</div>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {EXAMPLE_QUERIES.map((ex, i) => {
            const Icon = EXAMPLE_ICON[ex.id];
            return (
              <motion.button
                key={ex.id}
                type="button"
                onClick={() => onPick(ex.query)}
                initial={{ opacity: 0, transform: "translateY(6px)" }}
                animate={{ opacity: 1, transform: "translateY(0px)" }}
                transition={{ duration: 0.3, delay: 0.12 + i * 0.035, ease: [0.23, 1, 0.32, 1] }}
                className="group flex items-start gap-3 rounded-[12px] bg-white/[0.02] p-3 text-left ring-1 ring-inset ring-white/[0.06] transition-[background-color,box-shadow] duration-150 hover:bg-white/[0.04] hover:ring-white/[0.1] active:scale-[0.99]"
              >
                <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-[8px] bg-white/[0.04] text-muted ring-1 ring-inset ring-white/[0.07] transition-colors duration-150 group-hover:text-accent-2">
                  <Icon size={14} strokeWidth={1.6} aria-hidden />
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block text-[13px] font-medium text-text">{ex.label}</span>
                  <span className="mt-0.5 line-clamp-2 block text-xs leading-[18px] text-muted">{ex.query}</span>
                  <span className="mt-1.5 block font-mono text-[10.5px] text-faint">{ex.hint}</span>
                </span>
                <ArrowRight
                  size={14}
                  strokeWidth={1.6}
                  className="mt-1 shrink-0 text-faint opacity-0 transition-[opacity,transform] duration-150 group-hover:translate-x-0.5 group-hover:opacity-100"
                  aria-hidden
                />
              </motion.button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export function ConsoleView({ onInspect }: { onInspect: (traceId: string) => void }) {
  const session = useSession();
  const { turns, ask, focusedTurn, focus, confirmAction, cancelAction, clear } = session;
  const composer = useRef<ComposerHandle>(null);
  const [citation, setCitation] = useState<{ turnId: string; index: number } | null>(null);
  const [hitlTurnId, setHitlTurnId] = useState<string | null>(null);
  const autoOpened = useRef(new Set<string>());
  const bottom = useRef<HTMLDivElement>(null);

  // "/" enfoca la barra de consulta (si no se está escribiendo en otro campo).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
      if (e.key === "/" && !typing && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        composer.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Al llegar un borrador pendiente se abre el modal HITL una sola vez por turno.
  useEffect(() => {
    for (const t of turns) {
      if (t.status === "done" && t.response?.pending_action && t.confirm.status === "idle" && !autoOpened.current.has(t.id)) {
        autoOpened.current.add(t.id);
        setHitlTurnId(t.id);
      }
    }
  }, [turns]);

  const lastTurnId = turns[turns.length - 1]?.id;
  useEffect(() => {
    if (lastTurnId) bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [lastTurnId]);

  const submit = useCallback((q: string) => ask(q), [ask]);
  const citationTurn = citation ? turns.find((t) => t.id === citation.turnId) : null;
  const hitlTurn = hitlTurnId ? turns.find((t) => t.id === hitlTurnId) ?? null : null;

  const composerEl = <Composer ref={composer} onSubmit={submit} large={turns.length === 0} />;

  return (
    <div className="flex min-h-[calc(100dvh-56px)]">
      <section className="flex min-w-0 flex-1 flex-col">
        {turns.length === 0 ? (
          <EmptyHero composer={composerEl} onPick={submit} />
        ) : (
          <>
            <div className="mx-auto w-full max-w-[780px] flex-1 space-y-10 px-4 pb-8 pt-8 md:px-6">
              <div className="flex items-center justify-between">
                <div className="eyebrow">Sesión · {turns.length} {turns.length === 1 ? "consulta" : "consultas"}</div>
                <Button size="sm" variant="ghost" onClick={clear} disabled={session.busy}>
                  Nueva sesión
                </Button>
              </div>
              {turns.map((turn) => (
                <TurnView
                  key={turn.id}
                  turn={turn}
                  focused={focusedTurn?.id === turn.id}
                  onFocus={() => focus(turn.id)}
                  onOpenCitation={(index) => setCitation({ turnId: turn.id, index })}
                  onInspect={onInspect}
                  onCompare={() =>
                    ask(turn.request.query, {
                      version: turn.request.erp_version === "18.0" ? "17.0" : "18.0",
                      compareOf: turn.id,
                    })
                  }
                  onEscalate={() => composer.current?.setValue(buildEscalationQuery(turn.request.query))}
                  onReviewDraft={() => setHitlTurnId(turn.id)}
                />
              ))}
              <div ref={bottom} />
            </div>
            <div className="sticky bottom-0 z-10 bg-gradient-to-t from-bg via-bg/95 to-transparent pb-5 pt-6">
              <div className="mx-auto w-full max-w-[780px] px-4 md:px-6">{composerEl}</div>
            </div>
          </>
        )}
      </section>

      <aside className="sticky top-14 hidden h-[calc(100dvh-56px)] w-[316px] shrink-0 overflow-y-auto border-l hairline xl:block">
        <PipelineRail turn={focusedTurn} onInspect={onInspect} />
      </aside>

      <CitationDrawer
        sources={citationTurn?.response?.response.sources ?? []}
        role={citationTurn?.request.role ?? "portal_user"}
        index={citation?.index ?? null}
        onClose={() => setCitation(null)}
        onChange={(index) => setCitation((c) => (c ? { ...c, index } : c))}
      />

      <HitlModal
        open={!!hitlTurn}
        pending={hitlTurn?.response?.pending_action ?? null}
        confirm={hitlTurn?.confirm ?? { status: "idle" }}
        onClose={() => setHitlTurnId(null)}
        onConfirm={() => {
          if (!hitlTurn) return;
          void confirmAction(hitlTurn.id).then((ok) => ok && setHitlTurnId(null));
        }}
        onCancelAction={() => {
          if (hitlTurn) cancelAction(hitlTurn.id);
          setHitlTurnId(null);
        }}
        onRegenerate={(draft) => {
          if (hitlTurn) cancelAction(hitlTurn.id);
          setHitlTurnId(null);
          ask(buildDraftQuery(draft));
        }}
      />
    </div>
  );
}
