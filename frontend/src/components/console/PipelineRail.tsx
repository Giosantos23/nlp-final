/**
 * Carril "Pipeline en vivo": muestra las etapas de la traza a medida que llegan por SSE
 * (guard de entrada → router → recuperación → rerank → gate → permisos → generación →
 * guard de salida). Cada indicador cambia de estado con un fundido corto (150 ms) para que
 * el avance se perciba sin distraer.
 */
import { AnimatePresence, motion } from "framer-motion";
import { ArrowUpRight, Check, Minus, RotateCcw, X } from "lucide-react";
import { useMemo } from "react";
import { cx } from "../../lib/cx";
import { formatInt, formatMs, formatUsd } from "../../lib/format";
import { buildPipeline, idlePipeline, type PipelineStep, type StepStatus } from "../../lib/trace";
import type { Turn } from "../../state/session";
import { Button } from "../ui";

/** El gate τ "bloqueado" no es un ataque sino una abstención: se pinta en ámbar, no en rojo. */
function visualStatus(step: PipelineStep): StepStatus {
  return step.def.key === "evidence_gate" && step.status === "blocked" ? "retry" : step.status;
}

function Indicator({ status, soft = false }: { status: StepStatus; soft?: boolean }) {
  const base = "flex h-[18px] w-[18px] items-center justify-center rounded-full";
  let content: JSX.Element;
  switch (status) {
    case "ok":
      content = (
        <span className={cx(base, "bg-accent/15 text-accent-2 ring-1 ring-inset ring-accent/40")}>
          <Check size={11} strokeWidth={2.4} aria-hidden />
        </span>
      );
      break;
    case "blocked":
    case "error":
      content = (
        <span className={cx(base, "bg-danger/15 text-danger ring-1 ring-inset ring-danger/40")}>
          <X size={11} strokeWidth={2.4} aria-hidden />
        </span>
      );
      break;
    case "retry":
      content = (
        <span className={cx(base, "bg-warning/15 text-warning ring-1 ring-inset ring-warning/40")}>
          {soft ? <Minus size={10} strokeWidth={2.4} aria-hidden /> : <RotateCcw size={10} strokeWidth={2.2} aria-hidden />}
        </span>
      );
      break;
    case "skipped":
      content = (
        <span className={cx(base, "text-faint ring-1 ring-inset ring-white/[0.08]")}>
          <Minus size={10} strokeWidth={2} aria-hidden />
        </span>
      );
      break;
    case "running":
      content = (
        <span className={cx(base, "ring-1 ring-inset ring-accent/50 shadow-[0_0_12px_rgba(16,185,129,0.45)]")}>
          <span className="live-dot h-1.5 w-1.5 rounded-full bg-accent-2" />
        </span>
      );
      break;
    default:
      content = <span className={cx(base, "ring-1 ring-inset ring-white/[0.1]")} />;
  }
  return (
    <AnimatePresence mode="popLayout" initial={false}>
      <motion.span
        key={status}
        initial={{ opacity: 0, transform: "scale(0.85)" }}
        animate={{ opacity: 1, transform: "scale(1)" }}
        exit={{ opacity: 0, transform: "scale(0.9)" }}
        transition={{ duration: 0.16, ease: [0.23, 1, 0.32, 1] }}
        className="inline-flex"
      >
        {content}
      </motion.span>
    </AnimatePresence>
  );
}

const STATUS_WORD: Partial<Record<StepStatus, string>> = {
  running: "en curso",
  blocked: "bloqueado",
  error: "error",
  retry: "reintento",
  skipped: "no ejecutada",
};

function StepRow({ step, last }: { step: PipelineStep; last: boolean }) {
  const doneLine = step.status === "ok" || step.status === "retry";
  const gateBelow = step.def.key === "evidence_gate" && step.status === "blocked";
  const word = gateBelow ? "bajo τ" : STATUS_WORD[step.status];
  return (
    <li className="relative flex gap-3 pb-4 last:pb-0">
      {!last && (
        <span
          className={cx(
            "absolute left-[8.5px] top-[22px] h-[calc(100%-22px)] w-px transition-colors duration-200",
            doneLine ? "bg-accent/35" : "bg-white/[0.07]",
          )}
          aria-hidden
        />
      )}
      <Indicator status={visualStatus(step)} soft={gateBelow} />
      <div className="-mt-px min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          <span
            className={cx(
              "text-[13px] transition-colors duration-200",
              step.status === "pending" || step.status === "skipped" ? "text-faint" : "text-text",
            )}
          >
            {step.def.label}
          </span>
          {word && (
            <span
              className={cx(
                "text-2xs",
                step.status === "running"
                  ? "text-accent-2"
                  : step.status === "skipped"
                    ? "text-faint"
                    : gateBelow || step.status === "retry"
                      ? "text-warning"
                      : "text-danger",
              )}
            >
              {word}
            </span>
          )}
          <span className="ml-auto font-mono text-2xs text-faint">
            {step.events.length > 0 && step.status !== "skipped" ? formatMs(step.durationMs) : ""}
          </span>
        </div>
        <div className="mt-0.5 min-h-[16px] truncate text-2xs text-muted">
          {step.detail ?? (step.status === "running" ? step.def.running : "")}
        </div>
      </div>
    </li>
  );
}

export function PipelineRail({ turn, onInspect }: { turn: Turn | null; onInspect: (traceId: string) => void }) {
  const steps = useMemo(
    () => (turn ? buildPipeline(turn.events, turn.status !== "streaming", turn.response?.intent ?? null) : idlePipeline()),
    [turn],
  );
  const res = turn?.response ?? null;
  const elapsed = turn?.events.reduce((s, e) => s + e.duration_ms, 0) ?? 0;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between px-5 pb-3 pt-5">
        <div>
          <div className="eyebrow">Pipeline en vivo</div>
          <div className="mt-1 text-sm text-text">
            {!turn ? "En espera" : turn.status === "streaming" ? "Procesando consulta" : turn.status === "error" ? "Interrumpido" : "Completado"}
          </div>
        </div>
        <div className="text-right">
          <div className="font-mono text-lg tracking-tight text-text">{formatMs(res?.latency_ms ?? elapsed)}</div>
          <div className="text-2xs text-faint">latencia acumulada</div>
        </div>
      </div>

      <div className="mx-5 mb-4 h-px bg-white/[0.06]" />

      <ol className="flex-1 px-5" aria-live="polite" aria-label="Etapas del pipeline">
        {steps.map((step, i) => (
          <StepRow key={step.def.key} step={step} last={i === steps.length - 1} />
        ))}
      </ol>

      {!turn && (
        <p className="px-5 pb-2 text-2xs leading-4 text-faint">
          Las etapas se marcan en vivo a medida que el servidor las completa (SSE).
        </p>
      )}

      {res && (
        <div className="m-3 rounded-[12px] bg-white/[0.025] p-3 ring-1 ring-inset ring-white/[0.06]">
          <dl className="grid grid-cols-3 gap-2 text-center">
            <div>
              <dt className="text-2xs text-faint">Tokens</dt>
              <dd className="font-mono text-xs text-text">{formatInt(res.usage.input_tokens + res.usage.cache_read_tokens + res.usage.cache_creation_tokens + res.usage.output_tokens)}</dd>
            </div>
            <div>
              <dt className="text-2xs text-faint">Llamadas LLM</dt>
              <dd className="font-mono text-xs text-text">{res.usage.llm_calls}</dd>
            </div>
            <div>
              <dt className="text-2xs text-faint">Costo</dt>
              <dd className="font-mono text-xs text-text">{formatUsd(res.usage.cost_usd)}</dd>
            </div>
          </dl>
          <Button
            size="sm"
            variant="secondary"
            className="mt-3 w-full"
            onClick={() => onInspect(res.trace_id)}
            trailing={<ArrowUpRight size={13} strokeWidth={1.8} aria-hidden />}
          >
            Abrir en el inspector
          </Button>
        </div>
      )}
    </div>
  );
}
