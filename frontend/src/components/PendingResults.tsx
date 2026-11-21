/**
 * Estado vacío de las vistas de resultados (Evaluación, Seguridad) cuando la API responde
 * 404: todavía no se ejecutó la corrida. En lugar de un error, explica qué falta, qué
 * comando lo genera y de dónde lo leerá la vista en cuanto exista.
 */
import { RefreshCw, type LucideIcon } from "lucide-react";
import { useCopy } from "../hooks/useCopy";
import { Button, Panel } from "./ui";

export interface PendingStep {
  label: string;
  command: string;
}

export function PendingResults({
  icon: Icon,
  eyebrow,
  title,
  description,
  steps,
  path,
  onRetry,
}: {
  icon: LucideIcon;
  eyebrow: string;
  title: string;
  description: string;
  steps: PendingStep[];
  /** Ruta del repositorio que la vista lee (se muestra tal cual). */
  path: string;
  onRetry: () => void;
}) {
  const [copied, copy] = useCopy();
  return (
    <div className="mx-auto max-w-[760px] px-4 py-12 md:px-6 md:py-16">
      <Panel coreClassName="p-6 md:p-7">
        <div className="flex items-start gap-4">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-[11px] bg-white/[0.03] text-muted ring-1 ring-inset ring-white/[0.08]">
            <Icon size={18} strokeWidth={1.5} aria-hidden />
          </span>
          <div className="min-w-0">
            <div className="eyebrow mb-1">{eyebrow}</div>
            <h2 className="text-lg font-medium tracking-[-0.02em] text-text">{title}</h2>
            <p className="mt-1.5 text-sm leading-relaxed text-muted">{description}</p>
          </div>
        </div>

        <ol className="mt-6 space-y-2.5">
          {steps.map((step, i) => (
            <li key={step.command} className="grid grid-cols-[22px_1fr] items-start gap-3">
              <span className="mt-1 flex h-[22px] w-[22px] items-center justify-center rounded-full bg-white/[0.04] font-mono text-2xs text-faint ring-1 ring-inset ring-white/[0.07]">
                {i + 1}
              </span>
              <div className="min-w-0">
                <div className="text-xs text-muted">{step.label}</div>
                <button
                  type="button"
                  onClick={() => copy(step.command)}
                  title="Copiar comando"
                  className="mt-1 block w-full truncate rounded-[8px] bg-black/30 px-3 py-2 text-left font-mono text-xs text-text ring-1 ring-inset ring-white/[0.06] transition-colors duration-150 hover:ring-accent/30"
                >
                  <span className="select-none text-faint">$ </span>
                  {step.command}
                </button>
              </div>
            </li>
          ))}
        </ol>

        <div className="mt-6 flex flex-wrap items-center gap-3 border-t hairline pt-4">
          <span className="min-w-0 flex-1 text-2xs text-faint">
            La vista lee <span className="font-mono text-muted">{path}</span> en cuanto exista.
            {copied ? " Comando copiado." : ""}
          </span>
          <Button size="sm" variant="secondary" icon={RefreshCw} onClick={onRetry}>
            Volver a consultar
          </Button>
        </div>
      </Panel>
    </div>
  );
}
