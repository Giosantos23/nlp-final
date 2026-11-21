/**
 * Navegación lateral: identidad del producto, vistas con atajos (G + letra) y estado de la
 * conexión con la API. En pantallas angostas se muestra como cajón superpuesto.
 */
import { AnimatePresence, motion } from "framer-motion";
import { RefreshCw, X } from "lucide-react";
import { api } from "../../api/client";
import type { WarmupInfo } from "../../api/types";
import { useConnection } from "../../hooks/useConnection";
import { useWarmup } from "../../hooks/useWarmup";
import type { ViewKey } from "../../hooks/useHashRoute";
import { cx } from "../../lib/cx";
import { NAV_ITEMS } from "../../lib/nav";
import { LogoMark } from "../Logo";
import { Dot, IconButton, Kbd } from "../ui";

/** Texto del precalentamiento: qué se está cargando o cuánto tardó. */
function warmupText(info: WarmupInfo | null): { body: string; warming: boolean } {
  if (!info || info.state === "disabled") return { body: "Streaming SSE · agente real", warming: false };
  const done = info.steps.filter((s) => s.status === "ok" || s.status === "error" || s.status === "skipped").length;
  if (info.state === "pending" || info.state === "running") {
    const current = info.steps.find((s) => s.status === "running");
    return {
      body: `Precalentando modelos ${done}/${info.steps.length || "…"}${current ? ` · ${current.label}` : ""}`,
      warming: true,
    };
  }
  const seconds = info.total_ms !== null ? ` en ${(info.total_ms / 1000).toFixed(1)} s` : "";
  return info.state === "ready"
    ? { body: `Modelos en caliente (cargados${seconds}) · streaming SSE`, warming: false }
    : { body: `Precalentamiento parcial: ${info.steps.filter((s) => s.status === "error").map((s) => s.label).join(", ")}`, warming: false };
}

function ConnectionCard() {
  const mode = useConnection();
  const warm = warmupText(useWarmup());
  const copy = {
    live: { tone: warm.warming ? ("warning" as const) : ("accent" as const), title: "API conectada", body: warm.body },
    offline: {
      tone: "warning" as const,
      title: "Demo offline",
      body: "La API no responde; se usan escenarios precargados.",
    },
    mock: { tone: "violet" as const, title: "Modo demo", body: "VITE_MOCK=1 · escenarios precargados" },
  }[mode];
  return (
    <div className="rounded-[12px] bg-white/[0.025] p-3 ring-1 ring-inset ring-white/[0.06]">
      <div className="flex items-center gap-2">
        <Dot tone={copy.tone} live={mode === "live"} />
        <span className="text-xs font-medium text-text">{copy.title}</span>
        {mode === "offline" && (
          <button
            type="button"
            onClick={() => void api.probe()}
            className="ml-auto inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-2xs text-muted transition-colors duration-150 hover:text-text"
          >
            <RefreshCw size={11} strokeWidth={1.8} aria-hidden />
            Reintentar
          </button>
        )}
      </div>
      <p className="mt-1 text-2xs leading-4 text-faint">{copy.body}</p>
    </div>
  );
}

function SidebarBody({ current, onNavigate }: { current: ViewKey; onNavigate: (v: ViewKey) => void }) {
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2.5 px-4 pb-6 pt-5">
        <LogoMark />
        <div className="min-w-0 leading-tight">
          <div className="text-[15px] font-semibold tracking-[-0.02em] text-text">DocuERP</div>
          <div className="truncate text-2xs text-faint">Altamira Soluciones ERP</div>
        </div>
      </div>

      <div className="eyebrow px-5 pb-2">Espacio de trabajo</div>
      <nav aria-label="Vistas" className="flex flex-col gap-0.5 px-2.5">
        {NAV_ITEMS.map((item) => {
          const active = item.key === current;
          const Icon = item.icon;
          return (
            <button
              key={item.key}
              type="button"
              onClick={() => onNavigate(item.key)}
              aria-current={active ? "page" : undefined}
              className={cx(
                "group relative flex h-9 items-center gap-2.5 rounded-[9px] px-2.5 text-left text-sm",
                "transition-colors duration-150",
                active ? "text-text" : "text-muted hover:bg-white/[0.035] hover:text-text",
              )}
            >
              {active && (
                <motion.span
                  layoutId="nav-active"
                  className="absolute inset-0 rounded-[9px] bg-white/[0.06] ring-1 ring-inset ring-white/[0.07]"
                  transition={{ type: "spring", duration: 0.32, bounce: 0.12 }}
                />
              )}
              <Icon
                size={16}
                strokeWidth={1.5}
                aria-hidden
                className={cx("relative", active ? "text-accent-2" : "text-faint group-hover:text-muted")}
              />
              <span className="relative min-w-0 flex-1 truncate">{item.label}</span>
              <span className="absolute right-2 flex gap-0.5 opacity-0 transition-opacity duration-150 group-hover:opacity-100">
                <Kbd>G</Kbd>
                <Kbd>{item.shortcut}</Kbd>
              </span>
            </button>
          );
        })}
      </nav>

      <div className="mt-auto space-y-3 p-3">
        <ConnectionCard />
        <div className="flex items-center justify-between px-1 text-2xs text-faint">
          <span>v0.1 · CC3103 UVG</span>
          <span className="font-mono">Odoo 17/18</span>
        </div>
      </div>
    </div>
  );
}

export function Sidebar({
  current,
  onNavigate,
  mobileOpen,
  onCloseMobile,
}: {
  current: ViewKey;
  onNavigate: (v: ViewKey) => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}) {
  return (
    <>
      <aside className="sticky top-0 hidden h-[100dvh] w-[232px] shrink-0 border-r hairline lg:block">
        <SidebarBody current={current} onNavigate={onNavigate} />
      </aside>
      <AnimatePresence>
        {mobileOpen && (
          <div className="fixed inset-0 z-40 lg:hidden">
            <motion.div
              className="absolute inset-0 bg-black/60 backdrop-blur-sm"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              onClick={onCloseMobile}
            />
            <motion.aside
              className="absolute inset-y-0 left-0 w-[264px] border-r hairline bg-bg"
              initial={{ transform: "translateX(-100%)" }}
              animate={{ transform: "translateX(0%)" }}
              exit={{ transform: "translateX(-100%)" }}
              transition={{ duration: 0.26, ease: [0.32, 0.72, 0, 1] }}
            >
              <div className="absolute right-2 top-4">
                <IconButton icon={X} label="Cerrar navegación" onClick={onCloseMobile} />
              </div>
              <SidebarBody
                current={current}
                onNavigate={(v) => {
                  onNavigate(v);
                  onCloseMobile();
                }}
              />
            </motion.aside>
          </div>
        )}
      </AnimatePresence>
    </>
  );
}
