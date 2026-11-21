/**
 * Raíz de la aplicación: estructura (barra lateral + barra superior), enrutamiento por hash,
 * atajos globales (⌘K paleta, G + letra para navegar) y carga diferida de las vistas pesadas
 * (gráficos de Recharts) para que la consola aparezca rápido.
 */
import { Suspense, lazy, useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api/client";
import { CommandPalette } from "./components/shell/CommandPalette";
import { Sidebar } from "./components/shell/Sidebar";
import { Topbar } from "./components/shell/Topbar";
import { Skeleton } from "./components/ui";
import { useHashRoute, type ViewKey } from "./hooks/useHashRoute";
import { NAV_ITEMS } from "./lib/nav";
import { SessionProvider, useSession } from "./state/session";
import { ConsoleView } from "./views/Console";

const TraceView = lazy(() => import("./views/Trace"));
const EvalView = lazy(() => import("./views/Evaluation"));
const SecurityView = lazy(() => import("./views/Security"));
const CorpusView = lazy(() => import("./views/Corpus"));

const SHORTCUT_TO_VIEW: Record<string, ViewKey> = Object.fromEntries(
  NAV_ITEMS.map((n) => [n.shortcut.toLowerCase(), n.key]),
);

function ViewFallback() {
  return (
    <div className="mx-auto max-w-[1180px] space-y-4 px-6 py-8">
      <Skeleton className="h-6 w-56" />
      <div className="grid grid-cols-4 gap-3">
        {Array.from({ length: 4 }, (_, i) => (
          <Skeleton key={i} className="h-28" />
        ))}
      </div>
      <Skeleton className="h-72" />
    </div>
  );
}

function Shell() {
  const [route, navigate] = useHashRoute();
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [navOpen, setNavOpen] = useState(false);
  const { ask } = useSession();
  const pendingG = useRef<number | null>(null);

  useEffect(() => {
    void api.probe();
  }, []);

  // Atajos globales. ⌘K/Ctrl+K abre la paleta; "g" seguido de una letra navega.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((v) => !v);
        return;
      }
      const t = e.target as HTMLElement | null;
      const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
      if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
      const key = e.key.toLowerCase();
      if (pendingG.current !== null) {
        window.clearTimeout(pendingG.current);
        pendingG.current = null;
        const view = SHORTCUT_TO_VIEW[key];
        if (view) {
          e.preventDefault();
          navigate(view);
        }
        return;
      }
      if (key === "g") pendingG.current = window.setTimeout(() => (pendingG.current = null), 900);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navigate]);

  const current = NAV_ITEMS.find((n) => n.key === route.view) ?? NAV_ITEMS[0]!;
  const inspect = useCallback((traceId: string) => navigate("traza", traceId), [navigate]);
  const askFromPalette = useCallback(
    (q: string) => {
      navigate("consola");
      ask(q);
    },
    [ask, navigate],
  );

  useEffect(() => {
    document.title = `${current.label} · DocuERP`;
  }, [current.label]);

  return (
    <div className="relative flex min-h-[100dvh]">
      <div className="ambient" aria-hidden />
      <Sidebar current={route.view} onNavigate={navigate} mobileOpen={navOpen} onCloseMobile={() => setNavOpen(false)} />
      <div className="relative z-[1] flex min-w-0 flex-1 flex-col">
        <Topbar
          title={current.label}
          subtitle={current.subtitle}
          onOpenPalette={() => setPaletteOpen(true)}
          onOpenNav={() => setNavOpen(true)}
        />
        <main id="contenido" className="min-w-0 flex-1">
          {route.view === "consola" ? (
            <ConsoleView onInspect={inspect} />
          ) : (
            <Suspense fallback={<ViewFallback />}>
              {route.view === "traza" && <TraceView traceId={route.param} onBack={() => navigate("consola")} />}
              {route.view === "evaluacion" && <EvalView />}
              {route.view === "seguridad" && <SecurityView />}
              {route.view === "corpus" && <CorpusView />}
            </Suspense>
          )}
        </main>
      </div>
      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        onNavigate={navigate}
        onAsk={askFromPalette}
      />
    </div>
  );
}

export default function App() {
  return (
    <SessionProvider>
      <Shell />
    </SessionProvider>
  );
}
