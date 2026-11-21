/**
 * Enrutador mínimo basado en el hash de la URL (`#/traza/trc_…`).
 *
 * Con hash no hace falta configurar el servidor para rutas profundas y los enlaces se
 * pueden compartir. Cinco vistas no justifican una dependencia de enrutamiento completa.
 */
import { useCallback, useSyncExternalStore } from "react";

export type ViewKey = "consola" | "traza" | "evaluacion" | "seguridad" | "corpus";

export interface Route {
  view: ViewKey;
  /** Parámetro opcional (p. ej. el `trace_id` en `#/traza/<id>`). */
  param: string | null;
}

const VIEWS: ViewKey[] = ["consola", "traza", "evaluacion", "seguridad", "corpus"];

/** Interpreta un hash como ruta; cualquier valor desconocido cae en la consola. */
export function parseHash(hash: string): Route {
  const [rawView, ...rest] = hash.replace(/^#\/?/, "").split("/");
  const view = VIEWS.includes(rawView as ViewKey) ? (rawView as ViewKey) : "consola";
  const param = rest.length ? decodeURIComponent(rest.join("/")) : null;
  return { view, param: param || null };
}

function subscribe(fn: () => void): () => void {
  window.addEventListener("hashchange", fn);
  return () => window.removeEventListener("hashchange", fn);
}

export function useHashRoute(): [Route, (view: ViewKey, param?: string | null) => void] {
  const hash = useSyncExternalStore(subscribe, () => window.location.hash);
  const navigate = useCallback((view: ViewKey, param?: string | null) => {
    window.location.hash = param ? `/${view}/${encodeURIComponent(param)}` : `/${view}`;
  }, []);
  return [parseHash(hash), navigate];
}
