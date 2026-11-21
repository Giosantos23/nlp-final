/**
 * Consultas que la consola genera por el usuario (siempre visibles y editables antes de
 * enviarse, o derivadas de una edición explícita del borrador en el modal HITL).
 */
import type { TicketDraft } from "../api/types";

/** Consulta de regeneración de un borrador editado: resumen — descripción [pistas]. */
export function buildDraftQuery(draft: TicketDraft): string {
  const description = draft.description.trim();
  return (
    `Crea un ticket: ${draft.summary.trim()}${description ? ` — ${description}` : ""} ` +
    `[categoría=${draft.category}; prioridad=${draft.priority}; módulo=${draft.module}]`
  );
}

/** Texto de escalamiento que se precarga en la barra de consulta (el usuario decide enviarlo). */
export function buildEscalationQuery(originalQuery: string): string {
  const q = originalQuery.length > 140 ? `${originalQuery.slice(0, 137)}…` : originalQuery;
  return `Crea un ticket: escalar a N2 la consulta sin respuesta — «${q}»`;
}
