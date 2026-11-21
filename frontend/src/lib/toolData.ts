/**
 * Lectura tolerante de los resultados de herramientas que viajan en `TraceEvent.data`
 * (`tool.get_ticket_status`, `tool.classify_ticket`). Si el backend nombra un campo de otra
 * forma razonable (p. ej. `sla_due_at` plano) también se reconoce.
 */
import type { TraceEvent } from "../api/types";

export interface TicketView {
  ticket_id: string;
  summary: string;
  status: string;
  priority: string | null;
  category: string | null;
  module: string | null;
  account: string | null;
  assigned_to: string | null;
  updated_at: string | null;
  created_at: string | null;
  sla: {
    policy: string | null;
    resolution_due_at: string | null;
    paused: boolean;
    breached: boolean;
    elapsed_ratio: number | null;
  } | null;
}

export interface ClassificationView {
  category: string;
  module: string | null;
  priority: string | null;
  confidence: number;
  threshold: number | null;
  model: string | null;
  probabilities: { label: string; p: number }[];
}

const str = (v: unknown): string | null => (typeof v === "string" && v ? v : null);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const rec = (v: unknown): Record<string, unknown> | null =>
  v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : null;

/** Nombre legible del clasificador según `kind` (`docuerp.agent.tools.classify`). */
const CLASSIFIER_KIND: Record<string, string> = {
  embeddings: "bge-m3 + regresión logística",
  tfidf: "TF-IDF + regresión logística",
};

export function ticketFromEvent(event: TraceEvent | undefined): TicketView | null {
  if (!event) return null;
  const t = rec(event.data.ticket) ?? rec(event.data.result) ?? event.data;
  const id = str(t.ticket_id);
  const status = str(t.status);
  if (!id || !status) return null;
  const sla = rec(t.sla);
  const due = str(sla?.resolution_due_at) ?? str(t.sla_due_at) ?? str(t.sla_resolution_due_at);
  return {
    ticket_id: id,
    summary: str(t.summary) ?? str(t.title) ?? "",
    status,
    priority: str(t.priority),
    category: str(t.category),
    module: str(t.module),
    account: str(t.account) ?? str(t.account_name) ?? str(t.account_id),
    assigned_to: str(t.assigned_to) ?? str(t.assignee),
    updated_at: str(t.updated_at),
    created_at: str(t.created_at),
    sla:
      sla || due
        ? {
            policy: str(sla?.policy),
            resolution_due_at: due,
            paused: sla?.paused === true || t.sla_paused === true,
            breached: sla?.breached === true || t.sla_breached === true,
            elapsed_ratio: num(sla?.elapsed_ratio),
          }
        : null,
  };
}

export function classificationFromEvent(event: TraceEvent | undefined): ClassificationView | null {
  if (!event) return null;
  const d = rec(event.data.result) ?? event.data;
  const category = str(d.category);
  // El clasificador real informa `category_confidence` y `category_alternatives` (top 2 restantes).
  const confidence = num(d.confidence) ?? num(d.category_confidence);
  if (!category || confidence === null) return null;
  const rawProbs = Array.isArray(d.probabilities)
    ? d.probabilities
    : rec(d.probabilities)
      ? Object.entries(rec(d.probabilities) ?? {}).map(([label, p]) => ({ label, p }))
      : Array.isArray(d.category_alternatives)
        ? [{ label: category, p: confidence }, ...(d.category_alternatives as unknown[])]
        : [];
  const probabilities = rawProbs
    .map((x) => rec(x))
    .filter((x): x is Record<string, unknown> => x !== null)
    .map((x) => ({ label: str(x.label) ?? "?", p: num(x.p) ?? num(x.confidence) ?? 0 }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 5);
  const kind = str(d.kind);
  return {
    category,
    module: str(d.module),
    priority: str(d.priority),
    confidence,
    threshold: num(d.threshold),
    model: str(d.model) ?? (kind ? CLASSIFIER_KIND[kind] ?? kind : null),
    probabilities,
  };
}
