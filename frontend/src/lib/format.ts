/**
 * Formateadores y etiquetas en español para toda la interfaz.
 * Centralizarlos evita mezclar formatos (p. ej. "1,2 s" en un lugar y "1.2s" en otro).
 */
import type { ErpModule, Intent, TicketCategory, TicketPriority, UserRole } from "../api/types";

const LOCALE = "es-GT";

/** Milisegundos legibles: "0.4 ms", "382 ms", "3.87 s". */
export function formatMs(ms: number): string {
  if (!Number.isFinite(ms)) return "—";
  if (ms === 0) return "0 ms";
  if (ms < 1) return `${ms.toFixed(1)} ms`;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;
}

/** Cantidad con sustantivo en singular o plural: plural(1, "llamada") → "1 llamada". */
export function plural(n: number, singular: string, pluralForm = `${singular}s`): string {
  return `${n} ${n === 1 ? singular : pluralForm}`;
}

/** Proporción [0,1] como porcentaje: 0.876 → "87.6 %". */
export function formatPct(ratio: number, digits = 1): string {
  if (!Number.isFinite(ratio)) return "—";
  return `${(ratio * 100).toFixed(digits)} %`;
}

/** Puntaje con precisión fija (los scores de RRF necesitan más decimales). */
export function formatScore(score: number, digits = 3): string {
  if (!Number.isFinite(score)) return "—";
  return score.toFixed(digits);
}

/** Costo en dólares con la precisión necesaria para montos de fracciones de centavo. */
export function formatUsd(usd: number): string {
  if (!Number.isFinite(usd)) return "—";
  if (usd === 0) return "US$0";
  if (usd < 0.01) return `US$${usd.toFixed(4)}`;
  return `US$${usd.toFixed(2)}`;
}

/** Entero con separador de miles local: 81420 → "81,420". */
export function formatInt(n: number): string {
  return new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 0 }).format(n);
}

/** Número compacto: 81420 → "81.4 k". */
export function formatCompact(n: number): string {
  if (Math.abs(n) < 1000) return String(n);
  if (Math.abs(n) < 1_000_000) return `${(n / 1000).toFixed(1)} k`;
  return `${(n / 1_000_000).toFixed(2)} M`;
}

/** Fecha y hora cortas en la zona local. */
export function formatDateTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return new Intl.DateTimeFormat(LOCALE, {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(d);
}

/** Fecha corta (sin hora). */
export function formatDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return new Intl.DateTimeFormat(LOCALE, { day: "2-digit", month: "short", year: "numeric" }).format(d);
}

/** Duración restante "1 d 14 h", "3 h 05 min", "42 s"; negativa → "vencido hace …". */
export function formatRemaining(ms: number): string {
  const past = ms < 0;
  const abs = Math.abs(ms);
  const s = Math.floor(abs / 1000);
  const d = Math.floor(s / 86_400);
  const h = Math.floor((s % 86_400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  let text: string;
  if (d > 0) text = `${d} d ${h} h`;
  else if (h > 0) text = `${h} h ${String(m).padStart(2, "0")} min`;
  else if (m > 0) text = `${m} min`;
  else text = `${s % 60} s`;
  return past ? `vencido hace ${text}` : text;
}

/** Cuenta regresiva mm:ss para la expiración del token HITL. */
export function formatCountdown(ms: number): string {
  const total = Math.max(0, Math.ceil(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export const ROLE_LABEL: Record<UserRole, string> = {
  portal_user: "Portal",
  n1: "Analista N1",
  n2: "Analista N2",
  admin: "Admin",
};

export const ROLE_SHORT: Record<UserRole, string> = {
  portal_user: "Portal",
  n1: "N1",
  n2: "N2",
  admin: "Admin",
};

export const INTENT_LABEL: Record<Intent, string> = {
  doc_query: "Consulta documental",
  ticket_status: "Estado de ticket",
  ticket_create: "Crear ticket",
  ticket_classify: "Clasificar ticket",
  out_of_domain: "Fuera de dominio",
};

export const CATEGORY_LABEL: Record<TicketCategory, string> = {
  configuracion: "Configuración",
  error_funcional: "Error funcional",
  acceso_permisos: "Accesos y permisos",
  rendimiento: "Rendimiento",
  migracion: "Migración",
  capacitacion: "Capacitación",
  integracion: "Integración",
};

export const PRIORITY_LABEL: Record<TicketPriority, string> = {
  baja: "Baja",
  media: "Media",
  alta: "Alta",
  critica: "Crítica",
};

export const MODULE_LABEL: Record<ErpModule, string> = {
  inventario: "Inventario",
  compras: "Compras",
  ventas: "Ventas",
  facturacion: "Facturación",
  accesos: "Accesos",
  actualizacion: "Actualización",
  general: "General",
};

/** Etiqueta de un valor de vocabulario conocido o el valor tal cual si no se conoce. */
export function labelOf<T extends string>(map: Record<T, string>, value: string): string {
  return (map as Record<string, string>)[value] ?? value;
}

/** Recorta un hash o id largo para mostrarlo en tablas: "a1b2c3d4…9f0e". */
export function shortHash(hash: string, head = 8, tail = 4): string {
  return hash.length <= head + tail + 1 ? hash : `${hash.slice(0, head)}…${hash.slice(-tail)}`;
}
