/**
 * Tonos semánticos de la interfaz (SPEC §6, `design/tokens.json` → `semantic`):
 * respondida = accent, abstención = warning, bloqueo = danger, revisión humana = violet,
 * herramienta = info.
 */
import type { Outcome } from "./trace";

export type Tone = "accent" | "info" | "warning" | "danger" | "violet" | "neutral";

/** Fondo tenue + texto + anillo para insignias y contenedores de icono. */
export const TONE: Record<Tone, string> = {
  accent: "bg-accent/10 text-accent-2 ring-accent/25",
  info: "bg-info/10 text-info ring-info/25",
  warning: "bg-warning/10 text-warning ring-warning/25",
  danger: "bg-danger/10 text-danger ring-danger/25",
  violet: "bg-violet/10 text-violet ring-violet/25",
  neutral: "bg-white/[0.04] text-muted ring-white/[0.08]",
};

export const TONE_DOT: Record<Tone, string> = {
  accent: "bg-accent",
  info: "bg-info",
  warning: "bg-warning",
  danger: "bg-danger",
  violet: "bg-violet",
  neutral: "bg-faint",
};

export const TONE_TEXT: Record<Tone, string> = {
  accent: "text-accent-2",
  info: "text-info",
  warning: "text-warning",
  danger: "text-danger",
  violet: "text-violet",
  neutral: "text-muted",
};

/** Colores hex por tono (para SVG y gráficos, donde no aplican clases de Tailwind). */
export const TONE_HEX: Record<Tone, string> = {
  accent: "#10B981",
  info: "#60A5FA",
  warning: "#F5B544",
  danger: "#F87171",
  violet: "#A78BFA",
  neutral: "#8B94A3",
};

/** Tono de cada estado de respuesta. */
export const OUTCOME_TONE: Record<Outcome, Tone> = {
  answered: "accent",
  abstained: "warning",
  blocked: "danger",
  denied: "danger",
  review: "violet",
  tool: "info",
  pending_confirmation: "violet",
};

export const OUTCOME_LABEL: Record<Outcome, string> = {
  answered: "Respondida con evidencia",
  abstained: "Abstención",
  blocked: "Bloqueada por seguridad",
  denied: "Acceso restringido",
  review: "Revisión humana",
  tool: "Herramienta",
  pending_confirmation: "Pendiente de confirmación",
};
