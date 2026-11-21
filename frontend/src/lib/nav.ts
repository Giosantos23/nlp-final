/** Vistas de la aplicación con su icono, título, subtítulo y atajo (G + letra). */
import { Activity, FlaskConical, Library, ShieldCheck, SquareTerminal, type LucideIcon } from "lucide-react";
import type { ViewKey } from "../hooks/useHashRoute";

export interface NavItem {
  key: ViewKey;
  label: string;
  subtitle: string;
  icon: LucideIcon;
  shortcut: string;
}

export const NAV_ITEMS: NavItem[] = [
  {
    key: "consola",
    label: "Consola",
    subtitle: "Consultas con citas verificables, herramientas y confirmación humana",
    icon: SquareTerminal,
    shortcut: "C",
  },
  {
    key: "traza",
    label: "Inspector de traza",
    subtitle: "Latencias por etapa, listas de recuperación y contrato JSON",
    icon: Activity,
    shortcut: "T",
  },
  {
    key: "evaluacion",
    label: "Evaluación",
    subtitle: "Metas del anteproyecto, baseline vs final y ablaciones",
    icon: FlaskConical,
    shortcut: "E",
  },
  {
    key: "seguridad",
    label: "Seguridad",
    subtitle: "Matriz red-team por vector de ataque",
    icon: ShieldCheck,
    shortcut: "S",
  },
  {
    key: "corpus",
    label: "Corpus y gobernanza",
    subtitle: "Manifiesto de fuentes, licencias, vigencia y chunks",
    icon: Library,
    shortcut: "D",
  },
];
