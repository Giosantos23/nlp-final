/**
 * Piezas compartidas de gráficos (Recharts) con el estilo del sistema: rejilla recesiva,
 * ejes sin línea, tooltip oscuro donde el valor manda y el nombre de la serie acompaña.
 */
import type { TooltipProps } from "recharts";

type TooltipRow = { name?: string | number; value?: number | string; color?: string };

/** Tooltip con clave de línea corta por serie (no cajas) y el valor en negrita. */
export function ChartTooltip({
  active,
  payload,
  label,
  format = (v: number) => v.toFixed(2),
}: TooltipProps<number, string> & { format?: (v: number) => string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-[10px] bg-surface-2 px-3 py-2 text-xs shadow-float">
      {label !== undefined && <div className="mb-1.5 text-2xs text-faint">{String(label)}</div>}
      <ul className="space-y-1">
        {(payload as TooltipRow[]).map((p) => (
          <li key={String(p.name)} className="flex items-center gap-2">
            <span className="h-0.5 w-3 rounded-full" style={{ background: p.color }} aria-hidden />
            <span className="font-mono font-medium text-text">
              {typeof p.value === "number" ? format(p.value) : String(p.value ?? "—")}
            </span>
            <span className="text-muted">{String(p.name)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Leyenda HTML (identidad nunca solo por color: siempre con texto). */
export function Legend({ items }: { items: { label: string; color: string }[] }) {
  return (
    <ul className="flex flex-wrap items-center gap-4 text-2xs text-muted">
      {items.map((i) => (
        <li key={i.label} className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-[3px]" style={{ background: i.color }} aria-hidden />
          {i.label}
        </li>
      ))}
    </ul>
  );
}
