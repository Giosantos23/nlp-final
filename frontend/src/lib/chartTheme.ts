/** Parámetros visuales de los gráficos (colores de `design/tokens.json` → `chart`). */
export const CHART = {
  grid: "#232934",
  axis: "#8B94A3",
  final: "#10B981",
  /** Gris de contexto del baseline (validado contra la superficie oscura: ΔE CVD 20). */
  baseline: "#5C6574",
  second: "#A78BFA",
  font: 11,
} as const;

export const axisProps = {
  tick: { fill: CHART.axis, fontSize: CHART.font },
  tickLine: false,
  axisLine: false,
} as const;
