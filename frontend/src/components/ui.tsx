/**
 * Primitivas visuales compartidas (botones, insignias, paneles, atajos, esqueletos).
 *
 * Reglas de movimiento: solo se animan `transform`, `opacity` y colores; la presión de un
 * botón escala a 0.97 en 160 ms con curva ease-out fuerte; los hovers se limitan a
 * dispositivos con puntero fino para no "pegarse" en pantallas táctiles.
 */
import { forwardRef, type ButtonHTMLAttributes, type HTMLAttributes, type ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import { cx } from "../lib/cx";
import { TONE, TONE_DOT, type Tone } from "../lib/tone";

/* ------------------------------------------------------------------------------------ */
/* Botón                                                                                  */
/* ------------------------------------------------------------------------------------ */

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
type ButtonSize = "sm" | "md";

const VARIANT: Record<ButtonVariant, string> = {
  primary:
    "bg-accent text-[#03140D] shadow-[0_0_0_1px_rgba(16,185,129,0.6),0_6px_20px_-6px_rgba(16,185,129,0.55)] [@media(hover:hover)]:hover:bg-accent-2",
  secondary:
    "bg-white/[0.04] text-text ring-1 ring-inset ring-white/[0.08] [@media(hover:hover)]:hover:bg-white/[0.07] [@media(hover:hover)]:hover:ring-white/[0.12]",
  ghost: "text-muted [@media(hover:hover)]:hover:bg-white/[0.05] [@media(hover:hover)]:hover:text-text",
  danger:
    "bg-danger/10 text-danger ring-1 ring-inset ring-danger/30 [@media(hover:hover)]:hover:bg-danger/15",
};

const SIZE: Record<ButtonSize, string> = {
  sm: "h-7 gap-1.5 rounded-[8px] px-2.5 text-xs",
  md: "h-9 gap-2 rounded-[10px] px-3.5 text-sm",
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  icon?: LucideIcon;
  trailing?: ReactNode;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "secondary", size = "md", icon: Icon, trailing, className, children, type = "button", ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cx(
        "inline-flex select-none items-center justify-center whitespace-nowrap font-medium",
        "transition-[transform,background-color,color,box-shadow] duration-150 ease-out-strong",
        "active:scale-[0.97] disabled:pointer-events-none disabled:opacity-45",
        VARIANT[variant],
        SIZE[size],
        className,
      )}
      {...rest}
    >
      {Icon && <Icon size={size === "sm" ? 14 : 16} strokeWidth={1.6} aria-hidden />}
      {children}
      {trailing}
    </button>
  );
});

/** Botón cuadrado solo con icono (requiere `label` accesible). */
export function IconButton({
  icon: Icon,
  label,
  className,
  ...rest
}: { icon: LucideIcon; label: string } & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={cx(
        "inline-flex h-8 w-8 items-center justify-center rounded-[9px] text-muted",
        "transition-[transform,background-color,color] duration-150 ease-out-strong active:scale-[0.95]",
        "[@media(hover:hover)]:hover:bg-white/[0.06] [@media(hover:hover)]:hover:text-text",
        className,
      )}
      {...rest}
    >
      <Icon size={16} strokeWidth={1.6} aria-hidden />
    </button>
  );
}

/* ------------------------------------------------------------------------------------ */
/* Insignias y estados                                                                    */
/* ------------------------------------------------------------------------------------ */

export function Badge({
  tone = "neutral",
  icon: Icon,
  children,
  className,
  mono = false,
}: {
  tone?: Tone;
  icon?: LucideIcon;
  children: ReactNode;
  className?: string;
  mono?: boolean;
}) {
  return (
    <span
      className={cx(
        "inline-flex h-[22px] items-center gap-1.5 whitespace-nowrap rounded-full px-2 text-2xs font-medium ring-1 ring-inset",
        mono && "font-mono",
        TONE[tone],
        className,
      )}
    >
      {Icon && <Icon size={12} strokeWidth={1.8} aria-hidden />}
      {children}
    </span>
  );
}

/** Punto de estado con halo opcional (en vivo). */
export function Dot({ tone, live = false }: { tone: Tone; live?: boolean }) {
  return (
    <span className="relative inline-flex h-2 w-2 shrink-0" aria-hidden>
      {live && <span className={cx("live-dot absolute inset-0 rounded-full opacity-60 blur-[3px]", TONE_DOT[tone])} />}
      <span className={cx("relative h-2 w-2 rounded-full", TONE_DOT[tone], live && "live-dot")} />
    </span>
  );
}

/* ------------------------------------------------------------------------------------ */
/* Paneles                                                                                */
/* ------------------------------------------------------------------------------------ */

/** Tarjeta de doble bisel: carcasa exterior + núcleo interior concéntrico. */
export function Panel({
  children,
  className,
  coreClassName,
  ...rest
}: { children: ReactNode; coreClassName?: string } & HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cx("bezel", className)} {...rest}>
      <div className={cx("bezel-core", coreClassName)}>{children}</div>
    </div>
  );
}

/** Cabecera de sección con eyebrow, título y acciones a la derecha. */
export function SectionHeader({
  eyebrow,
  title,
  description,
  actions,
  className,
}: {
  eyebrow?: string;
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cx("flex flex-wrap items-end justify-between gap-4", className)}>
      <div className="min-w-0">
        {eyebrow && <div className="eyebrow mb-1.5">{eyebrow}</div>}
        <h2 className="text-md font-medium tracking-[-0.01em] text-text">{title}</h2>
        {description && <p className="mt-1 max-w-2xl text-sm text-muted">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

/** Tecla de atajo. */
export function Kbd({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <kbd
      className={cx(
        "inline-flex h-5 min-w-5 items-center justify-center rounded-[5px] bg-white/[0.06] px-1 font-mono text-[10.5px] text-muted ring-1 ring-inset ring-white/[0.08]",
        className,
      )}
    >
      {children}
    </kbd>
  );
}

/** Línea de esqueleto con shimmer. */
export function Skeleton({ className }: { className?: string }) {
  return <div className={cx("shimmer rounded-md", className)} aria-hidden />;
}

/** Estado vacío o de error con icono, título, texto y acción. */
export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  tone = "neutral",
  className,
}: {
  icon: LucideIcon;
  title: string;
  description?: ReactNode;
  action?: ReactNode;
  tone?: Tone;
  className?: string;
}) {
  return (
    <div className={cx("flex flex-col items-center justify-center px-6 py-14 text-center", className)}>
      <div
        className={cx(
          "mb-4 flex h-11 w-11 items-center justify-center rounded-[12px] ring-1 ring-inset",
          TONE[tone],
        )}
      >
        <Icon size={20} strokeWidth={1.5} aria-hidden />
      </div>
      <h3 className="text-base font-medium text-text">{title}</h3>
      {description && <p className="mt-1.5 max-w-sm text-sm text-muted">{description}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

/** Barra horizontal de proporción [0,1] con marca opcional (umbral). */
export function Meter({
  value,
  tone = "accent",
  marker,
  className,
  label,
}: {
  value: number;
  tone?: Tone;
  marker?: number;
  className?: string;
  label?: string;
}) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div
      className={cx("relative h-1.5 w-full rounded-full bg-white/[0.06]", className)}
      role="meter"
      aria-valuemin={0}
      aria-valuemax={1}
      aria-valuenow={Number(value.toFixed(3))}
      aria-label={label}
    >
      <div
        className={cx("absolute inset-y-0 left-0 rounded-full", TONE_DOT[tone])}
        style={{ width: `${pct}%` }}
      />
      {marker !== undefined && (
        <div
          className="absolute -top-1 h-3.5 w-px bg-text/70"
          style={{ left: `${Math.max(0, Math.min(1, marker)) * 100}%` }}
          aria-hidden
        />
      )}
    </div>
  );
}

/** Contenedor con desplazamiento horizontal para tablas en pantallas angostas. */
export function TableScroll({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cx("-mx-px overflow-x-auto", className)}>{children}</div>;
}
