/**
 * Tailwind lee los tokens de `design/tokens.json` (SPEC §6) para que el frontend y las
 * figuras de Matplotlib compartan exactamente la misma paleta. Cambiar un color allí lo
 * cambia en toda la interfaz.
 */
import type { Config } from "tailwindcss";
import tokens from "../design/tokens.json";

const c = tokens.color;

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: c.bg,
        surface: c.surface,
        "surface-2": c["surface-2"],
        "surface-3": "#1D222B",
        border: c.border,
        "border-strong": "#2E3542",
        text: c.text,
        muted: c.muted,
        faint: "#5C6574",
        accent: c.accent,
        "accent-2": c["accent-2"],
        info: c.info,
        warning: c.warning,
        danger: c.danger,
        violet: c.violet,
      },
      fontFamily: {
        sans: ["Geist Variable", ...tokens.font.sans],
        mono: ["Geist Mono Variable", ...tokens.font.mono],
      },
      fontSize: {
        "2xs": ["11px", { lineHeight: "16px", letterSpacing: "0.01em" }],
        xs: ["12px", { lineHeight: "16px" }],
        sm: ["13px", { lineHeight: "20px" }],
        base: ["14px", { lineHeight: "22px" }],
        md: ["16px", { lineHeight: "24px" }],
        lg: ["20px", { lineHeight: "28px", letterSpacing: "-0.01em" }],
        xl: ["28px", { lineHeight: "34px", letterSpacing: "-0.02em" }],
        "2xl": ["40px", { lineHeight: "44px", letterSpacing: "-0.03em" }],
      },
      borderRadius: {
        sm: tokens.radius.sm,
        md: tokens.radius.md,
        lg: tokens.radius.lg,
        xl: "18px",
      },
      boxShadow: {
        soft: tokens.shadow.soft,
        glow: tokens.shadow.glow,
        "inner-hl": "inset 0 1px 0 0 rgba(255,255,255,0.05)",
        float: "0 0 0 1px rgba(255,255,255,0.06), 0 16px 48px -12px rgba(0,0,0,0.7)",
      },
      transitionTimingFunction: {
        out: tokens.motion.ease,
        "out-strong": "cubic-bezier(0.23, 1, 0.32, 1)",
        "in-out": "cubic-bezier(0.77, 0, 0.175, 1)",
        drawer: "cubic-bezier(0.32, 0.72, 0, 1)",
      },
      // Pasos de opacidad extra para modificadores de color (p. ej. `bg-accent/12`).
      opacity: {
        "12": "0.12",
        "35": "0.35",
        "45": "0.45",
        "55": "0.55",
        "65": "0.65",
        "85": "0.85",
      },
      transitionDuration: {
        fast: tokens.motion.fast,
        base: tokens.motion.base,
        slow: tokens.motion.slow,
      },
    },
  },
  plugins: [],
};

export default config;
