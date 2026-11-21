/** Bloque JSON con resaltado de sintaxis (tokens tipados, sin HTML) y botón de copiar. */
import { Check, Copy } from "lucide-react";
import { useMemo } from "react";
import { useCopy } from "../hooks/useCopy";
import { cx } from "../lib/cx";
import { tokenizeJson, type JsonTokenKind } from "../lib/jsonTokens";

const COLOR: Record<JsonTokenKind, string> = {
  key: "text-info",
  string: "text-accent-2",
  number: "text-warning",
  boolean: "text-violet",
  null: "text-faint",
  punct: "text-faint",
  space: "",
};

export function JsonView({ value, className, maxHeight = 420 }: { value: unknown; className?: string; maxHeight?: number }) {
  const text = useMemo(() => JSON.stringify(value, null, 2), [value]);
  const tokens = useMemo(() => tokenizeJson(text), [text]);
  const [copied, copy] = useCopy();
  return (
    <div className={cx("relative rounded-[12px] bg-black/30 ring-1 ring-inset ring-white/[0.06]", className)}>
      <button
        type="button"
        onClick={() => copy(text)}
        className="absolute right-2 top-2 z-[1] inline-flex h-7 items-center gap-1.5 rounded-[8px] bg-white/[0.05] px-2 text-2xs text-muted ring-1 ring-inset ring-white/[0.08] transition-[color,transform] duration-150 hover:text-text active:scale-95"
      >
        {copied ? <Check size={12} strokeWidth={2} aria-hidden /> : <Copy size={12} strokeWidth={1.8} aria-hidden />}
        {copied ? "Copiado" : "Copiar"}
      </button>
      <pre className="overflow-auto p-4 pr-24 font-mono text-[12px] leading-[1.65]" style={{ maxHeight }}>
        <code>
          {tokens.map((t, i) => (
            <span key={i} className={COLOR[t.kind]}>
              {t.text}
            </span>
          ))}
        </code>
      </pre>
    </div>
  );
}
