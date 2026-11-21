/**
 * Barra de consulta estilo "command bar": textarea autoajustable, contexto de identidad y
 * versión, contador de caracteres (3..2000, como `QueryRequest`) y botón enviar/detener.
 * Enter envía; Shift+Enter inserta salto de línea; "/" enfoca desde cualquier parte.
 */
import { ArrowUp, Square } from "lucide-react";
import { forwardRef, useImperativeHandle, useLayoutEffect, useRef, useState } from "react";
import { QUERY_MAX_CHARS, QUERY_MIN_CHARS } from "../../api/types";
import { cx } from "../../lib/cx";
import { ROLE_LABEL } from "../../lib/format";
import { useSession } from "../../state/session";
import { Kbd } from "../ui";

export interface ComposerHandle {
  focus(): void;
  setValue(text: string): void;
}

export const Composer = forwardRef<ComposerHandle, { onSubmit: (q: string) => void; large?: boolean }>(
  function Composer({ onSubmit, large = false }, ref) {
    const { user, version, setVersion, busy, cancel } = useSession();
    const otherVersion = version === "18.0" ? "17.0" : "18.0";
    const [value, setValue] = useState("");
    const area = useRef<HTMLTextAreaElement>(null);

    useImperativeHandle(ref, () => ({
      focus: () => area.current?.focus(),
      setValue: (text: string) => {
        setValue(text);
        requestAnimationFrame(() => {
          area.current?.focus();
          area.current?.setSelectionRange(text.length, text.length);
        });
      },
    }));

    // Ajusta la altura al contenido (máx. ~7 líneas) antes de pintar, sin saltos visibles.
    useLayoutEffect(() => {
      const el = area.current;
      if (!el) return;
      el.style.height = "0px";
      el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
    }, [value]);

    const trimmed = value.trim();
    const tooLong = value.length > QUERY_MAX_CHARS;
    const canSend = !busy && trimmed.length >= QUERY_MIN_CHARS && !tooLong;

    const submit = () => {
      if (!canSend) return;
      onSubmit(trimmed);
      setValue("");
    };

    return (
      <div
        className={cx(
          "bezel transition-[box-shadow] duration-200 focus-within:shadow-[0_0_0_1px_rgba(16,185,129,0.35),0_0_32px_-8px_rgba(16,185,129,0.35)]",
        )}
      >
        <div className="bezel-core">
          <label htmlFor="docuerp-query" className="sr-only">
            Consulta para DocuERP
          </label>
          <textarea
            id="docuerp-query"
            ref={area}
            value={value}
            rows={1}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder="Pregunta por un procedimiento, consulta TCK-00042 o pide crear un ticket…"
            className={cx(
              "block w-full resize-none bg-transparent px-4 text-text placeholder:text-faint focus:outline-none",
              large ? "min-h-[64px] pt-4 text-[15px]" : "min-h-[48px] pt-3.5 text-sm",
            )}
          />
          <div className="flex items-center gap-2 px-3 pb-3 pt-1">
            <span className="inline-flex h-6 items-center gap-1.5 rounded-[7px] bg-white/[0.04] px-2 text-2xs text-muted ring-1 ring-inset ring-white/[0.06]">
              <span className="h-1.5 w-1.5 rounded-full bg-accent" aria-hidden />
              {ROLE_LABEL[user.role]}
              <span className="font-mono text-faint">{user.user_id}</span>
            </span>
            <button
              type="button"
              onClick={() => setVersion(otherVersion)}
              aria-label={`Versión Odoo ${version}; cambiar a ${otherVersion}`}
              title={`Cambiar a Odoo ${otherVersion}`}
              className="inline-flex h-6 items-center rounded-[7px] bg-white/[0.04] px-2 font-mono text-2xs text-muted ring-1 ring-inset ring-white/[0.06] transition-[color,background-color,transform] duration-150 hover:bg-white/[0.07] hover:text-text active:scale-[0.97]"
            >
              Odoo {version}
            </button>
            <span className="ml-auto hidden items-center gap-1.5 text-2xs text-faint sm:flex">
              <Kbd>↵</Kbd> enviar <Kbd>⇧↵</Kbd> salto
            </span>
            <span className={cx("font-mono text-2xs", tooLong ? "text-danger" : "text-faint")} aria-live="polite">
              {value.length > 0 ? `${value.length}/${QUERY_MAX_CHARS}` : ""}
            </span>
            {busy ? (
              <button
                type="button"
                onClick={cancel}
                aria-label="Detener consulta"
                className="flex h-8 w-8 items-center justify-center rounded-full bg-white/[0.08] text-text ring-1 ring-inset ring-white/[0.1] transition-transform duration-150 active:scale-95"
              >
                <Square size={12} strokeWidth={2} fill="currentColor" aria-hidden />
              </button>
            ) : (
              <button
                type="button"
                onClick={submit}
                disabled={!canSend}
                aria-label="Enviar consulta"
                className={cx(
                  "flex h-8 w-8 items-center justify-center rounded-full transition-[transform,background-color,opacity,box-shadow] duration-150 ease-out-strong active:scale-95",
                  canSend
                    ? "bg-accent text-[#03140D] shadow-[0_0_0_1px_rgba(16,185,129,0.6),0_6px_18px_-4px_rgba(16,185,129,0.6)] hover:bg-accent-2"
                    : "bg-white/[0.06] text-faint",
                )}
              >
                <ArrowUp size={16} strokeWidth={2} aria-hidden />
              </button>
            )}
          </div>
        </div>
      </div>
    );
  },
);
