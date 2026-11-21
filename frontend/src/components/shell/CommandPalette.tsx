/**
 * Paleta de comandos (⌘K): navegar, lanzar consultas de ejemplo y cambiar rol o versión.
 *
 * Decisión deliberada: abre y cierra SIN animación. Es una acción de teclado que se repite
 * muchas veces al día; animarla la haría sentir lenta (misma decisión que Raycast).
 */
import { CornerDownLeft, Layers, Search, Sparkles, UserRound } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import type { ErpVersion, UserRole } from "../../api/types";
import type { ViewKey } from "../../hooks/useHashRoute";
import { cx } from "../../lib/cx";
import { EXAMPLE_QUERIES } from "../../lib/examples";
import { ROLE_LABEL } from "../../lib/format";
import { NAV_ITEMS } from "../../lib/nav";
import { useSession } from "../../state/session";
import { Kbd } from "../ui";

interface Command {
  id: string;
  group: string;
  label: string;
  hint?: string;
  icon: typeof Search;
  run: () => void;
}

function fold(s: string): string {
  return s
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase();
}

export function CommandPalette({
  open,
  onClose,
  onNavigate,
  onAsk,
}: {
  open: boolean;
  onClose: () => void;
  onNavigate: (view: ViewKey) => void;
  onAsk: (query: string) => void;
}) {
  const { setRole, setVersion, user, version } = useSession();
  const [query, setQuery] = useState("");
  const [index, setIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open) {
      setQuery("");
      setIndex(0);
      inputRef.current?.focus();
    }
  }, [open]);

  const commands = useMemo<Command[]>(() => {
    const nav: Command[] = NAV_ITEMS.map((n) => ({
      id: `nav-${n.key}`,
      group: "Ir a",
      label: n.label,
      hint: `G ${n.shortcut}`,
      icon: n.icon,
      run: () => onNavigate(n.key),
    }));
    const examples: Command[] = EXAMPLE_QUERIES.map((e) => ({
      id: `ex-${e.id}`,
      group: "Consultas de ejemplo",
      label: e.query,
      hint: e.label,
      icon: Sparkles,
      run: () => onAsk(e.query),
    }));
    const roles: Command[] = (["portal_user", "n1", "n2", "admin"] as UserRole[]).map((r) => ({
      id: `role-${r}`,
      group: "Identidad",
      label: `Cambiar a ${ROLE_LABEL[r]}`,
      hint: r === user.role ? "actual" : undefined,
      icon: UserRound,
      run: () => setRole(r),
    }));
    const versions: Command[] = (["17.0", "18.0"] as ErpVersion[]).map((v) => ({
      id: `ver-${v}`,
      group: "Versión",
      label: `Usar Odoo ${v}`,
      hint: v === version ? "actual" : undefined,
      icon: Layers,
      run: () => setVersion(v),
    }));
    return [...nav, ...examples, ...roles, ...versions];
  }, [onAsk, onNavigate, setRole, setVersion, user.role, version]);

  const filtered = useMemo(() => {
    const q = fold(query.trim());
    if (!q) return commands;
    return commands.filter((c) => fold(`${c.label} ${c.hint ?? ""} ${c.group}`).includes(q));
  }, [commands, query]);

  // Si no hay coincidencias, Enter envía el texto como consulta libre.
  const freeQuery = query.trim().length >= 3 && filtered.length === 0;

  useEffect(() => {
    listRef.current?.querySelector(`[data-index="${index}"]`)?.scrollIntoView({ block: "nearest" });
  }, [index]);

  if (!open) return null;

  const execute = (cmd: Command | undefined) => {
    if (!cmd) return;
    onClose();
    cmd.run();
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setIndex((i) => Math.min(filtered.length - 1, i + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setIndex((i) => Math.max(0, i - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (freeQuery) {
        onClose();
        onAsk(query.trim());
      } else execute(filtered[index]);
    } else if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    }
  };

  let lastGroup = "";
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center px-4 pt-[14vh]" role="presentation">
      <div className="absolute inset-0 bg-black/55" onClick={onClose} aria-hidden />
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Paleta de comandos"
        className="relative w-full max-w-[620px] overflow-hidden rounded-[16px] bg-surface-2 shadow-float"
      >
        <div className="flex items-center gap-3 border-b hairline px-4">
          <Search size={16} strokeWidth={1.6} className="text-faint" aria-hidden />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setIndex(0);
            }}
            onKeyDown={onKeyDown}
            placeholder="Busca una vista, un ejemplo o escribe una consulta…"
            aria-label="Buscar comando"
            aria-controls="palette-list"
            aria-activedescendant={filtered[index] ? `cmd-${filtered[index].id}` : undefined}
            className="h-12 flex-1 bg-transparent text-sm text-text placeholder:text-faint focus:outline-none"
          />
          <Kbd>Esc</Kbd>
        </div>
        <div ref={listRef} id="palette-list" role="listbox" className="max-h-[52vh] overflow-y-auto p-1.5">
          {freeQuery && (
            <div className="flex items-center gap-3 rounded-[10px] bg-white/[0.05] px-3 py-2.5 text-sm text-text">
              <Sparkles size={15} strokeWidth={1.6} className="text-accent-2" aria-hidden />
              <span className="flex-1 truncate">Preguntar: «{query.trim()}»</span>
              <CornerDownLeft size={14} className="text-faint" aria-hidden />
            </div>
          )}
          {filtered.map((cmd, i) => {
            const header = cmd.group !== lastGroup ? cmd.group : null;
            lastGroup = cmd.group;
            const Icon = cmd.icon;
            const active = i === index;
            return (
              <div key={cmd.id}>
                {header && <div className="eyebrow px-3 pb-1.5 pt-3">{header}</div>}
                <div
                  id={`cmd-${cmd.id}`}
                  role="option"
                  aria-selected={active}
                  data-index={i}
                  onMouseMove={() => setIndex(i)}
                  onClick={() => execute(cmd)}
                  className={cx(
                    "flex cursor-pointer items-center gap-3 rounded-[10px] px-3 py-2 text-sm",
                    active ? "bg-white/[0.06] text-text" : "text-muted",
                  )}
                >
                  <Icon size={15} strokeWidth={1.6} className={active ? "text-accent-2" : "text-faint"} aria-hidden />
                  <span className="flex-1 truncate">{cmd.label}</span>
                  {cmd.hint && <span className="shrink-0 font-mono text-2xs text-faint">{cmd.hint}</span>}
                </div>
              </div>
            );
          })}
        </div>
        <div className="flex items-center gap-4 border-t hairline px-4 py-2 text-2xs text-faint">
          <span className="flex items-center gap-1.5">
            <Kbd>↑</Kbd>
            <Kbd>↓</Kbd> navegar
          </span>
          <span className="flex items-center gap-1.5">
            <Kbd>↵</Kbd> ejecutar
          </span>
          <span className="ml-auto">Rol y versión se aplican a la próxima consulta</span>
        </div>
      </div>
    </div>
  );
}
