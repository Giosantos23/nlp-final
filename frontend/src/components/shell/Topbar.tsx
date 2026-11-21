/**
 * Barra superior: título de la vista, selector de versión del ERP, selector de rol con
 * usuarios demo y disparador de la paleta de comandos (⌘K).
 */
import { AnimatePresence, motion } from "framer-motion";
import { Check, ChevronDown, Command, Menu, Search } from "lucide-react";
import type { ErpVersion, UserRole } from "../../api/types";
import { usePopover } from "../../hooks/usePopover";
import { cx } from "../../lib/cx";
import { ROLE_LABEL, ROLE_SHORT } from "../../lib/format";
import { useSession } from "../../state/session";
import { IconButton, Kbd } from "../ui";

const ROLE_SCOPE: Record<UserRole, string> = {
  portal_user: "Documentación pública · sus tickets",
  n1: "+ Runbooks N1 y SLA · tickets asignados",
  n2: "+ Runbooks N2 · cartera de cuentas",
  admin: "Todo el corpus · todos los tickets",
};

const ROLES: UserRole[] = ["portal_user", "n1", "n2", "admin"];

function initials(name: string): string {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase() ?? "")
    .join("");
}

function VersionSwitch() {
  const { version, setVersion } = useSession();
  const versions: ErpVersion[] = ["17.0", "18.0"];
  return (
    <div
      role="radiogroup"
      aria-label="Versión de Odoo"
      className="relative flex h-8 items-center rounded-[10px] bg-white/[0.03] p-0.5 ring-1 ring-inset ring-white/[0.07]"
    >
      {versions.map((v) => {
        const active = v === version;
        return (
          <button
            key={v}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => setVersion(v)}
            className={cx(
              "relative h-7 rounded-[8px] px-2.5 font-mono text-xs transition-colors duration-150",
              active ? "text-text" : "text-faint hover:text-muted",
            )}
          >
            {active && (
              <motion.span
                layoutId="version-pill"
                className="absolute inset-0 rounded-[8px] bg-white/[0.07] shadow-inner-hl ring-1 ring-inset ring-white/[0.08]"
                transition={{ type: "spring", duration: 0.28, bounce: 0.1 }}
              />
            )}
            <span className="relative">Odoo {v}</span>
          </button>
        );
      })}
    </div>
  );
}

function RoleMenu() {
  const { user, users, setUser } = useSession();
  const pop = usePopover<HTMLDivElement>();
  return (
    <div ref={pop.ref} className="relative">
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={pop.open}
        onClick={pop.toggle}
        className={cx(
          "flex h-8 items-center gap-2 rounded-[10px] pl-1 pr-2 ring-1 ring-inset ring-white/[0.07]",
          "bg-white/[0.03] transition-[background-color,transform] duration-150 hover:bg-white/[0.06] active:scale-[0.98]",
        )}
      >
        <span className="flex h-6 w-6 items-center justify-center rounded-[7px] bg-gradient-to-b from-accent/30 to-accent/10 text-[10px] font-semibold text-accent-2 ring-1 ring-inset ring-accent/30">
          {initials(user.name)}
        </span>
        <span className="hidden text-xs text-text sm:inline">{user.name}</span>
        <span className="rounded-[5px] bg-white/[0.06] px-1.5 py-px font-mono text-[10.5px] text-muted">
          {ROLE_SHORT[user.role]}
        </span>
        <ChevronDown size={14} strokeWidth={1.6} className="text-faint" aria-hidden />
      </button>
      <AnimatePresence>
        {pop.open && (
          <motion.div
            role="menu"
            aria-label="Cambiar identidad simulada"
            initial={{ opacity: 0, transform: "scale(0.97) translateY(-4px)" }}
            animate={{ opacity: 1, transform: "scale(1) translateY(0px)" }}
            exit={{ opacity: 0, transform: "scale(0.98) translateY(-2px)", transition: { duration: 0.1 } }}
            transition={{ duration: 0.16, ease: [0.23, 1, 0.32, 1] }}
            style={{ transformOrigin: "top right" }}
            className="absolute right-0 top-10 z-30 w-[320px] rounded-[14px] bg-surface-2 p-1.5 shadow-float"
          >
            <div className="px-2.5 pb-2 pt-1.5">
              <div className="eyebrow">Identidad simulada · RBAC</div>
              <p className="mt-1 text-2xs text-faint">El rol filtra documentos y herramientas en el servidor.</p>
            </div>
            {ROLES.map((role) => {
              const u = users.find((x) => x.role === role);
              if (!u) return null;
              const active = u.user_id === user.user_id;
              return (
                <button
                  key={u.user_id}
                  type="button"
                  role="menuitemradio"
                  aria-checked={active}
                  onClick={() => {
                    setUser(u.user_id);
                    pop.close();
                  }}
                  className={cx(
                    "flex w-full items-start gap-3 rounded-[10px] px-2.5 py-2 text-left transition-colors duration-150",
                    active ? "bg-white/[0.05]" : "hover:bg-white/[0.035]",
                  )}
                >
                  <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-[8px] bg-white/[0.05] text-[10.5px] font-semibold text-muted ring-1 ring-inset ring-white/[0.07]">
                    {initials(u.name)}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-2">
                      <span className="text-sm text-text">{ROLE_LABEL[role]}</span>
                      <span className="font-mono text-2xs text-faint">{u.user_id}</span>
                    </span>
                    <span className="block truncate text-2xs text-muted">
                      {u.name}
                      {u.account ? ` · ${u.account}` : ""}
                    </span>
                    <span className="mt-0.5 block text-2xs text-faint">{ROLE_SCOPE[role]}</span>
                  </span>
                  {active && <Check size={14} strokeWidth={2} className="mt-1 text-accent-2" aria-hidden />}
                </button>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function Topbar({
  title,
  subtitle,
  onOpenPalette,
  onOpenNav,
}: {
  title: string;
  subtitle: string;
  onOpenPalette: () => void;
  onOpenNav: () => void;
}) {
  return (
    <header className="sticky top-0 z-20 border-b hairline bg-bg/75 backdrop-blur-xl">
      <div className="flex h-14 items-center gap-3 px-4 md:px-6">
        <IconButton icon={Menu} label="Abrir navegación" onClick={onOpenNav} className="lg:hidden" />
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-2.5">
            <h1 className="truncate text-sm font-medium text-text">{title}</h1>
            <span className="hidden truncate text-xs text-faint md:inline">{subtitle}</span>
          </div>
        </div>
        <button
          type="button"
          onClick={onOpenPalette}
          className="hidden h-8 w-[220px] items-center gap-2 rounded-[10px] bg-white/[0.03] px-2.5 text-xs text-faint ring-1 ring-inset ring-white/[0.07] transition-colors duration-150 hover:text-muted xl:flex"
        >
          <Search size={14} strokeWidth={1.6} aria-hidden />
          <span className="flex-1 text-left">Buscar o ejecutar…</span>
          <Kbd>⌘</Kbd>
          <Kbd>K</Kbd>
        </button>
        <IconButton icon={Command} label="Paleta de comandos (⌘K)" onClick={onOpenPalette} className="xl:hidden" />
        <div className="hidden sm:block">
          <VersionSwitch />
        </div>
        <RoleMenu />
      </div>
    </header>
  );
}
