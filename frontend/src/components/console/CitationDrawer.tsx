/**
 * Panel lateral de una cita: metadatos del chunk (sección, versión, vigencia, roles, hash)
 * con su texto resaltado y el PDF renderizado en la página citada.
 *
 * Entra deslizándose desde la derecha (curva tipo cajón de iOS, 280 ms) y sale más rápido
 * (200 ms): el sistema responde rápido al cerrar.
 */
import { AnimatePresence, motion } from "framer-motion";
import { ChevronLeft, ChevronRight, FileText, Hash, X } from "lucide-react";
import { Suspense, lazy, useEffect, useState } from "react";
import { ApiError, api } from "../../api/client";
import type { Chunk, SourceRef, UserRole } from "../../api/types";
import { MODULE_LABEL, ROLE_SHORT, formatDateTime, labelOf, shortHash } from "../../lib/format";
import { Badge, IconButton, Skeleton } from "../ui";

const PdfViewer = lazy(() => import("./PdfViewer"));

type ChunkState = { status: "loading" } | { status: "ok"; chunk: Chunk } | { status: "error"; message: string };

function useChunk(chunkId: string | null, role: UserRole): ChunkState {
  const [state, setState] = useState<ChunkState>({ status: "loading" });
  useEffect(() => {
    if (!chunkId) return;
    let alive = true;
    setState({ status: "loading" });
    api
      .chunk(chunkId, role)
      .then((chunk) => alive && setState({ status: "ok", chunk }))
      .catch((e: unknown) =>
        alive && setState({ status: "error", message: e instanceof ApiError ? e.message : "No se pudo cargar el fragmento." }),
      );
    return () => {
      alive = false;
    };
  }, [chunkId, role]);
  return state;
}

function useFallbackUrl(source: string | null): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!source) return;
    let alive = true;
    api.fallbackFileUrl(source).then((u) => alive && setUrl(u)).catch(() => alive && setUrl(null));
    return () => {
      alive = false;
    };
  }, [source]);
  return url;
}

function ChunkCard({ state, source }: { state: ChunkState; source: SourceRef }) {
  if (state.status === "loading") {
    return (
      <div className="space-y-2 rounded-[12px] bg-white/[0.025] p-4 ring-1 ring-inset ring-white/[0.06]">
        <Skeleton className="h-3 w-1/2" />
        <Skeleton className="h-3 w-full" />
        <Skeleton className="h-3 w-[90%]" />
        <Skeleton className="h-3 w-[70%]" />
      </div>
    );
  }
  if (state.status === "error") {
    return (
      <div className="rounded-[12px] bg-warning/[0.06] p-4 text-xs text-muted ring-1 ring-inset ring-warning/20">
        {state.message}
        <div className="mt-2 font-mono text-2xs text-faint">{source.chunk_id}</div>
      </div>
    );
  }
  const c = state.chunk;
  return (
    <div className="rounded-[12px] bg-white/[0.025] ring-1 ring-inset ring-white/[0.06]">
      <div className="border-b hairline px-4 py-3">
        <div className="text-2xs text-faint">Sección</div>
        <div className="text-xs text-text">{c.section_path}</div>
      </div>
      <blockquote className="relative mx-4 my-4 border-l-2 border-accent/70 bg-accent/[0.06] py-2.5 pl-3.5 pr-3 text-[13px] leading-relaxed text-text/90">
        {c.text}
      </blockquote>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-2.5 px-4 pb-4 text-xs">
        <div>
          <dt className="text-2xs text-faint">Versión</dt>
          <dd className="font-mono text-muted">{c.version}</dd>
        </div>
        <div>
          <dt className="text-2xs text-faint">Páginas</dt>
          <dd className="font-mono text-muted">{c.page === c.page_end ? c.page : `${c.page}–${c.page_end}`}</dd>
        </div>
        <div>
          <dt className="text-2xs text-faint">Tokens</dt>
          <dd className="font-mono text-muted">{c.token_count}</dd>
        </div>
        <div>
          <dt className="text-2xs text-faint">Ingestado</dt>
          <dd className="text-muted">{formatDateTime(c.ingested_at)}</dd>
        </div>
        <div className="col-span-2">
          <dt className="text-2xs text-faint">Roles autorizados</dt>
          <dd className="mt-1 flex flex-wrap gap-1">
            {c.allowed_roles.map((r) => (
              <Badge key={r} mono>
                {ROLE_SHORT[r]}
              </Badge>
            ))}
          </dd>
        </div>
        <div className="col-span-2 flex items-center gap-1.5 font-mono text-2xs text-faint" title={c.content_hash}>
          <Hash size={11} strokeWidth={1.8} aria-hidden />
          sha256 {shortHash(c.content_hash, 12, 6)}
        </div>
      </dl>
    </div>
  );
}

export function CitationDrawer({
  sources,
  role,
  index,
  onClose,
  onChange,
}: {
  sources: SourceRef[];
  /** Rol de la consulta que produjo las citas: la API revalida el RBAC al abrirlas. */
  role: UserRole;
  index: number | null;
  onClose: () => void;
  onChange: (i: number) => void;
}) {
  const source = index !== null ? sources[index] ?? null : null;
  const chunkState = useChunk(source?.chunk_id ?? null, role);
  const fallback = useFallbackUrl(source?.source ?? null);
  const chunk = chunkState.status === "ok" ? chunkState.chunk : null;

  useEffect(() => {
    if (index === null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [index, onClose]);

  const isDocx = source ? /\.docx?$/i.test(source.source) : false;
  const canPreview = source ? api.canPreview(source.source) : false;

  return (
    <AnimatePresence>
      {source && index !== null && (
        <>
          <motion.div
            key="scrim"
            className="fixed inset-0 z-30 bg-black/40"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, transition: { duration: 0.15 } }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
            aria-hidden
          />
          <motion.aside
            key="drawer"
            role="dialog"
            aria-modal="true"
            aria-label={`Cita ${index + 1}: ${source.source}`}
            className="fixed inset-y-0 right-0 z-40 flex w-full max-w-[600px] flex-col border-l hairline bg-surface shadow-float"
            initial={{ transform: "translateX(100%)" }}
            animate={{ transform: "translateX(0%)" }}
            exit={{ transform: "translateX(100%)", transition: { duration: 0.2, ease: [0.32, 0.72, 0, 1] } }}
            transition={{ duration: 0.28, ease: [0.32, 0.72, 0, 1] }}
          >
            <header className="flex items-start gap-3 border-b hairline px-5 py-4">
              <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-[9px] bg-accent/15 text-accent-2 ring-1 ring-inset ring-accent/30">
                <FileText size={16} strokeWidth={1.6} aria-hidden />
              </span>
              <div className="min-w-0 flex-1">
                <div className="eyebrow">Cita {index + 1} de {sources.length}</div>
                <div className="truncate text-sm font-medium text-text">{source.source}</div>
                <div className="mt-1 flex flex-wrap items-center gap-1.5">
                  <Badge tone="accent" mono>
                    pág. {source.page}
                  </Badge>
                  {chunk && (
                    <>
                      <Badge tone={chunk.status === "vigente" ? "neutral" : "warning"}>{chunk.status}</Badge>
                      <Badge mono>{chunk.erp_version === "any" ? "todas las versiones" : `Odoo ${chunk.erp_version}`}</Badge>
                      <Badge>{labelOf(MODULE_LABEL, chunk.module)}</Badge>
                    </>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-0.5">
                <IconButton
                  icon={ChevronLeft}
                  label="Cita anterior"
                  disabled={index === 0}
                  onClick={() => onChange(index - 1)}
                  className="disabled:opacity-30"
                />
                <IconButton
                  icon={ChevronRight}
                  label="Cita siguiente"
                  disabled={index >= sources.length - 1}
                  onClick={() => onChange(index + 1)}
                  className="disabled:opacity-30"
                />
                <IconButton icon={X} label="Cerrar (Esc)" onClick={onClose} />
              </div>
            </header>

            <div className="flex-1 space-y-5 overflow-y-auto px-5 py-5">
              <ChunkCard state={chunkState} source={source} />
              <div className="font-mono text-2xs text-faint">{source.chunk_id}</div>
              {canPreview ? (
                <>
                  {isDocx && (
                    <p className="flex flex-wrap items-center gap-x-1.5 text-2xs leading-4 text-faint">
                      Vista PDF generada desde la misma fuente que el DOCX, con su paginación: portada en la pág. 1 y
                      una sección por página.
                      <a
                        className="text-accent-2 underline-offset-2 hover:underline"
                        href={api.documentFileUrl(source.source, role, { original: true })}
                      >
                        Descargar el DOCX original
                      </a>
                    </p>
                  )}
                  <Suspense fallback={<Skeleton className="aspect-[1/1.414] w-full" />}>
                    <PdfViewer
                      url={api.documentFileUrl(source.source, role)}
                      fallbackUrl={fallback}
                      page={source.page}
                      highlight={chunk?.text ?? null}
                    />
                  </Suspense>
                </>
              ) : (
                <div className="rounded-[12px] bg-white/[0.025] p-4 text-xs text-muted ring-1 ring-inset ring-white/[0.06]">
                  La vista previa solo está disponible para PDF.{" "}
                  <a className="text-accent-2 underline-offset-2 hover:underline" href={api.documentFileUrl(source.source, role)}>
                    Descargar {source.source}
                  </a>
                </div>
              )}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
