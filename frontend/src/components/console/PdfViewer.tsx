/**
 * Visor PDF (react-pdf / pdf.js) que abre la página citada y resalta el texto del chunk
 * sobre la capa de texto. Se carga de forma diferida: pdf.js pesa y solo hace falta al
 * abrir una cita.
 *
 * Seguridad: `customTextRenderer` devuelve HTML, así que el texto extraído del PDF se
 * escapa antes de envolverlo en <mark>. Un PDF malicioso no puede inyectar marcado.
 */
import { ChevronLeft, ChevronRight, ExternalLink, FileWarning } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/esm/Page/TextLayer.css";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import { chunkRange, foldForMatch, isPartOfChunk } from "../../lib/answerText";
import { EmptyState, IconButton, Skeleton } from "../ui";

pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] ?? c);
}

export default function PdfViewer({
  url,
  fallbackUrl,
  page,
  highlight,
}: {
  url: string;
  /** URL alternativa (PDF de demo) si la principal falla. */
  fallbackUrl: string | null;
  page: number;
  highlight: string | null;
}) {
  const [src, setSrc] = useState(url);
  const [failed, setFailed] = useState(false);
  const [pages, setPages] = useState<number | null>(null);
  const [current, setCurrent] = useState(page);
  const [width, setWidth] = useState(520);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setSrc(url);
    setFailed(false);
    setPages(null);
  }, [url]);
  useEffect(() => setCurrent(page), [page, url]);

  // El ancho de la página sigue al contenedor (cajón redimensionable / móvil).
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      if (entry) setWidth(Math.max(240, Math.floor(entry.contentRect.width)));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const folded = useMemo(() => (highlight ? foldForMatch(highlight) : ""), [highlight]);
  // Rango contiguo de renglones del chunk en la página citada (se calcula al leer el texto).
  const [range, setRange] = useState<[number, number] | null>(null);
  useEffect(() => setRange(null), [src, current, folded]);
  const onText = useCallback(
    (content: { items: unknown[] }) => {
      if (current !== page || !folded) return;
      const strs = content.items.map((it) => (it && typeof it === "object" && "str" in it ? String(it.str) : ""));
      setRange(chunkRange(strs, folded));
    },
    [current, page, folded],
  );
  const renderText = useCallback(
    ({ str, itemIndex }: { str: string; itemIndex: number }) => {
      const safe = escapeHtml(str);
      if (current !== page || !folded || !str.trim()) return safe;
      const hit = range ? itemIndex >= range[0] && itemIndex <= range[1] : isPartOfChunk(str, folded);
      return hit ? `<mark>${safe}</mark>` : safe;
    },
    [folded, current, page, range],
  );

  const onError = () => {
    if (fallbackUrl && src !== fallbackUrl) setSrc(fallbackUrl);
    else setFailed(true);
  };

  if (failed) {
    return (
      <EmptyState
        icon={FileWarning}
        tone="warning"
        title="No se pudo abrir el documento"
        description="El archivo fuente no está disponible en este entorno. El fragmento citado se muestra arriba."
      />
    );
  }

  return (
    <div>
      <div className="mb-3 flex items-center gap-1">
        <IconButton
          icon={ChevronLeft}
          label="Página anterior"
          disabled={current <= 1}
          onClick={() => setCurrent((p) => Math.max(1, p - 1))}
          className="disabled:opacity-30"
        />
        <span className="min-w-[96px] text-center font-mono text-xs text-muted">
          pág. {current}
          {pages ? ` / ${pages}` : ""}
        </span>
        <IconButton
          icon={ChevronRight}
          label="Página siguiente"
          disabled={pages !== null && current >= pages}
          onClick={() => setCurrent((p) => (pages ? Math.min(pages, p + 1) : p + 1))}
          className="disabled:opacity-30"
        />
        {current !== page && (
          <button
            type="button"
            onClick={() => setCurrent(page)}
            className="ml-1 rounded-md px-2 py-1 text-2xs text-accent-2 transition-colors duration-150 hover:bg-accent/10"
          >
            Volver a la cita
          </button>
        )}
        <a
          href={src}
          target="_blank"
          rel="noreferrer"
          className="ml-auto inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-2xs text-muted transition-colors duration-150 hover:text-text"
        >
          Abrir PDF <ExternalLink size={12} strokeWidth={1.8} aria-hidden />
        </a>
      </div>
      <div ref={box} className="pdf-page overflow-hidden rounded-[10px] bg-white ring-1 ring-white/10">
        <Document
          file={src}
          onLoadSuccess={(doc) => setPages(doc.numPages)}
          onLoadError={onError}
          onSourceError={onError}
          loading={<Skeleton className="aspect-[1/1.414] w-full rounded-none" />}
          error={<div className="aspect-[1/1.414] w-full bg-surface-2" />}
        >
          <Page
            key={`${src}-${current}`}
            pageNumber={current}
            width={width}
            renderAnnotationLayer={false}
            customTextRenderer={renderText}
            onGetTextSuccess={onText}
            onRenderTextLayerSuccess={() => {
              // Lleva a la vista el primer renglón resaltado (el chunk puede empezar a media página).
              if (current !== page) return;
              const first = box.current?.querySelector(".textLayer mark");
              first?.scrollIntoView({ block: "center", behavior: "smooth" });
            }}
            loading={<Skeleton className="aspect-[1/1.414] w-full rounded-none" />}
          />
        </Document>
      </div>
    </div>
  );
}
