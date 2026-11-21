/**
 * Renderiza el texto de la respuesta con formato ligero y marcadores de cita clicables.
 * El texto se interpreta con `parseAnswer` (nunca como HTML).
 */
import { Fragment, useMemo } from "react";
import type { SourceRef } from "../../api/types";
import { parseAnswer, type Inline } from "../../lib/answerText";

function CiteMark({ index, onOpen, source }: { index: number; onOpen: (i: number) => void; source?: SourceRef }) {
  return (
    <button
      type="button"
      onClick={() => onOpen(index)}
      aria-label={source ? `Abrir cita ${index + 1}: ${source.source}, página ${source.page}` : `Cita ${index + 1}`}
      className="relative -top-[0.45em] ml-0.5 inline-flex h-[17px] min-w-[17px] items-center justify-center rounded-[5px] bg-accent/15 px-1 font-mono text-[10px] font-medium leading-none text-accent-2 ring-1 ring-inset ring-accent/30 transition-[background-color,transform] duration-150 hover:bg-accent/25 active:scale-95"
    >
      {index + 1}
    </button>
  );
}

function renderInlines(inlines: Inline[], sources: SourceRef[], onOpen: (i: number) => void) {
  return inlines.map((item, i) => {
    switch (item.kind) {
      case "text":
        return <Fragment key={i}>{item.text}</Fragment>;
      case "bold":
        return (
          <strong key={i} className="font-medium text-text">
            {item.text}
          </strong>
        );
      case "italic":
        return (
          <em key={i} className="not-italic text-text">
            {item.text}
          </em>
        );
      case "code":
        return (
          <code key={i} className="rounded-[5px] bg-white/[0.06] px-1.5 py-0.5 font-mono text-[12.5px] text-accent-2/90">
            {item.text}
          </code>
        );
      case "cite":
        return <CiteMark key={i} index={item.index} onOpen={onOpen} source={sources[item.index]} />;
    }
  });
}

export function AnswerBody({
  text,
  sources,
  onOpenCitation,
}: {
  text: string;
  sources: SourceRef[];
  onOpenCitation: (index: number) => void;
}) {
  const blocks = useMemo(() => parseAnswer(text, sources), [text, sources]);
  return (
    <div className="space-y-3 text-[14.5px] leading-[1.7] text-text/85">
      {blocks.map((block, i) => {
        if (block.kind === "paragraph") return <p key={i}>{renderInlines(block.inlines, sources, onOpenCitation)}</p>;
        if (block.kind === "ordered") {
          return (
            <ol key={i} className="space-y-2">
              {block.items.map((item, j) => (
                <li key={j} className="flex gap-3">
                  <span className="mt-[3px] flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-white/[0.05] font-mono text-[10.5px] text-muted ring-1 ring-inset ring-white/[0.08]">
                    {j + 1}
                  </span>
                  <span className="min-w-0">{renderInlines(item, sources, onOpenCitation)}</span>
                </li>
              ))}
            </ol>
          );
        }
        return (
          <ul key={i} className="space-y-2">
            {block.items.map((item, j) => (
              <li key={j} className="flex gap-3">
                <span className="mt-[11px] h-1 w-1 shrink-0 rounded-full bg-accent/70" aria-hidden />
                <span className="min-w-0">{renderInlines(item, sources, onOpenCitation)}</span>
              </li>
            ))}
          </ul>
        );
      })}
    </div>
  );
}
