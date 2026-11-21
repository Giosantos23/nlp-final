/**
 * Mini-parser del texto de `Answer.answer` → bloques e inline tokens para React.
 *
 * Soporta lo que produce el generador: párrafos, listas numeradas y con viñetas, **negrita**,
 * *cursiva*, `código` y marcadores de cita `[1]` o `[chunk_id]`. No se usa `innerHTML`: el
 * texto del modelo nunca se interpreta como HTML (defensa contra XSS en la salida).
 */
import type { SourceRef } from "../api/types";

export type Inline =
  | { kind: "text"; text: string }
  | { kind: "bold"; text: string }
  | { kind: "italic"; text: string }
  | { kind: "code"; text: string }
  /** `index` es 0-based sobre `sources`. */
  | { kind: "cite"; index: number };

export type Block =
  | { kind: "paragraph"; inlines: Inline[] }
  | { kind: "ordered"; items: Inline[][] }
  | { kind: "bullets"; items: Inline[][] };

const CHUNK_ID = /^.+::p\d{4}::c\d{3}$/;
const TOKEN = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\[[^\]\n]+\])/g;

/** Convierte el contenido de un `[...]` en índices de cita, o `null` si no es una cita. */
function citationIndices(inner: string, sources: SourceRef[]): number[] | null {
  const parts = inner.split(/\s*[,;]\s*/);
  const out: number[] = [];
  for (const part of parts) {
    if (/^\d{1,2}$/.test(part)) {
      const n = Number(part) - 1;
      if (n < 0 || n >= sources.length) return null;
      out.push(n);
    } else if (CHUNK_ID.test(part)) {
      const idx = sources.findIndex((s) => s.chunk_id === part);
      if (idx === -1) return null;
      out.push(idx);
    } else {
      return null;
    }
  }
  return out;
}

/** Tokeniza una línea en fragmentos de formato y citas. */
export function parseInline(text: string, sources: SourceRef[]): Inline[] {
  const out: Inline[] = [];
  let last = 0;
  for (const m of text.matchAll(TOKEN)) {
    const token = m[0];
    const at = m.index ?? 0;
    if (at > last) out.push({ kind: "text", text: text.slice(last, at) });
    if (token.startsWith("**")) out.push({ kind: "bold", text: token.slice(2, -2) });
    else if (token.startsWith("`")) out.push({ kind: "code", text: token.slice(1, -1) });
    else if (token.startsWith("[")) {
      const idx = citationIndices(token.slice(1, -1), sources);
      if (idx) for (const i of idx) out.push({ kind: "cite", index: i });
      else out.push({ kind: "text", text: token });
    } else out.push({ kind: "italic", text: token.slice(1, -1) });
    last = at + token.length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  // Une textos contiguos y quita el espacio antes de una cita ("paso [1]" → "paso¹").
  const merged: Inline[] = [];
  for (const item of out) {
    const prev = merged[merged.length - 1];
    if (item.kind === "text" && prev?.kind === "text") prev.text += item.text;
    else merged.push(item);
  }
  return merged
    .map((item, i) => {
      const next = merged[i + 1];
      return item.kind === "text" && next?.kind === "cite" ? { ...item, text: item.text.replace(/\s+$/, "") } : item;
    })
    .filter((item) => item.kind !== "text" || item.text.length > 0);
}

/** Divide el texto en bloques (párrafos y listas). */
export function parseAnswer(text: string, sources: SourceRef[]): Block[] {
  const blocks: Block[] = [];
  let paragraph: string[] = [];
  const flush = () => {
    if (paragraph.length) {
      blocks.push({ kind: "paragraph", inlines: parseInline(paragraph.join(" "), sources) });
      paragraph = [];
    }
  };
  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();
    const ordered = /^\d+[.)]\s+(.*)$/.exec(line);
    const bullet = /^[-*•]\s+(.*)$/.exec(line);
    if (!line) {
      flush();
    } else if (ordered?.[1] !== undefined) {
      flush();
      const prev = blocks[blocks.length - 1];
      const item = parseInline(ordered[1], sources);
      if (prev?.kind === "ordered") prev.items.push(item);
      else blocks.push({ kind: "ordered", items: [item] });
    } else if (bullet?.[1] !== undefined) {
      flush();
      const prev = blocks[blocks.length - 1];
      const item = parseInline(bullet[1], sources);
      if (prev?.kind === "bullets") prev.items.push(item);
      else blocks.push({ kind: "bullets", items: [item] });
    } else {
      paragraph.push(line);
    }
  }
  flush();
  return blocks;
}

/** Normaliza texto para comparar fragmentos del PDF con el chunk (sin acentos ni espacios extra). */
export function foldForMatch(s: string): string {
  return s
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/[‐-―]/g, "-")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

/**
 * Decide si un fragmento de la capa de texto del PDF pertenece al chunk citado.
 * Los fragmentos muy cortos (≤ 3 caracteres) se ignoran para evitar falsos positivos.
 */
export function isPartOfChunk(item: string, foldedChunk: string): boolean {
  const f = foldForMatch(item);
  return f.length > 3 && foldedChunk.includes(f);
}

/** Largo mínimo (normalizado) de un renglón para servir de ancla del resaltado. */
const ANCHOR_MIN_CHARS = 16;

/**
 * Rango contiguo `[inicio, fin]` de elementos de la capa de texto que cubre el chunk.
 *
 * Comparar elemento por elemento marca también palabras sueltas (cursivas, términos en
 * negrita) que aparecen en el chunk pero están en otro párrafo. Por eso se toman como
 * anclas solo los renglones largos que están en el chunk y se resalta todo lo que hay entre
 * la primera y la última ancla. Devuelve `null` si no hay anclas (se usa la comparación
 * simple).
 */
export function chunkRange(items: string[], foldedChunk: string): [number, number] | null {
  let start = -1;
  let end = -1;
  items.forEach((item, i) => {
    const f = foldForMatch(item);
    if (f.length >= ANCHOR_MIN_CHARS && foldedChunk.includes(f)) {
      if (start === -1) start = i;
      end = i;
    }
  });
  return start === -1 ? null : [start, end];
}
