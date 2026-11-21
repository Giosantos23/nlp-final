/**
 * Tokenizador de JSON para el resaltado de sintaxis del inspector.
 *
 * Recibe el texto ya serializado (`JSON.stringify(x, null, 2)`) y lo corta en fragmentos
 * tipados; el componente los pinta con colores de los tokens de diseño. Ninguna parte del
 * contenido se inserta como HTML.
 */
export type JsonTokenKind = "key" | "string" | "number" | "boolean" | "null" | "punct" | "space";

export interface JsonToken {
  kind: JsonTokenKind;
  text: string;
}

const PATTERN =
  /("(?:\\.|[^"\\])*")(\s*:)?|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|\b(true|false)\b|\b(null)\b|([{}[\],:])|(\s+)/g;

export function tokenizeJson(text: string): JsonToken[] {
  const out: JsonToken[] = [];
  for (const m of text.matchAll(PATTERN)) {
    if (m[1] !== undefined) {
      if (m[2] !== undefined) {
        out.push({ kind: "key", text: m[1] });
        out.push({ kind: "punct", text: m[2] });
      } else {
        out.push({ kind: "string", text: m[1] });
      }
    } else if (m[3] !== undefined) out.push({ kind: "number", text: m[3] });
    else if (m[4] !== undefined) out.push({ kind: "boolean", text: m[4] });
    else if (m[5] !== undefined) out.push({ kind: "null", text: m[5] });
    else if (m[6] !== undefined) out.push({ kind: "punct", text: m[6] });
    else if (m[7] !== undefined) out.push({ kind: "space", text: m[7] });
  }
  return out;
}
