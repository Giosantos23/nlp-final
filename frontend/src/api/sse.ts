/**
 * Parser incremental de Server-Sent Events (SSE) para `POST /api/query/stream`.
 *
 * `EventSource` del navegador solo admite GET, así que leemos el cuerpo de un `fetch` POST
 * como flujo de bytes y lo cortamos en mensajes SSE. Formato (SPEC §11):
 *
 *     event: trace          ← un TraceEvent JSON por etapa completada
 *     data: {...}
 *
 *     event: final          ← el QueryResponse completo al terminar
 *     data: {...}
 *
 * Un mensaje termina con una línea vacía. Puede haber varias líneas `data:` (se unen con
 * salto de línea) y comentarios que empiezan con `:` (latidos), que se ignoran.
 */

export interface SseMessage {
  /** Nombre del evento (`message` si el servidor no envía `event:`). */
  event: string;
  /** Contenido de las líneas `data:` unidas con "\n". */
  data: string;
}

/**
 * Acumulador con estado: recibe trozos de texto arbitrarios (pueden cortar una línea a la
 * mitad) y devuelve los mensajes completos que ya se pueden procesar.
 */
export class SseParser {
  private buffer = "";

  /** Agrega un trozo recibido y devuelve los mensajes que quedaron completos. */
  push(chunk: string): SseMessage[] {
    // Normalizamos CRLF y CR sueltos a LF para tratar un solo tipo de salto de línea.
    this.buffer += chunk.replace(/\r\n?/g, "\n");
    const messages: SseMessage[] = [];
    let boundary = this.buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const raw = this.buffer.slice(0, boundary);
      this.buffer = this.buffer.slice(boundary + 2);
      const parsed = parseBlock(raw);
      if (parsed) messages.push(parsed);
      boundary = this.buffer.indexOf("\n\n");
    }
    return messages;
  }

  /** Procesa lo que quede en el búfer al cerrar el flujo (último mensaje sin línea vacía). */
  flush(): SseMessage[] {
    const rest = this.buffer.trim();
    this.buffer = "";
    if (!rest) return [];
    const parsed = parseBlock(rest);
    return parsed ? [parsed] : [];
  }
}

/** Convierte un bloque de líneas SSE en un mensaje; devuelve `null` si no trae `data`. */
function parseBlock(block: string): SseMessage | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    if (line === "" || line.startsWith(":")) continue; // comentario o latido
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    // Según la especificación SSE se elimina un único espacio después de los dos puntos.
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value || "message";
    else if (field === "data") dataLines.push(value);
  }
  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join("\n") };
}
