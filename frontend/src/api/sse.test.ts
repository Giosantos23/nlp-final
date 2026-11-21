import { describe, expect, it } from "vitest";
import { SseParser } from "./sse";

describe("SseParser", () => {
  it("separa mensajes trace y final completos", () => {
    const p = new SseParser();
    const out = p.push('event: trace\ndata: {"stage":"router"}\n\nevent: final\ndata: {"ok":true}\n\n');
    expect(out).toEqual([
      { event: "trace", data: '{"stage":"router"}' },
      { event: "final", data: '{"ok":true}' },
    ]);
  });

  it("acumula trozos que cortan una línea a la mitad", () => {
    const p = new SseParser();
    expect(p.push("event: tra")).toEqual([]);
    expect(p.push('ce\ndata: {"a"')).toEqual([]);
    expect(p.push(":1}\n\n")).toEqual([{ event: "trace", data: '{"a":1}' }]);
  });

  it("normaliza CRLF, une varias líneas data e ignora comentarios", () => {
    const p = new SseParser();
    const out = p.push(": latido\r\nevent: final\r\ndata: linea1\r\ndata: linea2\r\n\r\n");
    expect(out).toEqual([{ event: "final", data: "linea1\nlinea2" }]);
  });

  it("usa 'message' si no hay campo event y descarta bloques sin data", () => {
    const p = new SseParser();
    expect(p.push("data: hola\n\nevent: vacio\n\n")).toEqual([{ event: "message", data: "hola" }]);
  });

  it("flush entrega el último mensaje sin línea vacía final", () => {
    const p = new SseParser();
    expect(p.push("event: final\ndata: x")).toEqual([]);
    expect(p.flush()).toEqual([{ event: "final", data: "x" }]);
    expect(p.flush()).toEqual([]);
  });
});
