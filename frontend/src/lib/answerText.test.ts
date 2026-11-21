import { describe, expect, it } from "vitest";
import type { SourceRef } from "../api/types";
import { chunkRange, foldForMatch, isPartOfChunk, parseAnswer, parseInline } from "./answerText";

const sources: SourceRef[] = [
  { source: "a.pdf", page: 1, chunk_id: "a::p0001::c000" },
  { source: "b.pdf", page: 4, chunk_id: "b::p0004::c002" },
];

describe("parseInline", () => {
  it("reconoce negrita, cursiva, código y citas numéricas", () => {
    expect(parseInline("Usa **Ajustes** y *Almacén* con `WH/Stock` [2].", sources)).toEqual([
      { kind: "text", text: "Usa " },
      { kind: "bold", text: "Ajustes" },
      { kind: "text", text: " y " },
      { kind: "italic", text: "Almacén" },
      { kind: "text", text: " con " },
      { kind: "code", text: "WH/Stock" },
      { kind: "cite", index: 1 },
      { kind: "text", text: "." },
    ]);
  });

  it("acepta citas por chunk_id y listas separadas por coma", () => {
    const out = parseInline("Paso [b::p0004::c002, 1]", sources);
    expect(out).toEqual([
      { kind: "text", text: "Paso" },
      { kind: "cite", index: 1 },
      { kind: "cite", index: 0 },
    ]);
  });

  it("deja como texto corchetes que no son citas válidas", () => {
    expect(parseInline("Ver [3] y [nota]", sources)).toEqual([{ kind: "text", text: "Ver [3] y [nota]" }]);
  });

  it("no interpreta HTML: lo conserva como texto", () => {
    expect(parseInline("<img src=x onerror=alert(1)>", sources)).toEqual([
      { kind: "text", text: "<img src=x onerror=alert(1)>" },
    ]);
  });
});

describe("parseAnswer", () => {
  it("agrupa párrafos, listas numeradas y viñetas", () => {
    const blocks = parseAnswer("Intro\nsigue\n\n1. uno\n2. dos\n- a\n- b\nFinal", sources);
    expect(blocks.map((b) => b.kind)).toEqual(["paragraph", "ordered", "bullets", "paragraph"]);
    expect(blocks[0]).toEqual({ kind: "paragraph", inlines: [{ kind: "text", text: "Intro sigue" }] });
    const ordered = blocks[1];
    expect(ordered?.kind === "ordered" && ordered.items.length).toBe(2);
  });

  it("devuelve lista vacía para texto vacío", () => {
    expect(parseAnswer("", sources)).toEqual([]);
  });
});

describe("resaltado del PDF", () => {
  it("compara sin acentos ni espacios múltiples", () => {
    expect(foldForMatch("  Recepción   en  DOS pasos ")).toBe("recepcion en dos pasos");
  });

  it("isPartOfChunk ignora fragmentos muy cortos", () => {
    const chunk = foldForMatch("Two-step receipts move incoming goods to an Input location");
    expect(isPartOfChunk("move incoming goods", chunk)).toBe(true);
    expect(isPartOfChunk("to", chunk)).toBe(false);
    expect(isPartOfChunk("outgoing", chunk)).toBe(false);
  });

  it("chunkRange resalta un bloque contiguo y no palabras sueltas de otros párrafos", () => {
    // Página real (odoo-18.0-inventario p.138): las cursivas "Inventory" del primer párrafo
    // aparecen en el chunk, pero el renglón largo que las rodea no.
    const items = [
      "Depending on a company's needs, receiving and shipping products",
      "Inventory",
      "Tip: Incoming and outgoing shipments do not need to be configured",
      "Configuration",
      "In Odoo Inventory, both incoming and outgoing shipments are configured",
    ];
    const chunk = foldForMatch(
      "Tip: Incoming and outgoing shipments do not need to be configured with the same amount of steps. " +
        "Configuration In Odoo Inventory, both incoming and outgoing shipments are configured to process",
    );
    expect(chunkRange(items, chunk)).toEqual([2, 4]);
    expect(chunkRange(["Inventory", "otro"], chunk)).toBeNull();
  });
});
