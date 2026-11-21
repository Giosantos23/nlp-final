/**
 * Genera los PDF de demostración que usa el modo offline (VITE_MOCK=1).
 *
 * Lee `src/mocks/documents.json` y escribe un PDF por fuente en `public/mock-docs/`.
 * El texto de cada página citada es EXACTAMENTE el texto de los chunks de los fixtures,
 * así el visor puede resaltar el fragmento citado sobre la capa de texto de pdf.js.
 * Las páginas sin contenido definido se rellenan con una página neutra rotulada.
 *
 * Uso: `npm run mock:pdfs` (solo hace falta al cambiar documents.json).
 */
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { PDFDocument, StandardFonts, rgb } from "pdf-lib";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const outDir = join(root, "public", "mock-docs");

// Medidas de página A4 en puntos tipográficos y márgenes del cuerpo.
const PAGE_W = 595;
const PAGE_H = 842;
const MARGIN = 64;
const BODY_SIZE = 11;
const LEADING = 17;

/** Parte un texto en líneas que caben en `maxWidth` según la métrica real de la fuente. */
function wrap(text, font, size, maxWidth) {
  const words = text.split(/\s+/);
  const lines = [];
  let line = "";
  for (const word of words) {
    const candidate = line ? `${line} ${word}` : word;
    if (font.widthOfTextAtSize(candidate, size) > maxWidth && line) {
      lines.push(line);
      line = word;
    } else {
      line = candidate;
    }
  }
  if (line) lines.push(line);
  return lines;
}

/** Dibuja una página: cabecera del documento, título de sección, bloques y pie. */
function drawPage(pdf, fonts, doc, pageNumber, content) {
  const page = pdf.addPage([PAGE_W, PAGE_H]);
  const ink = rgb(0.11, 0.13, 0.16);
  const muted = rgb(0.42, 0.46, 0.52);
  const accent = rgb(0.06, 0.73, 0.51);

  // Franja superior con el título del documento y su versión.
  page.drawRectangle({ x: 0, y: PAGE_H - 6, width: PAGE_W, height: 6, color: accent });
  page.drawText(doc.title, { x: MARGIN, y: PAGE_H - 44, size: 9, font: fonts.bold, color: muted });
  const versionLabel = `${doc.doc_id} · ${doc.version}`;
  const vw = fonts.regular.widthOfTextAtSize(versionLabel, 9);
  page.drawText(versionLabel, {
    x: PAGE_W - MARGIN - vw,
    y: PAGE_H - 44,
    size: 9,
    font: fonts.regular,
    color: muted,
  });
  page.drawLine({
    start: { x: MARGIN, y: PAGE_H - 56 },
    end: { x: PAGE_W - MARGIN, y: PAGE_H - 56 },
    thickness: 0.6,
    color: rgb(0.86, 0.88, 0.9),
  });

  let y = PAGE_H - 96;
  const heading = content?.heading ?? `Sección ${pageNumber}`;
  for (const line of wrap(heading, fonts.bold, 15, PAGE_W - MARGIN * 2)) {
    page.drawText(line, { x: MARGIN, y, size: 15, font: fonts.bold, color: ink });
    y -= 22;
  }
  y -= 10;

  const blocks = content?.blocks ?? [
    {
      text:
        "Página de relleno del documento de demostración. El modo offline de DocuERP " +
        "solo incluye el contenido de las páginas que citan los escenarios precargados.",
    },
  ];
  for (const block of blocks) {
    for (const line of wrap(block.text, fonts.regular, BODY_SIZE, PAGE_W - MARGIN * 2)) {
      page.drawText(line, { x: MARGIN, y, size: BODY_SIZE, font: fonts.regular, color: ink });
      y -= LEADING;
    }
    y -= 12;
  }

  const footer = `${doc.source} · página ${pageNumber} de ${doc.pages_total}`;
  page.drawText(footer, { x: MARGIN, y: 40, size: 8.5, font: fonts.regular, color: muted });
}

async function main() {
  const raw = await readFile(join(root, "src", "mocks", "documents.json"), "utf8");
  const { documents } = JSON.parse(raw);
  await mkdir(outDir, { recursive: true });
  for (const doc of documents) {
    const pdf = await PDFDocument.create();
    pdf.setTitle(doc.title);
    pdf.setAuthor(doc.owner);
    pdf.setSubject("Documento de demostración del modo offline de DocuERP");
    const fonts = {
      regular: await pdf.embedFont(StandardFonts.Helvetica),
      bold: await pdf.embedFont(StandardFonts.HelveticaBold),
    };
    const byPage = new Map(doc.pages.map((p) => [p.page, p]));
    for (let n = 1; n <= doc.pages_total; n += 1) {
      drawPage(pdf, fonts, doc, n, byPage.get(n));
    }
    const bytes = await pdf.save();
    await writeFile(join(outDir, doc.source), bytes);
    console.log(`✓ ${doc.source} (${doc.pages_total} páginas, ${(bytes.length / 1024).toFixed(1)} KB)`);
  }
}

main().catch((error) => {
  console.error("No se pudieron generar los PDF de demostración:", error);
  process.exit(1);
});
