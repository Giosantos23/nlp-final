/**
 * Corpus de demostración para el modo offline.
 *
 * Los chunks se derivan de `documents.json`, el mismo archivo con el que
 * `scripts/make-mock-pdfs.mjs` genera los PDF de `public/mock-docs/`. Así el texto del chunk
 * coincide palabra por palabra con la capa de texto del PDF y el resaltado funciona.
 */
import raw from "./documents.json";
import type {
  Chunk,
  ChunkErpVersion,
  CorpusStats,
  DocStatus,
  DocType,
  DocumentsResponse,
  ErpModule,
  SourceEntry,
  UserRole,
} from "../api/types";

interface RawBlock {
  chunk?: number;
  section_path?: string;
  text: string;
}
interface RawPage {
  page: number;
  heading: string;
  blocks: RawBlock[];
}
interface RawDocument {
  doc_id: string;
  source: string;
  title: string;
  version: string;
  erp_version: string;
  status: string;
  module: string;
  doc_type: string;
  language: string;
  allowed_roles: string[];
  origin: string;
  license: string;
  owner: string;
  sensitivity: string;
  update_frequency: string;
  date: string;
  pages_total: number;
  pages: RawPage[];
}

const documents = (raw as { documents: RawDocument[] }).documents;

/**
 * Hash hexadecimal de 64 caracteres, determinista, SOLO para mostrar en la demo.
 * No es criptográfico: el sha256 real lo calcula la ingesta en Python.
 */
export function pseudoHash(input: string): string {
  let out = "";
  let seed = 0x811c9dc5;
  for (let round = 0; out.length < 64; round += 1) {
    let h = seed ^ round;
    for (let i = 0; i < input.length; i += 1) {
      h ^= input.charCodeAt(i);
      h = Math.imul(h, 0x01000193) >>> 0;
    }
    out += h.toString(16).padStart(8, "0");
    seed = h;
  }
  return out.slice(0, 64);
}

/** Construye el `chunk_id` determinista `{doc_id}::p{page:04d}::c{idx:03d}` (SPEC §4). */
export function makeChunkId(docId: string, page: number, idx: number): string {
  return `${docId}::p${String(page).padStart(4, "0")}::c${String(idx).padStart(3, "0")}`;
}

function buildChunks(): Record<string, Chunk> {
  const out: Record<string, Chunk> = {};
  for (const doc of documents) {
    for (const page of doc.pages) {
      for (const block of page.blocks) {
        if (block.chunk === undefined) continue;
        const chunkId = makeChunkId(doc.doc_id, page.page, block.chunk);
        out[chunkId] = {
          chunk_id: chunkId,
          doc_id: doc.doc_id,
          text: block.text,
          source: doc.source,
          page: page.page,
          page_end: page.page,
          version: doc.version,
          erp_version: doc.erp_version as ChunkErpVersion,
          status: doc.status as DocStatus,
          section_path: block.section_path ?? page.heading,
          module: doc.module as ErpModule,
          doc_type: doc.doc_type as DocType,
          language: doc.language === "es" ? "es" : "en",
          allowed_roles: doc.allowed_roles as UserRole[],
          content_hash: pseudoHash(block.text),
          ingested_at: "2026-11-14T18:20:41Z",
          token_count: Math.round(block.text.split(/\s+/).length * 1.38),
          flags: [],
        };
      }
    }
  }
  return out;
}

/** Registro `chunk_id → Chunk` de todos los fragmentos citables en la demo. */
export const MOCK_CHUNKS: Record<string, Chunk> = buildChunks();

/** Fuentes que tienen PDF de demostración en `public/mock-docs/`. */
export const MOCK_PDF_SOURCES: ReadonlySet<string> = new Set(documents.map((d) => d.source));

function toEntry(doc: RawDocument, chunks: number, tokens: number): SourceEntry {
  return {
    id: doc.doc_id,
    title: doc.title,
    source_file: doc.source,
    origin: doc.origin,
    license: doc.license,
    owner: doc.owner,
    format: "pdf",
    version: doc.version,
    erp_version: doc.erp_version as ChunkErpVersion,
    status: doc.status as DocStatus,
    date: doc.date,
    update_frequency: doc.update_frequency,
    allowed_roles: doc.allowed_roles as UserRole[],
    sensitivity: doc.sensitivity,
    module: doc.module as ErpModule,
    doc_type: doc.doc_type as DocType,
    language: doc.language === "es" ? "es" : "en",
    sha256: pseudoHash(doc.doc_id + doc.version),
    pages: doc.pages_total,
    chunks,
    tokens,
  };
}

/** Volumen representativo por documento (el corpus real lo reporta la ingesta). */
const VOLUME: Record<string, [number, number]> = {
  "odoo-18.0-inventario": [212, 81_420],
  "odoo-17.0-inventario": [198, 75_870],
  "RB-N1-004@v1.2": [9, 3_210],
  "RB-N2-003@v1.1": [12, 4_480],
  "POL-SLA-001@v2.0": [8, 2_760],
  "odoo-18.0-facturacion": [264, 101_350],
};

/** Fuentes adicionales del manifiesto (sin PDF de demo) para mostrar gobernanza completa. */
const EXTRA_SOURCES: SourceEntry[] = [
  {
    id: "odoo-17.0-facturacion",
    title: "Odoo 17.0 · Facturación",
    source_file: "odoo-17.0-facturacion.pdf",
    origin: "github.com/odoo/documentation (rama 17.0)",
    license: "CC BY-SA 4.0",
    owner: "Odoo S.A.",
    format: "pdf",
    version: "17.0@9f81e2a",
    erp_version: "17.0",
    status: "vigente",
    date: "2025-06-18",
    update_frequency: "trimestral",
    allowed_roles: ["portal_user", "n1", "n2", "admin"],
    sensitivity: "publica",
    module: "facturacion",
    doc_type: "odoo_doc",
    language: "en",
    sha256: pseudoHash("odoo-17.0-facturacion"),
    pages: 20,
    chunks: 241,
    tokens: 92_880,
  },
  {
    id: "odoo-18.0-compras",
    title: "Odoo 18.0 · Compras",
    source_file: "odoo-18.0-compras.pdf",
    origin: "github.com/odoo/documentation (rama 18.0)",
    license: "CC BY-SA 4.0",
    owner: "Odoo S.A.",
    format: "pdf",
    version: "18.0@a1b2c3d",
    erp_version: "18.0",
    status: "vigente",
    date: "2025-10-02",
    update_frequency: "trimestral",
    allowed_roles: ["portal_user", "n1", "n2", "admin"],
    sensitivity: "publica",
    module: "compras",
    doc_type: "odoo_doc",
    language: "en",
    sha256: pseudoHash("odoo-18.0-compras"),
    pages: 11,
    chunks: 126,
    tokens: 48_310,
  },
  {
    id: "odoo-18.0-ventas",
    title: "Odoo 18.0 · Ventas",
    source_file: "odoo-18.0-ventas.pdf",
    origin: "github.com/odoo/documentation (rama 18.0)",
    license: "CC BY-SA 4.0",
    owner: "Odoo S.A.",
    format: "pdf",
    version: "18.0@a1b2c3d",
    erp_version: "18.0",
    status: "vigente",
    date: "2025-10-02",
    update_frequency: "trimestral",
    allowed_roles: ["portal_user", "n1", "n2", "admin"],
    sensitivity: "publica",
    module: "ventas",
    doc_type: "odoo_doc",
    language: "en",
    sha256: pseudoHash("odoo-18.0-ventas"),
    pages: 13,
    chunks: 151,
    tokens: 57_960,
  },
  {
    id: "RB-N1-004@v1.1",
    title: "RB-N1-004 · Recepciones con verificación en almacén",
    source_file: "RB-N1-004_recepciones_v1.1.pdf",
    origin: "Altamira Soluciones ERP · runbook interno (sintético)",
    license: "Uso interno",
    owner: "Mesa de ayuda N1",
    format: "pdf",
    version: "v1.1",
    erp_version: "any",
    status: "obsoleto",
    date: "2026-02-03",
    update_frequency: "mensual",
    allowed_roles: ["n1", "n2", "admin"],
    sensitivity: "interna",
    module: "inventario",
    doc_type: "runbook",
    language: "es",
    sha256: pseudoHash("RB-N1-004@v1.1"),
    pages: 3,
    chunks: 7,
    tokens: 2_540,
  },
  {
    id: "RB-N1-007@v1.0",
    title: "RB-N1-007 · Restablecer acceso de usuarios del portal",
    source_file: "RB-N1-007_accesos_portal_v1.0.docx",
    origin: "Altamira Soluciones ERP · runbook interno (sintético)",
    license: "Uso interno",
    owner: "Mesa de ayuda N1",
    format: "docx",
    version: "v1.0",
    erp_version: "any",
    status: "vigente",
    date: "2026-05-20",
    update_frequency: "mensual",
    allowed_roles: ["n1", "n2", "admin"],
    sensitivity: "interna",
    module: "accesos",
    doc_type: "runbook",
    language: "es",
    sha256: pseudoHash("RB-N1-007@v1.0"),
    pages: 3,
    chunks: 8,
    tokens: 2_910,
  },
  {
    id: "RB-ADM-001@v1.0",
    title: "RB-ADM-001 · Rotación de credenciales de integración",
    source_file: "RB-ADM-001_rotacion_credenciales_v1.0.pdf",
    origin: "Altamira Soluciones ERP · runbook interno (sintético)",
    license: "Uso interno",
    owner: "Administración de plataforma",
    format: "pdf",
    version: "v1.0",
    erp_version: "any",
    status: "vigente",
    date: "2026-09-08",
    update_frequency: "semestral",
    allowed_roles: ["admin"],
    sensitivity: "restringida",
    module: "accesos",
    doc_type: "runbook",
    language: "es",
    sha256: pseudoHash("RB-ADM-001@v1.0"),
    pages: 4,
    chunks: 10,
    tokens: 3_620,
  },
];

const PRIMARY_SOURCES: SourceEntry[] = documents.map((doc) => {
  const [chunks, tokens] = VOLUME[doc.doc_id] ?? [doc.pages.length, doc.pages.length * 380];
  return toEntry(doc, chunks, tokens);
});

export const MOCK_SOURCES: SourceEntry[] = [...PRIMARY_SOURCES, ...EXTRA_SOURCES];

function buildStats(): CorpusStats {
  const totals = MOCK_SOURCES.reduce(
    (acc, s) => ({
      pages: acc.pages + (s.pages ?? 0),
      chunks: acc.chunks + (s.chunks ?? 0),
      tokens: acc.tokens + (s.tokens ?? 0),
    }),
    { pages: 0, chunks: 0, tokens: 0 },
  );
  const moduleTotals = new Map<ErpModule, number>();
  for (const s of MOCK_SOURCES) {
    moduleTotals.set(s.module, (moduleTotals.get(s.module) ?? 0) + (s.chunks ?? 0));
  }
  return {
    documents: MOCK_SOURCES.length,
    pages: totals.pages,
    chunks: totals.chunks,
    tokens: totals.tokens,
    flagged_chunks: 3,
    duplicates: 11,
    chunker: "structure (450 tokens, solape 15 %)",
    embedder: "bge-m3",
    by_module: [...moduleTotals.entries()]
      .map(([module, chunks]) => ({ module, chunks }))
      .sort((a, b) => b.chunks - a.chunks),
    token_histogram: [
      { bucket: "0–100", chunks: 61 },
      { bucket: "100–200", chunks: 138 },
      { bucket: "200–300", chunks: 247 },
      { bucket: "300–400", chunks: 402 },
      { bucket: "400–500", chunks: 351 },
      { bucket: "500–600", chunks: 47 },
    ],
  };
}

export const MOCK_DOCUMENTS: DocumentsResponse = {
  sources: MOCK_SOURCES,
  stats: buildStats(),
};
