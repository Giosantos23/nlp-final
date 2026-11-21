"""Orquestación de la ingesta y reporte (``reports/ingestion/ingestion_report.{json,md}``).

Las seis fases (SPEC §5.3 del enunciado), cronometradas por separado:

1. **carga**: manifiesto → documentos a procesar, ubicación del archivo y verificación de
   su sha256;
2. **extracción**: PDF/DOCX → páginas con bloques tipados (``extract.py``);
3. **limpieza**: normalización, encabezados/pies repetidos y escáner de inyección
   (``clean.py``);
4. **chunking**: ``structure`` | ``fixed256`` | ``fixed768`` (``chunk.py``);
5. **metadatos**: ``chunk_id``, hash, versión, roles y marcas (``metadata.py``);
6. **indexación**: JSONL de chunks + BM25 + Chroma (``index.py``); con
   ``skip_embeddings`` solo se escriben el JSONL y BM25.

Cada corrida guarda su reporte en ``reports/ingestion/runs/ingestion_{chunker}.{json,md}``;
``docuerp ingest report`` consolida todas las corridas en ``ingestion_report.{json,md}``.
"""

from __future__ import annotations

import itertools
import json
import logging
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from docuerp import paths
from docuerp.ingest import index as index_mod
from docuerp.ingest.chunk import Chunker, get_chunker
from docuerp.ingest.clean import SUSPICIOUS_FLAG, clean_document, scan_suspicious
from docuerp.ingest.extract import ExtractionError, extract
from docuerp.ingest.manifest import SourceEntry, load_manifest
from docuerp.ingest.metadata import build_chunks, utc_now_iso
from docuerp.ingest.odoo_pdf import sha256_file
from docuerp.schemas import Chunk

log = logging.getLogger(__name__)

SourcesFilter = Literal["all", "odoo", "internal"]

PHASES: tuple[str, ...] = ("carga", "extraccion", "limpieza", "chunking", "metadatos", "indexacion")
"""Fases cronometradas, en orden."""

TOKEN_BINS: tuple[int, ...] = (0, 64, 128, 256, 384, 512, 768, 1024)
"""Límites de los intervalos del histograma de tokens por chunk."""


class IngestionReport(BaseModel):
    """Resumen de una corrida de ingesta."""

    model_config = ConfigDict(extra="forbid")

    chunker: str
    embedder: str
    sources: SourcesFilter
    skip_embeddings: bool = False
    documents: int = 0
    pages: int = 0
    chunks: int = 0
    tokens_total: int = 0
    tokens_mean: float = 0.0
    token_stats: dict[str, float] = Field(default_factory=dict)
    token_histogram: dict[str, int] = Field(default_factory=dict)
    per_module_version: dict[str, dict[str, int]] = Field(default_factory=dict)
    flagged_chunks: list[str] = Field(default_factory=list)
    suspicious_findings: list[dict[str, str]] = Field(default_factory=list)
    duplicate_hashes: dict[str, list[str]] = Field(default_factory=dict)
    duplicates_within_doc: int = 0
    duplicates_cross_doc: int = 0
    per_document: list[dict[str, object]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    outputs: dict[str, str] = Field(default_factory=dict)
    timings_s: dict[str, float] = Field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def select_entries(entries: Iterable[SourceEntry], sources: SourcesFilter) -> list[SourceEntry]:
    """Filtra el manifiesto: ``odoo`` = documentación oficial; ``internal`` = el resto."""
    if sources not in ("all", "odoo", "internal"):
        raise ValueError(f"sources inválido: {sources!r} (all | odoo | internal)")
    selected: list[SourceEntry] = []
    for entry in entries:
        is_odoo = entry.doc_type == "odoo_doc"
        if sources == "all" or (sources == "odoo") == is_odoo:
            selected.append(entry)
    return selected


def default_search_dirs() -> list[Path]:
    """Carpetas donde se buscan los archivos citables (PDF de Odoo y runbooks)."""
    return [paths.BUILD / "pdf", paths.BUILD, paths.RUNBOOKS]


def resolve_source_path(entry: SourceEntry, search_dirs: list[Path] | None = None) -> Path | None:
    """Ubica ``entry.source_file`` en las carpetas de búsqueda (directo y luego recursivo)."""
    dirs = search_dirs or default_search_dirs()
    for directory in dirs:
        candidate = directory / entry.source_file
        if candidate.is_file():
            return candidate
    for directory in dirs:
        if directory.is_dir():
            found = sorted(directory.rglob(entry.source_file))
            if found:
                return found[0]
    return None


def token_statistics(counts: list[int]) -> dict[str, float]:
    """Mínimo, percentiles 10/50/90, máximo y media de los tokens por chunk."""
    if not counts:
        return {}
    ordered = sorted(counts)

    def pct(q: float) -> float:
        return float(ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))])

    return {
        "min": float(ordered[0]),
        "p10": pct(0.10),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "max": float(ordered[-1]),
        "mean": round(statistics.fmean(ordered), 1),
    }


def token_histogram(counts: list[int]) -> dict[str, int]:
    """Histograma por intervalos ``[a, b)`` de ``TOKEN_BINS`` (+ ``≥1024``)."""
    histogram: dict[str, int] = {}
    edges = list(TOKEN_BINS)
    for low, high in itertools.pairwise(edges):
        histogram[f"{low}-{high - 1}"] = sum(1 for c in counts if low <= c < high)
    histogram[f"≥{edges[-1]}"] = sum(1 for c in counts if c >= edges[-1])
    return histogram


def duplicate_groups(chunks: list[Chunk]) -> dict[str, list[str]]:
    """``content_hash`` → ``chunk_id`` de los chunks con texto idéntico (≥ 2)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        groups[chunk.content_hash].append(chunk.chunk_id)
    return {h: ids for h, ids in groups.items() if len(ids) > 1}


def _doc_of(chunk_id: str) -> str:
    return chunk_id.split("::", 1)[0]


def summarize_chunks(chunks: list[Chunk]) -> dict[str, Any]:
    """Estadísticas recalculables desde los chunks (las usa también ``ingest report``)."""
    counts = [c.token_count for c in chunks]
    per_mv: dict[str, dict[str, int]] = defaultdict(lambda: {"docs": 0, "chunks": 0, "tokens": 0})
    docs_by_key: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        key = f"{chunk.module}@{chunk.erp_version}"
        per_mv[key]["chunks"] += 1
        per_mv[key]["tokens"] += chunk.token_count
        docs_by_key[key].add(chunk.doc_id)
    for key, docs in docs_by_key.items():
        per_mv[key]["docs"] = len(docs)
    dups = duplicate_groups(chunks)
    within = sum(1 for ids in dups.values() if len({_doc_of(i) for i in ids}) == 1)
    return {
        "chunks": len(chunks),
        "tokens_total": sum(counts),
        "tokens_mean": round(statistics.fmean(counts), 1) if counts else 0.0,
        "token_stats": token_statistics(counts),
        "token_histogram": token_histogram(counts),
        "per_module_version": {k: dict(v) for k, v in sorted(per_mv.items())},
        "flagged_chunks": [c.chunk_id for c in chunks if SUSPICIOUS_FLAG in c.flags],
        "duplicate_hashes": dups,
        "duplicates_within_doc": within,
        "duplicates_cross_doc": len(dups) - within,
    }


# ---------------------------------------------------------------------------
# Corrida
# ---------------------------------------------------------------------------


def run_ingestion(
    chunker: str = "structure",
    embedder: str = "bge-m3",
    sources: SourcesFilter = "all",
    *,
    skip_embeddings: bool = False,
    build_bm25: bool = True,
    manifest_dir: Path | None = None,
    search_dirs: list[Path] | None = None,
    chunker_impl: Chunker | None = None,
    chunks_out: Path | None = None,
) -> IngestionReport:
    """Ejecuta las seis fases de la ingesta y devuelve el reporte.

    Args:
        chunker: ``structure`` | ``fixed256`` | ``fixed768``.
        embedder: ``bge-m3`` | ``e5-base`` | ``fake`` (solo si no se omiten embeddings).
        sources: ``all`` | ``odoo`` | ``internal``.
        skip_embeddings: solo produce ``chunks.{chunker}.jsonl`` (+ BM25), sin Chroma.
        build_bm25: construir también ``data/index/{chunker}/bm25.pkl``.
        manifest_dir / search_dirs / chunker_impl / chunks_out: inyección para pruebas.

    Raises:
        ValueError: si ningún documento del manifiesto pudo procesarse.
    """
    report = IngestionReport(
        chunker=chunker, embedder=embedder, sources=sources, skip_embeddings=skip_embeddings
    )
    report.started_at = utc_now_iso()
    timings: dict[str, float] = dict.fromkeys(PHASES, 0.0)
    wall = time.perf_counter()

    # 1. Carga ---------------------------------------------------------------
    t0 = time.perf_counter()
    entries = select_entries(load_manifest(manifest_dir), sources)
    impl = chunker_impl or get_chunker(chunker)
    jobs: list[tuple[SourceEntry, Path]] = []
    for entry in entries:
        path = resolve_source_path(entry, search_dirs)
        if path is None:
            report.warnings.append(f"{entry.id}: no se encontró {entry.source_file}; se omite")
            continue
        if entry.sha256 and sha256_file(path) != entry.sha256:
            report.warnings.append(
                f"{entry.id}: el sha256 de {path.name} no coincide con el manifiesto "
                "(¿se regeneró el archivo sin actualizar el manifiesto?)"
            )
        jobs.append((entry, path))
    timings["carga"] += time.perf_counter() - t0

    all_chunks: list[Chunk] = []
    stamp = report.started_at
    for entry, path in jobs:
        # 2. Extracción --------------------------------------------------------
        t0 = time.perf_counter()
        try:
            extracted = extract(path, entry.id)
        except ExtractionError as exc:
            report.warnings.append(f"{entry.id}: {exc}")
            continue
        timings["extraccion"] += time.perf_counter() - t0

        # 3. Limpieza -----------------------------------------------------------
        t0 = time.perf_counter()
        cleaned, findings = clean_document(extracted)
        timings["limpieza"] += time.perf_counter() - t0

        # 4. Chunking -------------------------------------------------------------
        t0 = time.perf_counter()
        drafts = impl.chunk(cleaned)
        timings["chunking"] += time.perf_counter() - t0

        # 5. Metadatos (incluye el escáner por chunk: detecta inyecciones partidas
        #    entre bloques que el escaneo por página ya vio, y las ubica en su chunk).
        t0 = time.perf_counter()
        flags = {i: [SUSPICIOUS_FLAG] for i, d in enumerate(drafts) if scan_suspicious(d.text)}
        chunks = build_chunks(drafts, entry, flags_by_index=flags, ingested_at=stamp)
        timings["metadatos"] += time.perf_counter() - t0

        all_chunks.extend(chunks)
        report.suspicious_findings.extend(findings)
        report.per_document.append(
            {
                "doc_id": entry.id,
                "source": entry.source_file,
                "module": entry.module,
                "erp_version": entry.erp_version,
                "doc_type": entry.doc_type,
                "pdf_pages": extracted.page_count,
                "content_pages": len(cleaned.pages),
                "skipped_pages": extracted.skipped_pages,
                "chunks": len(chunks),
                "tokens": sum(c.token_count for c in chunks),
                "flagged_chunks": sum(1 for c in chunks if c.flags),
            }
        )

    if not all_chunks:
        raise ValueError(
            "La ingesta no produjo chunks: revise el manifiesto (data/manifest/*.yaml) y que "
            "existan los archivos (p. ej. `python scripts/build_odoo_pdfs.py`)."
        )

    # 6. Indexación -------------------------------------------------------------
    t0 = time.perf_counter()
    report.outputs["chunks_jsonl"] = str(
        index_mod.write_chunks(all_chunks, chunker, path=chunks_out)
    )
    if build_bm25:
        # La recuperación valida esta caché por hash del JSONL y la reconstruye en su propio
        # formato la primera vez que la usa; ``load_bm25_index`` acepta ambos formatos.
        report.outputs["bm25"] = str(index_mod.build_bm25_index(all_chunks, chunker))
    if not skip_embeddings:
        report.outputs["chroma"] = str(index_mod.build_chroma_index(all_chunks, chunker, embedder))
    timings["indexacion"] += time.perf_counter() - t0

    summary = summarize_chunks(all_chunks)
    report.documents = len(report.per_document)
    report.pages = sum(int(str(d["pdf_pages"])) for d in report.per_document)
    for key in (
        "chunks",
        "tokens_total",
        "tokens_mean",
        "token_stats",
        "token_histogram",
        "per_module_version",
        "flagged_chunks",
        "duplicate_hashes",
        "duplicates_within_doc",
        "duplicates_cross_doc",
    ):
        setattr(report, key, summary[key])
    pages_by_key: Counter[str] = Counter()
    for doc in report.per_document:
        pages_by_key[f"{doc['module']}@{doc['erp_version']}"] += int(str(doc["pdf_pages"]))
    for key, stats in report.per_module_version.items():
        stats["pages"] = pages_by_key.get(key, 0)
    timings["total"] = time.perf_counter() - wall
    report.timings_s = {k: round(v, 3) for k, v in timings.items()}
    report.finished_at = utc_now_iso()
    for warning in report.warnings:
        log.warning(warning)
    return report


# ---------------------------------------------------------------------------
# Reportes
# ---------------------------------------------------------------------------


def reports_dir() -> Path:
    """``reports/ingestion``."""
    return paths.REPORTS / "ingestion"


def _md_table(headers: list[str], rows: list[list[object]]) -> str:
    """Tabla Markdown simple."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def render_run_markdown(report: IngestionReport) -> str:
    """Reporte Markdown de una corrida."""
    out = [
        f"# Ingesta — chunker `{report.chunker}`",
        "",
        f"- Fuentes: `{report.sources}` · embedder: `{report.embedder}`"
        + (" (embeddings omitidos)" if report.skip_embeddings else ""),
        f"- Inicio: {report.started_at} · fin: {report.finished_at}",
        f"- Documentos: **{report.documents}** · páginas: **{report.pages}** · chunks: "
        f"**{report.chunks}** · tokens: {report.tokens_total} (media {report.tokens_mean})",
        "",
        "## Chunks por módulo y versión",
        "",
        _md_table(
            ["Módulo@versión", "Docs", "Páginas", "Chunks", "Tokens"],
            [
                [k, v.get("docs", 0), v.get("pages", 0), v.get("chunks", 0), v.get("tokens", 0)]
                for k, v in report.per_module_version.items()
            ],
        ),
        "",
        "## Distribución de tokens por chunk",
        "",
        _md_table(list(report.token_stats), [list(report.token_stats.values())]),
        "",
        _md_table(list(report.token_histogram), [list(report.token_histogram.values())]),
        "",
        "## Documentos",
        "",
        _md_table(
            ["doc_id", "Págs PDF", "Págs con texto", "Chunks", "Tokens", "Marcados"],
            [
                [
                    d["doc_id"],
                    d["pdf_pages"],
                    d["content_pages"],
                    d["chunks"],
                    d["tokens"],
                    d["flagged_chunks"],
                ]
                for d in report.per_document
            ],
        ),
        "",
        "## Escáner de instrucciones sospechosas",
        "",
        f"Chunks marcados (`{SUSPICIOUS_FLAG}`): **{len(report.flagged_chunks)}**",
        "",
    ]
    if report.suspicious_findings:
        out.append(
            _md_table(
                ["doc_id", "Página", "Patrón", "Fragmento"],
                [
                    [f["doc_id"], f["page"], f["pattern"], f["excerpt"].replace("|", "/")]
                    for f in report.suspicious_findings
                ],
            )
        )
    else:
        out.append("Sin hallazgos.")
    out += [
        "",
        "## Duplicados por hash",
        "",
        f"Grupos de texto idéntico: **{len(report.duplicate_hashes)}** "
        f"(dentro de un documento: {report.duplicates_within_doc}; entre documentos, p. ej. "
        f"páginas idénticas en 17.0 y 18.0: {report.duplicates_cross_doc}).",
        "",
        "## Tiempos por fase (s)",
        "",
        _md_table(list(report.timings_s), [list(report.timings_s.values())]),
    ]
    if report.warnings:
        out += ["", "## Avisos", "", *[f"- {w}" for w in report.warnings]]
    return "\n".join(out) + "\n"


def write_report(report: IngestionReport, out_dir: Path | None = None) -> tuple[Path, Path]:
    """Escribe el reporte en JSON y Markdown; devuelve ambas rutas.

    Por defecto en ``reports/ingestion/runs/ingestion_{chunker}.{json,md}``.
    """
    directory = out_dir or (reports_dir() / "runs")
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"ingestion_{report.chunker}.json"
    md_path = directory / f"ingestion_{report.chunker}.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_run_markdown(report), encoding="utf-8")
    return json_path, md_path


def build_consolidated_report(
    chunkers: list[str] | None = None,
    *,
    runs_dir: Path | None = None,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Consolida en ``ingestion_report.{json,md}`` todas las corridas disponibles.

    Las estadísticas de chunks se **recalculan** desde ``data/processed/chunks.*.jsonl``;
    páginas por documento, hallazgos del escáner y tiempos se toman del reporte de cada
    corrida (``runs/ingestion_{chunker}.json``).

    Raises:
        FileNotFoundError: si no hay ningún JSONL de chunks.
    """
    runs = runs_dir or (reports_dir() / "runs")
    if chunkers is None:
        chunkers = sorted(
            p.name.removeprefix("chunks.").removesuffix(".jsonl")
            for p in paths.PROCESSED.glob("chunks.*.jsonl")
        )
    sections: dict[str, dict[str, Any]] = {}
    for name in chunkers:
        try:
            chunks = index_mod.load_chunks(name)
        except FileNotFoundError:
            continue
        summary = summarize_chunks(chunks)
        run_file = runs / f"ingestion_{name}.json"
        run = (
            IngestionReport.model_validate_json(run_file.read_text(encoding="utf-8"))
            if run_file.is_file()
            else None
        )
        if run is not None:
            pages = {f"{d['module']}@{d['erp_version']}": 0 for d in run.per_document}
            for d in run.per_document:
                pages[f"{d['module']}@{d['erp_version']}"] += int(str(d["pdf_pages"]))
            for key, stats in summary["per_module_version"].items():
                stats["pages"] = pages.get(key, 0)
        summary["duplicate_groups"] = len(summary.pop("duplicate_hashes"))
        sections[name] = {
            **summary,
            "documents": len({c.doc_id for c in chunks}),
            "pages": sum(int(str(d["pdf_pages"])) for d in run.per_document) if run else None,
            "embedder": run.embedder if run else None,
            "skip_embeddings": run.skip_embeddings if run else None,
            "sources": run.sources if run else None,
            "suspicious_findings": run.suspicious_findings if run else [],
            "timings_s": run.timings_s if run else {},
            "per_document": run.per_document if run else [],
            "warnings": run.warnings if run else [],
            "run_finished_at": run.finished_at if run else None,
        }
    if not sections:
        raise FileNotFoundError(
            "No hay chunks procesados en data/processed: ejecute `docuerp ingest run` primero."
        )
    payload = {"generated_at": utc_now_iso(), "chunkers": sections}
    directory = out_dir or reports_dir()
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "ingestion_report.json"
    md_path = directory / "ingestion_report.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_consolidated_markdown(payload), encoding="utf-8")
    return json_path, md_path


def render_consolidated_markdown(payload: dict[str, Any]) -> str:
    """Markdown del reporte consolidado (una sección por chunker)."""
    sections: dict[str, dict[str, Any]] = payload["chunkers"]
    out = [
        "# Reporte de ingesta — DocuERP",
        "",
        f"Generado: {payload['generated_at']}",
        "",
        "## Resumen por chunker",
        "",
        _md_table(
            [
                "Chunker",
                "Docs",
                "Páginas",
                "Chunks",
                "Tokens",
                "Media",
                "p50",
                "p90",
                "Máx",
                "Marcados",
                "Dup. (grupos)",
                "Total s",
            ],
            [
                [
                    name,
                    s["documents"],
                    s["pages"] if s["pages"] is not None else "—",
                    s["chunks"],
                    s["tokens_total"],
                    s["tokens_mean"],
                    s["token_stats"].get("p50", "—"),
                    s["token_stats"].get("p90", "—"),
                    s["token_stats"].get("max", "—"),
                    len(s["flagged_chunks"]),
                    s["duplicate_groups"],
                    s["timings_s"].get("total", "—"),
                ]
                for name, s in sections.items()
            ],
        ),
    ]
    for name, s in sections.items():
        out += [
            "",
            f"## `{name}`",
            "",
            f"Fuentes: `{s['sources']}` · embedder: `{s['embedder']}`"
            + (" (embeddings omitidos)" if s["skip_embeddings"] else ""),
            "",
            "### Chunks por módulo y versión",
            "",
            _md_table(
                ["Módulo@versión", "Docs", "Páginas", "Chunks", "Tokens"],
                [
                    [k, v.get("docs", 0), v.get("pages", 0), v.get("chunks", 0), v.get("tokens", 0)]
                    for k, v in s["per_module_version"].items()
                ],
            ),
            "",
            "### Distribución de tokens",
            "",
            _md_table(list(s["token_histogram"]), [list(s["token_histogram"].values())]),
            "",
            "### Duplicados por hash",
            "",
            f"{s['duplicate_groups']} grupos (dentro de un documento: {s['duplicates_within_doc']};"
            f" entre documentos: {s['duplicates_cross_doc']}).",
            "",
            "### Escáner de instrucciones sospechosas",
            "",
            f"Chunks marcados: {len(s['flagged_chunks'])}"
            + (f" ({', '.join(s['flagged_chunks'][:20])})" if s["flagged_chunks"] else ""),
            "",
        ]
        if s["suspicious_findings"]:
            out.append(
                _md_table(
                    ["doc_id", "Página", "Patrón", "Fragmento"],
                    [
                        [f["doc_id"], f["page"], f["pattern"], f["excerpt"].replace("|", "/")]
                        for f in s["suspicious_findings"]
                    ],
                )
            )
        else:
            out.append("Sin hallazgos en el corpus.")
        if s["timings_s"]:
            out += [
                "",
                "### Tiempos por fase (s)",
                "",
                _md_table(list(s["timings_s"]), [list(s["timings_s"].values())]),
            ]
        if s["warnings"]:
            out += ["", "### Avisos", "", *[f"- {w}" for w in s["warnings"]]]
    return "\n".join(out) + "\n"
