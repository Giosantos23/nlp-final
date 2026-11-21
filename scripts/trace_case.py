"""Traza de recuperación de un caso del banco: rango de cada chunk en cada etapa.

Sirve para la autopsia de fallas: muestra dónde queda el chunk que contiene la respuesta en
BM25, denso, fusión RRF y rerank, y qué entró al top-k que vio el generador. Uso::

    PYTHONPATH=src python scripts/trace_case.py BAS-01 "RB-N1-001@v1.1::p0004::c000"

Escribe ``reports/eval/trace_<caso>.json`` y ``.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docuerp.config import get_settings  # noqa: E402
from docuerp.eval.benchmark import load_benchmark  # noqa: E402
from docuerp.retrieval.contracts import RetrievalQuery  # noqa: E402
from docuerp.retrieval.pipeline import build_retriever  # noqa: E402

DEPTH = 20


def main(case_id: str, target: str) -> None:
    cfg = get_settings().retrieval
    case = {c.id: c for c in load_benchmark()}[case_id]
    retriever = build_retriever(cfg.mode, cfg.chunker, cfg.embedder)
    result = retriever.retrieve(
        RetrievalQuery(
            text=case.question,
            role=case.role,
            erp_version=case.erp_version,
            top_k=DEPTH,
            candidates_k=cfg.candidates_k,
        )
    )
    stages = {}
    for stage, hits in result.stages.items():
        ids = [h.chunk.chunk_id for h in hits]
        rank = ids.index(target) + 1 if target in ids else None
        stages[stage] = {
            "rank": rank,
            "score": hits[rank - 1].score if rank else None,
            "top": [
                {
                    "rank": i + 1,
                    "chunk_id": h.chunk.chunk_id,
                    "page": h.chunk.page,
                    "section": h.chunk.section_path,
                    "score": round(h.score, 4),
                }
                for i, h in enumerate(hits[:8])
            ],
        }
    out = {
        "case_id": case_id,
        "question": case.question,
        "role": case.role,
        "erp_version": case.erp_version,
        "target_chunk": target,
        "top_k_generator": cfg.top_k,
        "retriever": {"mode": cfg.mode, "chunker": cfg.chunker, "embedder": cfg.embedder},
        "target_in_generator_context": any(
            h.chunk.chunk_id == target for h in result.hits[: cfg.top_k]
        ),
        "stages": stages,
    }
    dest = ROOT / "reports" / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    stem = f"trace_{case_id}"
    (dest / f"{stem}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    lines = [
        f"# Traza de recuperación — {case_id}",
        "",
        f"Pregunta ({case.role}, Odoo {case.erp_version}): {case.question}",
        "",
        f"Chunk con la respuesta: `{target}`. El generador ve el top-{cfg.top_k} del rerank; "
        f"¿entra? **{'sí' if out['target_in_generator_context'] else 'no'}**.",
        "",
        "| Etapa | Rango del chunk | Score |",
        "|---|---|---|",
    ]
    for stage, data in stages.items():
        score = "—" if data["score"] is None else f"{data['score']:.3f}"
        lines.append(f"| {stage} | {data['rank'] or f'> {DEPTH}'} | {score} |")
    lines += [
        "",
        "Top del rerank:",
        "",
        "| # | chunk | pág. | sección | score |",
        "|---|---|---|---|---|",
    ]
    for row in stages.get("rerank", {}).get("top", []):
        lines.append(
            f"| {row['rank']} | `{row['chunk_id']}` | {row['page']} | {row['section']} "
            f"| {row['score']:.3f} |"
        )
    (dest / f"{stem}.md").write_text("\n".join(lines) + "\n", "utf-8")
    print("\n".join(lines[6 : 8 + len(stages)]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
