"""Acuerdo entre anotadores del banco de evaluación.

Lee las cuatro pasadas de ``eval/annotation/`` (autor, anotador A, anotador B y
adjudicación) y mide, por campo:

* acuerdo observado y κ de Cohen por par (autor–A, autor–B, A–B);
* κ de Fleiss para los tres anotadores;
* Jaccard medio de la evidencia a nivel de página ``(fuente, página)`` y de fuente;
* cuánto cambió la adjudicación respecto del voto de mayoría.

Escribe ``reports/benchmark/annotation_agreement.{json,md}``. Uso::

    PYTHONPATH=src python scripts/annotation_agreement.py
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
ANNOT = ROOT / "eval" / "annotation"
OUT = ROOT / "reports" / "benchmark"

PASSES = {
    "autor": "author.jsonl",
    "anotador_a": "annotator_a.jsonl",
    "anotador_b": "annotator_b.jsonl",
}
CATEGORICAL = ("expected_behavior", "answerable", "expected_intent", "expected_tool")
LABELS_ES = {
    "expected_behavior": "Comportamiento esperado",
    "answerable": "Respondible",
    "expected_intent": "Intención",
    "expected_tool": "Herramienta",
}


def _load(name: str) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in (ANNOT / name).read_text("utf-8").splitlines() if line]
    return {row["id"]: row for row in rows}


def _evidence(row: dict[str, Any]) -> set[tuple[str, int]]:
    refs = row.get("expected_evidence", row.get("evidence")) or []
    return {(ref["source"], int(ref["page"])) for ref in refs}


def _jaccard(a: set[Any], b: set[Any]) -> float | None:
    """Jaccard; ``None`` si ambos están vacíos (caso sin evidencia: no aporta)."""
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def _kappa(x: list[str], y: list[str]) -> float:
    if len(set(x) | set(y)) == 1:  # una sola clase en ambos: acuerdo perfecto trivial
        return 1.0
    return float(cohen_kappa_score(x, y))


def fleiss_kappa(ratings: list[list[str]]) -> float:
    """κ de Fleiss para N ítems × n anotadores (categorías nominales)."""
    n = len(ratings[0])
    categories = sorted({r for item in ratings for r in item})
    counts = [Counter(item) for item in ratings]
    p_j = [sum(c[cat] for c in counts) / (len(ratings) * n) for cat in categories]
    p_i = [(sum(v * v for v in c.values()) - n) / (n * (n - 1)) for c in counts]
    p_bar = sum(p_i) / len(p_i)
    p_e = sum(p * p for p in p_j)
    if p_e == 1.0:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)


def _mean(values: list[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def main() -> dict[str, Any]:
    passes = {who: _load(name) for who, name in PASSES.items()}
    adjudicated = _load("adjudicated.jsonl")
    ids = sorted(adjudicated)
    for who, rows in passes.items():
        missing = set(ids) - set(rows)
        if missing:
            raise SystemExit(f"{who} no etiquetó: {sorted(missing)}")

    result: dict[str, Any] = {"n_cases": len(ids), "fields": {}, "evidence": {}}
    for field in CATEGORICAL:
        values = {who: [str(passes[who][i].get(field)) for i in ids] for who in passes}
        pairs = {}
        for a, b in combinations(passes, 2):
            agree = sum(x == y for x, y in zip(values[a], values[b], strict=True)) / len(ids)
            pairs[f"{a}~{b}"] = {"agreement": agree, "kappa": _kappa(values[a], values[b])}
        ratings = [[values[who][k] for who in passes] for k in range(len(ids))]
        unanimous = sum(len(set(r)) == 1 for r in ratings)
        majority = [Counter(r).most_common(1)[0] for r in ratings]
        adj_values = [str(adjudicated[i].get(field)) for i in ids]
        overridden = [
            ids[k]
            for k, (label, votes) in enumerate(majority)
            if votes >= 2 and label != adj_values[k]
        ]
        disagreements = [ids[k] for k, r in enumerate(ratings) if len(set(r)) > 1]
        result["fields"][field] = {
            "pairs": pairs,
            "fleiss_kappa": fleiss_kappa(ratings),
            "unanimous": unanimous,
            "disagreements": disagreements,
            "adjudication_overrode_majority": overridden,
        }

    for level in ("page", "source"):
        pairs = {}
        for a, b in combinations(passes, 2):
            scores = []
            for i in ids:
                ea, eb = _evidence(passes[a][i]), _evidence(passes[b][i])
                if level == "source":
                    ea, eb = {s for s, _ in ea}, {s for s, _ in eb}
                scores.append(_jaccard(ea, eb))
            pairs[f"{a}~{b}"] = _mean(scores)
        vs_adj = {}
        for who in passes:
            scores = []
            for i in ids:
                ea, eb = _evidence(passes[who][i]), _evidence(adjudicated[i])
                if level == "source":
                    ea, eb = {s for s, _ in ea}, {s for s, _ in eb}
                scores.append(_jaccard(ea, eb))
            vs_adj[who] = _mean(scores)
        mean_pairs = _mean(list(pairs.values()))
        result["evidence"][level] = {"pairs": pairs, "mean_pairwise": mean_pairs, "vs_adj": vs_adj}

    changed = [
        i
        for i in ids
        if any(str(passes["autor"][i].get(f)) != str(adjudicated[i].get(f)) for f in CATEGORICAL)
        or _evidence(passes["autor"][i]) != _evidence(adjudicated[i])
        or passes["autor"][i]["question"] != adjudicated[i]["question"]
    ]
    any_disagreement = sorted(
        {i for f in result["fields"].values() for i in f["disagreements"]}
        | {i for i in ids if len({frozenset(_evidence(passes[w][i])) for w in passes}) > 1}
    )
    result["cases_with_disagreement"] = any_disagreement
    result["cases_changed_by_adjudication"] = changed

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "annotation_agreement.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    (OUT / "annotation_agreement.md").write_text(_markdown(result), "utf-8")
    return result


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Acuerdo entre anotadores del banco",
        "",
        f"{result['n_cases']} casos (42 test + 18 dev) etiquetados por tres pasadas "
        "independientes (autor, anotador A y anotador B) y adjudicados después. "
        "Generado por `scripts/annotation_agreement.py`.",
        "",
        "## Campos categóricos",
        "",
        "| Campo | κ autor–A | κ autor–B | κ A–B | κ Fleiss | Unánimes | Mayoría corregida |",
        "|---|---|---|---|---|---|---|",
    ]
    for field, data in result["fields"].items():
        p = data["pairs"]
        lines.append(
            f"| {LABELS_ES[field]} | {_fmt(p['autor~anotador_a']['kappa'])} "
            f"| {_fmt(p['autor~anotador_b']['kappa'])} "
            f"| {_fmt(p['anotador_a~anotador_b']['kappa'])} | {_fmt(data['fleiss_kappa'])} "
            f"| {data['unanimous']}/{result['n_cases']} "
            f"| {len(data['adjudication_overrode_majority'])} |"
        )
    lines += [
        "",
        "Acuerdo observado por par:",
        "",
        "| Campo | autor–A | autor–B | A–B |",
        "|---|---|---|---|",
    ]
    for field, data in result["fields"].items():
        p = data["pairs"]
        lines.append(
            f"| {LABELS_ES[field]} | {p['autor~anotador_a']['agreement']:.1%} "
            f"| {p['autor~anotador_b']['agreement']:.1%} "
            f"| {p['anotador_a~anotador_b']['agreement']:.1%} |"
        )
    lines += [
        "",
        "## Evidencia",
        "",
        "Jaccard medio sobre los casos con evidencia (los casos sin evidencia en ambas "
        "pasadas no cuentan).",
        "",
        "| Nivel | autor–A | autor–B | A–B | Media | autor vs adj. | A vs adj. | B vs adj. |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for level, data in result["evidence"].items():
        p, v = data["pairs"], data["vs_adj"]
        name = "(fuente, página)" if level == "page" else "fuente"
        lines.append(
            f"| {name} | {_fmt(p['autor~anotador_a'])} | {_fmt(p['autor~anotador_b'])} "
            f"| {_fmt(p['anotador_a~anotador_b'])} | {_fmt(data['mean_pairwise'])} "
            f"| {_fmt(v['autor'])} | {_fmt(v['anotador_a'])} | {_fmt(v['anotador_b'])} |"
        )
    lines += [
        "",
        "## Desacuerdos",
        "",
        f"- Casos con algún desacuerdo: {len(result['cases_with_disagreement'])} de "
        f"{result['n_cases']}.",
        f"- Casos que la adjudicación cambió respecto del autor: "
        f"{len(result['cases_changed_by_adjudication'])} "
        f"({', '.join(result['cases_changed_by_adjudication']) or 'ninguno'}).",
    ]
    for field, data in result["fields"].items():
        if data["disagreements"]:
            lines.append(f"- {LABELS_ES[field]}: {', '.join(data['disagreements'])}.")
        if data["adjudication_overrode_majority"]:
            lines.append(
                f"  - La adjudicación se apartó de la mayoría en: "
                f"{', '.join(data['adjudication_overrode_majority'])}."
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    summary = main()
    behavior = summary["fields"]["expected_behavior"]
    print(
        f"κ Fleiss comportamiento={behavior['fleiss_kappa']:.3f} · "
        f"Jaccard páginas={summary['evidence']['page']['mean_pairwise']:.3f} · "
        f"desacuerdos={len(summary['cases_with_disagreement'])}"
    )
