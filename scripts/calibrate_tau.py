"""Calibración del umbral τ del gate de evidencia.

El agente se abstiene si el score máximo del reranker (sigmoide) queda bajo τ. Este script
mide ese score con el recuperador final (``settings.retrieval``) sobre un conjunto de
calibración **disjunto del test**:

* casos del split ``dev`` del banco con intención documental (responder vs abstenerse);
* ``eval/calibration/tau_probes.jsonl``: preguntas respondibles con evidencia verificada y
  preguntas cercanas al dominio sin respuesta en el corpus.

Elige τ maximizando la exactitud balanceada (TPR de respondibles que pasan + TNR de no
respondibles que se abstienen) y, en empate, el centro de la meseta óptima más ancha.
Como control *post hoc* (no interviene en la elección) reporta el mismo umbral sobre el
split ``test``. Escribe ``reports/retrieval/tau_calibration.{json,md}`` y la figura.

Uso::

    PYTHONPATH=src python scripts/calibrate_tau.py [--apply]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docuerp import paths  # noqa: E402
from docuerp.config import get_settings  # noqa: E402
from docuerp.eval.benchmark import BenchmarkCase, load_benchmark  # noqa: E402
from docuerp.retrieval.contracts import RetrievalQuery  # noqa: E402
from docuerp.retrieval.pipeline import build_retriever  # noqa: E402

PROBES = ROOT / "eval" / "calibration" / "tau_probes.jsonl"
OUT = ROOT / "reports" / "retrieval"
GRID = [round(i / 100, 2) for i in range(0, 101)]


def _doc_label(case: BenchmarkCase) -> str | None:
    """``answerable`` / ``unanswerable`` para casos documentales; ``None`` si no aplica."""
    if case.expected_intent != "doc_query":
        return None
    if case.expected_behavior == "answer" and case.answerable:
        return "answerable"
    if case.expected_behavior in ("abstain", "deny") or case.category in (
        "unanswerable",
        "sec_hallucination",
    ):
        return "unanswerable"
    return None


def _items() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calib: list[dict[str, Any]] = []
    for case in load_benchmark(split="dev"):
        label = _doc_label(case)
        if label:
            calib.append(
                {
                    "id": case.id,
                    "origin": "dev",
                    "question": case.question,
                    "role": case.role,
                    "erp_version": case.erp_version,
                    "label": label,
                    "evidence": [e.model_dump() for e in case.evidence],
                }
            )
    if PROBES.is_file():
        for k, line in enumerate(PROBES.read_text("utf-8").splitlines()):
            if line.strip():
                probe = json.loads(line)
                probe.setdefault("id", f"PRB-{k + 1:03d}")
                probe["origin"] = "probe"
                calib.append(probe)
    test: list[dict[str, Any]] = []
    for case in load_benchmark(split="test"):
        label = _doc_label(case)
        if label:
            test.append(
                {
                    "id": case.id,
                    "origin": "test",
                    "question": case.question,
                    "role": case.role,
                    "erp_version": case.erp_version,
                    "label": label,
                    "evidence": [e.model_dump() for e in case.evidence],
                }
            )
    return calib, test


def _score(items: list[dict[str, Any]], retriever: Any, top_k: int, cand_k: int) -> None:
    for item in items:
        started = time.perf_counter()
        result = retriever.retrieve(
            RetrievalQuery(
                text=item["question"],
                role=item["role"],
                erp_version=item["erp_version"],
                top_k=top_k,
                candidates_k=cand_k,
            )
        )
        item["top_score"] = result.top_score if result.top_score is not None else 0.0
        item["latency_ms"] = (time.perf_counter() - started) * 1000
        wanted = {(e["source"], int(e["page"])) for e in item.get("evidence", [])}
        item["evidence_in_top"] = (
            any(
                (h.chunk.source, p) in wanted
                for h in result.hits
                for p in range(h.chunk.page, (h.chunk.page_end or h.chunk.page) + 1)
            )
            if wanted
            else None
        )
        item["top_source"] = result.hits[0].chunk.source if result.hits else None
        item["_hits"] = result.hits


async def _generate(items: list[dict[str, Any]], concurrency: int) -> None:
    """Genera la respuesta con el contexto recuperado **sin gate** y guarda si el LLM se
    abstuvo por su cuenta (regla 5 del prompt). Así se calibra la decisión del sistema."""
    from docuerp.generation.generator import generate_answer
    from docuerp.llm import get_provider

    provider = get_provider()
    semaphore = asyncio.Semaphore(concurrency)

    async def one(item: dict[str, Any]) -> None:
        hits = item.get("_hits") or []
        if not hits:
            item["llm_abstained"] = True
            return
        async with semaphore:
            answer, _ = await generate_answer(
                item["question"],
                hits,
                intent="doc_query",
                provider=provider,
                erp_version=item["erp_version"],
            )
        item["llm_abstained"] = bool(answer.abstained)

    await asyncio.gather(*(one(i) for i in items))


def _system_curve(items: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Responde si pasa el gate (score ≥ τ) **y** el LLM no se abstuvo."""
    pos = [i for i in items if i["label"] == "answerable"]
    neg = [i for i in items if i["label"] == "unanswerable"]
    rows = []
    for tau in GRID:

        def answered(i: dict[str, Any], tau: float = tau) -> bool:
            return i["top_score"] >= tau and not i["llm_abstained"]

        tpr = sum(answered(i) for i in pos) / len(pos) if pos else 0.0
        tnr = sum(not answered(i) for i in neg) / len(neg) if neg else 0.0
        rows.append({"tau": tau, "tpr": tpr, "tnr": tnr, "balanced_accuracy": (tpr + tnr) / 2})
    return rows


def _curve(items: list[dict[str, Any]]) -> list[dict[str, float]]:
    pos = [i["top_score"] for i in items if i["label"] == "answerable"]
    neg = [i["top_score"] for i in items if i["label"] == "unanswerable"]
    rows = []
    for tau in GRID:
        tpr = sum(s >= tau for s in pos) / len(pos) if pos else 0.0
        tnr = sum(s < tau for s in neg) / len(neg) if neg else 0.0
        rows.append({"tau": tau, "tpr": tpr, "tnr": tnr, "balanced_accuracy": (tpr + tnr) / 2})
    return rows


def _select(curve: list[dict[str, float]]) -> float:
    best = max(r["balanced_accuracy"] for r in curve)
    # Mesetas contiguas con la exactitud óptima; se toma el centro de la más ancha.
    plateaus: list[list[float]] = []
    for r in curve:
        if abs(r["balanced_accuracy"] - best) < 1e-12:
            if plateaus and abs(plateaus[-1][-1] - (r["tau"] - 0.01)) < 1e-9:
                plateaus[-1].append(r["tau"])
            else:
                plateaus.append([r["tau"]])
    widest = max(plateaus, key=len)
    return round((widest[0] + widest[-1]) / 2, 2)


def _at(curve: list[dict[str, float]], tau: float) -> dict[str, float]:
    return min(curve, key=lambda r: abs(r["tau"] - tau))


def _auc(items: list[dict[str, Any]]) -> float | None:
    pos = [i["top_score"] for i in items if i["label"] == "answerable"]
    neg = [i["top_score"] for i in items if i["label"] == "unanswerable"]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _figure(
    calib: list[dict[str, Any]],
    curve: list[dict[str, float]],
    tau: float,
    gate_curve: list[dict[str, float]] | None = None,
) -> None:
    import runpy

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = runpy.run_path(str(paths.DESIGN / "matplotlib_style.py"))
    style["apply_style"]()
    colors = style["COLORS"]
    tokens = json.loads(paths.DESIGN_TOKENS.read_text("utf-8"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    pos = [i["top_score"] for i in calib if i["label"] == "answerable"]
    neg = [i["top_score"] for i in calib if i["label"] == "unanswerable"]
    bins = [i / 20 for i in range(21)]
    ax1.hist(pos, bins=bins, alpha=0.85, color=tokens["chart"]["final"], label="Respondibles")
    ax1.hist(neg, bins=bins, alpha=0.75, color=tokens["chart"]["baseline"], label="No respondibles")
    ax1.axvline(tau, color=colors.get("text", "#e6e6e6"), ls="--", lw=1.2)
    ax1.set_xlabel("Score máximo del reranker (sigmoide)")
    ax1.set_ylabel("Consultas")
    ax1.set_title(f"Distribución de scores (τ = {tau:.2f})")
    ax1.legend(frameon=False)
    taus = [r["tau"] for r in curve]
    ax2.plot(
        taus, [r["tpr"] for r in curve], color=tokens["chart"]["final"], label="TPR (responde)"
    )
    ax2.plot(
        taus,
        [r["tnr"] for r in curve],
        color=tokens["chart"]["baseline"],
        label="TNR (se abstiene)",
    )
    ax2.plot(
        taus,
        [r["balanced_accuracy"] for r in curve],
        color=colors.get("text", "#e6e6e6"),
        lw=1.6,
        label="Exactitud balanceada",
    )
    if gate_curve is not None:
        ax2.plot(
            taus,
            [r["balanced_accuracy"] for r in gate_curve],
            color=colors.get("muted", "#8a8f98"),
            lw=1.2,
            ls=":",
            label="Exactitud balanceada (solo gate)",
        )
    ax2.axvline(tau, color=colors.get("text", "#e6e6e6"), ls="--", lw=1.2)
    ax2.set_xlabel("τ")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Curva de calibración (dev + sondas)")
    ax2.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "tau_calibration.png")
    fig.savefig(figures / "tau_calibration.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="escribe τ en settings.yaml")
    parser.add_argument(
        "--gate-only",
        action="store_true",
        help="calibra solo el gate (sin generar); por defecto se calibra el sistema completo",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--tag", default="", help="sufijo de los archivos de salida (experimentos, p. ej. top_k)"
    )
    args = parser.parse_args()

    settings = get_settings()
    cfg = settings.retrieval
    retriever = build_retriever(cfg.mode, cfg.chunker, cfg.embedder)
    calib, test = _items()
    _score(calib, retriever, cfg.top_k, cfg.candidates_k)
    _score(test, retriever, cfg.top_k, cfg.candidates_k)

    gate_curve = _curve(calib)
    gate_tau = _select(gate_curve)
    system = not args.gate_only
    if system:
        asyncio.run(_generate(calib + test, args.concurrency))
        curve = _system_curve(calib)
        test_curve = _system_curve(test)
    else:
        curve = gate_curve
        test_curve = _curve(test)
    tau = _select(curve)
    for item in calib + test:
        item.pop("_hits", None)
    result = {
        "retriever": {
            "mode": cfg.mode,
            "chunker": cfg.chunker,
            "embedder": cfg.embedder,
            "top_k": cfg.top_k,
        },
        "decision": "sistema (gate + abstención del LLM)" if system else "solo gate",
        "generator_model": settings.llm.model if system else None,
        "criterion": "máxima exactitud balanceada en dev + sondas; empate → centro de la "
        "meseta óptima más ancha",
        "gate_only_tau": gate_tau,
        "gate_only_at_tau": _at(gate_curve, gate_tau),
        "gate_only_curve": gate_curve,
        "n": {
            "calibration_answerable": sum(i["label"] == "answerable" for i in calib),
            "calibration_unanswerable": sum(i["label"] == "unanswerable" for i in calib),
            "test_answerable": sum(i["label"] == "answerable" for i in test),
            "test_unanswerable": sum(i["label"] == "unanswerable" for i in test),
        },
        "auc_calibration": _auc(calib),
        "auc_test_post_hoc": _auc(test),
        "tau_previous": cfg.tau,
        "tau_selected": tau,
        "at_previous": _at(curve, cfg.tau),
        "at_selected": _at(curve, tau),
        "test_post_hoc_at_previous": _at(test_curve, cfg.tau),
        "test_post_hoc_at_selected": _at(test_curve, tau),
        "evidence_in_top5_answerable": _rate(
            [i["evidence_in_top"] for i in calib if i["label"] == "answerable"]
        ),
        "curve": curve,
        "items": calib,
        "test_items": test,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"tau_calibration_{args.tag}" if args.tag else "tau_calibration"
    (OUT / f"{stem}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    (OUT / f"{stem}.md").write_text(_markdown(result), "utf-8")
    if not args.tag:
        _figure(calib, curve, tau, gate_curve if system else None)
    if args.apply and abs(tau - cfg.tau) > 1e-9:
        yaml_path = ROOT / "configs" / "settings.yaml"
        text = yaml_path.read_text("utf-8")
        old = f"  tau: {cfg.tau:.2f}"
        if old not in text:
            raise SystemExit(f"No encontré «{old}» en settings.yaml")
        yaml_path.write_text(text.replace(old, f"  tau: {tau:.2f}", 1), "utf-8")
    print(
        f"τ={tau:.2f} (antes {cfg.tau:.2f}) · exactitud balanceada "
        f"{result['at_selected']['balanced_accuracy']:.3f} · AUC {result['auc_calibration']:.3f}"
    )


def _rate(values: list[bool | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def _markdown(r: dict[str, Any]) -> str:
    def row(name: str, data: dict[str, float]) -> str:
        return (
            f"| {name} | {data['tau']:.2f} | {data['tpr']:.3f} | {data['tnr']:.3f} "
            f"| {data['balanced_accuracy']:.3f} |"
        )

    n = r["n"]
    system = r["generator_model"] is not None

    def answered(i: dict[str, Any]) -> bool:
        passed = i["top_score"] >= r["tau_selected"]
        return passed and not i.get("llm_abstained", False) if system else passed

    misses = [i for i in r["items"] if (i["label"] == "answerable") != answered(i)]
    lines = [
        "# Calibración del umbral τ del gate de evidencia",
        "",
        f"Recuperador: `{r['retriever']['mode']}` · chunker `{r['retriever']['chunker']}` · "
        f"embedder `{r['retriever']['embedder']}`. Generado por `scripts/calibrate_tau.py`.",
        "",
        f"Conjunto de calibración (disjunto del test): {n['calibration_answerable']} "
        f"respondibles y {n['calibration_unanswerable']} no respondibles (casos dev del banco "
        "y sondas verificadas de `eval/calibration/tau_probes.jsonl`).",
        f"Decisión calibrada: {r['decision']}"
        + (f" con `{r['generator_model']}`" if system else "")
        + f". Criterio: {r['criterion']}.",
        f"Solo con el score del gate, el mismo criterio daría τ = {r['gate_only_tau']:.2f} "
        f"(exactitud balanceada {r['gate_only_at_tau']['balanced_accuracy']:.3f}).",
        "",
        "| Conjunto | τ | TPR (responde) | TNR (se abstiene) | Exactitud balanceada |",
        "|---|---|---|---|---|",
        row("Calibración · τ anterior", r["at_previous"]),
        row("Calibración · τ elegido", r["at_selected"]),
        row("Test (post hoc) · τ anterior", r["test_post_hoc_at_previous"]),
        row("Test (post hoc) · τ elegido", r["test_post_hoc_at_selected"]),
        "",
        f"AUC del score como separador: calibración {r['auc_calibration']:.3f}"
        + (
            f", test post hoc {r['auc_test_post_hoc']:.3f}."
            if r["auc_test_post_hoc"] is not None
            else "."
        ),
        f"Evidencia esperada dentro del top-5 en las respondibles de calibración: "
        f"{(r['evidence_in_top5_answerable'] or 0):.1%}.",
        "",
        "El test no interviene en la elección: se reporta solo para mostrar que el umbral "
        "generaliza.",
        "",
        "![Calibración de τ](figures/tau_calibration.png)",
        "",
        f"## Errores con τ = {r['tau_selected']:.2f}",
        "",
        "| id | etiqueta | score | LLM se abstuvo | pregunta |",
        "|---|---|---|---|---|",
    ]
    for i in sorted(misses, key=lambda x: x["top_score"]):
        q = i["question"].replace("|", "/")
        abst = {True: "sí", False: "no"}.get(i.get("llm_abstained"), "—")
        lines.append(f"| {i['id']} | {i['label']} | {i['top_score']:.3f} | {abst} | {q[:110]} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
