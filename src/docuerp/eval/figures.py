"""Figuras del reporte de evaluación (estética *dark tech* con los tokens de diseño).

Cada función recibe datos ya agregados (los de ``metrics.json``), guarda la figura en PNG y
SVG dentro de ``figures_dir`` y devuelve la ruta del PNG, o ``None`` si no hay datos
suficientes para dibujarla (así el reporte se genera aunque falte alguna corrida).

El tema se carga desde ``design/matplotlib_style.py`` y ``design/tokens.json``. Matplotlib
se importa de forma perezosa con el backend ``Agg`` (sin ventana) para poder correr en
servidores y en CI.
"""

from __future__ import annotations

import runpy
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from docuerp import paths
from docuerp.eval import metrics
from docuerp.eval.benchmark import BenchmarkCase
from docuerp.eval.runs import read_json

Summary = Mapping[str, Any]
"""Resumen ``{mean, ci_low, ci_high, n}`` de una métrica."""

BAR_METRICS: tuple[tuple[str, str], ...] = (
    ("behavior_correct", "Comportamiento\ncorrecto"),
    ("citation_precision", "Precisión\nde citas"),
    ("faithfulness", "Fidelidad"),
    ("correctness", "Corrección\n(juez, 0–1)"),
    ("abstention_correct", "Abstención\ncorrecta"),
    ("tool_selection", "Selección de\nherramienta"),
    ("attack_blocked", "Ataques\nbloqueados"),
)
"""Métricas del gráfico de barras baseline vs final (llave, etiqueta)."""

STAGE_LABELS: dict[str, str] = {
    "guards": "Guardias",
    "router": "Enrutador",
    "retrieval": "Recuperación",
    "rerank": "Rerank / MMR",
    "gates": "Gates",
    "tools": "Herramientas",
    "generation": "Generación (LLM)",
    "other": "Otras",
}


@lru_cache(maxsize=1)
def _theme() -> dict[str, Any]:
    """Carga el tema una sola vez: paleta, serie de colores y colores baseline/final."""
    module = runpy.run_path(str(paths.DESIGN / "matplotlib_style.py"))
    tokens = read_json(paths.DESIGN_TOKENS)
    return {
        "apply": module["apply_style"],
        "colors": module["COLORS"],
        "series": module["SERIES"],
        "baseline": tokens["chart"]["baseline"],
        "final": tokens["chart"]["final"],
    }


def _pyplot() -> Any:
    """Importa ``matplotlib.pyplot`` con backend sin ventana y aplica el tema oscuro."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _theme()["apply"]()
    return plt


def _save(fig: Any, figures_dir: Path, name: str) -> Path:
    """Guarda ``fig`` como ``name.png`` y ``name.svg`` y la cierra."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / f"{name}.png"
    fig.savefig(png)
    fig.savefig(figures_dir / f"{name}.svg")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return png


def _mean(summary: Summary | None) -> float | None:
    """Media de un resumen, o ``None`` si no hay datos."""
    return None if summary is None else summary.get("mean")


def _errors(summary: Summary) -> tuple[float, float]:
    """Brazos del IC (inferior, superior) alrededor de la media, nunca negativos."""
    mean = float(summary["mean"])
    low = summary.get("ci_low")
    high = summary.get("ci_high")
    return (
        max(0.0, mean - float(low if low is not None else mean)),
        max(0.0, float(high if high is not None else mean) - mean),
    )


def bars_baseline_vs_final(
    baseline: Mapping[str, Summary] | None,
    final: Mapping[str, Summary] | None,
    figures_dir: Path,
) -> Path | None:
    """Barras agrupadas baseline vs final con IC 95 % bootstrap para las métricas clave."""
    series = [
        ("Baseline (BM25)", baseline, _theme()["baseline"]),
        ("Final", final, _theme()["final"]),
    ]
    series = [s for s in series if s[1]]
    keys = [
        (k, label)
        for k, label in BAR_METRICS
        if any(_mean((agg or {}).get(k)) is not None for _, agg, _ in series)
    ]
    if not series or not keys:
        return None
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    width = 0.8 / len(series)
    for s_index, (label, agg, color) in enumerate(series):
        xs, ys, lows, highs = [], [], [], []
        for k_index, (key, _) in enumerate(keys):
            summary = (agg or {}).get(key)
            if _mean(summary) is None:
                continue
            xs.append(k_index + (s_index - (len(series) - 1) / 2) * width)
            ys.append(float(summary["mean"]))  # type: ignore[index]
            low, high = _errors(summary)  # type: ignore[arg-type]
            lows.append(low)
            highs.append(high)
        ax.bar(xs, ys, width=width * 0.92, color=color, label=label, zorder=3)
        ax.errorbar(
            xs,
            ys,
            yerr=[lows, highs],
            fmt="none",
            ecolor=_theme()["colors"]["text"],
            elinewidth=1.2,
            capsize=3,
            zorder=4,
        )
        for x, y in zip(xs, ys, strict=True):
            ax.text(
                x,
                min(y + 0.04, 1.02),
                f"{y:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=_theme()["colors"]["text"],
            )
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([label for _, label in keys])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Valor (0–1)")
    ax.set_title("Baseline vs sistema final · media con IC 95 % bootstrap")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    return _save(fig, figures_dir, "bars_baseline_vs_final")


def recall_curves(
    retrieval: Mapping[str, Mapping[str, Any]] | None, figures_dir: Path, *, max_configs: int = 8
) -> Path | None:
    """Curvas de Recall@k (k = 1, 3, 5, 10) por configuración de recuperación."""
    if not retrieval:
        return None
    ks = list(metrics.RETRIEVAL_KS)
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    palette = _theme()["series"]
    ranked = sorted(
        retrieval.items(),
        key=lambda item: -(_mean(item[1]["aggregate"].get("recall@5")) or 0.0),
    )[:max_configs]
    for index, (label, data) in enumerate(ranked):
        ys = [_mean(data["aggregate"].get(f"recall@{k}")) for k in ks]
        if any(y is None for y in ys):
            continue
        ax.plot(ks, ys, marker="o", color=palette[index % len(palette)], label=label)
    ax.axhline(0.85, color=_theme()["colors"]["warning"], linestyle="--", linewidth=1.0)
    ax.text(
        ks[-1],
        0.86,
        "meta Recall@5 = 0.85",
        ha="right",
        va="bottom",
        fontsize=8,
        color=_theme()["colors"]["warning"],
    )
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_title("Recall@k por configuración de recuperación")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, figures_dir, "recall_curves")


def _ablation_groups(
    retrieval: Mapping[str, Mapping[str, Any]], pivot: str
) -> dict[str, list[tuple[str, Mapping[str, Any]]]]:
    """Agrupa configuraciones que difieren en un solo factor.

    ``pivot`` ∈ {``mode``, ``chunker``, ``embedder``}: para cada valor de los otros dos
    factores se obtiene la lista de configuraciones que varían solo ``pivot``.
    """
    factors = ("mode", "chunker", "embedder")
    groups: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for data in retrieval.values():
        rest = "/".join(str(data[f]) for f in factors if f != pivot)
        groups.setdefault(rest, []).append((str(data[pivot]), data))
    return groups


def ablation(retrieval: Mapping[str, Mapping[str, Any]] | None, figures_dir: Path) -> Path | None:
    """Ablación de reranker (modo), chunking y embedder: Recall@5 y MRR con IC.

    Para cada factor se elige el grupo con más variantes (el resto de factores fijo).
    """
    if not retrieval:
        return None
    panels: list[tuple[str, str, list[tuple[str, Mapping[str, Any]]]]] = []
    for pivot, title in (
        ("mode", "Recuperador / reranker"),
        ("chunker", "Chunking"),
        ("embedder", "Embedder"),
    ):
        groups = _ablation_groups(retrieval, pivot)
        rest, items = max(groups.items(), key=lambda kv: len(kv[1]))
        if len(items) >= 2:
            panels.append((title, rest, items))
    if not panels:
        return None
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6), squeeze=False)
    colors = [_theme()["final"], _theme()["colors"]["info"]]
    for ax, (title, rest, items) in zip(axes[0], panels, strict=True):
        order = sorted(items, key=lambda it: it[0])
        for m_index, (metric, label) in enumerate((("recall@5", "Recall@5"), ("mrr", "MRR"))):
            xs, ys, lows, highs = [], [], [], []
            for i, (_, data) in enumerate(order):
                summary = data["aggregate"].get(metric)
                if _mean(summary) is None:
                    continue
                xs.append(i + (m_index - 0.5) * 0.38)
                ys.append(float(summary["mean"]))
                low, high = _errors(summary)
                lows.append(low)
                highs.append(high)
            ax.bar(xs, ys, width=0.34, color=colors[m_index], label=label, zorder=3)
            ax.errorbar(
                xs,
                ys,
                yerr=[lows, highs],
                fmt="none",
                ecolor=_theme()["colors"]["text"],
                elinewidth=1.0,
                capsize=3,
                zorder=4,
            )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([name for name, _ in order], rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{title}\n({rest})", fontsize=11)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9, bbox_to_anchor=(1.0, 1.12))
    fig.suptitle(
        "Ablaciones de recuperación · media con IC 95 %",
        x=0.01,
        y=1.12,
        ha="left",
        color=_theme()["colors"]["text"],
    )
    return _save(fig, figures_dir, "ablation")


def latency_stack(operations: Mapping[str, Mapping[str, Any]], figures_dir: Path) -> Path | None:
    """Barras horizontales apiladas: latencia media por grupo de etapas y variante."""
    variants = {
        name: op["latency_by_stage_ms"]
        for name, op in operations.items()
        if op.get("latency_by_stage_ms")
    }
    if not variants:
        return None
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(10, 3.2 + 0.5 * len(variants)))
    names = list(variants)
    palette = _theme()["series"]
    left = [0.0] * len(names)
    for g_index, group in enumerate(metrics.STAGE_GROUPS):
        values = [float((variants[n].get(group) or {}).get("mean") or 0.0) for n in names]
        if not any(values):
            continue
        ax.barh(
            names,
            values,
            left=left,
            color=palette[g_index % len(palette)],
            label=STAGE_LABELS[group],
            zorder=3,
            height=0.55,
        )
        left = [a + b for a, b in zip(left, values, strict=True)]
    for y, total in enumerate(left):
        ax.text(
            total,
            y,
            f"  {total:,.0f} ms",
            va="center",
            fontsize=9,
            color=_theme()["colors"]["text"],
        )
    ax.set_xlabel("Latencia media por consulta (ms)")
    ax.set_title("Latencia por etapa (media)")
    ax.set_xlim(0, max(left) * 1.18)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    return _save(fig, figures_dir, "latency_stages")


def latency_hist(
    latencies: Mapping[str, Sequence[float]],
    figures_dir: Path,
    *,
    p50_target_ms: float = 4000.0,
    p95_target_ms: float = 10000.0,
) -> Path | None:
    """Histograma de latencia total con P50/P95 marcados y las metas del anteproyecto."""
    data = {name: [v / 1000.0 for v in values] for name, values in latencies.items() if values}
    if not data:
        return None
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = {"baseline": _theme()["baseline"], "final": _theme()["final"]}
    palette = _theme()["series"]
    top = max(max(v) for v in data.values())
    bins = [i * top / 24 for i in range(25)] if top > 0 else 10
    for index, (name, values) in enumerate(data.items()):
        color = colors.get(name, palette[index % len(palette)])
        ax.hist(values, bins=bins, color=color, alpha=0.65, label=name, zorder=3)
        for q, style in ((50, "-"), (95, ":")):
            ax.axvline(
                metrics.percentile(values, q), color=color, linestyle=style, linewidth=1.6, zorder=4
            )
    for target, text in ((p50_target_ms, "meta P50"), (p95_target_ms, "meta P95")):
        ax.axvline(
            target / 1000.0,
            color=_theme()["colors"]["warning"],
            linestyle="--",
            linewidth=1.0,
            zorder=2,
        )
        ax.text(
            target / 1000.0,
            ax.get_ylim()[1] * 0.97,
            f" {text}",
            fontsize=8,
            va="top",
            color=_theme()["colors"]["warning"],
        )
    ax.set_xlabel("Latencia total (s) · línea continua = P50, punteada = P95")
    ax.set_ylabel("Consultas")
    ax.set_title("Distribución de latencia total")
    ax.legend()
    return _save(fig, figures_dir, "latency_hist")


def _heatmap(ax: Any, confusion: Mapping[str, Any], title: str) -> None:
    """Dibuja una matriz de confusión con los conteos anotados."""
    from matplotlib.colors import LinearSegmentedColormap

    colors = _theme()["colors"]
    cmap = LinearSegmentedColormap.from_list("docuerp", [colors["surface"], colors["accent"]])
    labels = confusion["labels"]
    matrix = confusion["matrix"]
    ax.grid(False)
    ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    peak = max((max(row) for row in matrix), default=0)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value:
                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=colors["bg"] if value > peak / 2 else colors["text"],
                )
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Esperado")
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(True)


def confusion_matrices(confusion: Mapping[str, Any] | None, figures_dir: Path) -> Path | None:
    """Matrices de confusión de intenciones y de herramientas (sistema final)."""
    if not confusion:
        return None
    panels = [
        (confusion[key], title)
        for key, title in (("intent", "Intenciones"), ("tool", "Herramientas"))
        if confusion.get(key) and confusion[key]["labels"]
    ]
    if not panels:
        return None
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.2), squeeze=False)
    for ax, (data, title) in zip(axes[0], panels, strict=True):
        _heatmap(ax, data, title)
    return _save(fig, figures_dir, "confusion_matrices")


def security_matrix(
    cases: Sequence[BenchmarkCase],
    per_case: Mapping[str, Mapping[str, Mapping[str, float | None]]],
    figures_dir: Path,
) -> Path | None:
    """Matriz de seguridad: cada caso de seguridad × variante, coloreada por tasa de acierto.

    Args:
        cases: casos del banco (se usan solo los de seguridad).
        per_case: ``variante → caso → métricas``; se grafica ``behavior_correct``.
        figures_dir: carpeta de salida.
    """
    sec = [c for c in cases if c.is_security_case]
    variants = [v for v, table in per_case.items() if any(c.id in table for c in sec)]
    if not sec or not variants:
        return None
    from matplotlib.colors import LinearSegmentedColormap

    colors = _theme()["colors"]
    plt = _pyplot()
    cmap = LinearSegmentedColormap.from_list("pass", [colors["danger"], colors["accent"]])
    order = sorted(sec, key=lambda c: (c.category, c.id))
    grid = [[per_case[v].get(c.id, {}).get("behavior_correct") for v in variants] for c in order]
    fig, ax = plt.subplots(figsize=(3 + 2.2 * len(variants), 0.42 * len(order) + 2.0))
    ax.grid(False)
    shown = [[float("nan") if x is None else x for x in row] for row in grid]
    ax.imshow(shown, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{c.id} · {c.category.removeprefix('sec_')}" for c in order], fontsize=9)
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            text = "—" if value is None else ("✔" if value >= 0.999 else f"{value:.2f}")
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=colors["bg"])
    ax.set_title("Matriz de seguridad · comportamiento esperado cumplido por caso")
    for spine in ax.spines.values():
        spine.set_visible(True)
    return _save(fig, figures_dir, "security_matrix")
