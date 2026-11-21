"""Tema oscuro de Matplotlib derivado de ``design/tokens.json`` (SPEC §6).

Uso::

    from design.matplotlib_style import apply_style, COLORS
    apply_style()

o, sin que ``design`` sea importable como paquete::

    import runpy; runpy.run_path("design/matplotlib_style.py")["apply_style"]()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TOKENS_PATH = Path(__file__).resolve().with_name("tokens.json")


def load_tokens(path: Path = TOKENS_PATH) -> dict[str, Any]:
    """Lee los tokens de diseño."""
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


_TOKENS = load_tokens()
COLORS: dict[str, str] = _TOKENS["color"]
"""Paleta base (bg, surface, text, accent…)."""
SERIES: list[str] = _TOKENS["chart"]["series"]
"""Ciclo de colores para series categóricas."""


def apply_style(*, font_scale: float = 1.0, transparent: bool = False) -> dict[str, Any]:
    """Aplica el tema oscuro DocuERP a ``matplotlib.rcParams`` y lo devuelve.

    Args:
        font_scale: multiplicador de tamaños de fuente (p. ej. 1.2 para diapositivas).
        transparent: si ``True``, guarda figuras con fondo transparente.
    """
    import matplotlib as mpl
    from cycler import cycler
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    sans = [f for f in _TOKENS["font"]["sans"] if f in available] or ["DejaVu Sans"]
    mono = [f for f in _TOKENS["font"]["mono"] if f in available] or ["DejaVu Sans Mono"]

    c = COLORS
    params: dict[str, Any] = {
        "figure.facecolor": c["bg"],
        "figure.edgecolor": c["bg"],
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.facecolor": "none" if transparent else c["bg"],
        "savefig.edgecolor": "none" if transparent else c["bg"],
        "savefig.bbox": "tight",
        "savefig.transparent": transparent,
        "axes.facecolor": c["surface"],
        "axes.edgecolor": c["border"],
        "axes.linewidth": 1.0,
        "axes.labelcolor": c["text"],
        "axes.titlecolor": c["text"],
        "axes.titleweight": "semibold",
        "axes.titlesize": 13 * font_scale,
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 11 * font_scale,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=SERIES),
        "grid.color": _TOKENS["chart"]["grid"],
        "grid.linewidth": 0.8,
        "grid.alpha": 0.9,
        "xtick.color": c["muted"],
        "ytick.color": c["muted"],
        "xtick.labelcolor": c["muted"],
        "ytick.labelcolor": c["muted"],
        "xtick.labelsize": 10 * font_scale,
        "ytick.labelsize": 10 * font_scale,
        "text.color": c["text"],
        "font.family": "sans-serif",
        "font.sans-serif": sans,
        "font.monospace": mono,
        "font.size": 11 * font_scale,
        "legend.frameon": True,
        "legend.facecolor": c["surface-2"],
        "legend.edgecolor": c["border"],
        "legend.labelcolor": c["text"],
        "legend.fontsize": 10 * font_scale,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "patch.edgecolor": c["bg"],
        "boxplot.boxprops.color": c["muted"],
        "boxplot.medianprops.color": c["accent"],
        "image.cmap": "viridis",
    }
    mpl.rcParams.update(params)
    return params


__all__ = ["COLORS", "SERIES", "TOKENS_PATH", "apply_style", "load_tokens"]
