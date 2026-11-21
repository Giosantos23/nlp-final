"""Diagramas del informe técnico (SVG con la identidad *dark tech* de DocuERP).

Dos figuras se dibujan a mano, con coordenadas explícitas, para que cada caja y cada flecha
correspondan a un componente real del código:

* :func:`architecture_svg`: pipeline de ingesta, rama documental, rama de herramientas con
  confirmación humana y elementos transversales (estado compartido, contrato, trazas).
* :func:`agent_graph_svg`: el ``StateGraph`` de LangGraph de ``docuerp.agent.graph``.

Los colores salen de ``design/tokens.json``. El SVG usa la familia Geist (cargada por la
página que lo contiene) con alternativas Inter y del sistema. ``scripts/build_report.py``
incrusta el SVG en el PDF (vectorial) y lo rasteriza a PNG con Chromium para el DOCX.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKENS = json.loads((ROOT / "design" / "tokens.json").read_text(encoding="utf-8"))
C = TOKENS["color"]

SANS = "Geist, Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'Geist Mono', 'JetBrains Mono', ui-monospace, Menlo, monospace"
BG = "#0B0D10"
PANEL = C["surface"]
PANEL_2 = "#16181D"


@dataclass(frozen=True)
class Rect:
    """Rectángulo de una caja (coordenadas del lienzo)."""

    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h


# ---------------------------------------------------------------------------
# Primitivas
# ---------------------------------------------------------------------------


def _defs() -> str:
    """Marcadores de flecha, patrón de puntos y brillo esmeralda."""
    markers = []
    for name, color in (
        ("flow", "#C9CED6"),
        ("exit", C["warning"]),
        ("data", C["info"]),
        ("hitl", C["violet"]),
        ("accent", C["accent"]),
    ):
        markers.append(
            f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,1 L9,5 L0,9 z" fill="{color}"/></marker>'
        )
    return (
        "<defs>"
        + "".join(markers)
        + '<pattern id="dots" width="18" height="18" patternUnits="userSpaceOnUse">'
        '<circle cx="1" cy="1" r="0.9" fill="#FFFFFF" fill-opacity="0.045"/></pattern>'
        '<filter id="glow" x="-20%" y="-20%" width="140%" height="140%">'
        '<feGaussianBlur stdDeviation="5" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
        '<linearGradient id="panelGrad" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{PANEL_2}"/><stop offset="1" stop-color="{PANEL}"/>'
        "</linearGradient>"
        "</defs>"
    )


def _text(
    x: float,
    y: float,
    content: str,
    *,
    size: float = 12,
    color: str = C["text"],
    weight: int = 400,
    family: str = SANS,
    anchor: str = "start",
    spacing: float = 0,
) -> str:
    """Un ``<text>`` con el contenido escapado."""
    extra = f' letter-spacing="{spacing}"' if spacing else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}"{extra}>'
        f"{escape(content)}</text>"
    )


def _diamond(x: float, y: float, size: float, color: str) -> str:
    """Rombo pequeño (marca de compuerta de decisión) centrado en ``(x, y)``."""
    s = size / 2
    return (
        f'<path d="M{x:.1f},{y - s:.1f} L{x + s:.1f},{y:.1f} L{x:.1f},{y + s:.1f} '
        f'L{x - s:.1f},{y:.1f} z" fill="{color}" fill-opacity="0.18" stroke="{color}" '
        'stroke-width="1.4"/>'
    )


def _box(
    r: Rect,
    title: str,
    lines: list[str] | tuple[str, ...] = (),
    *,
    kind: str = "default",
    tag: str | None = None,
    mono_lines: bool = False,
    title_size: float = 14,
    line_size: float = 12,
) -> str:
    """Caja con título y líneas de detalle.

    ``kind``: ``default`` | ``gate`` (rombo esmeralda) | ``io`` (borde esmeralda con brillo)
    | ``exit`` (ámbar punteado) | ``hitl`` (violeta punteado) | ``soft`` (sin borde marcado).
    """
    stroke, dash, fill, width = C["border"], "", "url(#panelGrad)", 1.2
    if kind == "io":
        stroke, width = C["accent"], 1.6
    elif kind == "gate":
        stroke = "#1F6B53"
    elif kind == "exit":
        stroke, dash = C["warning"], ' stroke-dasharray="5 4"'
    elif kind == "hitl":
        stroke, dash, width = C["violet"], ' stroke-dasharray="5 4"', 1.4
    parts = []
    if kind == "io":
        parts.append(
            f'<rect x="{r.x}" y="{r.y}" width="{r.w}" height="{r.h}" rx="11" fill="none" '
            f'stroke="{C["accent"]}" stroke-opacity="0.45" stroke-width="3" filter="url(#glow)"/>'
        )
    parts.append(
        f'<rect x="{r.x}" y="{r.y}" width="{r.w}" height="{r.h}" rx="11" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash}/>'
    )
    tx = r.x + 14
    ty = r.y + 22
    if kind == "gate":
        parts.append(_diamond(r.x + 19, ty - 4.5, 11, C["accent"]))
        tx += 14
    title_color = C["warning"] if kind == "exit" else C["text"]
    if kind == "hitl":
        title_color = C["violet"]
    parts.append(_text(tx, ty, title, size=title_size, weight=600, color=title_color))
    if tag:
        tw = 7.2 * len(tag) + 14
        parts.append(
            f'<rect x="{r.right - tw - 10}" y="{r.y + 9}" width="{tw}" height="17" rx="8.5" '
            f'fill="{C["accent"]}" fill-opacity="0.14" stroke="{C["accent"]}" '
            'stroke-opacity="0.45" stroke-width="0.8"/>'
        )
        parts.append(
            _text(
                r.right - tw / 2 - 10,
                r.y + 21.3,
                tag,
                size=9.5,
                weight=600,
                color=C["accent-2"],
                anchor="middle",
                spacing=0.8,
            )
        )
    family = MONO if mono_lines else SANS
    for i, line in enumerate(lines):
        parts.append(
            _text(
                r.x + 14,
                r.y + 42 + i * (line_size + 4.5),
                line,
                size=line_size,
                color=C["muted"],
                family=family,
            )
        )
    return "".join(parts)


def _path(
    points: list[tuple[float, float]],
    *,
    style: str = "flow",
    width: float = 1.6,
) -> str:
    """Polilínea ortogonal con punta de flecha al final."""
    color = {
        "flow": "#C9CED6",
        "exit": C["warning"],
        "data": C["info"],
        "hitl": C["violet"],
        "accent": C["accent"],
    }[style]
    dash = ' stroke-dasharray="6 5"' if style in {"exit", "hitl"} else ""
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linejoin="round"{dash} marker-end="url(#arrow-{style})"/>'
    )


def _label(x: float, y: float, content: str, *, color: str = C["accent-2"], **kw: object) -> str:
    """Etiqueta de arista en monoespaciada pequeña."""
    return _text(x, y, content, size=10.5, color=color, family=MONO, **kw)  # type: ignore[arg-type]


def _column_header(x: float, w: float, label: str) -> str:
    """Rótulo de columna con línea fina."""
    return (
        _text(x, 30, label, size=10.5, weight=600, color=C["accent-2"], spacing=1.4)
        + f'<line x1="{x}" y1="39" x2="{x + w}" y2="39" stroke="{C["border"]}" '
        'stroke-width="1"/>'
    )


def _svg(width: int, height: int, body: str) -> str:
    """Documento SVG con fondo grafito y patrón de puntos."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img">'
        + _defs()
        + f'<rect width="{width}" height="{height}" rx="16" fill="{BG}"/>'
        + f'<rect width="{width}" height="{height}" rx="16" fill="url(#dots)"/>'
        + body
        + "</svg>"
    )


# ---------------------------------------------------------------------------
# Figura: arquitectura
# ---------------------------------------------------------------------------


def architecture_svg() -> str:
    """Arquitectura completa: ingesta, consulta, herramientas y elementos transversales."""
    width, height = 1040, 1012
    col_i, col_a, col_b, col_c = (24, 196), (264, 250), (576, 204), (810, 206)
    out: list[str] = [
        _column_header(col_i[0], col_i[1], "INGESTA · FUERA DE LÍNEA"),
        _column_header(col_a[0], col_a[1], "CONSULTA · RAMA DOCUMENTAL"),
        _column_header(col_b[0], col_b[1], "HERRAMIENTAS Y RESPALDOS"),
        _column_header(col_c[0], col_c[1], "TRANSVERSAL"),
    ]

    # --- Ingesta -----------------------------------------------------------
    xi, wi = col_i
    ing = [
        (
            Rect(xi, 56, wi, 88),
            "Fuentes",
            ["12 PDF Odoo 17.0 / 18.0", "10 documentos internos", "manifiestos YAML + sha256"],
        ),
        (Rect(xi, 168, wi, 70), "Extracción", ["PyMuPDF · python-docx", "encabezados → sección"]),
        (Rect(xi, 262, wi, 70), "Limpieza", ["NFC · pies repetidos", "escáner de inyección"]),
        (Rect(xi, 356, wi, 70), "Segmentación", ["estructural, 450 tokens", "15 % de solape"]),
        (
            Rect(xi, 450, wi, 70),
            "Metadatos",
            ["chunk_id · versión · roles", "vigencia · content_hash"],
        ),
    ]
    for i, (r, title, lines) in enumerate(ing):
        out.append(_box(r, title, lines))
        if i + 1 < len(ing):
            out.append(_path([(r.cx, r.bottom), (r.cx, ing[i + 1][0].y)]))
    idx = Rect(xi, 544, wi, 150)
    out.append(_box(idx, "Índices", [], tag="MODELO"))
    out.append(_path([(idx.cx, ing[-1][0].bottom), (idx.cx, idx.y)]))
    for j, (name, sub) in enumerate(
        (("Chroma · HNSW", "bge-m3 · 1 024 dim."), ("BM25", "ids como token completo"))
    ):
        cy = idx.y + 44 + j * 50
        cx = idx.x + 24
        out.append(
            f'<ellipse cx="{cx + 10}" cy="{cy}" rx="10" ry="4" fill="none" '
            f'stroke="{C["info"]}" stroke-width="1.3"/>'
            f'<path d="M{cx},{cy} L{cx},{cy + 20} A10,4 0 0 0 {cx + 20},{cy + 20} L{cx + 20},{cy}" '
            f'fill="{C["info"]}" fill-opacity="0.12" stroke="{C["info"]}" stroke-width="1.3"/>'
        )
        out.append(_text(cx + 32, cy + 8, name, size=12.5, weight=600))
        out.append(_text(cx + 32, cy + 23, sub, size=11, color=C["muted"]))
    out.append(
        _text(
            idx.x + 14,
            idx.bottom - 10,
            "mismo chunk_id en ambos",
            size=10.5,
            color=C["muted"],
            family=MONO,
        )
    )

    # --- Rama documental -----------------------------------------------------
    xa, wa = col_a
    steps = [
        (
            "Usuario o API",
            ["QueryRequest: consulta, rol,", "user_id, versión 17.0 / 18.0"],
            "io",
            None,
        ),
        (
            "Guard de entrada",
            ["esquema · anti-ofuscación", "patrones · similitud · PII"],
            "gate",
            None,
        ),
        (
            "Enrutador híbrido",
            ["reglas → LR sobre bge-m3", "→ detección fuera de dominio"],
            "gate",
            "MODELO",
        ),
        (
            "Recuperación híbrida",
            ["pre-filtro rol · versión · vigencia", "BM25 ∥ denso → RRF (k = 60)"],
            "default",
            "MODELO",
        ),
        (
            "Reordenamiento",
            ["bge-reranker-v2-m3", "top-20 → top-5 · sigmoide"],
            "default",
            "MODELO",
        ),
        (
            "Gate de evidencia",
            ["hay fragmentos autorizados", "y puntaje ≥ τ (τ = 0)"],
            "gate",
            None,
        ),
        (
            "Validación de permisos",
            ["revalida el rol y sanea", "el contexto por oración"],
            "gate",
            None,
        ),
        (
            "Generación con LLM",
            ["Haiku · JSON Schema · canario", "<contexto> tratado como datos"],
            "default",
            "MODELO",
        ),
        (
            "Guard de salida",
            ["citas ∈ recuperados · página", "canario · PII · abstención"],
            "gate",
            None,
        ),
        ("Respuesta y traza", ["Answer + QueryResponse", "eventos SSE por etapa"], "io", None),
    ]
    rects = [Rect(xa, 56 + i * 96, wa, 70) for i in range(len(steps))]
    for r, (title, lines, kind, tag) in zip(rects, steps, strict=True):
        out.append(_box(r, title, lines, kind=kind, tag=tag))
    for i in range(len(rects) - 1):
        out.append(_path([(rects[i].cx, rects[i].bottom), (rects[i].cx, rects[i + 1].y)]))
    out.append(_label(rects[2].cx + 8, rects[2].bottom + 16, "doc_query"))

    # Índices → recuperación (datos).
    out.append(
        _path(
            [
                (idx.right, idx.y + 70),
                (236, idx.y + 70),
                (236, rects[3].cy),
                (rects[3].x, rects[3].cy),
            ],
            style="data",
        )
    )

    # Salidas tempranas: bus por el canal izquierdo hacia la respuesta.
    bus_x = 252
    for k in (1, 5, 6):
        y = rects[k].y + 22
        out.append(
            f'<path d="M{rects[k].x},{y} L{bus_x},{y}" stroke="{C["warning"]}" '
            'stroke-width="1.5" stroke-dasharray="6 5" fill="none"/>'
        )
    resp = rects[9]
    out.append(
        _path([(bus_x, rects[1].y + 22), (bus_x, resp.y + 24), (resp.x, resp.y + 24)], style="exit")
    )

    # --- Herramientas --------------------------------------------------------
    xb, wb = col_b
    ood = Rect(xb, 152, wb, 70)
    out.append(_box(ood, "Rechazo controlado", ["fuera de dominio →", "mensaje fijo"], kind="exit"))
    router = rects[2]
    out.append(
        _path([(router.right, router.y + 22), (558, router.y + 22), (558, ood.cy), (ood.x, ood.cy)])
    )
    out.append(_label(ood.x + 4, router.y + 18, "out_of_domain", color=C["warning"]))
    reg = Rect(xb, 248 + 24, wb, 70)
    out.append(
        _box(
            reg,
            "Registro de herramientas",
            ["RBAC · límites por", "consulta y sesión · 10 s"],
            kind="gate",
        )
    )
    out.append(_path([(router.right, router.y + 52), (reg.x, router.y + 52)]))
    out.append(_label(router.right + 6, router.y + 66, "ticket_*"))
    group = Rect(xb, 370, wb, 248)
    out.append(
        f'<rect x="{group.x}" y="{group.y}" width="{group.w}" height="{group.h}" rx="12" '
        f'fill="none" stroke="{C["border"]}" stroke-width="1" stroke-dasharray="2 4"/>'
    )
    out.append(_path([(reg.cx, reg.bottom), (reg.cx, group.y)]))
    tools = [
        ("get_ticket_status", ["SQLite solo lectura", "alcance por rol"]),
        ("classify_ticket", ["TF-IDF / bge-m3 + LR", "confianza < 0.60 → revisión"]),
        ("create_ticket", ["borrador + token HMAC", "10 min · un solo uso"]),
    ]
    for j, (name, lines) in enumerate(tools):
        tr = Rect(xb + 10, group.y + 10 + j * 78, wb - 20, 70)
        out.append(_box(tr, name, lines, title_size=13))
    hitl = Rect(xb, 642, wb, 88)
    out.append(
        _box(
            hitl,
            "Confirmación humana",
            [
                "el usuario confirma en otra",
                "solicitud: firma, vigencia y",
                "uso único → escritura",
            ],
            kind="hitl",
        )
    )
    out.append(_path([(group.cx, group.bottom), (group.cx, hitl.y)], style="hitl"))
    out.append(_label(group.cx + 8, group.bottom + 16, "token", color=C["violet"]))
    perm = rects[6]
    out.append(
        _path(
            [
                (group.x, group.bottom - 30),
                (544, group.bottom - 30),
                (544, perm.y + 50),
                (perm.right, perm.y + 50),
            ]
        )
    )
    out.append(_label(group.x - 4, group.bottom - 36, "resultado", anchor="end"))
    offline = Rect(xb, 754, wb, 70)
    out.append(
        _box(
            offline,
            "Proveedor offline",
            ["respaldo extractivo ante", "timeout o caída del LLM"],
            kind="exit",
        )
    )
    gen = rects[7]
    out.append(_path([(gen.right, gen.cy + 8), (offline.x, gen.cy + 8)], style="exit"))
    audit = Rect(xb, 920, wb, 70)
    out.append(
        _box(audit, "Auditoría JSONL", ["PII enmascarada · HMAC", "de usuario · sin tokens"])
    )
    out.append(_path([(resp.right, resp.cy + 6), (audit.x, resp.cy + 6)]))

    # --- Transversal -----------------------------------------------------------
    xc, wc = col_c
    state = Rect(xc, 56, wc, 196)
    out.append(
        _box(
            state,
            "Estado compartido",
            [
                "AgentState (TypedDict)",
                "· solicitud y consulta saneada",
                "· intención y confianza",
                "· fragmentos recuperados",
                "\u00a0\u00a0y autorizados",
                "· resultados de herramientas",
                "· pasos ≤ 4, herramientas ≤ 3",
                "· tokens, costo y eventos",
                "· short_circuit, degraded",
            ],
            line_size=11.5,
        )
    )
    contract = Rect(xc, 276, wc, 154)
    out.append(
        _box(
            contract,
            "Contrato de salida",
            [
                "answer: str",
                "sources: [source, page,",
                "\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0chunk_id]",
                "abstained: bool",
                "tool_used: str | null",
                "needs_human_review: bool",
            ],
            mono_lines=True,
            line_size=11,
        )
    )
    legend = Rect(xc, 454, wc, 246)
    out.append(_box(legend, "Leyenda", []))
    items = [
        ("flow", "flujo principal"),
        ("exit", "salida temprana: bloqueo,"),
        (None, "abstención o denegación"),
        ("data", "lectura de índices"),
        ("hitl", "humano en el ciclo"),
    ]
    y = legend.y + 48
    for style, label in items:
        if style is not None:
            out.append(_path([(xc + 14, y - 4), (xc + 52, y - 4)], style=style, width=1.5))
        out.append(_text(xc + 62, y, label, size=11.5, color=C["muted"]))
        y += 24
    out.append(_diamond(xc + 33, y - 4, 11, C["accent"]))
    out.append(_text(xc + 62, y, "compuerta de decisión", size=11.5, color=C["muted"]))
    y += 26
    out.append(
        f'<rect x="{xc + 14}" y="{y - 15}" width="40" height="16" rx="8" fill="{C["accent"]}" '
        f'fill-opacity="0.14" stroke="{C["accent"]}" stroke-opacity="0.45" stroke-width="0.8"/>'
    )
    out.append(_text(xc + 62, y - 2, "modelo neuronal", size=11.5, color=C["muted"]))
    y += 26
    out.append(
        f'<rect x="{xc + 14}" y="{y - 14}" width="40" height="18" rx="6" fill="none" '
        f'stroke="{C["accent"]}" stroke-width="1.6"/>'
    )
    out.append(_text(xc + 62, y, "entrada y salida", size=11.5, color=C["muted"]))

    trace = Rect(xc, 724, wc, 266)
    out.append(
        _box(
            trace,
            "Trazabilidad",
            [
                "cada nodo emite un",
                "TraceEvent con su duración",
                "y datos sanitizados",
                "",
                "la consola muestra BM25,",
                "denso, RRF, rerank, gate τ",
                "y el JSON contractual",
                "",
                "una excepción en un nodo",
                "se convierte en respuesta",
                "segura con revisión humana",
            ],
            line_size=11.5,
        )
    )
    return _svg(width, height, "".join(out))


# ---------------------------------------------------------------------------
# Figura: grafo del agente (LangGraph)
# ---------------------------------------------------------------------------


def agent_graph_svg() -> str:
    """``StateGraph`` de ``docuerp.agent.graph`` con sus aristas condicionales."""
    width, height = 1000, 470
    nw, nh = 160, 62
    nodes = {
        "input_guard": (235, 95, "guard de entrada", True),
        "router": (440, 95, "enrutador híbrido", False),
        "retrieve": (665, 95, "search_documents", True),
        "evidence_gate": (880, 95, "gate estructural · τ = 0", True),
        "reject": (440, 250, "fuera de dominio", False),
        "tool": (665, 250, "herramientas de tickets", True),
        "permission_check": (880, 250, "revalidación RBAC", True),
        "respond": (440, 400, "QueryResponse + auditoría", False),
        "output_guard": (665, 400, "guard de salida", False),
        "generate": (880, 400, "LLM o plantilla", True),
    }
    rects = {k: Rect(x - nw / 2, y - nh / 2, nw, nh) for k, (x, y, _s, _f) in nodes.items()}
    out: list[str] = []

    def edge(a: tuple[float, float], b: tuple[float, float], *mid: tuple[float, float]) -> None:
        out.append(_path([a, *mid, b], style="flow", width=1.5))

    r = rects
    start = (70, 95)
    end = (235, 400)
    edge((start[0] + 24, start[1]), (r["input_guard"].x, 95))
    edge((r["input_guard"].right, 95), (r["router"].x, 95))
    edge((r["router"].right, 88), (r["retrieve"].x, 88))
    out.append(_label(r["router"].right + 5, 81, "doc_query"))
    edge((r["router"].right, 108), (r["tool"].x, 250), (552, 108), (552, 250))
    for i, lab in enumerate(("ticket_status", "ticket_classify", "ticket_create")):
        out.append(_label(560, 150 + i * 14, lab))
    edge((r["router"].cx, r["router"].bottom), (r["reject"].cx, r["reject"].y))
    out.append(_label(r["router"].cx + 8, 178, "out_of_domain", color=C["warning"]))
    edge((r["retrieve"].right, 95), (r["evidence_gate"].x, 95))
    edge(
        (r["evidence_gate"].cx, r["evidence_gate"].bottom),
        (r["permission_check"].cx, r["permission_check"].y),
    )
    edge((r["tool"].right, 250), (r["permission_check"].x, 250))
    edge(
        (r["permission_check"].cx, r["permission_check"].bottom),
        (r["generate"].cx, r["generate"].y),
    )
    edge((r["generate"].x, 400), (r["output_guard"].right, 400))
    edge((r["output_guard"].x, 400), (r["respond"].right, 400))
    edge((r["reject"].cx, r["reject"].bottom), (r["respond"].cx, r["respond"].y))
    edge((r["respond"].x, 400), (end[0] + 24, 400))

    for key, (_x, _y, sub, can_short) in nodes.items():
        rr = r[key]
        stroke = C["warning"] if can_short else C["border"]
        dash = ' stroke-dasharray="5 3"' if can_short else ""
        if key == "respond":
            stroke, dash = C["accent"], ""
            out.append(
                f'<rect x="{rr.x}" y="{rr.y}" width="{rr.w}" height="{rr.h}" rx="12" fill="none" '
                f'stroke="{C["accent"]}" stroke-opacity="0.5" stroke-width="3" '
                'filter="url(#glow)"/>'
            )
        out.append(
            f'<rect x="{rr.x}" y="{rr.y}" width="{rr.w}" height="{rr.h}" rx="12" '
            f'fill="url(#panelGrad)" stroke="{stroke}" stroke-width="1.4"{dash}/>'
        )
        out.append(
            _text(rr.cx, rr.y + 26, key, size=13.5, weight=600, family=MONO, anchor="middle")
        )
        out.append(_text(rr.cx, rr.y + 46, sub, size=12, color=C["muted"], anchor="middle"))

    for (cx, cy), label in ((start, "START"), (end, "END")):
        out.append(
            f'<circle cx="{cx}" cy="{cy}" r="24" fill="{C["accent"]}" fill-opacity="0.14" '
            f'stroke="{C["accent"]}" stroke-width="1.5"/>'
        )
        out.append(
            _text(
                cx,
                cy + 4,
                label,
                size=10.5,
                weight=600,
                family=MONO,
                color=C["accent-2"],
                anchor="middle",
            )
        )

    note = Rect(24, 170, 300, 150)
    out.append(
        f'<rect x="{note.x}" y="{note.y}" width="{note.w}" height="{note.h}" rx="12" '
        f'fill="{PANEL}" fill-opacity="0.7" stroke="{C["border"]}" stroke-width="1"/>'
    )
    out.append(
        f'<rect x="{note.x + 14}" y="{note.y + 16}" width="26" height="14" rx="5" fill="none" '
        f'stroke="{C["warning"]}" stroke-width="1.4" stroke-dasharray="5 3"/>'
    )
    lines = [
        "borde ámbar: el nodo puede saltar",
        "a respond (bloqueo, abstención,",
        "denegación o degradación)",
        "presupuesto: ≤ 4 pasos y ≤ 3",
        "llamadas a herramientas",
        "excepción en un nodo → respuesta",
        "segura con needs_human_review",
    ]
    ys = [28, 44, 60, 86, 102, 128, 144]
    for line, dy in zip(lines, ys, strict=True):
        x = note.x + 50 if dy <= 60 else note.x + 14
        out.append(_text(x, note.y + dy, line, size=12.5, color=C["muted"]))
    return _svg(width, height, "".join(out))


DIAGRAMS = {
    "arquitectura": architecture_svg,
    "grafo_agente": agent_graph_svg,
}
"""Diagramas disponibles: nombre → función que devuelve el SVG."""
