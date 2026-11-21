"""Conversión de la documentación RST de Odoo a PDF por módulo y versión.

Flujo completo::

    data/raw/odoo-docs/{17.0,18.0}/content/**/*.rst
        │  (1) orden de páginas según los ``toctree`` (se toleran páginas ausentes)
        ▼
    HTML por página con docutils + roles/directivas de Sphinx registrados
        │  (2) un ``<article>`` por página RST, que empieza en página nueva con su h1
        ▼
    HTML del módulo (portada + páginas + CSS común del proyecto)
        │  (3) Chromium headless (Playwright) → PDF con pie "archivo · pág N"
        ▼
    data/build/pdf/odoo-{versión}-{módulo}.pdf

¿Por qué pasar por PDF si ya tenemos el RST? Porque el contrato de citas del proyecto es
``(source, page)``: el analista debe poder abrir el PDF en la página citada. Además, el mismo
extractor (``extract.py``) sirve para los runbooks internos, que también se publican en PDF.

La hoja de estilo usa **tamaños de fuente fijos** por nivel de encabezado (h1 22 pt, h2 17 pt,
h3 14 pt, h4 12 pt, cuerpo 10.5 pt). El extractor reconoce los encabezados por ese tamaño, así
que ``PDF_BASE_CSS`` y ``HEADING_SIZES_PT`` son el **contrato tipográfico común** a todos los
PDF del proyecto (Odoo y runbooks).

Licencia: el contenido es de Odoo S.A. bajo CC BY-SA 4.0; la portada de cada PDF lo atribuye
y declara la modificación (compilado a PDF), como exige la sección 3(a) de la licencia.
"""

from __future__ import annotations

import fnmatch
import hashlib
import html
import io
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from docutils import nodes
from docutils.core import publish_doctree, publish_parts
from docutils.io import DocTreeInput
from docutils.parsers.rst import Directive, directives, roles
from docutils.parsers.rst.states import Inliner
from docutils.readers.doctree import Reader as DoctreeReader

from docuerp import paths

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Contrato tipográfico común (lo reutilizan los PDF de runbooks)
# ---------------------------------------------------------------------------

HEADING_SIZES_PT: dict[int, float] = {1: 22.0, 2: 17.0, 3: 14.0, 4: 12.0}
"""Tamaño de fuente (pt) por nivel de encabezado; ``extract.py`` lo usa para inferir niveles."""

BODY_SIZE_PT = 10.5
"""Tamaño del texto de cuerpo."""

BREADCRUMB_SIZE_PT = 9.0
"""Tamaño de la miga de pan (ruta de páginas padre) que precede al h1 de cada página."""

CODE_SIZE_PT = 9.0
"""Tamaño de los bloques de código (fuente monoespaciada)."""

TABLE_SIZE_PT = 9.5
"""Tamaño del texto dentro de tablas."""

FOOTER_SIZE_PT = 8.0
"""Tamaño del pie de página ``archivo · pág N``."""

COVER_TITLE_SIZE_PT = 28.0
"""Tamaño del título de portada: el extractor omite la página 1 si contiene este tamaño."""

BREADCRUMB_SEPARATOR = " › "
"""Separador visible de la miga de pan."""

MENU_SEPARATOR = " ‣ "
"""Separador con el que se renderiza ``:menuselection:`` (``A --> B`` → ``A ‣ B``)."""

FIGURE_PREFIX = "[Figura: "
"""Prefijo del marcador que sustituye a cada imagen (las imágenes no se descargaron)."""

LICENSE_NAME = "CC BY-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
ODOO_OWNER = "Odoo S.A."
ODOO_REPO_URL = "https://github.com/odoo/documentation"

PDF_BASE_CSS = f"""
@page {{ size: A4; margin: 18mm 17mm 20mm 17mm; }}
html {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
body {{
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: {BODY_SIZE_PT}pt; line-height: 1.42; color: #1d2228; margin: 0;
}}
h1, h2, h3, h4, h5, h6 {{
  font-weight: 700; color: #0b3d2e; line-height: 1.2; break-after: avoid;
  page-break-after: avoid; margin: 0.9em 0 0.4em 0;
}}
h1 {{ font-size: {HEADING_SIZES_PT[1]}pt; margin-top: 0; }}
h2 {{ font-size: {HEADING_SIZES_PT[2]}pt; }}
h3 {{ font-size: {HEADING_SIZES_PT[3]}pt; }}
h4, h5, h6 {{ font-size: {HEADING_SIZES_PT[4]}pt; }}
h1 code, h2 code, h3 code, h4 code, h5 code, h6 code {{ font-size: inherit; }}
p, li, dd, dt {{ font-size: {BODY_SIZE_PT}pt; }}
p {{ margin: 0.35em 0 0.55em 0; }}
a {{ color: #0f6e52; text-decoration: none; }}
ul, ol {{ margin: 0.3em 0 0.6em 0; padding-left: 1.6em; }}
ul.marked, ol.marked {{ list-style: none; padding-left: 1.3em; }}
li {{ margin: 0.15em 0; }}
li > p {{ margin: 0.1em 0; }}
code, kbd, tt {{ font-family: Menlo, "Courier New", monospace; font-size: {BODY_SIZE_PT}pt; }}
td code, th code, td kbd, th kbd {{ font-size: {TABLE_SIZE_PT}pt; }}
pre {{
  font-family: Menlo, "Courier New", monospace; font-size: {CODE_SIZE_PT}pt;
  white-space: pre-wrap; word-break: break-word; background: #f3f5f7;
  border: 1px solid #d9dee4; border-radius: 4px; padding: 6px 8px; margin: 0.5em 0;
}}
pre code {{ font-size: {CODE_SIZE_PT}pt; }}
table {{ border-collapse: collapse; margin: 0.9em 0 1.1em 0; width: 100%; }}
th, td {{
  font-size: {TABLE_SIZE_PT}pt; border: 0.75pt solid #9aa5b1; padding: 3px 5px;
  vertical-align: top; text-align: left;
}}
th, td p, td li, th p {{ font-size: {TABLE_SIZE_PT}pt; }}
th {{ background: #e8eef3; }}
caption {{ text-align: left; font-weight: 700; caption-side: top; margin-bottom: 2pt; }}
tr {{ break-inside: avoid; page-break-inside: avoid; }}
.breadcrumb {{ font-size: {BREADCRUMB_SIZE_PT}pt; color: #5b6673; margin: 0 0 4pt 0; }}
.rst-page {{ break-before: page; page-break-before: always; }}
.admonition {{
  border-left: 3pt solid #10b981; background: #f1faf6; padding: 4px 10px; margin: 0.6em 0;
  break-inside: avoid-page;
}}
.admonition.warning, .admonition.danger, .admonition.important {{
  border-left-color: #d97706; background: #fdf7ec;
}}
.admonition-title {{ font-weight: 700; }}
.tab {{ border-top: 0.75pt solid #c4ccd4; margin: 0.6em 0; padding-top: 2px; }}
.tab-title {{ font-weight: 700; }}
.figure-placeholder {{ font-style: italic; color: #4a5562; }}
blockquote {{ margin: 0.3em 0 0.3em 1.2em; }}
.cover {{ height: 245mm; display: flex; flex-direction: column; justify-content: center; }}
.cover-title {{ font-size: {COVER_TITLE_SIZE_PT}pt; font-weight: 700; color: #0b3d2e; }}
.cover-subtitle {{ font-size: 15pt; color: #10b981; margin: 6pt 0 18pt 0; }}
.cover-meta {{ font-size: 11pt; margin: 2pt 0; }}
.cover-license {{ font-size: 9.5pt; color: #4a5562; margin-top: 28pt; max-width: 150mm; }}
"""
"""Hoja de estilo fija y común a todos los PDF del proyecto (ver docstring del módulo)."""

# ---------------------------------------------------------------------------
# Catálogo de módulos y versiones
# ---------------------------------------------------------------------------

MODULE_ROOTS: dict[str, tuple[str, ...]] = {
    "inventario": ("applications/inventory_and_mrp/inventory.rst",),
    "compras": ("applications/inventory_and_mrp/purchase.rst",),
    "ventas": ("applications/sales/sales.rst",),
    "facturacion": (
        "applications/finance/accounting/customer_invoices.rst",
        "applications/finance/accounting/vendor_bills.rst",
    ),
    "accesos": (
        "applications/general/users.rst",
        "applications/general/developer_mode.rst",
    ),
    "actualizacion": (
        "administration/upgrade.rst",
        "administration/standard_extended_support.rst",
    ),
}
"""Páginas raíz (relativas a ``content/``) de cada módulo. El resto sale de sus toctrees."""

MODULE_TITLES: dict[str, str] = {
    "inventario": "Inventario",
    "compras": "Compras",
    "ventas": "Ventas",
    "facturacion": "Facturación (clientes y proveedores)",
    "accesos": "Usuarios, derechos de acceso y modo desarrollador",
    "actualizacion": "Actualización y soporte extendido",
}
"""Título en español de cada módulo para la portada y el manifiesto."""


@dataclass(frozen=True)
class OdooSnapshot:
    """Instantánea del repositorio ``odoo/documentation`` usada para construir los PDF."""

    version: str
    commit: str
    commit_date: str

    @property
    def short_commit(self) -> str:
        """Primeros 7 caracteres del commit (para ``version`` = ``18.0@160a12c``)."""
        return self.commit[:7]


ODOO_SNAPSHOTS: dict[str, OdooSnapshot] = {
    "17.0": OdooSnapshot(
        "17.0", "fbf3e3a6ef0bda7c30b28f213848deca17c2a872", "2026-10-23T14:40:22Z"
    ),
    "18.0": OdooSnapshot(
        "18.0", "160a12ca9f533e0dfe0027c856ab18ff335fe55c", "2026-11-16T12:14:38Z"
    ),
}
"""Commits descargados (ver ``docs/research/odoo_corpus.md``); se verifican contra ``.git``."""


def odoo_root(version: str) -> Path:
    """Carpeta del clon de la documentación de Odoo para ``version``."""
    return paths.RAW / "odoo-docs" / version


def content_dir(version: str) -> Path:
    """Carpeta ``content/`` (raíz de las rutas de ``:doc:`` absolutas)."""
    return odoo_root(version) / "content"


def read_head_commit(repo_dir: Path) -> str | None:
    """Lee el commit de ``HEAD`` directamente de los archivos de ``.git`` (sin ejecutar git).

    Soporta ``HEAD`` desacoplado (hash directo), referencias sueltas en ``refs/`` y
    ``packed-refs`` (incluido el remoto ``origin/<rama>`` de los clones superficiales).
    Devuelve ``None`` si no hay repositorio o no se puede resolver.
    """
    git_dir = repo_dir / ".git"
    head_file = git_dir / "HEAD"
    if not head_file.is_file():
        return None
    head = head_file.read_text(encoding="utf-8").strip()
    if not head.startswith("ref:"):
        return head or None
    ref = head.split(":", 1)[1].strip()
    loose = git_dir / ref
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip()
    branch = ref.rsplit("/", 1)[-1]
    packed = git_dir / "packed-refs"
    if packed.is_file():
        candidates = {ref, f"refs/remotes/origin/{branch}"}
        for line in packed.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] in candidates:
                return parts[0]
    return None


def resolve_snapshot(version: str) -> OdooSnapshot:
    """Devuelve la instantánea de ``version`` y avisa si el clon local no coincide."""
    if version not in ODOO_SNAPSHOTS:
        raise ValueError(f"Versión de Odoo no soportada: {version!r}. Opciones: 17.0, 18.0")
    snapshot = ODOO_SNAPSHOTS[version]
    local = read_head_commit(odoo_root(version))
    if local and local != snapshot.commit:
        log.warning(
            "El clon local de %s está en %s y no en %s: se usa el commit local",
            version,
            local[:7],
            snapshot.short_commit,
        )
        return OdooSnapshot(version, local, snapshot.commit_date)
    return snapshot


# ---------------------------------------------------------------------------
# Lectura ligera de RST: títulos, etiquetas, toctrees y sustituciones
# ---------------------------------------------------------------------------

_PUNCT_LINE_RE = re.compile(r"^([=\-~*^\"'`#+_.:<>])\1{2,}\s*$")
_LABEL_RE = re.compile(r"^\.\.\s+_([^:]+):\s*$")
_SUBST_DEF_RE = re.compile(r"^\.\.\s+\|([^|]+)\|\s+replace::\s+(.+)$")
_SUBST_REF_RE = re.compile(r"(?<![\w|])\|([^|\s][^|]*?)\|(?![\w|])")
_TOP_FIELD_RE = re.compile(r"^:[\w-]+:.*$")
_ROLE_USE_RE = re.compile(r"(?<![\w`]):([a-zA-Z][\w-]*(?::[\w-]+)?):`")
_DIRECTIVE_USE_RE = re.compile(r"^\s*\.\.\s+([a-zA-Z][\w-]*(?::[\w-]+)?)::", re.MULTILINE)
_EXPLICIT_TARGET_RE = re.compile(r"^(?P<title>.*?)\s*<(?P<target>[^<>]+)>$", re.DOTALL)


def _is_title_underline(line: str, title: str) -> bool:
    """Indica si ``line`` es un subrayado RST válido para ``title``."""
    return bool(_PUNCT_LINE_RE.match(line)) and len(line.rstrip()) >= len(title.rstrip())


def rst_title(text: str) -> str | None:
    """Devuelve el primer título de sección de un texto RST (o ``None``).

    Reconoce títulos con subrayado y con sobrelineado + subrayado::

        ======        Título
        Título        ======
        ======
    """
    lines = text.splitlines()
    for i in range(len(lines) - 1):
        line = lines[i]
        if _PUNCT_LINE_RE.match(line) and i + 2 < len(lines):
            candidate = lines[i + 1].strip()
            if candidate and _is_title_underline(lines[i + 2], candidate):
                return candidate
        stripped = line.strip()
        if (
            stripped
            and not line.startswith((" ", "\t", ".."))
            and not _PUNCT_LINE_RE.match(line)
            and _is_title_underline(lines[i + 1], stripped)
        ):
            return stripped
    return None


def _labels_in(text: str) -> dict[str, str]:
    """Mapea cada etiqueta ``.. _label:`` del texto al título de sección que la sigue."""
    lines = text.splitlines()
    found: dict[str, str] = {}
    for i, line in enumerate(lines):
        match = _LABEL_RE.match(line)
        if not match:
            continue
        # Avanza sobre líneas vacías y otras etiquetas apiladas hasta el siguiente bloque.
        j = i + 1
        while j < len(lines) and (not lines[j].strip() or _LABEL_RE.match(lines[j])):
            j += 1
        title = rst_title("\n".join(lines[j : j + 3]))
        if title:
            found[match.group(1).strip()] = title
    return found


def parse_toctree_entries(text: str) -> list[str]:
    """Extrae las entradas (en orden) de todos los ``.. toctree::`` del texto RST.

    Las entradas pueden ser ``ruta``, ``Título <ruta>`` o patrones con ``:glob:``; las
    opciones (``:titlesonly:``, ``:hidden:``…) se ignoran.
    """
    entries: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if re.match(r"^\s*\.\.\s+toctree::", lines[i]):
            base_indent = len(lines[i]) - len(lines[i].lstrip())
            i += 1
            while i < len(lines):
                line = lines[i]
                if line.strip() and (len(line) - len(line.lstrip())) <= base_indent:
                    break
                stripped = line.strip()
                if stripped and not stripped.startswith(":"):
                    match = _EXPLICIT_TARGET_RE.match(stripped)
                    entries.append(match.group("target") if match else stripped)
                i += 1
            continue
        i += 1
    return entries


@dataclass
class RstCatalog:
    """Índice de una versión: títulos por documento, etiquetas y sustituciones globales."""

    content: Path
    titles: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    substitutions: dict[str, str] = field(default_factory=dict)

    def doc_title(self, docname: str) -> str | None:
        """Título de ``docname`` (ruta sin ``.rst`` relativa a ``content/``)."""
        return self.titles.get(docname)


def _docname(path: Path, content: Path) -> str:
    """Convierte ``content/a/b.rst`` en ``a/b`` (nombre de documento estilo Sphinx)."""
    return path.relative_to(content).with_suffix("").as_posix()


@lru_cache(maxsize=4)
def build_catalog(content: Path) -> RstCatalog:
    """Recorre todos los ``.rst`` de ``content`` y construye el :class:`RstCatalog`."""
    if not content.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de contenido de Odoo: {content}")
    catalog = RstCatalog(content=content)
    for path in sorted(content.rglob("*.rst")):
        text = path.read_text(encoding="utf-8")
        name = _docname(path, content)
        title = rst_title(_strip_top_fields(text))
        if title:
            catalog.titles[name] = title
        catalog.labels.update(_labels_in(text))
        for line in text.splitlines():
            match = _SUBST_DEF_RE.match(line)
            if match:
                catalog.substitutions.setdefault(match.group(1), match.group(2).strip())
    return catalog


def _strip_top_fields(text: str) -> str:
    """Quita el bloque de campos de metadatos inicial (``:nosearch:``, ``:show-toc:``…).

    docutils lo interpretaría como *docinfo* y lo pintaría como tabla; en Sphinx son
    metadatos de página sin contenido visible.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines) and (not lines[i].strip() or _TOP_FIELD_RE.match(lines[i])):
        i += 1
    return "\n".join(lines[i:])


# ---------------------------------------------------------------------------
# Orden de páginas por toctree
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RstPage:
    """Página RST a incluir en un PDF, con su miga de pan (títulos de páginas ancestro)."""

    path: Path
    docname: str
    title: str
    breadcrumb: tuple[str, ...] = ()


@dataclass
class PageCollection:
    """Resultado de recorrer los toctrees de un módulo."""

    pages: list[RstPage]
    missing: list[str]
    orphans: list[str]


def _resolve_entry(entry: str, current: str, content: Path) -> list[str]:
    """Resuelve una entrada de toctree a nombres de documento existentes (admite glob)."""
    entry = entry.removesuffix(".rst")
    if entry.startswith("/"):
        target = entry.lstrip("/")
    else:
        parent = Path(current).parent
        target = (parent / entry).as_posix()
    # Normaliza "a/../b" sin tocar el sistema de archivos.
    parts: list[str] = []
    for part in target.split("/"):
        if part == "..":
            if parts:
                parts.pop()
        elif part not in ("", "."):
            parts.append(part)
    target = "/".join(parts)
    if any(ch in target for ch in "*?["):
        matches = sorted(
            _docname(p, content)
            for p in content.rglob("*.rst")
            if fnmatch.fnmatch(_docname(p, content), target)
        )
        return [m for m in matches if m != current]
    return [target] if (content / f"{target}.rst").is_file() else []


def collect_module_pages(version: str, module: str) -> PageCollection:
    """Ordena las páginas de ``module`` recorriendo los toctrees en profundidad (DFS).

    - Las entradas que apuntan a páginas no descargadas se registran en ``missing``.
    - Las páginas de la carpeta del módulo que ningún toctree alcanza (``orphans``) se
      agregan al final para no perder contenido.
    """
    if module not in MODULE_ROOTS:
        raise ValueError(f"Módulo desconocido: {module!r}. Opciones: {sorted(MODULE_ROOTS)}")
    content = content_dir(version)
    catalog = build_catalog(content)
    pages: list[RstPage] = []
    missing: list[str] = []
    seen: set[str] = set()

    def visit(docname: str, breadcrumb: tuple[str, ...]) -> None:
        if docname in seen:
            return
        seen.add(docname)
        path = content / f"{docname}.rst"
        title = catalog.doc_title(docname) or Path(docname).name.replace("_", " ").title()
        pages.append(RstPage(path=path, docname=docname, title=title, breadcrumb=breadcrumb))
        for entry in parse_toctree_entries(path.read_text(encoding="utf-8")):
            resolved = _resolve_entry(entry, docname, content)
            if not resolved:
                missing.append(f"{docname} → {entry}")
                continue
            for child in resolved:
                visit(child, (*breadcrumb, title))

    for root in MODULE_ROOTS[module]:
        root_name = root.removesuffix(".rst")
        if not (content / root).is_file():
            missing.append(f"(raíz) {root_name}")
            continue
        visit(root_name, ())

    orphans: list[str] = []
    for root in MODULE_ROOTS[module]:
        folder = content / root.removesuffix(".rst")
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*.rst")):
            name = _docname(path, content)
            if name in seen:
                continue
            orphans.append(name)
            seen.add(name)
            # Miga de pan reconstruida con los títulos de las carpetas ancestro.
            crumbs: list[str] = []
            parent = Path(name).parent
            while parent.as_posix() not in (".", ""):
                parent_title = catalog.doc_title(parent.as_posix())
                if parent_title:
                    crumbs.insert(0, parent_title)
                parent = parent.parent
            title = catalog.doc_title(name) or path.stem.replace("_", " ").title()
            pages.append(RstPage(path=path, docname=name, title=title, breadcrumb=tuple(crumbs)))
    return PageCollection(pages=pages, missing=missing, orphans=orphans)


# ---------------------------------------------------------------------------
# Roles de Sphinx
# ---------------------------------------------------------------------------

RoleFn = Callable[..., tuple[list[nodes.Node], list[nodes.system_message]]]


def _split_explicit(text: str) -> tuple[str | None, str]:
    """Separa ``"Título <destino>"`` en ``("Título", "destino")``; sin título → ``(None, t)``."""
    match = _EXPLICIT_TARGET_RE.match(text.strip())
    if match and match.group("title"):
        return match.group("title").strip(), match.group("target").strip()
    if match:
        return None, match.group("target").strip()
    return None, text.strip()


def _context(inliner: Inliner) -> dict[str, Any]:
    """Contexto de conversión (catálogo, documento actual) guardado en los *settings*."""
    ctx = getattr(inliner.document.settings, "docuerp_context", None)
    return ctx if isinstance(ctx, dict) else {}


def _humanize_target(target: str) -> str:
    """Convierte ``inventory/product_management/tracking-inventory`` en ``tracking inventory``."""
    tail = target.rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"[-_]+", " ", tail).strip() or target


def _role_guilabel(
    name: str, rawtext: str, text: str, lineno: int, inliner: Inliner, options=None, content=None
) -> tuple[list[nodes.Node], list[nodes.system_message]]:
    """``:guilabel:`Save``` → **Save** (negrita: es una etiqueta de la interfaz)."""
    return [nodes.strong(rawtext, text.replace("&&", "&"), classes=["guilabel"])], []


def _role_menuselection(
    name: str, rawtext: str, text: str, lineno: int, inliner: Inliner, options=None, content=None
) -> tuple[list[nodes.Node], list[nodes.system_message]]:
    """``:menuselection:`A --> B``` → **A ‣ B** (ruta de menú con flecha)."""
    parts = [part.strip() for part in re.split(r"\s*-->\s*", text)]
    return [nodes.strong(rawtext, MENU_SEPARATOR.join(parts), classes=["menuselection"])], []


def _role_doc(
    name: str, rawtext: str, text: str, lineno: int, inliner: Inliner, options=None, content=None
) -> tuple[list[nodes.Node], list[nodes.system_message]]:
    """``:doc:`texto <ruta>``` → *texto*; sin texto explícito se usa el título del destino."""
    title, target = _split_explicit(text)
    if title is None:
        ctx = _context(inliner)
        catalog: RstCatalog | None = ctx.get("catalog")
        current: str = ctx.get("docname", "")
        resolved = _resolve_entry(target, current, catalog.content) if catalog is not None else []
        title = (catalog.doc_title(resolved[0]) if catalog and resolved else None) or (
            _humanize_target(target)
        )
    return [nodes.emphasis(rawtext, title, classes=["xref"])], []


def _role_ref(
    name: str, rawtext: str, text: str, lineno: int, inliner: Inliner, options=None, content=None
) -> tuple[list[nodes.Node], list[nodes.system_message]]:
    """``:ref:`texto <etiqueta>``` → *texto*; sin texto se usa el título de la sección."""
    title, target = _split_explicit(text)
    if title is None:
        catalog: RstCatalog | None = _context(inliner).get("catalog")
        title = (catalog.labels.get(target) if catalog else None) or _humanize_target(target)
    return [nodes.emphasis(rawtext, title, classes=["xref"])], []


def _role_abbr(
    name: str, rawtext: str, text: str, lineno: int, inliner: Inliner, options=None, content=None
) -> tuple[list[nodes.Node], list[nodes.system_message]]:
    """``:abbr:`SO (Sales Order)``` → ``SO (Sales Order)``: la expansión queda en el texto."""
    return [nodes.Text(" ".join(text.split()))], []


def _role_icon(
    name: str, rawtext: str, text: str, lineno: int, inliner: Inliner, options=None, content=None
) -> tuple[list[nodes.Node], list[nodes.system_message]]:
    """``:icon:`fa-cog``` → ``[cog]``: se conserva el nombre del ícono sin el prefijo CSS."""
    icon = re.sub(r"^(fa|oi)-", "", text.strip().split()[0] if text.strip() else "")
    if not icon:
        return [], []
    return [nodes.Text(f"[{icon.replace('-', ' ')}]")], []


def _role_literal(
    name: str, rawtext: str, text: str, lineno: int, inliner: Inliner, options=None, content=None
) -> tuple[list[nodes.Node], list[nodes.system_message]]:
    """Roles de código (``:kbd:``, ``:file:``, ``:math:``…) → texto monoespaciado."""
    return [nodes.literal(rawtext, text)], []


def _role_emphasis(
    name: str, rawtext: str, text: str, lineno: int, inliner: Inliner, options=None, content=None
) -> tuple[list[nodes.Node], list[nodes.system_message]]:
    """``:dfn:`término``` → *término*."""
    return [nodes.emphasis(rawtext, text)], []


def _role_link_text(
    name: str, rawtext: str, text: str, lineno: int, inliner: Inliner, options=None, content=None
) -> tuple[list[nodes.Node], list[nodes.system_message]]:
    """Roles de enlace genéricos (``:download:``, ``:term:``…) → solo el texto visible."""
    title, target = _split_explicit(text)
    return [nodes.Text(title or target)], []


SPHINX_ROLES: dict[str, RoleFn] = {
    "guilabel": _role_guilabel,
    "menuselection": _role_menuselection,
    "doc": _role_doc,
    "ref": _role_ref,
    "abbr": _role_abbr,
    "icon": _role_icon,
    "math": _role_literal,
    "kbd": _role_literal,
    "file": _role_literal,
    "command": _role_literal,
    "samp": _role_literal,
    "envvar": _role_literal,
    "program": _role_literal,
    "dfn": _role_emphasis,
    "download": _role_link_text,
    "term": _role_link_text,
    "numref": _role_link_text,
    "option": _role_literal,
}
"""Roles de Sphinx del inventario del corpus (más algunos comunes) y su renderizado."""

# ---------------------------------------------------------------------------
# Directivas de Sphinx
# ---------------------------------------------------------------------------


class _AnyOptions(dict[str, Callable[[str | None], str]]):
    """``option_spec`` que acepta cualquier opción (``:align:``, ``:class:``, ``:large:``…).

    docutils rechaza opciones no declaradas; Sphinx y el tema de Odoo agregan muchas que
    no afectan al texto, así que todas se aceptan como cadenas sin validar.
    """

    def __missing__(self, key: str) -> Callable[[str | None], str]:
        return directives.unchanged

    def __bool__(self) -> bool:
        # docutils solo busca opciones si ``option_spec`` es "verdadero"; un dict vacío no
        # lo es, así que se fuerza a ``True`` para que ``:alt:`` y compañía se reconozcan.
        return True


_ADMONITION_TITLES: dict[str, str] = {
    "note": "Note",
    "tip": "Tip",
    "important": "Important",
    "warning": "Warning",
    "danger": "Danger",
    "caution": "Caution",
    "attention": "Attention",
    "hint": "Hint",
    "error": "Error",
    "seealso": "See also",
    "example": "Example",
    "exercise": "Exercise",
}
"""Título visible de cada caja de aviso (el corpus está en inglés)."""


class AdmonitionDirective(Directive):
    """Cajas ``note/tip/important/warning/seealso/example/...``: título + contenido.

    No admite argumentos (igual que en Sphinx): el texto de la misma línea
    (``.. note:: Texto``) y las líneas siguientes forman el contenido. El título se escribe
    en línea al inicio del primer párrafo (``Note: Texto…``) para que el texto extraído del
    PDF conserve de qué tipo de aviso se trata.
    """

    has_content = True
    option_spec = _AnyOptions()

    def run(self) -> list[nodes.Node]:
        kind = self.name.lower()
        box = nodes.container(classes=["admonition", kind])
        self.state.nested_parse(self.content, self.content_offset, box)
        _prepend_box_title(box, [nodes.Text(_ADMONITION_TITLES.get(kind, kind.title()))])
        return [box]


def _prepend_box_title(box: nodes.Element, title: list[nodes.Node]) -> None:
    """Antepone ``Título:`` en negrita al primer párrafo de ``box`` (o crea uno si no hay)."""
    strong = nodes.strong(classes=["admonition-title"])
    strong.extend(title)
    first = box.children[0] if box.children else None
    if isinstance(first, nodes.paragraph) and not first["classes"]:
        first.insert(0, nodes.Text(" "))
        strong += nodes.Text(":")
        first.insert(0, strong)
    else:
        heading = nodes.paragraph(classes=["admonition-title"])
        heading += strong
        box.insert(0, heading)


class TitledBoxDirective(Directive):
    """``.. admonition:: Título`` y ``.. spoiler:: Título``: caja con título propio."""

    has_content = True
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = _AnyOptions()

    def run(self) -> list[nodes.Node]:
        kind = self.name.lower()
        box = nodes.container(classes=["admonition", kind])
        self.state.nested_parse(self.content, self.content_offset, box)
        title: list[nodes.Node] = [nodes.Text(kind.title())]
        if self.arguments:
            inline, messages = self.state.inline_text(self.arguments[0], self.lineno)
            title = list(inline)
            box.extend(messages)
        _prepend_box_title(box, title)
        return [box]


class FigureDirective(Directive):
    """``.. image::`` / ``.. figure::`` → ``[Figura: alt]`` (+ leyenda de la figura).

    Las imágenes no se descargaron (solo texto), pero el texto alternativo describe la
    pantalla y es útil para la búsqueda.
    """

    required_arguments = 1
    final_argument_whitespace = True
    has_content = True
    option_spec = _AnyOptions()

    def run(self) -> list[nodes.Node]:
        alt = " ".join((self.options.get("alt") or "").split())
        if not alt:
            stem = Path(self.arguments[0].strip()).stem
            alt = re.sub(r"[-_]+", " ", stem).strip()
        paragraph = nodes.paragraph(classes=["figure-placeholder"])
        paragraph += nodes.Text(f"{FIGURE_PREFIX}{alt}]")
        result: list[nodes.Node] = [paragraph]
        if self.content:
            caption = nodes.container(classes=["figure-caption"])
            self.state.nested_parse(self.content, self.content_offset, caption)
            result.append(caption)
        return result


class TabsDirective(Directive):
    """``.. tabs::`` → contenedor; cada pestaña se muestra como una sub-sección visible."""

    has_content = True
    option_spec = _AnyOptions()

    def run(self) -> list[nodes.Node]:
        box = nodes.container(classes=["tabs"])
        self.state.nested_parse(self.content, self.content_offset, box)
        return [box]


class TabDirective(Directive):
    """``.. tab::`` / ``.. group-tab::`` / ``.. code-tab::`` → título en negrita + contenido."""

    has_content = True
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = _AnyOptions()

    def run(self) -> list[nodes.Node]:
        box = nodes.container(classes=["tab"])
        title = nodes.paragraph(classes=["tab-title"])
        label = self.arguments[0] if self.arguments else "Tab"
        inline, messages = self.state.inline_text(label, self.lineno)
        title += nodes.Text("Tab: ")
        title.extend(inline)
        box += title
        box.extend(messages)
        if self.name.lower() == "code-tab":
            box += nodes.literal_block("\n".join(self.content), "\n".join(self.content))
        else:
            self.state.nested_parse(self.content, self.content_offset, box)
        return [box]


class CardsDirective(Directive):
    """``.. cards::`` → lista con viñetas (una entrada por tarjeta)."""

    has_content = True
    option_spec = _AnyOptions()

    def run(self) -> list[nodes.Node]:
        bullet = nodes.bullet_list(classes=["cards"])
        self.state.nested_parse(self.content, self.content_offset, bullet)
        return [bullet]


class CardDirective(Directive):
    """``.. card:: Título`` → elemento de lista: **Título** (etiqueta) — descripción."""

    has_content = True
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = _AnyOptions()

    def run(self) -> list[nodes.Node]:
        item = nodes.list_item(classes=["card"])
        head = nodes.paragraph()
        if self.arguments:
            inline, messages = self.state.inline_text(self.arguments[0], self.lineno)
            strong = nodes.strong()
            strong.extend(inline)
            head += strong
            item.extend(messages)
        tag = (self.options.get("tag") or "").strip()
        if tag:
            head += nodes.Text(f" ({tag})")
        item += head
        self.state.nested_parse(self.content, self.content_offset, item)
        if isinstance(self.state.parent, nodes.bullet_list):
            return [item]
        wrapper = nodes.bullet_list(classes=["cards"])
        wrapper += item
        return [wrapper]


class CodeBlockDirective(Directive):
    """``.. code-block:: xml`` → bloque literal (sin resaltado, fuente monoespaciada)."""

    has_content = True
    optional_arguments = 1
    option_spec = _AnyOptions()

    def run(self) -> list[nodes.Node]:
        code = "\n".join(self.content)
        language = self.arguments[0] if self.arguments else "text"
        return [nodes.literal_block(code, code, classes=["code", language])]


class MathDirective(Directive):
    """``.. math::`` → fórmula LaTeX como bloque literal (legible y extraíble como texto)."""

    has_content = True
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = _AnyOptions()

    def run(self) -> list[nodes.Node]:
        formula = "\n".join([*self.arguments, *self.content]).strip()
        return [nodes.literal_block(formula, formula, classes=["math"])]


class NoOpDirective(Directive):
    """Directivas sin contenido visible en PDF (``toctree``, ``meta``, ``rst-class``…)."""

    has_content = True
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = _AnyOptions()

    def run(self) -> list[nodes.Node]:
        return []


class ContainerDirective(Directive):
    """Directiva desconocida: se conserva su contenido como RST anidado (no se pierde texto)."""

    has_content = True
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = _AnyOptions()

    def run(self) -> list[nodes.Node]:
        box = nodes.container(classes=[self.name.lower()])
        if self.arguments:
            inline, messages = self.state.inline_text(self.arguments[0], self.lineno)
            head = nodes.paragraph()
            head.extend(inline)
            box += head
            box.extend(messages)
        self.state.nested_parse(self.content, self.content_offset, box)
        return [box]


SPHINX_DIRECTIVES: dict[str, type[Directive]] = {
    **dict.fromkeys(_ADMONITION_TITLES, AdmonitionDirective),
    "admonition": TitledBoxDirective,
    "spoiler": TitledBoxDirective,
    "image": FigureDirective,
    "figure": FigureDirective,
    "tabs": TabsDirective,
    "tab": TabDirective,
    "group-tab": TabDirective,
    "code-tab": TabDirective,
    "cards": CardsDirective,
    "card": CardDirective,
    "code-block": CodeBlockDirective,
    "sourcecode": CodeBlockDirective,
    "code": CodeBlockDirective,
    "math": MathDirective,
    "toctree": NoOpDirective,
    "meta": NoOpDirective,
    "rst-class": NoOpDirective,
    "highlight": NoOpDirective,
    "only": ContainerDirective,
    "versionadded": ContainerDirective,
    "versionchanged": ContainerDirective,
    "deprecated": ContainerDirective,
}
"""Directivas del inventario del corpus y su renderizado (las de docutils se sustituyen)."""

_REGISTERED = False


def register_sphinx_roles() -> None:
    """Registra en docutils los roles y directivas de Sphinx (idempotente).

    Los nombres de directiva en docutils no distinguen mayúsculas, así que ``.. Note::``
    (presente una vez en el corpus) también queda cubierta.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    for name, fn in SPHINX_ROLES.items():
        roles.register_local_role(name, fn)
    for name, cls in SPHINX_DIRECTIVES.items():
        directives.register_directive(name, cls)
    _REGISTERED = True


def _register_unknown_constructs(text: str) -> list[str]:
    """Registra como genéricos los roles/directivas usados en ``text`` que no se conocen.

    Así un constructo nuevo en una versión futura de la documentación no rompe la
    conversión: el rol se muestra como texto y la directiva conserva su contenido.
    Devuelve los nombres registrados en esta llamada (para el registro de avisos).
    """
    added: list[str] = []
    # Los registros internos de docutils (``_role_registry``, ``_directive_registry``)
    # contienen los constructos nativos (``literal``, ``strong``, ``table``…).
    native_roles = set(roles._role_registry) | set(roles._roles)
    for name in sorted(set(_ROLE_USE_RE.findall(text))):
        if name not in SPHINX_ROLES and name.lower() not in native_roles:
            roles.register_local_role(name, _role_link_text)
            SPHINX_ROLES[name] = _role_link_text
            added.append(f":{name}:")
    native_directives = set(directives._directive_registry) | set(directives._directives)
    for name in sorted(set(_DIRECTIVE_USE_RE.findall(text))):
        lowered = name.lower()
        if lowered in SPHINX_DIRECTIVES or lowered in native_directives:
            continue
        directives.register_directive(lowered, ContainerDirective)
        SPHINX_DIRECTIVES[lowered] = ContainerDirective
        added.append(f".. {name}::")
    return added


# ---------------------------------------------------------------------------
# RST → HTML
# ---------------------------------------------------------------------------


def prepare_rst_source(text: str, catalog: RstCatalog | None = None) -> str:
    """Preprocesa el RST antes de docutils.

    1. Quita los campos de metadatos iniciales (``:nosearch:``…).
    2. Fija el rol por defecto ``literal`` (como ``default_role`` del ``conf.py`` de Odoo).
    3. Agrega las definiciones de sustitución ``|X|`` usadas pero no definidas en el
       archivo, tomándolas de otras páginas de la misma versión (Sphinx las comparte vía
       ``rst_epilog``; docutils necesita verlas en el propio documento).
    """
    body = _strip_top_fields(text)
    defined = {m.group(1) for m in map(_SUBST_DEF_RE.match, body.splitlines()) if m}
    missing_defs: list[str] = []
    if catalog is not None:
        for name in sorted(set(_SUBST_REF_RE.findall(body)) - defined):
            replacement = catalog.substitutions.get(name)
            if replacement:
                missing_defs.append(f".. |{name}| replace:: {replacement}")
    prefix = ".. default-role:: literal\n\n"
    suffix = ("\n\n" + "\n".join(missing_defs) + "\n") if missing_defs else "\n"
    return prefix + body + suffix


def rst_to_html(
    rst_path: Path, *, base_dir: Path | None = None, catalog: RstCatalog | None = None
) -> str:
    """Convierte un archivo RST a un fragmento HTML (cuerpo) con los roles registrados.

    Args:
        rst_path: archivo ``.rst``.
        base_dir: carpeta ``content/`` para resolver ``:doc:`` y nombres de documento
            (por defecto, el ancestro llamado ``content`` o la carpeta del archivo).
        catalog: catálogo de títulos/etiquetas/sustituciones; se construye si falta.

    Raises:
        FileNotFoundError: si ``rst_path`` no existe.
    """
    if not rst_path.is_file():
        raise FileNotFoundError(f"No existe el archivo RST: {rst_path}")
    register_sphinx_roles()
    if base_dir is None:
        base_dir = next((p for p in rst_path.parents if p.name == "content"), rst_path.parent)
    if catalog is None:
        catalog = build_catalog(base_dir)
    raw = rst_path.read_text(encoding="utf-8")
    source = prepare_rst_source(raw, catalog)
    added = _register_unknown_constructs(source)
    if added:
        log.warning("%s: constructos desconocidos tratados como genéricos: %s", rst_path, added)
    warnings = io.StringIO()
    try:
        docname = _docname(rst_path, base_dir)
    except ValueError:
        docname = rst_path.stem
    settings = {
        "report_level": 5,  # sin mensajes de sistema en el HTML
        "halt_level": 5,  # nunca abortar: se tolera RST imperfecto
        "warning_stream": warnings,
        "doctitle_xform": False,  # el título de la página queda como <h1> en el cuerpo
        "sectsubtitle_xform": False,
        "initial_header_level": 1,
        "file_insertion_enabled": False,  # seguridad: sin ``include`` de archivos
        "raw_enabled": False,  # seguridad: sin HTML crudo
        "syntax_highlight": "none",
        "embed_stylesheet": False,
        "smart_quotes": False,
        "docuerp_context": {"catalog": catalog, "docname": docname},
    }
    # Dos pasos: RST → árbol (doctree) → se agregan los marcadores de lista → HTML.
    document = publish_doctree(source, source_path=str(rst_path), settings_overrides=settings)
    add_list_markers(document)
    parts = publish_parts(
        source=document,
        source_class=DocTreeInput,
        reader=DoctreeReader(parser="null"),
        writer="html5",
        settings_overrides=settings,
    )
    return parts["body"]


_ROMAN = ((10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"))


def _enumerator(enumtype: str, number: int) -> str:
    """Texto del número de una lista enumerada (``arabic``, ``loweralpha``, ``lowerroman``…)."""
    if enumtype in ("loweralpha", "upperalpha") and 1 <= number <= 26:
        letter = chr(ord("a") + number - 1)
        return letter if enumtype == "loweralpha" else letter.upper()
    if enumtype in ("lowerroman", "upperroman") and 1 <= number < 40:
        remaining, roman = number, ""
        for value, symbol in _ROMAN:
            while remaining >= value:
                roman += symbol
                remaining -= value
        return roman if enumtype == "lowerroman" else roman.upper()
    return str(number)


def _prefix_list_item(item: nodes.Element, marker: str) -> None:
    """Escribe ``marker`` como texto al inicio del primer párrafo del elemento de lista."""
    first = item.children[0] if item.children else None
    if isinstance(first, nodes.paragraph):
        first.insert(0, nodes.Text(marker))
    else:
        paragraph = nodes.paragraph()
        paragraph += nodes.Text(marker.rstrip())
        item.insert(0, paragraph)


def add_list_markers(document: nodes.document) -> None:
    """Convierte las viñetas y números de lista en **texto** del documento.

    Chromium dibuja los marcadores de ``<ul>``/``<ol>`` como gráficos, así que no aparecen
    en la capa de texto del PDF. Al escribirlos como texto (``1. ``, ``• ``, ``◦ ``), el
    extractor puede reconocer pasos numerados y viñetas, y el chunker evita cortarlos.
    """
    for enumerated in list(document.findall(nodes.enumerated_list)):
        start = int(enumerated.get("start", 1))
        enumtype = str(enumerated.get("enumtype", "arabic"))
        prefix = str(enumerated.get("prefix", ""))
        suffix = str(enumerated.get("suffix", ".")) or "."
        items = [c for c in enumerated.children if isinstance(c, nodes.list_item)]
        for offset, item in enumerate(items):
            _prefix_list_item(item, f"{prefix}{_enumerator(enumtype, start + offset)}{suffix} ")
        enumerated["classes"].append("marked")
    for bullet in list(document.findall(nodes.bullet_list)):
        depth = sum(1 for a in _ancestors(bullet) if isinstance(a, nodes.bullet_list))
        marker = "• " if depth % 2 == 0 else "◦ "
        for item in (c for c in bullet.children if isinstance(c, nodes.list_item)):
            _prefix_list_item(item, marker)
        bullet["classes"].append("marked")


def _ancestors(node: nodes.Node) -> list[nodes.Node]:
    """Lista de ancestros de ``node`` (del padre hacia la raíz)."""
    found: list[nodes.Node] = []
    parent = node.parent
    while parent is not None:
        found.append(parent)
        parent = parent.parent
    return found


# ---------------------------------------------------------------------------
# Ensamblado del HTML del módulo
# ---------------------------------------------------------------------------


def pdf_file_name(version: str, module: str) -> str:
    """Nombre citable del PDF: ``odoo-18.0-inventario.pdf``."""
    return f"odoo-{version}-{module}.pdf"


def doc_id_for(version: str, module: str) -> str:
    """``doc_id`` del manifiesto: ``odoo-18.0-inventario``."""
    return f"odoo-{version}-{module}"


def cover_html(
    *,
    title: str,
    subtitle: str,
    meta_lines: Iterable[str],
    license_html: str,
) -> str:
    """Portada genérica (también la pueden usar los runbooks) con el título a 28 pt."""
    meta = "".join(f'<p class="cover-meta">{line}</p>' for line in meta_lines)
    return (
        '<section class="cover">'
        f'<div class="cover-title">{html.escape(title)}</div>'
        f'<div class="cover-subtitle">{html.escape(subtitle)}</div>'
        f"{meta}"
        f'<div class="cover-license">{license_html}</div>'
        "</section>"
    )


def render_document_html(title: str, cover: str, body: str) -> str:
    """Documento HTML completo con la hoja de estilo común del proyecto."""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title><style>{PDF_BASE_CSS}</style></head>"
        f"<body>{cover}{body}</body></html>"
    )


def _odoo_cover(version: str, module: str, snapshot: OdooSnapshot, built_at: str) -> str:
    """Portada de un PDF de Odoo: módulo, versión, commit, fechas y atribución CC BY-SA."""
    commit_url = f"{ODOO_REPO_URL}/tree/{snapshot.commit}"
    license_html = (
        f"Contenido: documentación oficial de Odoo, © {ODOO_OWNER}, publicada bajo la licencia "
        f'<a href="{LICENSE_URL}">Creative Commons Attribution-ShareAlike 4.0 International '
        f"({LICENSE_NAME})</a>. Fuente: {html.escape(commit_url)}. "
        "<b>Material modificado</b>: las páginas RST se compilaron a PDF por DocuERP "
        "(roles y directivas de Sphinx convertidos a texto; imágenes sustituidas por su "
        "texto alternativo). Esta adaptación se distribuye bajo la misma licencia "
        f"{LICENSE_NAME}. Odoo S.A. no respalda esta adaptación. El material se ofrece "
        "«tal cual», sin garantías, según la sección 5 de la licencia."
    )
    return cover_html(
        title=f"Odoo {version} — {MODULE_TITLES[module]}",
        subtitle="Documentación oficial de Odoo compilada para DocuERP",
        meta_lines=[
            f"<b>Módulo:</b> {html.escape(module)}",
            f"<b>Versión de Odoo:</b> {html.escape(version)}",
            f"<b>Commit:</b> {html.escape(snapshot.commit)}",
            f"<b>Fecha del commit:</b> {html.escape(snapshot.commit_date)}",
            f"<b>Fecha de compilación:</b> {html.escape(built_at)}",
            f"<b>Archivo:</b> {html.escape(pdf_file_name(version, module))}",
        ],
        license_html=license_html,
    )


def build_module_html(
    version: str,
    module: str,
    files: list[Path],
    *,
    breadcrumbs: dict[Path, tuple[str, ...]] | None = None,
    snapshot: OdooSnapshot | None = None,
    built_at: str | None = None,
) -> str:
    """Ensambla el HTML completo (portada + una sección por RST) de un módulo.

    Cada archivo RST va en un ``<article class="rst-page">`` que fuerza un salto de página,
    precedido por su miga de pan (``Inventory › Product management``) cuando tiene ancestros.
    El índice navegable lo genera Chromium como marcadores del PDF (``outline``).
    """
    if module not in MODULE_ROOTS:
        raise ValueError(f"Módulo desconocido: {module!r}")
    if not files:
        raise ValueError(f"El módulo {module} ({version}) no tiene páginas RST")
    snapshot = snapshot or resolve_snapshot(version)
    built_at = built_at or datetime.now(UTC).strftime("%Y-%m-%d")
    content = content_dir(version)
    catalog = build_catalog(content) if content.is_dir() else None
    articles: list[str] = []
    for path in files:
        crumbs = (breadcrumbs or {}).get(path, ())
        crumb_html = (
            f'<p class="breadcrumb">{html.escape(BREADCRUMB_SEPARATOR.join(crumbs))}</p>'
            if crumbs
            else ""
        )
        base = content if content.is_dir() and content in path.parents else None
        body = rst_to_html(path, base_dir=base, catalog=catalog if base else None)
        articles.append(
            f'<article class="rst-page" data-doc="{html.escape(path.name)}">'
            f"{crumb_html}{body}</article>"
        )
    cover = _odoo_cover(version, module, snapshot, built_at)
    return render_document_html(
        f"Odoo {version} — {MODULE_TITLES[module]}", cover, "".join(articles)
    )


# ---------------------------------------------------------------------------
# HTML → PDF (Chromium vía Playwright)
# ---------------------------------------------------------------------------


def footer_template(label: str) -> str:
    """Plantilla de pie de Chromium: ``archivo · pág N`` (``pageNumber`` lo rellena Chromium)."""
    return (
        f'<div style="width:100%;font-family:Helvetica,Arial,sans-serif;'
        f'font-size:{FOOTER_SIZE_PT}pt;color:#6b7480;text-align:center;">'
        f'{html.escape(label)} · pág <span class="pageNumber"></span></div>'
    )


def html_to_pdf(html_doc: str, out_path: Path, *, footer_label: str | None = None) -> Path:
    """Renderiza HTML a PDF con Chromium (Playwright) y devuelve la ruta escrita.

    Args:
        html_doc: documento HTML completo (con CSS en línea, sin recursos externos).
        out_path: ruta del PDF a escribir (se crean las carpetas).
        footer_label: texto del pie (por defecto, el nombre del archivo de salida).

    Raises:
        RuntimeError: si Chromium no está instalado (``playwright install chromium``).
    """
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    out_path.parent.mkdir(parents=True, exist_ok=True)
    label = footer_label or out_path.name
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page()
                page.set_content(html_doc, wait_until="load")
                page.pdf(
                    path=str(out_path),
                    format="A4",
                    print_background=True,
                    display_header_footer=True,
                    header_template="<span></span>",
                    footer_template=footer_template(label),
                    margin={"top": "18mm", "bottom": "20mm", "left": "17mm", "right": "17mm"},
                    prefer_css_page_size=True,
                    outline=True,
                    tagged=True,
                )
            finally:
                browser.close()
    except PlaywrightError as exc:
        raise RuntimeError(
            f"No se pudo generar {out_path.name} con Chromium: {exc}. "
            "¿Falta `playwright install chromium`?"
        ) from exc
    return out_path


# ---------------------------------------------------------------------------
# Construcción de los 12 PDF y su manifiesto
# ---------------------------------------------------------------------------


@dataclass
class BuiltPdf:
    """Registro de un PDF generado (para el manifiesto y el reporte de construcción)."""

    version: str
    module: str
    path: Path
    pages: int
    rst_files: int
    missing: list[str]
    orphans: list[str]
    sha256: str
    snapshot: OdooSnapshot


def sha256_file(path: Path) -> str:
    """sha256 hexadecimal del contenido de un archivo (lectura por bloques)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pdf_page_count(path: Path) -> int:
    """Número de páginas de un PDF (PyMuPDF)."""
    import pymupdf

    with pymupdf.open(path) as pdf:
        return int(pdf.page_count)


def build_module_pdf(version: str, module: str, out_dir: Path) -> BuiltPdf:
    """Genera ``odoo-{version}-{module}.pdf`` y devuelve su registro."""
    snapshot = resolve_snapshot(version)
    collection = collect_module_pages(version, module)
    files = [page.path for page in collection.pages]
    crumbs = {page.path: page.breadcrumb for page in collection.pages}
    html_doc = build_module_html(version, module, files, breadcrumbs=crumbs, snapshot=snapshot)
    out_path = html_to_pdf(html_doc, out_dir / pdf_file_name(version, module))
    built = BuiltPdf(
        version=version,
        module=module,
        path=out_path,
        pages=_pdf_page_count(out_path),
        rst_files=len(files),
        missing=collection.missing,
        orphans=collection.orphans,
        sha256=sha256_file(out_path),
        snapshot=snapshot,
    )
    log.info(
        "%s: %d páginas RST → %d páginas PDF (%d entradas de toctree ausentes)",
        out_path.name,
        built.rst_files,
        built.pages,
        len(built.missing),
    )
    return built


def build_odoo_pdfs(
    versions: list[str] | None = None,
    modules: list[str] | None = None,
    out_dir: Path | None = None,
) -> list[Path]:
    """Genera todos los PDF ``odoo-{ver}-{modulo}.pdf`` y devuelve sus rutas.

    Además escribe (o actualiza) ``data/manifest/odoo.yaml`` con los documentos generados
    y ``data/build/pdf/odoo_build.json`` con páginas, archivos RST y toctrees ausentes.
    """
    built = build_odoo_pdf_records(versions=versions, modules=modules, out_dir=out_dir)
    return [record.path for record in built]


def build_odoo_pdf_records(
    versions: list[str] | None = None,
    modules: list[str] | None = None,
    out_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> list[BuiltPdf]:
    """Como :func:`build_odoo_pdfs`, pero devuelve los registros completos."""
    versions = versions or sorted(ODOO_SNAPSHOTS)
    modules = modules or list(MODULE_ROOTS)
    out_dir = out_dir or (paths.BUILD / "pdf")
    records = [build_module_pdf(v, m, out_dir) for v in versions for m in modules]
    write_odoo_manifest(records, manifest_path or (paths.MANIFEST / "odoo.yaml"))
    _write_build_log(records, out_dir / "odoo_build.json")
    return records


def manifest_entry(record: BuiltPdf) -> dict[str, Any]:
    """Entrada del manifiesto (campos de ``SourceEntry``) para un PDF de Odoo."""
    snap = record.snapshot
    return {
        "id": doc_id_for(record.version, record.module),
        "title": f"Odoo {record.version} — {MODULE_TITLES[record.module]}",
        "source_file": record.path.name,
        "origin": (
            f"{ODOO_REPO_URL}/tree/{snap.commit} "
            f"(content/: {', '.join(MODULE_ROOTS[record.module])}; RST compilado a PDF)"
        ),
        "license": f"{LICENSE_NAME} ({LICENSE_URL}); modificado: compilado a PDF",
        "owner": ODOO_OWNER,
        "format": "pdf",
        "version": f"{record.version}@{snap.short_commit}",
        "erp_version": record.version,
        "status": "vigente",
        "date": snap.commit_date,
        "update_frequency": "por versión mayor",
        "allowed_roles": ["portal_user", "n1", "n2", "admin"],
        "sensitivity": "public",
        "module": record.module,
        "doc_type": "odoo_doc",
        "language": "en",
        "sha256": record.sha256,
    }


def write_odoo_manifest(records: list[BuiltPdf], manifest_path: Path) -> Path:
    """Escribe ``odoo.yaml`` fusionando con entradas previas de documentos no regenerados."""
    existing: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        for entry in data.get("documents", []) or []:
            if isinstance(entry, dict) and "id" in entry:
                existing[str(entry["id"])] = entry
    for record in records:
        entry = manifest_entry(record)
        existing[entry["id"]] = entry
    ordered = [
        existing[key]
        for key in sorted(
            existing,
            key=lambda k: (
                str(existing[k].get("erp_version")),
                list(MODULE_ROOTS).index(str(existing[k].get("module")))
                if existing[k].get("module") in MODULE_ROOTS
                else 99,
            ),
        )
    ]
    header = (
        "# Manifiesto de la documentación oficial de Odoo (generado por\n"
        "# scripts/build_odoo_pdfs.py). Formato de origen: RST → PDF (compilado por DocuERP).\n"
        "# Licencia CC BY-SA 4.0, © Odoo S.A.; el sha256 es el del PDF generado.\n"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump({"documents": ordered}, allow_unicode=True, sort_keys=False, width=100)
    manifest_path.write_text(header + body, encoding="utf-8")
    return manifest_path


def _write_build_log(records: list[BuiltPdf], path: Path) -> Path:
    """Guarda el registro de construcción (páginas, RST, toctrees ausentes, huérfanos)."""
    import json

    payload = [
        {
            "file": r.path.name,
            "version": r.version,
            "module": r.module,
            "pdf_pages": r.pages,
            "rst_files": r.rst_files,
            "missing_toctree_entries": r.missing,
            "orphan_pages": r.orphans,
            "sha256": r.sha256,
            "commit": r.snapshot.commit,
        }
        for r in records
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
