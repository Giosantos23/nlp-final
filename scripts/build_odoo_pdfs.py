"""Construye los 12 PDF de la documentación oficial de Odoo (6 módulos × 2 versiones).

Uso::

    python scripts/build_odoo_pdfs.py                      # todo
    python scripts/build_odoo_pdfs.py --versions 18.0      # solo una versión
    python scripts/build_odoo_pdfs.py --modules compras,ventas

Requisitos: el corpus RST en ``data/raw/odoo-docs/{17.0,18.0}`` (sparse checkout de
``odoo/documentation``) y Chromium de Playwright (``playwright install chromium``).

Salidas:

- ``data/build/pdf/odoo-{versión}-{módulo}.pdf``
- ``data/build/pdf/odoo_build.json`` (páginas, archivos RST, entradas de toctree ausentes)
- ``data/manifest/odoo.yaml`` (manifiesto con sha256 de cada PDF)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from docuerp.ingest.odoo_pdf import MODULE_ROOTS, ODOO_SNAPSHOTS, build_odoo_pdf_records


def _csv(value: str) -> list[str]:
    """Convierte ``"a,b"`` en ``["a", "b"]`` (ignora vacíos)."""
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada; devuelve el código de salida del proceso."""
    parser = argparse.ArgumentParser(description="RST de Odoo → PDF por módulo y versión")
    parser.add_argument("--versions", default=",".join(sorted(ODOO_SNAPSHOTS)))
    parser.add_argument("--modules", default=",".join(MODULE_ROOTS))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    versions = _csv(args.versions)
    modules = _csv(args.modules)
    unknown = sorted(set(modules) - set(MODULE_ROOTS)) + sorted(set(versions) - set(ODOO_SNAPSHOTS))
    if unknown:
        print(f"Valores desconocidos: {', '.join(unknown)}", file=sys.stderr)
        return 2

    started = time.perf_counter()
    try:
        records = build_odoo_pdf_records(versions=versions, modules=modules)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started

    print(f"\n{'Archivo':<34} {'RST':>4} {'Págs':>5} {'Ausentes':>8}")
    for record in records:
        name, missing = record.path.name, len(record.missing)
        print(f"{name:<34} {record.rst_files:>4} {record.pages:>5} {missing:>8}")
    total_pages = sum(r.pages for r in records)
    print(f"\n{len(records)} PDF, {total_pages} páginas en {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
