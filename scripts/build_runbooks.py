"""Genera los PDF/DOCX de los runbooks sintéticos y el manifiesto ``internal.yaml``.

Equivale a ``docuerp data build-runbooks``. Uso::

    PYTHONPATH=src python scripts/build_runbooks.py
"""

from __future__ import annotations

import logging

from docuerp.data.runbooks import build_runbooks


def main() -> None:
    """Construye los runbooks e imprime las rutas generadas."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for path in build_runbooks():
        print(path)


if __name__ == "__main__":
    main()
