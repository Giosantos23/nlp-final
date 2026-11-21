"""Ejecuta el banco red-team de DocuERP (equivale a ``docuerp security redteam``).

Uso (desde la raíz del repositorio):
    python security/run_redteam.py --guards-only --embedder fake
    python security/run_redteam.py            # e2e contra docuerp.agent.service
"""

from __future__ import annotations

import sys

from typer.main import get_command

from docuerp.security.cli import app


def main(argv: list[str] | None = None) -> None:
    """Reenvía los argumentos al subcomando ``redteam`` de la CLI de seguridad."""
    args = sys.argv[1:] if argv is None else argv
    get_command(app).main(["redteam", *args], standalone_mode=True)


if __name__ == "__main__":
    main()
