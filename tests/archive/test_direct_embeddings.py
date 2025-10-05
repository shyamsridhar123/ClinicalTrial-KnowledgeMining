#!/usr/bin/env python3
"""Deprecated manual embedding demo.

This helper moved to ``scripts/direct_embeddings_demo.py`` to keep the pytest
suite focused on automated coverage. Invoke the new location directly when you
need to generate ad-hoc embeddings from the command line.
"""

from __future__ import annotations

import sys


def _print_redirect() -> None:
    message = (
        "[docintel] The direct embedding demo has moved to "
        "scripts/direct_embeddings_demo.py.\n"
        "Run `pixi run -- python scripts/direct_embeddings_demo.py` instead."
    )
    print(message, file=sys.stderr)


if __name__ == "__main__":
    _print_redirect()
    sys.exit(1)