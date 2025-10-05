#!/usr/bin/env python3
"""Compatibility CLI that delegates to the official docintel.embed entrypoint."""

from __future__ import annotations

from pathlib import Path
import sys


# Ensure the package can be resolved when the script is executed from the repo root.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from docintel.embed import main as _embed_main  # noqa: E402 - import after path mutation


def main() -> None:
    """Invoke the canonical embedding CLI."""

    _embed_main()


if __name__ == "__main__":
    main()