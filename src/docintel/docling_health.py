"""CLI helper to verify Granite Docling availability via Modular MAX."""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Optional

from .config import get_parsing_settings
from .parsing.client import DoclingClient


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


async def _run_async(*, verbose: bool, client: Optional[DoclingClient] = None) -> bool:
    settings = get_parsing_settings()
    docling_client = client or DoclingClient(settings)

    if verbose:
        logging.getLogger(__name__).info(
            "Checking Docling endpoint",
            extra={
                "base_url": str(settings.docling_max_base_url),
                "model": settings.docling_model_name,
            },
        )

    healthy = await docling_client.health_check()
    logger = logging.getLogger(__name__)
    if healthy:
        logger.info(
            "Docling MAX endpoint is reachable",
            extra={
                "base_url": str(settings.docling_max_base_url),
                "model": settings.docling_model_name,
            },
        )
    else:  # pragma: no cover - relies on network failures
        logger.error(
            "Docling MAX endpoint is not reachable",
            extra={
                "base_url": str(settings.docling_max_base_url),
                "model": settings.docling_model_name,
            },
        )
    return healthy


def run(*, verbose: bool = True, client: Optional[DoclingClient] = None) -> bool:
    _configure_logging()
    return asyncio.run(_run_async(verbose=verbose, client=client))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Docling MAX health check")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational logs and only emit errors",
    )
    args = parser.parse_args(argv)

    verbose = not args.quiet
    result = run(verbose=verbose)
    if not result:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
