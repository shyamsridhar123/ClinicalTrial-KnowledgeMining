"""CLI entrypoint for running the ingestion pipeline."""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any, Dict

from .ingestion import IngestionPhase
from .config import get_settings
from .pipeline import PipelineContext, PipelineRunner
from .storage import build_storage_layout


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


async def _run_async(max_studies: int) -> Dict[str, Any]:
    settings = get_settings()
    storage = build_storage_layout(settings)
    context = PipelineContext(
        ingestion_settings=settings,
        storage_layout=storage,
    )
    runner = PipelineRunner([IngestionPhase(max_studies=max_studies)], context)
    results = await runner.run()
    ingestion_result = next((result for result in results if result.name == "ingestion"), None)
    if not ingestion_result or not ingestion_result.succeeded:
        raise RuntimeError("Ingestion phase failed")
    report = ingestion_result.details.get("report", {})
    return report


def run(max_studies: int = 25) -> Dict[str, Any]:
    """Execute the ingestion pipeline synchronously."""

    _configure_logging()
    return asyncio.run(_run_async(max_studies=max_studies))


def main(argv: list[str] | None = None) -> None:
    """Console script entry point."""

    parser = argparse.ArgumentParser(description="ClinicalTrials.gov ingestion pipeline")
    parser.add_argument(
        "--max-studies",
        type=int,
        default=25,
        help="Maximum number of studies to ingest in a single run (default: 25)",
    )
    args = parser.parse_args(argv)

    report = run(max_studies=args.max_studies)
    logging.getLogger(__name__).info("Ingestion completed", extra={"summary": report})


if __name__ == "__main__":  # pragma: no cover
    main()
