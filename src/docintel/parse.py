"""CLI entrypoint for running the Docling parsing pipeline phase."""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any, Dict, Optional

from .config import get_parsing_settings, get_settings
from .pipeline import PipelineContext, PipelineRunner
from .parsing import ParsingPhase
from .storage import build_processing_layout, build_storage_layout, ensure_processing_layout


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


async def _run_async(*, force_reparse: bool, max_workers: Optional[int]) -> Dict[str, Any]:
    ingestion_settings = get_settings()
    storage_layout = build_storage_layout(ingestion_settings)

    parsing_settings = get_parsing_settings()
    processing_layout = build_processing_layout(parsing_settings)
    ensure_processing_layout(processing_layout)

    context = PipelineContext(
        ingestion_settings=ingestion_settings,
        parsing_settings=parsing_settings,
        storage_layout=storage_layout,
        processing_layout=processing_layout,
    )
    phase = ParsingPhase(force_reparse=force_reparse, max_workers_override=max_workers)
    runner = PipelineRunner([phase], context)
    results = await runner.run()
    parsing_result = next((result for result in results if result.name == phase.name), None)
    if not parsing_result or not parsing_result.succeeded:
        raise RuntimeError("Parsing phase failed")
    return parsing_result.details.get("report", {})


def run(*, force_reparse: bool = False, max_workers: Optional[int] = None) -> Dict[str, Any]:
    _configure_logging()
    return asyncio.run(_run_async(force_reparse=force_reparse, max_workers=max_workers))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Docling parsing pipeline")
    parser.add_argument(
        "--force-reparse",
        action="store_true",
        help="Reprocess documents even if artefacts already exist",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Override the maximum number of concurrent parsing tasks",
    )
    args = parser.parse_args(argv)

    report = run(force_reparse=args.force_reparse, max_workers=args.max_workers)
    logging.getLogger(__name__).info("Parsing completed", extra={"summary": report})


if __name__ == "__main__":  # pragma: no cover
    main()
