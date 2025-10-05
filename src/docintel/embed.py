"""CLI entrypoint for generating embeddings over parsed chunks."""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any, Dict, Optional

from .config import get_embedding_settings, get_parsing_settings, get_settings
from .embeddings import EmbeddingPhase
from .pipeline import PipelineContext, PipelineRunner
from .storage import (
    build_embedding_layout,
    build_processing_layout,
    build_storage_layout,
    ensure_embedding_layout,
    ensure_storage_layout,
    ensure_processing_layout,
)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


async def _run_async(
    *,
    force_reembed: bool,
    batch_size: Optional[int],
    quantization_encoding: Optional[str],
    store_float32: Optional[bool],
) -> Dict[str, Any]:
    """Execute the embedding pipeline phase using shared pipeline abstractions."""

    ingestion_settings = get_settings()
    storage_layout = build_storage_layout(ingestion_settings)
    ensure_storage_layout(storage_layout)

    parsing_settings = get_parsing_settings()
    processing_layout = build_processing_layout(parsing_settings)
    ensure_processing_layout(processing_layout)

    base_embedding_settings = get_embedding_settings()
    overrides: Dict[str, Any] = {}
    if batch_size is not None:
        overrides["embedding_batch_size"] = batch_size
    if quantization_encoding is not None:
        overrides["embedding_quantization_encoding"] = quantization_encoding
    if store_float32 is not None:
        overrides["embedding_quantization_store_float32"] = store_float32

    if overrides:
        embedding_settings = base_embedding_settings.model_copy(update=overrides)
    else:
        embedding_settings = base_embedding_settings
    embedding_settings.ensure_embedding_storage()

    embedding_layout = build_embedding_layout(embedding_settings)
    ensure_embedding_layout(embedding_layout)

    context = PipelineContext(
        ingestion_settings=ingestion_settings,
        parsing_settings=parsing_settings,
        embedding_settings=embedding_settings,
        storage_layout=storage_layout,
        processing_layout=processing_layout,
        embedding_layout=embedding_layout,
    )

    phase = EmbeddingPhase(force_reembed=force_reembed, batch_size_override=batch_size)
    runner = PipelineRunner([phase], context)
    results = await runner.run()
    embedding_result = next((result for result in results if result.name == phase.name), None)
    if not embedding_result or not embedding_result.succeeded:
        raise RuntimeError("Embedding phase failed")

    report = embedding_result.details.get("report", {})
    stats = report.get("statistics", {})

    return {
        "status": "success",
        "documents_processed": int(stats.get("documents_processed", 0)),
        "documents_skipped": int(stats.get("documents_skipped", 0)),
        "documents_failed": int(stats.get("documents_failed", 0)),
        "total_embeddings": int(stats.get("chunks_embedded", 0)),
        "report": report,
    }


def run(
    *,
    force_reembed: bool = False,
    batch_size: Optional[int] = None,
    quantization_encoding: Optional[str] = None,
    store_float32: Optional[bool] = None,
) -> Dict[str, Any]:
    _configure_logging()
    return asyncio.run(
        _run_async(
            force_reembed=force_reembed,
            batch_size=batch_size,
            quantization_encoding=quantization_encoding,
            store_float32=store_float32,
        )
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate embeddings for parsed clinical trial documents")
    parser.add_argument(
        "--force-reembed",
        action="store_true",
        help="Recompute embeddings even if vector artefacts already exist",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the embedding batch size when invoking Modular MAX",
    )
    parser.add_argument(
        "--quantization-encoding",
        choices=["none", "bfloat16", "int8"],
        default=None,
        help=(
            "Override the quantization encoding for persisted embeddings. "
            "Defaults to configuration value when omitted."
        ),
    )
    parser.add_argument(
        "--store-float32",
        dest="store_float32",
        action="store_true",
        help="Persist float32 vectors alongside quantized payloads.",
    )
    parser.add_argument(
        "--no-store-float32",
        dest="store_float32",
        action="store_false",
        help="Omit float32 vectors when quantization is enabled to minimise storage footprint.",
    )
    parser.set_defaults(store_float32=None)
    args = parser.parse_args(argv)

    report = run(
        force_reembed=args.force_reembed,
        batch_size=args.batch_size,
        quantization_encoding=args.quantization_encoding,
        store_float32=args.store_float32,
    )
    logging.getLogger(__name__).info("Embedding completed", extra={"summary": report})


if __name__ == "__main__":  # pragma: no cover
    main()
