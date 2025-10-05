"""CLI entry point for ingesting clinical vocabularies into the repository graph."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from hashlib import new as hash_new
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import aiohttp
import psycopg

from docintel.config import get_config
from docintel.repository import (
    RepositoryIngestionOrchestrator,
    SnomedLoader,
    UMLSLoader,
    RxNormLoader,
)

LOGGER = logging.getLogger("docintel.repository.ingest")


@dataclass
class DownloadResult:
    """Represents an archive downloaded from a remote source."""

    path: Path
    checksum: str
    size_bytes: int


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest UMLS, RxNorm, and SNOMED vocabularies")
    parser.add_argument("--umls-root", type=str, help="Directory containing UMLS RRF exports")
    parser.add_argument("--rxnorm-root", type=str, help="Directory containing RxNorm RRF exports")
    parser.add_argument("--snomed-root", type=str, help="Directory containing SNOMED CT RF2 exports")
    parser.add_argument("--umls-uri", type=str, help="HTTPS URI to download a pre-authorised UMLS subset archive")
    parser.add_argument("--rxnorm-uri", type=str, help="HTTPS URI to download an RxNorm archive")
    parser.add_argument("--snomed-uri", type=str, help="HTTPS URI to download a SNOMED CT archive")
    parser.add_argument("--dry-run", action="store_true", help="Parse sources without writing to the database")
    parser.add_argument("--limit-nodes", type=int, help="Limit the number of nodes processed per vocabulary (testing)")
    parser.add_argument("--limit-edges", type=int, help="Limit the number of edges processed per vocabulary (testing)")
    parser.add_argument(
        "--vocab",
        choices=["umls", "rxnorm", "snomed"],
        action="append",
        help="Restrict ingestion to specific vocabularies",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


async def _download_archive(
    url: str,
    destination: Path,
    *,
    chunk_size: int,
    timeout: float,
    algorithm: str,
) -> DownloadResult:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported download URI: {url}")
    LOGGER.info("Downloading %s -> %s", url, destination)
    timeout_cfg = aiohttp.ClientTimeout(total=timeout)
    digest = hash_new(algorithm)
    size_bytes = 0
    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        async with session.get(url) as response:
            response.raise_for_status()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as handle:
                async for chunk in response.content.iter_chunked(chunk_size):
                    handle.write(chunk)
                    digest.update(chunk)
                    size_bytes += len(chunk)
    return DownloadResult(path=destination, checksum=digest.hexdigest(), size_bytes=size_bytes)


def _extract_archive(archive: Path, staging_dir: Path) -> Path:
    staging_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Extracting %s into %s", archive, staging_dir)
    suffixes = archive.suffixes
    if archive.suffix == ".zip" or suffixes and suffixes[-1] == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staging_dir)
    elif suffixes[-1] == ".gz" and len(suffixes) >= 2 and suffixes[-2] == ".tar":
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(staging_dir)
    elif suffixes[-1] in {".tgz", ".tar"}:
        mode = "r:gz" if suffixes[-1] == ".tgz" else "r"
        with tarfile.open(archive, mode) as tf:
            tf.extractall(staging_dir)
    elif suffixes[-1] == ".gz":
        import gzip

        target = staging_dir / archive.stem
        with gzip.open(archive, "rb") as source, target.open("wb") as sink:
            sink.write(source.read())
    else:
        raise ValueError(f"Unsupported archive format: {archive}")
    items = list(staging_dir.iterdir())
    if len(items) == 1 and items[0].is_dir():
        return items[0]
    return staging_dir


def _coerce_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    return path if path.exists() else None


def _build_loader(vocabulary: str, root: Path):
    if vocabulary == "umls":
        return UMLSLoader(root)
    if vocabulary == "rxnorm":
        return RxNormLoader(root)
    if vocabulary == "snomed":
        return SnomedLoader(root)
    raise ValueError(f"Unsupported vocabulary: {vocabulary}")


def _resolve_sources(args: argparse.Namespace, settings) -> Dict[str, Dict[str, object]]:
    staging_root = settings.ensure_staging_root()
    mapping = {
        "umls": (args.umls_root, args.umls_uri, settings.umls_source_root, settings.umls_download_uri),
        "rxnorm": (args.rxnorm_root, args.rxnorm_uri, settings.rxnorm_source_root, settings.rxnorm_download_uri),
        "snomed": (args.snomed_root, args.snomed_uri, settings.snomed_source_root, settings.snomed_download_uri),
    }

    sources: Dict[str, Dict[str, object]] = {}
    for vocab, (cli_path, cli_uri, cfg_path, cfg_uri) in mapping.items():
        if args.vocab and vocab not in args.vocab:
            continue
        resolved_path = _coerce_path(cli_path) or settings.resolve_path(cfg_path)
        sources[vocab] = {
            "path": resolved_path,
            "uri": cli_uri or cfg_uri,
            "staging": staging_root / vocab,
        }
    return sources


def _ensure_source(vocab: str, record: Dict[str, object], settings) -> Dict[str, object]:
    existing_path: Optional[Path] = record["path"]
    uri: Optional[str] = record["uri"]
    staging: Path = record["staging"]
    staging.mkdir(parents=True, exist_ok=True)

    if existing_path and existing_path.exists():
        LOGGER.info("Using existing %s source at %s", vocab, existing_path)
        return {"root": existing_path, "download_metadata": {}}

    if not uri:
        raise FileNotFoundError(
            f"No source directory or download URI provided for {vocab}. "
            f"Supply --{vocab}-root or --{vocab}-uri."
        )

    filename = Path(urlparse(uri).path).name or f"{vocab}.archive"
    archive_path = staging / filename
    download_result = asyncio.run(
        _download_archive(
            uri,
            archive_path,
            chunk_size=settings.download_chunk_size_bytes,
            timeout=settings.download_timeout_seconds,
            algorithm=settings.checksum_algorithm,
        )
    )
    extracted_root = staging / f"{archive_path.stem}_extracted"
    root = _extract_archive(download_result.path, extracted_root)
    return {
        "root": root,
        "download_metadata": {
            "archive_path": str(download_result.path),
            "archive_checksum": download_result.checksum,
            "archive_size_bytes": download_result.size_bytes,
            "downloaded_at": datetime.utcnow().isoformat(),
            "source_uri": uri,
        },
    }


def main() -> int:
    args = parse_args()
    _configure_logging(args.verbose)

    config = get_config()
    repo_settings = config.repository_ingestion
    sources = _resolve_sources(args, repo_settings)

    selected = [v for v in ("umls", "rxnorm", "snomed") if v in sources]
    if args.vocab:
        selected = [v for v in selected if v in args.vocab]

    if not selected:
        LOGGER.error("No vocabularies selected for ingestion")
        return 1

    dry_run = args.dry_run or repo_settings.dry_run
    connection: Optional[psycopg.Connection] = None
    try:
        if not dry_run:
            connection = psycopg.connect(config.docintel_dsn)

        orchestrator = RepositoryIngestionOrchestrator(
            connection,
            schema=config.vector_db.schema,
            batch_size=repo_settings.batch_size,
            edge_batch_size=repo_settings.edge_batch_size,
            checksum_algorithm=repo_settings.checksum_algorithm,
            processing_stage="vocab_ingestion",
        )

        overall_success = True
        for vocab in selected:
            try:
                resolved = _ensure_source(vocab, sources[vocab], repo_settings)
            except Exception as exc:  # pragma: no cover - runtime failure path
                LOGGER.exception("Failed to prepare %s source: %s", vocab, exc)
                overall_success = False
                continue

            loader = _build_loader(vocab, resolved["root"])
            metadata = resolved.get("download_metadata", {})

            try:
                result = orchestrator.ingest(
                    loader,
                    dry_run=dry_run,
                    node_limit=args.limit_nodes,
                    edge_limit=args.limit_edges,
                    metadata=metadata,
                )
            except Exception as exc:  # pragma: no cover - runtime failure path
                LOGGER.exception("Ingestion failed for %s: %s", vocab, exc)
                overall_success = False
                if connection is not None:
                    orchestrator._record_processing_log(  # pylint: disable=protected-access
                        vocab,
                        status="failed",
                        message=f"Repository ingestion failed for {loader.describe()}",
                        metadata={"error": str(exc), **metadata},
                    )
            else:
                LOGGER.info(
                    "%s | nodes=%d edges=%d dry_run=%s",
                    result.vocabulary,
                    result.nodes_processed,
                    result.edges_processed,
                    result.dry_run,
                )

        if not overall_success:
            return 2
        return 0
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    sys.exit(main())
