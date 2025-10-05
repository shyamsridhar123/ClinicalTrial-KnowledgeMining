"""Async orchestrator that drives Granite Docling parsing for downloaded documents."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional

from ..config import ParsingSettings
from ..storage import ProcessingLayout, StorageLayout
from .artifacts import ArtifactWriter
from .client import DoclingClient, DoclingClientError, DoclingParseResult
from .ocr import OcrEngine


@dataclass(slots=True)
class DocumentJob:
    """Metadata associated with a parsing job."""

    nct_id: str
    document_path: Path

    @property
    def document_name(self) -> str:
        return self.document_path.name


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".html",
    ".htm",
}


class ParsingOrchestrator:
    """Coordinate Docling parsing, OCR fallback, and artefact persistence."""

    def __init__(
        self,
        settings: ParsingSettings,
        ingestion_layout: StorageLayout,
        processing_layout: ProcessingLayout,
        docling_client: DoclingClient,
        ocr_engine: OcrEngine,
        artifact_writer: ArtifactWriter,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._settings = settings
        self._ingestion_layout = ingestion_layout
        self._processing_layout = processing_layout
        self._client = docling_client
        self._ocr = ocr_engine
        self._writer = artifact_writer
        self._logger = logger or logging.getLogger(__name__)
        # Optimize worker count based on GPU availability and system resources
        self._max_workers = self._calculate_optimal_workers(settings.max_concurrent_parses)
        self._result_cache = {}  # Simple in-memory cache for processed results

    def _calculate_optimal_workers(self, configured_workers: int) -> int:
        """Calculate optimal worker count based on system resources and GPU availability."""
        try:
            import torch  # type: ignore[import-not-found]
            import os
            
            # Base worker count on system resources
            cpu_count = os.cpu_count() or 4
            
            if torch.cuda.is_available():
                # GPU available - can handle more concurrent work
                gpu_count = torch.cuda.device_count()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                
                # Optimize for GPU throughput: 2-4 workers per GPU depending on VRAM
                if gpu_memory >= 8.0:  # 8+ GB VRAM
                    optimal_workers = min(gpu_count * 4, cpu_count)
                elif gpu_memory >= 4.0:  # 4-8 GB VRAM
                    optimal_workers = min(gpu_count * 2, cpu_count // 2)
                else:  # < 4 GB VRAM
                    optimal_workers = gpu_count
                    
                self._logger.info(
                    f"docling | GPU optimization | gpus={gpu_count} | vram={gpu_memory:.1f}GB | workers={optimal_workers}"
                )
            else:
                # CPU only - more conservative
                optimal_workers = max(1, cpu_count // 2)
                self._logger.info(f"docling | CPU-only mode | cpus={cpu_count} | workers={optimal_workers}")
            
            # Respect configured limits but optimize within bounds
            if configured_workers > 0:
                return min(configured_workers, optimal_workers)
            return optimal_workers
            
        except Exception as exc:
            self._logger.warning(f"docling | worker optimization failed: {exc} | using default=2")
            return max(1, configured_workers) if configured_workers > 0 else 2

    async def run(self, *, force_reparse: bool = False, max_workers: Optional[int] = None) -> Dict[str, Any]:
        """Process available documents and return a summary report."""

        worker_limit = max_workers or self._max_workers
        semaphore = asyncio.Semaphore(worker_limit)
        jobs = list(self._discover_jobs())
        documents_summary: List[Dict[str, Any]] = []
        stats = {
            "total_documents": len(jobs),
            "processed": 0,
            "skipped_existing": 0,
            "failed": 0,
            "ocr_used": 0,
            "chunk_count": 0,
            "total_duration_ms": 0.0,
        }

        async def _run_job(job: DocumentJob) -> None:
            async with semaphore:
                # Check cache first for identical documents
                cache_key = f"{job.document_path.stat().st_size}_{job.document_path.stat().st_mtime}"
                if not force_reparse and cache_key in self._result_cache:
                    self._logger.debug(f"docling | cache hit | {job.document_path}")
                    cached = self._result_cache[cache_key].copy()
                    cached["document"] = str(job.document_path)
                    cached["status"] = "skipped"
                    cached.setdefault("reason", "cached_result")
                    result = cached
                else:
                    result = await self._process_job(job, force_reparse=force_reparse)
                    # Cache successful results
                    if result["status"] == "processed":
                        self._result_cache[cache_key] = result.copy()
                        
                documents_summary.append(result)
                if result["status"] == "processed":
                    stats["processed"] += 1
                    stats["ocr_used"] += 1 if result.get("ocr_used") else 0
                    stats["chunk_count"] += result.get("chunk_count", 0)
                    stats["total_duration_ms"] += result.get("duration_ms", 0.0)
                elif result["status"] == "skipped":
                    stats["skipped_existing"] += 1
                elif result["status"] == "failed":
                    stats["failed"] += 1

        # Process jobs in optimized batches for better GPU utilization
        batch_size = min(worker_limit * 2, 10)  # Process in batches
        for i in range(0, len(jobs), batch_size):
            batch = jobs[i:i + batch_size]
            self._logger.info(f"docling | processing batch {i//batch_size + 1} | jobs={len(batch)}")
            await asyncio.gather(*(_run_job(job) for job in batch))
            
            # Clear cache periodically to prevent memory buildup
            if len(self._result_cache) > 100:
                self._result_cache.clear()
                self._logger.debug("docling | cleared result cache")

        stats["total_duration_ms"] = round(stats["total_duration_ms"], 2)
        report = {
            "collection_date": datetime.now(timezone.utc).isoformat(),
            "statistics": stats,
            "documents": sorted(documents_summary, key=lambda item: item["document"]),
        }
        self._write_report(report)
        return report

    def _discover_jobs(self) -> Iterable[DocumentJob]:
        documents_dir = self._ingestion_layout.documents
        if not documents_dir.exists():
            return []
        jobs: List[DocumentJob] = []
        for nct_dir in sorted(documents_dir.iterdir()):
            if not nct_dir.is_dir():
                continue
            nct_id = nct_dir.name
            for doc_path in sorted(nct_dir.glob("*")):
                if doc_path.is_dir():
                    continue
                if doc_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                jobs.append(DocumentJob(nct_id=nct_id, document_path=doc_path))
        return jobs

    async def _process_job(self, job: DocumentJob, *, force_reparse: bool) -> Dict[str, Any]:
        if not force_reparse and self._writer.exists(job.nct_id, job.document_name):
            self._logger.info(
                "docling-job=%s | skipping existing artefacts | path=%s",
                job.nct_id,
                job.document_path,
            )
            return {
                "status": "skipped",
                "document": str(job.document_path),
                "reason": "already_parsed",
            }

        start = perf_counter()
        ocr_used = False
        self._logger.info(
            "docling-job=%s | starting parse | path=%s",
            job.nct_id,
            job.document_path,
        )
        try:
            parse_result = await self._client.parse_document(document_path=job.document_path)
        except DoclingClientError as exc:
            self._logger.error("Docling parsing failed for %s: %s", job.document_path, exc)
            if not self._settings.ocr_enabled:
                return {
                    "status": "failed",
                    "document": str(job.document_path),
                    "error": str(exc),
                }
            self._logger.info(
                "docling-job=%s | attempting OCR fallback after docling failure | path=%s",
                job.nct_id,
                job.document_path,
            )
            ocr_text = await self._ocr.extract_text(job.document_path)
            if not ocr_text:
                self._logger.error(
                    "docling-job=%s | OCR fallback produced no text after docling failure | path=%s",
                    job.nct_id,
                    job.document_path,
                )
                return {
                    "status": "failed",
                    "document": str(job.document_path),
                    "error": str(exc),
                }
            ocr_used = True
            parse_result = await self._client.parse_document(
                document_path=job.document_path,
                ocr_text=ocr_text,
            )

        if _is_blank(parse_result.plain_text) and self._settings.ocr_enabled:
            ocr_text = await self._ocr.extract_text(job.document_path)
            if ocr_text:
                ocr_used = True
                self._logger.info(
                    "docling-job=%s | running OCR fallback | path=%s",
                    job.nct_id,
                    job.document_path,
                )
                try:
                    parse_result = await self._client.parse_document(
                        document_path=job.document_path,
                        ocr_text=ocr_text,
                    )
                except DoclingClientError as exc:
                    self._logger.error("Docling re-parse after OCR failed for %s: %s", job.document_path, exc)
                    return {
                        "status": "failed",
                        "document": str(job.document_path),
                        "error": str(exc),
                    }
            else:
                self._logger.warning(
                    "docling-job=%s | OCR fallback produced no text | path=%s",
                    job.nct_id,
                    job.document_path,
                )

        if not parse_result.chunks:
            # Use enhanced semantic chunking for clinical documents
            from .semantic_chunking import create_semantic_chunks
            
            parse_result.chunks = await asyncio.to_thread(
                create_semantic_chunks,
                parse_result.plain_text,
                document_id=job.nct_id,
                target_token_size=self._settings.chunk_token_size,
                overlap_tokens=self._settings.chunk_overlap,
            )
            self._logger.info(
                "docling-job=%s | generated semantic chunks | count=%d | enhanced=clinical-aware",
                job.nct_id,
                len(parse_result.chunks),
            )

        duration_ms = (perf_counter() - start) * 1000.0
        provenance = {
            "document_path": str(job.document_path),
            "nct_id": job.nct_id,
            "document_name": job.document_name,
            "size_bytes": job.document_path.stat().st_size if job.document_path.exists() else 0,
            "parse_duration_ms": round(duration_ms, 2),
            "ocr_used": ocr_used,
            "fallback_strategy": parse_result.metadata.get("status"),
            "chunk_count": len(parse_result.chunks),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "docling_metadata": parse_result.metadata,
        }

        artifact_paths = self._writer.write(job.nct_id, job.document_name, parse_result, provenance)
        # Build artifact mapping with correct labels
        artifact_map = {
            "structured": str(artifact_paths.structured),
            "markdown": str(artifact_paths.markdown),
            "html": str(artifact_paths.html),
            "text": str(artifact_paths.text),
            "tables": str(artifact_paths.tables),
            "figures": str(artifact_paths.figures_metadata),  # Correct: figures_metadata -> "figures"
            "chunks": str(artifact_paths.chunks),
            "provenance": str(artifact_paths.provenance),
        }
        self._logger.info(
            "docling-job=%s | parse complete | duration_ms=%.2f | chunks=%d | artefacts=%s",
            job.nct_id,
            round(duration_ms, 2),
            len(parse_result.chunks),
            artifact_map,
        )
        return {
            "status": "processed",
            "document": str(job.document_path),
            "nct_id": job.nct_id,
            "document_name": job.document_name,
            "duration_ms": round(duration_ms, 2),
            "ocr_used": ocr_used,
            "chunk_count": len(parse_result.chunks),
            "artefacts": artifact_map,
        }

    def _write_report(self, report: Dict[str, Any]) -> None:
        report_path = self._processing_layout.root / "parse_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json_dumps(report),
            encoding="utf-8",
        )


def _is_blank(value: Optional[str]) -> bool:
    return value is None or not value.strip()


def _fallback_chunks(text: str, *, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    """Optimized chunking with better memory efficiency and semantic boundary detection."""
    if not text or not text.strip():
        return []
    
    # For very large documents, use sentence-aware chunking for better semantic boundaries
    if len(text) > 50000:  # 50KB+ documents
        return _semantic_chunks(text, chunk_size=chunk_size, overlap=overlap)
    
    words = text.split()
    if not words:
        return []
        
    chunk_size = max(1, chunk_size)
    overlap = min(overlap, chunk_size - 1) if chunk_size > 1 else 0
    step = max(1, chunk_size - overlap)
    chunks: List[Dict[str, Any]] = []
    
    # Process chunks with better memory efficiency
    for index in range(0, len(words), step):
        window = words[index : index + chunk_size]
        if not window:
            continue
        chunk_text = " ".join(window)
        chunks.append(
            {
                "id": f"chunk-{len(chunks):04d}",  # Zero-padded for better sorting
                "text": chunk_text,
                "token_count": len(window),
                "start_word_index": index,
                "char_count": len(chunk_text),
            }
        )
    return chunks


def _semantic_chunks(text: str, *, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    """Advanced chunking that respects sentence boundaries for better semantic coherence."""
    import re
    
    # Split into sentences using improved regex
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, text.strip())
    
    if not sentences:
        return []
    
    chunks: List[Dict[str, Any]] = []
    current_chunk = []
    current_word_count = 0
    sentence_index = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)
        
        # If adding this sentence would exceed chunk size, finalize current chunk
        if current_chunk and current_word_count + sentence_word_count > chunk_size:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "id": f"chunk-{len(chunks):04d}",
                "text": chunk_text,
                "token_count": current_word_count,
                "sentence_start": sentence_index - len(current_chunk) + 1,
                "sentence_end": sentence_index,
                "char_count": len(chunk_text),
            })
            
            # Start new chunk with overlap
            if overlap > 0 and current_chunk:
                overlap_words = " ".join(current_chunk).split()[-overlap:]
                current_chunk = overlap_words
                current_word_count = len(overlap_words)
            else:
                current_chunk = []
                current_word_count = 0
        
        current_chunk.append(sentence)
        current_word_count += sentence_word_count
        sentence_index += 1
    
    # Add final chunk if it has content
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({
            "id": f"chunk-{len(chunks):04d}",
            "text": chunk_text,
            "token_count": current_word_count,
            "sentence_start": sentence_index - len(current_chunk) + 1,
            "sentence_end": sentence_index,
            "char_count": len(chunk_text),
        })
    
    return chunks


def json_dumps(payload: Dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True)


__all__ = ["ParsingOrchestrator", "DocumentJob"]
