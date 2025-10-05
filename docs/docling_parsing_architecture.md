# Docling Parsing Architecture

## Overview

This document captures the design for the multimodal parsing stage that converts raw clinical trial documents into structured artifacts, as required by the Clinical Trial Knowledge Mining TRD ("TRD"). The stage builds on the ingestion pipeline and prepares downstream NLP, knowledge graph, and search components.

- **Scope:** PDFs and other supported formats output by the ingestion step (ClinicalTrials.gov ProvidedDocs).
- **Goal:** Deliver JSON, Markdown, HTML, and provenance metadata while preserving layout hierarchy and enabling OCR fallback.
- **Non-goals:** Entity extraction, embeddings, and knowledge graph population (covered by subsequent stages).

## Requirements Traceability

| Requirement | Source | Design Hook |
| --- | --- | --- |
| Preserve layout hierarchy, formulas, figures | TRD §5.2 "Multimodal Parsing & Extraction"[^1] | Docling SDK GPU pipeline + structured artifact writer |
| Execute through Pixi-managed environment | TRD §6.1 "Environment Baseline"[^1] | All commands invoked via `pixi run -- …`; docs include reminders |
| Use Modular MAX with OpenAI-compatible endpoints | TRD §6.2 "Modular MAX Integration"[^1], Modular MAX Quickstart[^2] | MAX retained for text/embedding workloads; parsing client keeps optional health probe |
| Provide OCR fallback for scanned documents | TRD §5.1 "Document Ingestion & Preprocessing"[^1] | Orchestrator integrates Tesseract pipeline |
| Capture provenance + audit trail | TRD §§7.1 & 10 | Provenance records alongside artefacts |
| Output ready for vectorization | TRD §5.4 | Chunk manifest recorded for downstream embedding stage |

[^1]: *Clinical Trial Knowledge Mining Platform — Technical Requirements Document (Modular-Accelerated Edition)*.
[^2]: Modular MAX Quickstart, "Start a model endpoint" and "Run inference with the endpoint" (<https://docs.modular.com/max/get-started/>)

## High-Level Data Flow

1. **Discovery:** Enumerate downloaded documents from ingestion storage (`data/ingestion/pdfs/<NCT>/<file>`).
2. **Scheduling:** Push parse jobs onto an async queue; default concurrency configurable (aligns with GPU resources).
3. **Parsing:** For each job, invoke the Granite Docling 258M SDK directly. The standalone SDK path eliminates the MAX VLM overhead while keeping GPU acceleration through PyTorch + CUDA.
4. **OCR Fallback:** If the SDK signals a structured extraction failure or the document is flagged as scanned, run Tesseract OCR pipeline and retry Docling with OCR output as input.
5. **Resilience:** On specific Docling core exceptions (e.g., `basic_string::at` from table structure assembly), retry conversion with table-structure disabled. If the retry also fails, fall back to GPU-friendly PyMuPDF text extraction so downstream embedding can proceed. All fallbacks are flagged in metadata so operators can quantify impact and raise upstream bug reports.
5. **Artifact Generation:** Persist structured JSON, Markdown, HTML, plus page-level text, figure manifests, and table CSV extracts.
6. **Provenance + Metrics:** Update per-study provenance JSON (inputs, timestamps, versions, MAX model hash, GPU info) and append to processing report.
7. **Chunk Manifest:** Produce chunk definitions (1,000 tokens, 200 overlap) for the embedding stage, referencing section hierarchy.
8. **Completion Hooks:** Emit events/logs for downstream pipelines and optional webhook notifications.

## Module Breakdown

### `docintel.docling_client`

- Encapsulates Docling SDK initialisation.
- Features:
  - Configurable pipeline options (table extraction, OCR toggle, image generation) exposed via `ParsingSettings`.
  - Automatic device selection (prefers CUDA) and logging of the active GPU per Modular guidance.
  - Optional MAX health probes retained for operators who still run the MAX server for auxiliary tasks.
  - Response validation to detect parse success vs. structured error payloads.
  - Automatic retry with a table-structure-disabled converter when Docling triggers the upstream `basic_string::at` bug; metadata captures the fallback type for downstream awareness.

### `docintel.parser`

- Orchestrates document processing using `DoclingClient` (SDK).
- Responsibilities:
  - Resolve file paths, determine if OCR pre-processing is required (e.g., via ingestion metadata or heuristics such as low text density).
  - Manage asynchronous concurrency with `asyncio.TaskGroup` or `asyncio.Semaphore` to respect GPU throughput.
  - Invoke OCR fallback pipeline when Docling returns `unsupported_format`, `image_only`, or timeouts.
  - When Docling raises a hard failure, automatically trigger the OCR pipeline and accept a text-only artefact so the run still completes; provenance captures whether the result came from Docling, PyMuPDF, or OCR.
  - Write outputs via `ArtifactWriter` helpers.
  - Update per-document `ParseResult` dataclasses capturing status, durations, and generated artifact paths.

### `docintel.ocr`

- Wraps Tesseract OCR invocation (executed within Pixi environment) to produce intermediate searchable PDFs or plain text for Docling retries.
- Tracks OCR statistics (pages processed, confidence scores).

### `docintel.artifacts`

- Provides helpers for writing structured outputs:
  - JSON: full document tree, metadata, bounding boxes.
  - Markdown: human-readable summary preserving heading levels and tables.
  - HTML: original layout w/ figure references, alt-text placeholders.
  - Tables: optional CSV exports keyed by table ID.
  - Chunk manifest: list of sections with token spans and metadata fields required for embeddings.
- Enforces atomic writes via temporary files under `processing/temp`.

### CLI Entry Point `docintel.parse`

- Command-line interface consistent with ingestion CLI.
- Key options: `--max-workers`, `--nct-filter`, `--force-reparse`, `--skip-ocr`, `--output-root`.
- Provides progress logging and summary stats similar to ingestion (counts, durations, OCR usage, failure reasons).
- Executed via `pixi run -- python -m docintel.parse` per runtime policy.
- Logs explicitly note when the SDK falls back to CPU because CUDA is unavailable, aligning with expectations set in the README.

## Configuration Extensions

- Augment `DataCollectionSettings` or introduce a dedicated `ParsingSettings` (via Pydantic) with the following fields:

- `docling_model_name` (default `ibm-granite/granite-docling-258M`).
- `docling_images_scale`, `docling_generate_page_images`, etc., to expose performance levers surfaced in the SDK helper.
- `max_concurrent_parses` (default 1–2 on RTX A500, tunable per GPU memory).
- `ocr_enabled` flag + `tesseract_langs` (default `eng`).
- `processed_storage_root` (default `data/processing`).
- `cache_dir` for intermediate OCR conversions.
- `chunk_token_size` / `chunk_overlap` (default 1,000 / 200 per TRD §5.4).

Settings reuse Pixi environment variables with `DOCINTEL_` prefix (e.g., `DOCINTEL_DOCLING_MAX_BASE_URL` for optional MAX health checks).

## Storage Layout

All parsing outputs live under `data/processing` (override via settings):

```
data/
  ingestion/
    pdfs/
    metadata/
    ...
  processing/
    structured/             # JSON exports (doc tree)
    markdown/
    html/
    tables/
    text/
    ocr/
    chunks/                 # chunk manifest JSONL per document
    provenance/             # per-document provenance records
    logs/
    temp/
  reports/
    parse_report.json       # cumulative job summaries
```

Each document stores artifacts within a subdirectory named `<NCT_ID>/<DOCUMENT_STEM>/…` to accommodate multiple files per study. Provenance files include doc hash, ingestion source, parse timestamps, Docling package version, OCR usage, and software versions.

## Error Handling & Resilience

- **Conversion Errors:** Map SDK exceptions to typed errors (e.g., `DoclingConversionError`, `DoclingTimeoutError`). Provide exponential backoff within limits.
- **Timeouts:** Detect inference timeouts and allow configurable retry counts before triggering OCR fallback.
- **Partial Outputs:** If Docling returns partial content, persist it with warnings and mark job as `partial_success`.
- **Validation:** Ensure generated artifacts are non-empty; otherwise fail the job and capture diagnostics.
- **Idempotency:** Skip processing when existing artifacts are newer than the source file unless `--force-reparse` is supplied.

## Observability & Reporting

- Structured logs with document identifiers, durations, GPU utilization (queried via `nvidia-smi` when available).
- Metrics emitted to Prometheus exporters (to be integrated later): parse duration, OCR rate, success/failure counts.
- `parse_report.json` mirrors ingestion report schema with additional fields: `docling_latency_ms`, `ocr_used`, `artifact_bytes`, `chunk_count`, and the detected execution device (`cuda` vs `cpu`).
- Optional webhook notifications to `/webhook` endpoint with job status payloads.

## Performance Considerations

- Respect GPU memory by limiting concurrent parses (`max_concurrent_parses`) and optionally chunking long documents before conversion.
- Keep OCR disabled by default to avoid CPU bottlenecks; enable only for image-only studies.
- Consider pre-warming the Docling cache for predictable startup latency. MAX warm-up is optional and only needed if the operator runs auxiliary services through MAX.
- Monitor parse reports for the new `fallback` metadata field (values include `disable_table_structure`, `pymupdf_fallback`, and `ocr_fallback`) to quantify resilience behaviour and to drive upstream Docling tickets when needed.

## Security & Compliance

- Ensure processed artifacts inherit ingestion storage encryption (filesystem + potential object storage sync down the line).
- Redact PHI/PII in logs and reports; only store hashed identifiers when necessary.
- Retain provenance and logs ≥7 years per TRD §10; include processing user identity for audit trails.

## Future Extensions

- Integrate Mammoth orchestration for scaling across multiple MAX instances.
- Add Docling model health probes and automated failover to legacy pipeline.
- Incorporate GPU profiling via NVTX markers when invoking Mojo kernels for NLP stage.

## Next Steps

1. Implement configuration extensions and storage helpers.
2. Build `DoclingClient`, OCR wrapper, artifact writers, and orchestrator modules.
3. Wire up `docintel.parse` CLI and integrate with ingestion outputs.
4. Add unit/integration tests (MAX client mocks, OCR fallback simulation).
5. Update documentation and run end-to-end validation via Pixi-managed commands.
