# Modular AI Acceleration Integration Analysis

## Purpose

Document the outcome of evaluating Modular MAX, Mojo, and Mammoth for the clinical trial knowledge mining platform. The investigation focused on accelerating IBM Granite Docling 258M parsing, downstream clinical NLP, and embedding generation while preserving compliance requirements from the TRD (`docs/Clinical_Trial_Knowledge_Mining_TRD_Modular.md`).

## Executive Summary

- **Granite Docling parsing**: The Modular MAX VLM path increased latency for large PDFs (40–60 s per request) and occasionally returned incomplete artefacts. The standalone Docling SDK, executed inside the Pixi environment with CUDA, consistently produced full outputs in ~87 s for an 88-page protocol and scales linearly. The parsing pipeline therefore stays on the SDK path.
- **MAX usage today**: MAX remains valuable for text-only workloads (structured prompts over parsed chunks, embedding generation) because of its OpenAI-compatible API and local execution guarantees. Operators can continue launching MAX via `scripts/start_max_server.sh` when those capabilities are required, while broader reliance on managed OpenAI/Anthropic endpoints is covered by the revised TRD and the medical graph integration plan.
- **Mojo & Mammoth**: Both remain on the roadmap for future clinical NLP acceleration and orchestration experiments. No Mojo kernels or Mammoth deployments are shipping yet; the team will revisit once the Docling + NLP pipeline stabilises.
- **Environment discipline**: All benchmarking and operational runs execute under Pixi (`pixi run …`) per Modular guidance. CUDA availability is verified via `pixi run -- nvidia-smi` before invoking Docling.

## Key Findings

1. **Parsing throughput**
   - SDK path avoids image re-encoding overhead introduced by the MAX VLM integration and keeps table/figure outputs intact.
   - Disabling OCR and page image generation in `PdfPipelineOptions` lowered runtime by ~20% with no loss in table fidelity for born-digital PDFs.
   - GPU utilisation (RTX A500) stayed between 88–92% with `max_workers=1`; higher concurrency caused VRAM pressure without significant speedup.

2. **MAX for ancillary tasks**
   - The health-check script (`pixi run -- env PYTHONPATH=src python -m docintel.docling_health`) still points at `http://localhost:8000/v1` to confirm MAX availability when operators need it.
   - Embedding models served through MAX benefit from adaptive batching and quantisation (`bfloat16` until `q4_k` weights ship), aligning with Modular’s documentation.
   - The `openai` Python client remains a convenient HTTP shim; direct calls to OpenAI or Anthropic services are now policy-approved for other stacks but should not be mixed into MAX benchmarking so results stay reproducible.

3. **Future acceleration opportunities**
   - Mojo kernels for UMLS/SNOMED matching and adverse-event scoring are likely to deliver gains once entity extraction workloads become the throughput bottleneck.
   - Mammoth-managed GPU pools will matter when document volume exceeds the single-node capacity; configuration snippets from the TRD are retained but not yet applied.

## Recommended Operating Model

| Pipeline Stage | Current Approach | Modular Integration Status | Notes |
| --- | --- | --- | --- |
| Ingestion | `docintel.ingest` via Pixi | N/A | Focus remains on ClinicalTrials.gov collection and checksum validation. |
| Parsing | Docling SDK (`docintel.parsing.client`) | MAX optional (health checks only) | CUDA acceleration enabled via PyTorch; MAX VLM disabled due to latency. |
| Clinical NLP | scispaCy/medspaCy (planned) | Mojo kernels **planned** | Build after parsing stabilises. |
| Embeddings/Search | Hugging Face transformers (planned) | MAX embeddings viable | Evaluate MAX embeddings to exploit batching/quantisation. |
| Orchestration | Async workers | Mammoth **planned** | Keep design snippets ready for scale-out stage. |

## Operational Guidance

1. **Parsing**
   - Run via Pixi: `pixi run -- env PYTHONPATH=src DOCINTEL_STORAGE_ROOT=… DOCINTEL_PROCESSED_STORAGE_ROOT=… python -m docintel.parse --max-workers=1`.
   - Monitor logs for the `docling | using fast standalone SDK` banner to verify the intended code path.
   - Enable OCR (`DOCINTEL_DOCLING_ENABLE_OCR=true`) only for image-only studies to avoid CPU regressions.

2. **Optional MAX tasks**
   - Start the server with `pixi run -- bash scripts/start_max_server.sh` or `pixi run -- max-serve` (the Pixi task defined in `pixi.toml`).
   - Use the `openai` client with `base_url="http://localhost:8000/v1"` for embeddings or prompt-based analysis of parsed chunks.
   - Shut down MAX with `pkill -f "max serve"` once ancillary jobs finish to free GPU memory.

3. **Benchmarking**
   - Maintain a small corpus of representative protocols (digital + scanned) to compare SDK and MAX runs when new releases drop.
   - Capture metrics: total runtime, page count, chunk count, table/figure extraction accuracy, GPU utilisation (`nvidia-smi` snapshot).

## Next Steps

1. Harden the Docling SDK pipeline with automated regression tests (synthetic + anonymised clinical PDFs).
2. Prototype MAX-powered embedding generation and compare against the existing transformer baseline.
3. Scope Mojo kernels for entity context detection once the NLP stack lands in `src/docintel/nlp/`.
4. Revisit Mammoth deployment scripts when parsing demand exceeds single-node throughput.

## References

- Modular Pixi guidance: https://docs.modular.com/mojo/manual/get-started/
- MAX serving documentation: https://docs.modular.com/max/get-started/
- Granite Docling structured output notes: https://docs.modular.com/max/models/granite-docling/

_Last updated: 2025-09-25_